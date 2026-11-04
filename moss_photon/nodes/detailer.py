"""MossPhotonIdentityDetailer (SPEC 4.2 item 3, 5 and 7; MP-58): redraw small faces with the character LoRA, paste them
back through a feathered ellipse, keep the new face only when ArcFace says it is closer to the references.

Thin node. The pipeline lives in ``finishing/face_detailer.py`` (numpy only, fakes in unit tests); the detector, the
ArcFace scorer and the diffusion redraw live in ``finishing/backends.py``. No heavy import at module level: ultralytics,
insightface, numpy, torch and comfy are all imported inside functions, so the pack loads without any of them. A missing
face detector disables the node with a clear message (``missing_guard_dependency``) when it runs; nothing else is affected.

The incoming MODEL is never mutated: sampling runs on ``model.clone()``. The prompt is only the face plus the trigger
word of the character in the identity profile; the profile's text is not written to the report (sha256 only).
"""
import logging
import time

from ..core import adapter, canon, consts, errors, fingerprint
from ..core import report as report_mod
from ..core import solver as core_solver
from ..finishing import deps

_log = logging.getLogger("MossPhoton")

NODE_ID = "MossPhotonIdentityDetailer"
_SEED_MAX = 0xFFFFFFFFFFFFFFFF
MAX_FACES_LIMIT = 16
PROMPT_WITH_TRIGGER = "close-up photo of %s's face, natural skin texture"
PROMPT_NO_TRIGGER = "close-up photo of a person's face, natural skin texture"

_DESCRIPTION = (
    "Finishing pass for small faces (SPEC 4.2). Faces whose short side is below size_gate_px are cut out (1.8x square), "
    "redrawn at 1024 px with the turbo stack of the connected MODEL (8 steps, denoise 0.25), colour-matched in LAB and "
    "pasted back through a feathered ellipse; pixels outside the ellipse are bit-equal to the input. With reference photos "
    "(and consent) a redraw is kept only if the ArcFace score rises; without them it is kept and marked unscorable. "
    "Larger faces are left alone: at 2 MP the pass lowered ArcFace (MP-56). Needs ultralytics + a face model; "
    "insightface is needed for references. Experimental.")


def _probe():
    """Seam for tests: what is installed. Cheap, imports nothing heavy."""
    return deps.probe()


def _make_backend(avail, settings, model, vae, positive, negative):
    """Seam for tests: the real backend (heavy imports happen here and in its methods)."""
    from ..finishing import backends
    return backends.ComfyBackend(avail, settings, model=model, vae=vae, positive=positive, negative=negative)


def _require_detector(avail):
    if avail.detector_ok:
        return
    raise errors.MissingGuardDependency(errors.make_issue(
        "missing_guard_dependency", NODE_ID, "detector",
        "%s is disabled: %s" % (NODE_ID, avail.detector_reason), deps.DETECTOR_FIX))


def _require_valid_report(rep):
    """Raise ``unsupported_feature`` for a newer schema, ``output_error`` for an invalid report."""
    if isinstance(rep, dict):
        report_mod.redact(rep)  # version gate; the copy is dropped
    problems = report_mod.validate_report(rep)
    if problems:
        raise errors.OutputError(errors.make_issue(
            "output_error", NODE_ID, "report", "the report is not a valid PHOTON_REPORT: " + "; ".join(problems[:5]),
            "connect the report output of a MossPhoton sampler"))


def trigger_of(profile):
    """First trigger hint of the first ``character`` adapter of the identity profile, or ``None``."""
    for entry in profile.get("stack") or []:
        if entry.get("role") == "character":
            for hint in entry.get("trigger_hints") or []:
                if str(hint).strip():
                    return str(hint).strip()
    return None


def _require_turbo_profile(profile):
    roles = [e.get("role") for e in profile.get("stack") or []]
    if "distill" not in roles:
        raise errors.UnsupportedFeature(errors.make_issue(
            "unsupported_feature", NODE_ID, "identity_profile",
            "the profile has no distill adapter (roles: %s); the face pass is verified on the turbo stack only (8 steps, cfg 1)"
            % (", ".join(r for r in roles if r) or "none"),
            "chain a distill LoRA with MossPhotonLoRAProfile (role=distill) before the character LoRA, as in the turbo_identity workflow"))


def _check_model_and_profile(model, profile):
    """Preflight of the MODEL and its profile; returns the non-error issues (foreign patches and modifiers, notes)."""
    if not isinstance(profile, dict):
        raise errors.MissingLora(errors.make_issue(
            "missing_lora", NODE_ID, "identity_profile", "no identity_profile is connected",
            "connect the identity_profile output of the last MossPhotonLoRAProfile of the chain"))
    facts = adapter.probe_model(model)
    issues = adapter.validate_model_facts(facts, node=NODE_ID, slg_requested=False)
    errors.raise_first_error(issues)
    check = core_solver.check_profile(profile, facts, node=NODE_ID)
    errors.raise_first_error(list(check.issues))
    _require_turbo_profile(profile)
    return [i for i in issues if i.severity != "error"] + [i for i in check.issues if i.severity != "error"]


def _images_as_numpy(tensor):
    """``[B, H, W, C]`` torch -> list of float32 ``[H, W, C]`` numpy views (read only by the caller)."""
    import numpy as np
    arr = tensor.detach().cpu().numpy()
    if arr.ndim != 4 or arr.shape[-1] < 3:
        raise errors.OutputError(errors.make_issue(
            "output_error", NODE_ID, "image", "expected an IMAGE [B,H,W,C>=3], got shape %s" % (tuple(arr.shape),),
            "connect an IMAGE"))
    return [np.ascontiguousarray(arr[b], dtype=np.float32) for b in range(arr.shape[0])]


class MossPhotonIdentityDetailer:
    """Redraws faces below ``size_gate_px`` with the MODEL's character LoRA and keeps the result by the accept rule."""

    CATEGORY = consts.CAT_FINISHING
    FUNCTION = "detail"
    RETURN_TYPES = ("IMAGE", "PHOTON_REPORT", "STRING")
    RETURN_NAMES = ("image", "report", "summary")
    OUTPUT_TOOLTIPS = (
        "The image with the accepted faces pasted in. Outside the feathered ellipse of each face it is bit-equal to the input.",
        "The incoming report with a detailer section (per face: bbox, size gate, score before/after, decision, denoise, time, colour match).",
        "Text summary.")
    DESCRIPTION = _DESCRIPTION

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Decoded image(s). Never modified; the result is a new tensor."}),
                "model": ("MODEL", {"tooltip": "The MODEL of the same MossPhotonLoRAProfile chain that produced the image "
                                               "(turbo distill + character). Cloned, never changed."}),
                "clip": ("CLIP", {"tooltip": "The MOSS text encoder, for the face prompt."}),
                "vae": ("VAE", {"tooltip": "The VAE that decoded the image (encode/decode of the face crop)."}),
                "report": ("PHOTON_REPORT", {"tooltip": "Report from a MossPhoton sampler; returned with a detailer section."}),
                "identity_profile": ("PHOTON_IDENTITY_PROFILE", {
                    "tooltip": "From the same MossPhotonLoRAProfile chain: the trigger word for the prompt and the "
                               "check that the MODEL carries these adapters."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _SEED_MAX, "control_after_generate": True,
                                 "tooltip": "Noise seed; each face gets a fixed offset from it."}),
                "size_gate_px": ("INT", {"default": 110, "min": 16, "max": 4096,
                                         "tooltip": "The pass runs only on faces whose bbox SHORT side is below this many "
                                                    "pixels. Larger faces are left untouched (info face_large_skipped). "
                                                    "MP-56: at 2 MP (faces ~120x165) the pass only lowered ArcFace."}),
                "denoise": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.6, "step": 0.01, "round": 0.001,
                                      "tooltip": "Img2img strength of the redraw (turbo, 8 steps). MP-56: 0.2-0.3 sharpens "
                                                 "small faces; 0.35 and up changed the face and lowered ArcFace."}),
            },
            "optional": {
                "reference_images": ("IMAGE", {"tooltip": "Photos of the character (any number, faces large enough for "
                                                          "ArcFace). With them a redraw is kept only if the ArcFace score "
                                                          "(mean similarity to these photos) rises. Only counted/read when "
                                                          "consent_reference_photos is on; never stored or logged."}),
                "consent_reference_photos": ("BOOLEAN", {"default": False,
                                                         "tooltip": "Allow the reference photos to be read (SPEC 7)."}),
                "other_reference_images_1": ("IMAGE", {"tooltip": "Photos of a second person in the frame. A face that matches "
                                                                  "them better is left untouched, not redrawn with the "
                                                                  "character's LoRA."}),
                "other_reference_images_2": ("IMAGE", {"tooltip": "Photos of a third person (same use)."}),
                "max_faces": ("INT", {"default": 8, "min": 1, "max": MAX_FACES_LIMIT,
                                      "tooltip": "Largest faces first. Smaller ones are not looked at."}),
                "accept_without_references": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OFF (default): without usable reference photos nothing is redrawn, the image passes through "
                               "(a redraw that cannot be checked could replace the person). ON: redraw anyway, keep the "
                               "result when the new crop passes the visibility/pose gates, and mark it unscorable."}),
                "character_match_min": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001,
                                                  "tooltip": "With references: a face whose mean ArcFace similarity to the "
                                                             "character is below this is nobody's and stays untouched. "
                                                             "Not a measure of identity, only a binding cut-off."}),
            },
        }

    def detail(self, image, model, clip, vae, report, identity_profile, seed, size_gate_px, denoise,
               reference_images=None, consent_reference_photos=False, other_reference_images_1=None,
               other_reference_images_2=None, max_faces=8, accept_without_references=False, character_match_min=0.30):
        """Returns ``{"ui": {"text": [summary]}, "result": (image, report, summary)}``.

        Raises ``missing_guard_dependency`` (no face detector, or references without ArcFace), ``missing_lora`` /
        ``incompatible_lora`` / ``unsupported_feature`` (profile does not match the MODEL or has no distill adapter),
        ``missing_required_input`` (references with no usable face), ``unsupported_feature`` (newer report schema) and
        ``output_error`` (invalid report or image). Inputs are never mutated.
        """
        t_start = time.perf_counter()
        avail = _probe()
        _require_detector(avail)
        import numpy as np
        import torch
        from ..finishing import face_detailer as fd

        _require_valid_report(report)
        issues = _check_model_and_profile(model, identity_profile)

        settings = fd.Settings(size_gate_px=float(size_gate_px), denoise=float(denoise), max_faces=int(max_faces),
                               match_min=float(character_match_min),
                               accept_without_references=bool(accept_without_references))
        trigger = trigger_of(identity_profile)
        prompt = PROMPT_WITH_TRIGGER % trigger if trigger else PROMPT_NO_TRIGGER
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        backend = _make_backend(avail, settings, model.clone(), vae, positive, negative)

        # reference photos -> character matrices (index 0 = the character the MODEL carries)
        characters, n_images, n_with_face = [], 0, 0
        connected = reference_images is not None
        if connected and not consent_reference_photos:
            issues.append(errors.make_issue(
                "reference_photos_ignored", NODE_ID, "reference_images",
                "reference photos are connected but consent_reference_photos is off; they were not read",
                "turn on consent_reference_photos on MossPhotonIdentityDetailer to use them"))
        elif connected:
            if not avail.arcface_ok:
                raise errors.MissingGuardDependency(errors.make_issue(
                    "missing_guard_dependency", NODE_ID, "reference_images",
                    "reference photos need the ArcFace scorer: %s" % avail.arcface_reason, deps.ARCFACE_FIX))
            photos = [fd.to_u8(p) for p in _images_as_numpy(reference_images)]
            n_images = len(photos)
            primary, n_with_face = fd.reference_matrix(backend, photos)
            if primary is None:
                raise errors.MissingRequiredInput(errors.make_issue(
                    "missing_required_input", NODE_ID, "reference_images",
                    "no face was found in any of the %d reference photo(s)" % n_images,
                    "connect photos with a clear, frontal face, or disconnect reference_images to run unscored"))
            characters.append(primary)
            for k, extra in enumerate((other_reference_images_1, other_reference_images_2), start=1):
                if extra is None:
                    continue
                matrix, _ = fd.reference_matrix(backend, [fd.to_u8(p) for p in _images_as_numpy(extra)])
                if matrix is None:
                    issues.append(errors.make_issue(
                        "reference_no_face", NODE_ID, "other_reference_images_%d" % k,
                        "no face was found in other_reference_images_%d; that person is not protected" % k,
                        "connect clearer photos of that person"))
                else:
                    characters.append(matrix)

        frames = _images_as_numpy(image)
        input_hashes = [canon.sha256_hex(f.tobytes()) for f in frames]
        outs, faces = [], []
        for b, frame in enumerate(frames):
            out, recs, frame_issues = fd.detail_image(frame, settings, backend, characters, seed=int(seed), image_index=b)
            faces.extend(recs)
            issues.extend(frame_issues)
            outs.append(image[b] if out is frame else torch.from_numpy(np.ascontiguousarray(out)).to(
                device=image.device, dtype=image.dtype))
        result = torch.stack(outs, dim=0) if outs else image

        section = fd.build_section(
            settings, int(seed), images=len(frames), faces=faces, issues=issues,
            detector={"available": True, "model": deps.DETECTOR_MODEL},
            arcface={"available": bool(avail.arcface_ok), "pack": deps.ARCFACE_PACK if avail.arcface_ok else None},
            references={"connected": connected, "consent": bool(consent_reference_photos), "used": bool(characters),
                        "n_images": int(n_images), "n_with_face": int(n_with_face), "n_characters": len(characters)},
            prompt_sha256=canon.sha256_hex(prompt), trigger_used=trigger is not None,
            time_s=time.perf_counter() - t_start,
            recipe=fd.recipe_block(
                settings, int(seed), model=_model_identity(model), lora=_lora_list(identity_profile),
                vae_sha256=_vae_hash(vae), input_sha256=input_hashes, prompt_sha256=canon.sha256_hex(prompt),
                reference_sha256=(canon.sha256_hex(b"".join(np.ascontiguousarray(m, dtype=np.float32).tobytes()
                                                               for m in characters)) if characters else None)))
        out_report = report_mod.with_detailer(report, section)
        summary = _summary(section)
        for issue in issues:
            if issue.severity == "warning":
                _log.warning(issue.format())
        return {"ui": {"text": [summary]}, "result": (result, out_report, summary)}


def _model_identity(model):
    """``{"sha256", "basename", "size"}`` of the checkpoint behind the MODEL from the hash cache only (never hashes 10 GB
    here); unknown parts are ``None``."""
    try:
        ident = fingerprint.identify_checkpoint(model, mode="cached_only", node=NODE_ID)["identity"]
        return {"sha256": ident.get("sha256"), "basename": ident.get("basename"), "size": ident.get("size")}
    except Exception:  # noqa: BLE001 - a fingerprint is a label, never a gate
        return {"sha256": None, "basename": None, "size": None}


def _lora_list(profile):
    out = []
    for entry in profile.get("stack") or []:
        lora = entry.get("lora") or {}
        out.append({"role": str(entry.get("role")), "basename": str(lora.get("basename")), "size": int(lora.get("size", 0)),
                    "sha256": lora.get("sha256"), "strength_model": float(entry.get("strength_model", 0.0))})
    return out


def _vae_hash(vae):
    from ..finishing import backends
    return backends.vae_sha256(vae)


def _summary(section):
    c = section["counts"]
    lines = ["MossPhoton IdentityDetailer: %d face(s): %d accepted, %d rejected, %d skipped, %d unscorable; %.1f s" % (
        c["faces"], c["accepted"], c["rejected"], c["skipped"], c["unscorable"], section["time_s"])]
    for f in section["faces"]:
        score = ""
        if f["score_before"] is not None or f["score_after"] is not None:
            score = ", ArcFace %s -> %s" % ("-" if f["score_before"] is None else "%.4f" % f["score_before"],
                                            "-" if f["score_after"] is None else "%.4f" % f["score_after"])
        lines.append("face %d (image %d): %.0f px, %s%s" % (f["index"], f["image_index"], f["short_side_px"],
                                                            f["decision"], score))
    for w in section["warnings"]:
        if w["severity"] == "warning":
            lines.append("warning: " + w["code"])
    return "\n".join(lines)


NODE_CLASS_MAPPINGS = {NODE_ID: MossPhotonIdentityDetailer}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "Moss Photon Identity Detailer"}
