"""MossPhotonHiresFix (MP-91): the second pass of a hires fix (latent upscale + a short, low-denoise redraw) in which the FACE
enters the redraw late, and the character LoRA can be boosted for this pass only.

It replaces ``LatentUpscaleBy`` + ``KSampler`` of the stock graph and keeps the recipe: with ``upscale_method`` latent_bicubic,
``face_protect`` off and ``identity_boost`` 0 the result is bit-equal to that pair of nodes (same upscale, same noise, same
``comfy.sample.sample``). The bicubic latent upscale damages the frame (a semi-transparent doubled image on brows and eyes, visible
in a plain decode); the stock graph hides it by redrawing from sigma 0.86, but a face HELD at the source would keep it. So the default
upscale is pixel_lanczos (decode, Lanczos in pixels, encode), and a face is never held on a bicubic latent: with latent_bicubic and
faces found the run takes the pixel path and says so (``hires_bicubic_not_held``). What the node adds, each of it optional:

* the face is held at the source while the schedule is above ``face_sigma``: the detector finds the faces on the decoded input,
  ``finishing/hires.py`` builds the release map, the sampler gets it as ``noise_mask`` and a ``denoise_mask_function`` that turns
  it into a mask per sigma (``mask_value``). The pixels outside the faces are not touched by the blending at all;
* ``identity_boost`` is added to the strength of the character LoRA for this pass: the entries of a LoRA Phase plan with the phase
  ``identity`` (the pass counts as the finish phase: ``composition`` and ``composition+identity`` entries do not act, which is
  said in the summary) and the file in ``character_lora`` (for a MODEL patched by ordinary loaders, which carry no plan).

No heavy import at module level (numpy, torch, comfy and ultralytics are imported inside functions), so the pack loads without any
of them. The incoming MODEL is never mutated: the pass runs on ``model.clone()``, one clone, a direct child of the incoming MODEL.
Nothing here stops the run for a setting: a detector that is missing or fails, no face, a ``face_sigma`` outside the schedule, a
boost with nothing to boost and a mask that is already taken are warnings in the summary and a run without the part that could not
be built. It stops only for what is impossible (the wrong MODEL or latent, a LoRA file that is missing or maps nothing, a
non-finite result), through helpers that are on the no-raise list; this module has no raise site of its own.
"""
import logging
import sys

from ..core import adapter, consts, errors, guard, lora, lora_plan
from ..finishing import deps

_log = logging.getLogger("MossPhoton")

NODE_ID = "MossPhotonHiresFix"
_SEED_MAX = 0xFFFFFFFFFFFFFFFF
FINISH = "finish"                   # the darkroom phase this pass counts as (core/lora_plan.PHASE_SEGMENTS)
MAX_FACES = 8                       # the largest faces of an image are protected, as the detailer's default
MP_RECOMMENDED, MP_DOCUMENTED = 2.0, 4.0   # MOSS: recommended up to 2 MP, extrapolation up to 4 MP
UPSCALE_METHODS = ("latent_bicubic", "pixel_lanczos")

# The lists of comfy.samplers.KSampler at the supported ComfyUI (tests/integration/test_hires_cpu.py pins the equality). Inside a
# running ComfyUI the live lists are used (``_live``), so a newer ComfyUI offers its newer samplers; this copy is what a process
# without ComfyUI (the unit tests, the snapshot of /object_info) sees. comfy.samplers is never imported from here.
SAMPLER_NAMES = (
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2", "exp_heun_2_x0",
    "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
    "ipndm", "ipndm_v", "deis", "cfgpp_ud10_ab", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3",
    "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
)
SCHEDULER_NAMES = ("simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic",
                   "kl_optimal")

_DESCRIPTION = (
    "Second pass of a hires fix for MOSS-Image: replaces LatentUpscaleBy + KSampler (then VAEDecode). With upscale_method "
    "latent_bicubic, face_protect off and identity_boost 0 it is exactly that pair; the default upscale is pixel_lanczos, because "
    "a bicubic latent upscale leaves a doubled image that a held face would keep. face_protect holds the faces at the source until the schedule reaches face_sigma, "
    "so the features of the face come from the first pass and only the fine detail is redrawn; identity_boost adds to the "
    "strength of the character LoRA for this pass only. The values are not calibrated (face_sigma was measured on two seeds and one prompt). Face protection needs ultralytics and "
    "a face model (without them the pass runs unprotected, with a warning). Experimental.")


def _probe():
    """Seam for tests: what is installed. Cheap, imports nothing heavy."""
    return deps.probe()


def _detect_faces(avail, image, conf):
    """Seam for tests: ``[(x0, y0, x1, y1, conf)]`` in pixels for a float ``[H, W, 3]`` image (the YOLO detector)."""
    from ..finishing import backends
    return backends.detect_faces(image, avail.detector_path, conf)


def _live(attribute, static):
    """The ``comfy.samplers.KSampler`` list when comfy.samplers is already loaded (a running ComfyUI), else the static copy."""
    module = sys.modules.get("comfy.samplers")
    names = getattr(getattr(module, "KSampler", None), attribute, None)
    return list(names) if names else list(static)


def _lora_choices():
    """``["none"] + the loras folder``; just ``["none"]`` when ComfyUI's ``folder_paths`` is not importable."""
    try:
        import folder_paths
        names = list(folder_paths.get_filename_list("loras"))
    except Exception:  # noqa: BLE001 - INPUT_TYPES must work with no ComfyUI (unit tests)
        names = []
    return ["none"] + [n for n in names if n != "none"]


def _interrupt_check():
    try:
        import comfy.model_management
        return comfy.model_management.throw_exception_if_processing_interrupted
    except Exception:  # noqa: BLE001 - hashing then simply cannot be interrupted
        return None


def _poll_interrupt():
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


def _is_interrupt(exc):
    """True for ComfyUI's ``InterruptProcessingException``: it is an ``Exception``, and every ``except`` of this module that
    catches one lets it through."""
    try:
        import comfy.model_management
    except Exception:  # noqa: BLE001 - no ComfyUI, no interrupt
        return False
    return isinstance(exc, comfy.model_management.InterruptProcessingException)


def _resolve_plan(model, entries):
    """The ``lora_plan.Resolved`` entries of the LoRA Phase plan of ``model`` (the loop of ``core/solver._resolve_plan``): an
    entry of strength 0 is not loaded; a missing, changed or unmappable file is the loader's error, before any forward."""
    out = []
    for i, entry in enumerate(entries):
        patch_dict, metadata = ({}, None) if float(entry["strength"]) == 0.0 else lora.plan_patches(model, entry)
        out.append(lora_plan.Resolved(
            index=i, basename=entry["basename"], size=entry["size"], sha256=entry["sha256"],
            strength=float(entry["strength"]), phase=entry["phase"], patch_dict=patch_dict, metadata=metadata))
    return tuple(out)


def _same_file(resolved, desc):
    """A plan entry and a file (``describe_file``) are one LoRA: equal sha256, or - when a side has none - basename and size."""
    if resolved.sha256 and desc["sha256"]:
        return resolved.sha256 == desc["sha256"]
    return resolved.basename == desc["basename"] and resolved.size == desc["size"]


def _vae_ratio(vae):
    """Pixels per latent cell of the VAE; the Flux 2 VAE of MOSS is 16."""
    try:
        ratio = vae.spacial_compression_encode()
        if isinstance(ratio, int) and not isinstance(ratio, bool) and ratio > 0:
            return ratio
    except Exception:  # noqa: BLE001 - a stand-in VAE without the method: the model's own ratio
        pass
    return adapter.SPATIAL_RATIO


def _decode(vae, samples):
    """``vae.decode`` as the stock VAEDecode calls it: ``[B, H, W, 3]`` pixels."""
    pixels = vae.decode(samples)
    if len(pixels.shape) == 5:     # video VAEs: fold the frame axis into the batch
        pixels = pixels.reshape(-1, pixels.shape[-3], pixels.shape[-2], pixels.shape[-1])
    return pixels[:, :, :, :3]


def _sigmas(clone, steps, sampler_name, scheduler, denoise):
    """The sigmas ``comfy.sample.sample`` runs for these widgets, as python floats (comfy's own KSampler computes them)."""
    import comfy.samplers
    ks = comfy.samplers.KSampler(clone, steps=steps, device=clone.load_device, sampler=sampler_name, scheduler=scheduler,
                                 denoise=denoise, model_options=clone.model_options)
    return [float(s) for s in ks.sigmas.tolist()]


def _mask_function(face_sigma):
    """The ``denoise_mask_function`` of the clone. It holds numbers and one function, no tensor: ``model_options`` is
    deep-copied by ``ModelPatcher.clone``."""
    from ..finishing import hires as fh
    mask_value, band, face_sigma = fh.mask_value, fh.BAND, float(face_sigma)

    def denoise_mask_function(sigma, denoise_mask, extra_options):
        s0 = float(extra_options["sigmas"][0])
        if hasattr(sigma, "reshape"):          # [B] -> [B, 1, 1, 1], to meet the mask [B, C, H, W]
            sigma = sigma.reshape((-1,) + (1,) * (denoise_mask.ndim - 1))
        return mask_value(denoise_mask, sigma, s0, face_sigma, band)

    return denoise_mask_function


def _size_issue(width, height):
    """The note about the size of the result, or ``None``."""
    mp = width * height / 1e6
    mk = errors.make_issue
    if mp > MP_DOCUMENTED:
        return mk("megapixels_out_of_documented_range", NODE_ID, "scale",
                  "the result is %dx%d px (%.2f MP), above the 4 MP up to which MOSS is documented" % (width, height, mp),
                  "lower scale or start from a smaller first pass", severity="warning")
    if mp > MP_RECOMMENDED:
        return mk("megapixels_out_of_documented_range", NODE_ID, "scale",
                  "the result is %dx%d px (%.2f MP), above the 2 MP MOSS is recommended for (extrapolation is documented up to "
                  "4 MP)" % (width, height, mp), "none needed; lower scale to stay within 2 MP", severity="info")
    return None


def _method_for(upscale_method, holding):
    """The upscale a run really uses: pixel_lanczos whenever a face is to be held (a hold on a bicubic latent keeps its damage),
    else what was asked for."""
    return "pixel_lanczos" if holding else upscale_method


def _plural(n, one, many):
    return one if n == 1 else many


class MossPhotonHiresFix:
    """Upscale + redraw with the face entering late and an optional boost of the character LoRA; see the module docstring."""

    CATEGORY = consts.CAT_FINISHING
    FUNCTION = "hires"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "summary")
    OUTPUT_TOOLTIPS = (
        "The redrawn latent at scale x the input size. Connect it to VAEDecode. With face_protect off and identity_boost 0 it "
        "is the output of LatentUpscaleBy + KSampler with the same settings.",
        "Text summary: size, the sigmas that ran, faces per image, what was boosted and every warning.")
    DESCRIPTION = _DESCRIPTION
    SEARCH_ALIASES = ["hires fix", "highres fix", "second pass", "latent upscale", "face protect", "upscale and refine"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The MOSS model of the first pass, with its LoRAs. Cloned, never changed. A "
                                               "MossPhotonLoRAPhase plan is applied here as the finish phase."}),
                "positive": ("CONDITIONING", {"tooltip": "The positive conditioning of the first pass."}),
                "negative": ("CONDITIONING", {"tooltip": "The negative conditioning of the first pass."}),
                "vae": ("VAE", {"tooltip": "The VAE of the model. Used only when the pixels are needed: to find the faces and "
                                           "for the pixel_lanczos upscale (the default)."}),
                "latent": ("LATENT", {"tooltip": "The latent of the first pass (the sampler's output). Never modified."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _SEED_MAX, "control_after_generate": True,
                                 "tooltip": "Noise seed of the second pass, as in KSampler."}),
                "scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 2.0, "step": 0.05,
                                    "tooltip": "Upscale factor of the latent; the size is rounded like LatentUpscaleBy does. "
                                               "1.0 redraws without upscaling. MOSS is recommended up to 2 MP and documented "
                                               "up to 4 MP: the summary says when the result is larger."}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 50,
                                  "tooltip": "Sampling steps of the second pass, as in KSampler."}),
                "denoise": ("FLOAT", {"default": 0.32, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001,
                                      "tooltip": "As in KSampler. On linear_quadratic this is not the fraction of noise: 0.32 "
                                                 "starts the redraw near sigma 0.86 (the summary prints the sigmas that ran). "
                                                 "The default is the recipe of the stock graph; it is not calibrated yet."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "Guidance, as in KSampler. 1.0 skips the negative prompt."}),
                "sampler_name": (_live("SAMPLERS", SAMPLER_NAMES), {"default": "res_multistep",
                                                                    "tooltip": "As in KSampler."}),
                "scheduler": (_live("SCHEDULERS", SCHEDULER_NAMES), {"default": "linear_quadratic",
                                                                     "tooltip": "As in KSampler."}),
                "face_protect": ("BOOLEAN", {"default": True,
                                             "tooltip": "Hold the faces at the source until the schedule reaches face_sigma "
                                                        "(needs ultralytics and a face model; without them the pass runs "
                                                        "unprotected, with a warning). Off with latent_bicubic = the stock pair of nodes."}),
                "face_sigma": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "The sigma at which the face enters the redraw. Above it the face is held "
                                                    "at the source (re-noised), so its structure comes from the first pass; "
                                                    "from it down the face is redrawn with the rest. High = the face is "
                                                    "redrawn early (little protection); at or above the first sigma of the run "
                                                    "nothing is protected; at or below the last one the face is not redrawn "
                                                    "at all. Measured on 2 seeds and one prompt, not calibrated: a lower value "
                                                    "keeps the face closer to the first pass, with slightly less new detail."}),
                "identity_boost": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05,
                                             "tooltip": "Added to the strength of the character LoRA for this pass only: the "
                                                        "MossPhotonLoRAPhase entries with phase identity and the file in "
                                                        "character_lora. A known distill LoRA is never boosted. 0 = off. The "
                                                        "default is a first guess; its effect is not measured yet on the "
                                                        "pixel_lanczos path."}),
            },
            "optional": {
                "character_lora": (_lora_choices(), {"default": "none",
                                                     "tooltip": "The character LoRA to boost when the MODEL was patched by "
                                                                "ordinary loaders, which carry no plan and so no way to know "
                                                                "the character. Boosted once if it is also a plan entry. Read "
                                                                "only when identity_boost is above 0. none = nothing extra."}),
                "upscale_method": (list(UPSCALE_METHODS), {
                    "default": "pixel_lanczos",
                    "tooltip": "pixel_lanczos = decode, Lanczos upscale in pixels, encode again (the default: the bicubic latent "
                               "upscale leaves a semi-transparent doubled image on brows and eyes, which a held face would "
                               "keep). latent_bicubic = the upscale of LatentUpscaleBy, the stock recipe; with face_protect on "
                               "and a face found the run uses pixel_lanczos anyway and says so."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, character_lora="none", **kwargs):
        """Size and mtime of the character LoRA file, so an edited file reruns the node."""
        if character_lora in (None, "", "none"):
            return "none"
        try:
            import os
            import folder_paths
            st = os.stat(folder_paths.get_full_path_or_raise("loras", character_lora))
            return "%d:%d" % (st.st_size, st.st_mtime_ns)
        except Exception:  # noqa: BLE001 - the run reports a missing file, not IS_CHANGED
            return "missing"

    def hires(self, model, positive, negative, vae, latent, seed, scale, steps, denoise, cfg, sampler_name, scheduler,
              face_protect, face_sigma, identity_boost, character_lora="none", upscale_method="pixel_lanczos"):
        """Returns ``{"ui": {"text": [summary]}, "result": (latent, summary)}``.

        Stops only for the impossible: ``unsupported_model`` and the other model errors, ``invalid_latent``, ComfyUI's
        ``FileNotFoundError`` for a ``character_lora`` that is not there, ``incompatible_lora`` for a LoRA file that maps nothing
        or changed, ``non_finite_output``. Everything else is a warning in the summary. Inputs are never mutated.
        """
        import comfy.sample
        import comfy.utils
        import latent_preview
        from ..finishing import hires as fh
        mk = errors.make_issue
        issues = []

        # 1. the MODEL and the latent: the wrong ones are impossible to run on
        facts = adapter.probe_model(model)
        model_issues = adapter.validate_model_facts(facts, node=NODE_ID, slg_requested=False)
        errors.raise_first_error(model_issues)
        issues.extend(model_issues)
        fixed = adapter.fix_empty_latent(model, latent, node=NODE_ID)
        latent_issues = [i for i in adapter.validate_latent_facts(adapter.probe_latent(latent, fixed),
                                                                 channels=adapter.LATENT_CHANNELS, node=NODE_ID)
                         if i.code != "unsupported_source_input"]    # an incoming noise_mask is handled below, not refused
        errors.raise_first_error(latent_issues)
        issues.extend(latent_issues)

        samples = latent["samples"]
        h0, w0 = int(samples.shape[-2]), int(samples.shape[-1])
        new_w, new_h = round(w0 * scale), round(h0 * scale)      # the expression of LatentUpscaleBy
        ratio = _vae_ratio(vae)
        size_issue = _size_issue(new_w * ratio, new_h * ratio)
        if size_issue is not None:
            issues.append(size_issue)

        # 2. one clone, a direct child of the incoming MODEL: the LoRA plan as the finish phase, then the boost
        clone = model.clone()
        boost = float(identity_boost)
        resolved = _resolve_plan(model, lora_plan.entries_of(model))
        lora_plan.apply_to(clone, resolved, darkroom_phase=FINISH, boost=boost)
        acting = lora_plan.active(resolved, FINISH)
        skipped = [r for r in resolved if r.strength != 0.0 and r.index not in acting]
        if skipped:
            issues.append(mk(
                "hires_lora_phase_skipped", NODE_ID, "model",
                "%d LoRA Phase %s do not act in this pass (it counts as the finish phase): %s"
                % (len(skipped), _plural(len(skipped), "entry", "entries"),
                   ", ".join("%s (%s, %g)" % (r.basename, r.phase, r.strength) for r in skipped)),
                "none needed; use the phase identity, texture or all for a LoRA that should act in the second pass"))
        boosted = []                      # (what, strength before, strength after)
        if boost > 0.0:
            boosted = [(resolved[i].basename + " (plan, " + resolved[i].phase + ")", resolved[i].strength,
                        resolved[i].strength + boost) for i in acting if lora_plan.takes_boost(resolved[i])]
            if character_lora not in (None, "", "none"):
                import folder_paths
                path = folder_paths.get_full_path_or_raise("loras", character_lora)
                desc = lora.describe_file(path, interrupt_check=_interrupt_check())
                if lora.distill_entry(desc["sha256"]) is not None:
                    issues.append(mk(
                        "hires_boost_distill_skipped", NODE_ID, "character_lora",
                        "%s is a known distill LoRA (%s...): it is never boosted, the boost would change the distill recipe"
                        % (desc["basename"], desc["sha256"][:8]), "pick the character LoRA in character_lora"))
                elif any(_same_file(resolved[i], desc) and lora_plan.takes_boost(resolved[i]) for i in acting):
                    issues.append(mk(
                        "hires_boost_already_in_plan", NODE_ID, "character_lora",
                        "%s is a LoRA Phase entry with phase identity that takes the boost already: it is boosted once"
                        % desc["basename"], "none needed"))
                else:
                    patch_dict, _metadata = lora.plan_patches(model, {
                        "lora_name": character_lora, "size": desc["size"], "sha256": desc["sha256"]})
                    clone.add_patches(patch_dict, boost)
                    boosted.append((desc["basename"] + " (character_lora)", None, boost))
            if not boosted:
                issues.append(mk(
                    "hires_boost_no_target", NODE_ID, "identity_boost",
                    "identity_boost %.2f has nothing to boost: no LoRA Phase entry with phase identity acts in this pass and "
                    "no usable character_lora is selected" % boost,
                    "give the MODEL a MossPhotonLoRAPhase entry with phase identity, or pick the character LoRA in "
                    "character_lora, or set identity_boost to 0"))

        # 3. what the pass will run: the sigmas, and whether the faces can be protected
        sigmas = _sigmas(clone, steps, sampler_name, scheduler, denoise)
        first, last = fh.sigma_window(sigmas)
        n_steps = sum(1 for s in sigmas if s > 0.0)
        protect, avail, face_sigma = bool(face_protect), None, float(face_sigma)
        if protect:
            avail = _probe()
            if not avail.detector_ok:
                issues.append(mk(
                    "hires_no_detector", NODE_ID, "face_protect",
                    "no face detector, the pass runs without face protection: %s" % avail.detector_reason, deps.DETECTOR_FIX))
                protect = False
        if protect and ("noise_mask" in latent or "denoise_mask_function" in model.model_options):
            issues.append(mk(
                "hires_mask_not_combined", NODE_ID, "face_protect",
                "the %s already carries a mask; the face mask would replace it, so the pass runs without face protection"
                % ("latent" if "noise_mask" in latent else "MODEL"),
                "remove the mask upstream, or set face_protect to off"))
            protect = False
        if protect:
            verdict = fh.judge_face_sigma(face_sigma, sigmas)
            if verdict in (fh.ABOVE, fh.NO_WINDOW):
                issues.append(mk(
                    "hires_face_sigma_above_schedule", NODE_ID, "face_sigma",
                    "face_sigma %.2f is at or above the first sigma %.3f of the run: the face is free from the first step, the "
                    "protection does nothing and is not built" % (face_sigma, first if first is not None else 0.0),
                    "lower face_sigma below %.3f, or set face_protect to off" % (first if first is not None else 0.0)))
                protect = False
            elif verdict == fh.BELOW:
                issues.append(mk(
                    "hires_face_sigma_below_schedule", NODE_ID, "face_sigma",
                    "face_sigma %.2f is at or below the last sigma %.3f of the run: the face is never released and stays as in "
                    "the source (it is not redrawn)" % (face_sigma, last),
                    "raise face_sigma above %.3f if the face should be redrawn" % last))

        # 4. the pixels, only when something needs them: the faces, or the pixel upscale
        pixel_mode = upscale_method == "pixel_lanczos"
        pixels = None
        if protect or pixel_mode:
            _poll_interrupt()
            pixels = _decode(vae, samples)

        boxes_per_image = None
        if protect:
            boxes_per_image = self._find_faces(avail, pixels, issues)
            if boxes_per_image is None:
                protect = False
            elif not any(boxes_per_image):
                issues.append(mk(
                    "hires_no_face", NODE_ID, "face_protect",
                    "no face found in the %d image(s), the pass runs without face protection" % len(boxes_per_image),
                    "none needed; set face_protect to off to silence this"))
                protect = False
            elif _method_for(upscale_method, holding=True) != upscale_method:
                # a face is never held on a bicubic latent: the damage of that upscale would be kept in the face
                issues.append(mk(
                    "hires_bicubic_not_held", NODE_ID, "upscale_method",
                    "holding a face on a bicubic latent doubles the image (the bicubic latent upscale leaves a semi-transparent "
                    "copy on brows and eyes, and the held face keeps it), so this run used pixel_lanczos",
                    "use upscale_method pixel_lanczos, or set face_protect to off to get the stock recipe with latent_bicubic"))
                pixel_mode = True

        # 5. the upscale
        up = latent.copy()
        if pixel_mode:
            big = comfy.utils.common_upscale(pixels.movedim(-1, 1), new_w * ratio, new_h * ratio, "lanczos",
                                             "disabled").movedim(1, -1)
            up["samples"] = vae.encode(big[:, :, :, :3])
            up.pop("downscale_ratio_spacial", None)
            up.pop("downscale_ratio_temporal", None)
        else:
            up["samples"] = comfy.utils.common_upscale(samples, new_w, new_h, "bicubic", "disabled")
        pixels = None

        # 6. the sampling: the call of nodes.common_ksampler, with the face mask when there is one
        latent_image = comfy.sample.fix_empty_latent_channels(
            clone, up["samples"], up.get("downscale_ratio_spacial", None), up.get("downscale_ratio_temporal", None))
        noise = comfy.sample.prepare_noise(latent_image, seed, up["batch_index"] if "batch_index" in up else None)
        noise_mask = up["noise_mask"] if "noise_mask" in up else None
        if protect:
            noise_mask = self._release_tensor(fh, boxes_per_image, latent_image)
            clone.set_model_denoise_mask_function(_mask_function(face_sigma))
        callback = latent_preview.prepare_callback(clone, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        out_samples = comfy.sample.sample(clone, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                          denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                                          force_full_denoise=False, noise_mask=noise_mask, callback=callback,
                                          disable_pbar=disable_pbar, seed=seed)
        guard.check_finite(out_samples, node=NODE_ID, where="hires latent")
        out = up.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = out_samples

        summary = self._summary(
            scale=scale, method="pixel_lanczos" if pixel_mode else upscale_method, in_hw=(h0, w0), out_hw=(new_h, new_w),
            ratio=ratio, first=first, last=last,
            n_steps=n_steps, sampler_name=sampler_name, scheduler=scheduler, denoise=denoise, cfg=cfg,
            face_protect=bool(face_protect), protect=protect, face_sigma=face_sigma, boxes=boxes_per_image, boosted=boosted,
            boost=boost, issues=issues)
        for issue in issues:
            if issue.severity == "warning":
                _log.warning(issue.format())
        return {"ui": {"text": [summary]}, "result": (out, summary)}

    @staticmethod
    def _find_faces(avail, pixels, issues):
        """Faces per image of the decoded input: ``[[(x0, y0, x1, y1), ...], ...]`` as FRACTIONS of the image, or ``None`` when
        the detector failed (a warning is added; ComfyUI's interrupt is let through)."""
        import numpy as np
        from ..finishing import face_detailer, geometry
        from ..finishing import hires as fh
        conf = face_detailer.Settings().det_conf
        height, width = int(pixels.shape[1]), int(pixels.shape[2])
        found = []
        for b in range(int(pixels.shape[0])):
            _poll_interrupt()
            image = np.ascontiguousarray(pixels[b].detach().float().cpu().numpy(), dtype=np.float32)
            try:
                detections = _detect_faces(avail, image, conf)
            except Exception as exc:  # noqa: BLE001 - a broken detector must not stop the pass
                if _is_interrupt(exc):
                    raise
                issues.append(errors.make_issue(
                    "hires_detector_failed", NODE_ID, "face_protect",
                    "the face detector failed (%s: %s), the pass runs without face protection"
                    % (type(exc).__name__, " ".join(str(exc).split())[:200]), deps.DETECTOR_FIX))
                return None
            kept = geometry.select_faces(detections, max_faces=MAX_FACES, min_conf=conf)
            found.append(fh.boxes_to_fractions(kept, width, height))
        return found

    @staticmethod
    def _release_tensor(fh, boxes_per_image, latent_image):
        """The release maps of the batch as a ``[B, 1, H, W]`` tensor at the latent size of the redraw; an image without a face
        is all ones (it gets exactly 1.0 at every sigma, so the blending leaves it alone)."""
        import numpy as np
        import torch
        height, width = int(latent_image.shape[-2]), int(latent_image.shape[-1])
        maps = [fh.release_map(height, width, boxes) for boxes in boxes_per_image]
        return torch.from_numpy(np.stack(maps, axis=0)[:, None, :, :].copy())

    @staticmethod
    def _summary(*, scale, method, in_hw, out_hw, ratio, first, last, n_steps, sampler_name, scheduler, denoise, cfg,
                 face_protect, protect, face_sigma, boxes, boosted, boost, issues):
        (h0, w0), (h1, w1) = in_hw, out_hw
        lines = ["MossPhoton HiresFix: scale %g (%s), latent %dx%d -> %dx%d, pixels %dx%d -> %dx%d (%.2f MP)" % (
            scale, method, w0, h0, w1, h1, w0 * ratio, h0 * ratio, w1 * ratio, h1 * ratio, w1 * ratio * h1 * ratio / 1e6)]
        if first is not None:
            lines.append("sigmas %.3f .. %.3f (%d steps, %s, %s, denoise %g, cfg %g)" % (
                first, last, n_steps, sampler_name, scheduler, denoise, cfg))
        if boxes is not None:
            lines.append("faces: " + "; ".join("image %d: %d %s" % (b, len(found), _plural(len(found), "face", "faces"))
                                               for b, found in enumerate(boxes)))
        if protect:
            lines.append("face protection: on, face_sigma %.2f (a face is held at the source above it)" % face_sigma)
        elif face_protect:
            lines.append("face protection: not built, see the warnings")
        else:
            lines.append("face protection: off")
        for what, before, after in boosted:
            if before is None:      # a loader's LoRA: its strength is not readable from the MODEL
                lines.append("boost: %+.2f on %s, on top of the loader's strength" % (boost, what))
            else:
                lines.append("boost: %+.2f on %s: strength %g -> %g" % (boost, what, before, after))
        for issue in issues:
            if issue.severity in ("warning", "info"):
                lines.append("%s: %s" % (issue.severity, issue.format()))
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {NODE_ID: MossPhotonHiresFix}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "Moss Photon Hires Fix"}
