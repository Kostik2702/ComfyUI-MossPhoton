"""MossPhotonLoRAProfile (design 4.4, 5.2; docs/03-turbo.md 5.4): one LoRA per node, chained through ``profile_in``.

The LoRA is mapped once by the native ``comfy.lora.load_lora`` and applied the way ``comfy.sd.load_lora_for_models``
does, on clones only, transformer keys only unless the text-encoder flag is set. Coverage is counted per file from the native mapper (B1). The returned MODEL carries the
attachment string (M1) so a sampler can tell a native loader that ran before, between or after the chain.
"""
import copy
import logging
import os

from ..core import adapter, consts, errors, lora
from ..core import fingerprint as fp

NODE_ID = lora.NODE
_log = logging.getLogger("MossPhoton")


def _lora_choices():
    """``["none"] + the loras folder``; just ``["none"]`` when ComfyUI's ``folder_paths`` is not importable."""
    try:
        import folder_paths
        names = list(folder_paths.get_filename_list("loras"))
    except Exception:  # noqa: BLE001 - INPUT_TYPES must work with no ComfyUI (unit tests)
        names = []
    return ["none"] + [n for n in names if n != "none"]


def _resolve_path(lora_name):
    """Full path of a LoRA in the loras folder (raises ComfyUI's own ``FileNotFoundError`` when it is gone)."""
    import folder_paths
    return folder_paths.get_full_path_or_raise("loras", lora_name)


def _interrupt_check():
    try:
        import comfy.model_management
        return comfy.model_management.throw_exception_if_processing_interrupted
    except Exception:  # noqa: BLE001 - hashing then simply cannot be interrupted
        return None


def _issue(code, param, message, fix, *, severity=None, detail=None):
    return errors.make_issue(code, NODE_ID, param, message, fix, severity=severity, detail=detail)


class MossPhotonLoRAProfile:
    """Apply one LoRA with a constant weight, record it in a PHOTON_IDENTITY_PROFILE and chain further adapters."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The MOSS model. It is never modified: the LoRA is applied to a clone."}),
                "lora_name": (_lora_choices(), {
                    "default": "none",
                    "tooltip": "LoRA file from the loras folder; none = pass the model through unchanged."}),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Constant LoRA weight on the transformer (multiplier of the low-rank delta). "
                               "The 8-step distill LoRA wants 1.0; there is no verified value for character LoRAs."}),
                "role": (list(lora.ROLES), {
                    "default": lora.DEFAULT_ROLE,
                    "tooltip": "What the adapter is for. Presets read the roles: identity presets need a character "
                               "profile, turbo presets need a distill profile."}),
                "profile": (list(lora.WEIGHT_PROFILE_KINDS), {
                    "default": "constant",
                    "tooltip": "Weight profile over the sampling steps. Only constant exists in v1."}),
            },
            "optional": {
                "trigger": ("STRING", {
                    "default": "",
                    "tooltip": "Trigger word(s), comma separated. Recorded privately for the trigger check, "
                               "never added to any prompt and never written to a report."}),
                "profile_in": ("PHOTON_IDENTITY_PROFILE", {
                    "tooltip": "Profile of the previous adapter in the chain; connect the MODEL from the same node."}),
                "clip": ("CLIP", {"tooltip": "Needed only with apply_text_encoder."}),
                "apply_text_encoder": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also apply the text-encoder part of the file to a clone of clip. Off = "
                               "transformer only (the verified path). Logged and recorded when on."}),
                "strength_clip": ("FLOAT", {
                    "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "LoRA weight on the text encoder; used only with apply_text_encoder."}),
                "allow_low_coverage": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Accept a file that maps less than 50% of its modules to MOSS (never 0%)."}),
                "fingerprint": (list(fp.HASH_MODES), {
                    "default": "cached_or_compute",
                    "tooltip": "sha256 of the LoRA file: cached_or_compute (hash on a cache miss), cached_only, "
                               "off (the hash is then unknown and a distill LoRA cannot be verified)."}),
            },
        }

    RETURN_TYPES = ("MODEL", "PHOTON_IDENTITY_PROFILE", "CLIP")
    RETURN_NAMES = ("model", "identity_profile", "clip")
    OUTPUT_TOOLTIPS = (
        "A clone of the model with the LoRA applied; carries the chain attachment. The input model is untouched.",
        "Identity profile: the whole stack in apply order, coverage per file, provenance of the patches.",
        "The input clip, or a patched clone of it when apply_text_encoder is on; empty when no clip is connected.")
    FUNCTION = "apply"
    CATEGORY = consts.CAT_SETUP
    DESCRIPTION = (
        "Loads one LoRA through ComfyUI's own loader on a clone, checks how much of the file maps to MOSS and "
        "records it with the stack order in an identity profile. Chain several nodes with profile_in; the order of "
        "the nodes is the order of the patches and it changes the result in the last bits.")

    @classmethod
    def IS_CHANGED(cls, lora_name="none", **kwargs):
        """Size and mtime of the LoRA file and of its sidecar, so an edited file reruns the node."""
        if lora_name in (None, "", "none"):
            return "none"
        try:
            path = _resolve_path(lora_name)
        except Exception:  # noqa: BLE001 - validation reports a missing file, not IS_CHANGED
            return "missing"
        parts = []
        for candidate in (path, lora.sidecar_path(path)):
            try:
                st = os.stat(candidate)
                parts.append("%d:%d" % (st.st_size, st.st_mtime_ns))
            except OSError:
                parts.append("-")
        return "|".join(parts)

    def apply(self, model, lora_name, strength_model, role, profile, trigger="", profile_in=None, clip=None,
              apply_text_encoder=False, strength_clip=0.0, allow_low_coverage=False, fingerprint="cached_or_compute"):
        """Returns ``(model, identity_profile, clip)``.

        Raises ``unsupported_feature`` (weight profile other than constant, newer / phased ``profile_in``),
        ``unsupported_model`` and the other model errors of ``validate_model_facts``, ``missing_required_input``
        (text-encoder flag without ``clip``), ``incompatible_lora`` (coverage verdict error, architecture named in
        the metadata, the unsupported SFT data-space DMD file, a second distill LoRA, ``profile_in`` of another
        chain) and ``FileNotFoundError`` from ComfyUI for a missing file.
        """
        if profile not in lora.WEIGHT_PROFILE_KINDS:
            raise errors.UnsupportedFeature(_issue(
                "unsupported_feature", "profile",
                "weight profile %r is not available in v1 (only %s)" % (profile, ", ".join(lora.WEIGHT_PROFILE_KINDS)),
                "set profile to constant"))
        if role not in lora.ROLES:
            raise errors.UnsupportedFeature(_issue(
                "unsupported_feature", "role", "role must be one of %s, got %r" % (list(lora.ROLES), role),
                "pick one of the listed roles"))
        te_requested = bool(apply_text_encoder)
        te_strength = te_requested and float(strength_clip) != 0.0
        if lora_name in (None, "", "none") or (float(strength_model) == 0.0 and not te_strength):
            return self._pass_through(model, lora_name, profile_in, clip)

        facts = adapter.probe_model(model)
        errors.raise_first_error(adapter.validate_model_facts(facts, node=NODE_ID, slg_requested=False))
        if profile_in is not None:
            lora.check_profile_in(profile_in, facts.lora_stack_attachment, node=NODE_ID)
        if te_requested and clip is None:
            raise errors.MissingRequiredInput(_issue(
                "missing_required_input", "clip", "apply_text_encoder is on but no clip is connected",
                "connect the CLIP of the text encoder to this node or switch apply_text_encoder off"))

        path = _resolve_path(lora_name)
        desc = lora.describe_file(path, mode=fingerprint, interrupt_check=_interrupt_check())
        sha = desc["sha256"]
        issues = lora.distill_issues(role, sha, desc["basename"], node=NODE_ID)
        errors.raise_first_error(issues)  # the unsupported SFT data-space DMD file stops before the tensors are loaded
        known = lora.distill_entry(sha)
        if known is not None and role != "distill":
            _log.warning("[MossPhoton] %s is the known 8-step distill LoRA %s but its role is %s; presets read "
                         "the role, so set role to distill", desc["basename"], known["name"], role)
        sidecar = None
        try:
            sidecar = lora.read_sidecar(path)
        except ValueError as exc:
            _log.warning("[MossPhoton] %s; the sidecar is ignored", exc)

        sd, meta = lora.load_lora_state(path)
        parsed = lora.parse_lora_metadata(meta)
        if parsed["architecture"] == "other":
            raise errors.IncompatibleLora(_issue(
                "incompatible_lora", "lora_name",
                "%s names the architecture %r in its metadata, not MOSS" % (desc["basename"], parsed["architecture_value"]),
                "use a LoRA trained for moss_image",
                detail={"architecture": parsed["architecture_value"]}))

        mapping, patch_dict = lora.map_lora_patches(model, sd, clip=clip if te_requested else None)
        verdict = lora.coverage_verdict(mapping.modules_total, mapping.modules_mapped, allow_low=bool(allow_low_coverage))
        if verdict["verdict"] == "error":
            raise errors.IncompatibleLora(_issue(
                "incompatible_lora", "lora_name",
                "this file maps %d/%d modules to MOSS" % (mapping.modules_mapped, mapping.modules_total),
                "use a LoRA trained for moss_image (file may be misfiled: %s)" % desc["basename"],
                detail={"modules_mapped": mapping.modules_mapped, "modules_total": mapping.modules_total,
                        "tensors_total": mapping.tensors_total}))
        if verdict["code"] is not None:
            issues.append(_issue(
                verdict["code"], "lora_name",
                "%s maps %d/%d modules to MOSS (%.0f%%)" % (desc["basename"], mapping.modules_mapped,
                                                           mapping.modules_total, 100.0 * verdict["ratio"]),
                "check that the file was trained for moss_image; set allow_low_coverage only if you know it is right",
                detail={"ratio": verdict["ratio"]}))
        if mapping.layers_missing:
            issues.append(_issue(
                "lora_layers_missing", "lora_name",
                "%s skips layers %s of %d" % (desc["basename"], list(mapping.layers_missing), mapping.layers_total),
                "none needed; a LoRA without layer 22 is normal (the reference SLG layer)",
                detail={"layers_missing": list(mapping.layers_missing)}))
        if mapping.te_modules_ignored:
            issues.append(_issue(
                "te_keys_ignored", "apply_text_encoder",
                "%s has %d text-encoder modules and apply_text_encoder is off; they are not applied"
                % (desc["basename"], mapping.te_modules_ignored),
                "connect clip and switch apply_text_encoder on to apply them"))
        if mapping.te_entries:
            issues.append(_issue(
                "te_lora_applied", "apply_text_encoder",
                "text-encoder part of %s applied to a clone of clip (%d entries, strength %g)"
                % (desc["basename"], mapping.te_entries, float(strength_clip)),
                "switch apply_text_encoder off for the verified transformer-only path"))
            _log.warning("[MossPhoton] text-encoder LoRA applied: %s (%d entries, strength %g)",
                         desc["basename"], mapping.te_entries, float(strength_clip))

        if profile_in is not None and profile_in["model_identity"] is not None:
            model_identity = copy.deepcopy(profile_in["model_identity"])
        else:
            ident = fp.identify_checkpoint(model, mode=fingerprint, node=NODE_ID, interrupt_check=_interrupt_check())
            model_sha = ident["identity"]["sha256"]
            model_identity = {"family": ident["family"], "sha256": model_sha} if model_sha else None
        family = model_identity["family"] if model_identity else "unknown-moss"

        if known is not None:
            trained_on = known["trained_on"]
        elif sidecar is not None and sidecar["trained_on"]:
            trained_on = sidecar["trained_on"]
        else:
            trained_on = parsed["trained_on"]
        base_match = lora.base_match_of(trained_on, family)
        if base_match is False:
            verified_distill = known is not None and known["status"] == "verified_recipe"
            issues.append(_issue(
                "lora_base_mismatch", "lora_name",
                "%s was trained on moss-image-%s, the model is %s"
                % ("distill LoRA" if verified_distill else desc["basename"], trained_on, family),
                "none needed; the turbo gate shows whether the combination holds" if verified_distill
                else "use a LoRA trained on the same base as the model or accept an unverified result",
                severity="info" if verified_distill else None))

        prior_stack = copy.deepcopy(profile_in["stack"]) if profile_in is not None else []
        index = len(prior_stack)
        strength_clip_used = float(strength_clip) if te_requested else 0.0
        hints = [t.strip() for t in str(trigger or "").split(",") if t.strip()]
        entry = lora.build_stack_entry(
            index=index, role=role,
            lora={"basename": desc["basename"], "size": desc["size"], "sha256": sha, "metadata": parsed["whitelist"]},
            strength_model=float(strength_model), strength_clip=strength_clip_used, mapping=mapping, verdict=verdict,
            trained_on=trained_on, base_match=base_match,
            sidecar=lora.sidecar_block(sidecar, known if known is not None and known["status"] == "verified_recipe" else None),
            trigger_hints=hints, warnings=issues)
        stack_issues = [i for i in lora.check_stack(prior_stack + [entry], node=NODE_ID)
                        if i.detail and i.detail.get("index") == index]
        errors.raise_first_error(stack_issues)
        entry["warnings"] = entry["warnings"] + lora.issues_to_dicts(stack_issues)

        entries_before = lora.patch_entries_of(model)
        new_model, new_clip = lora.apply_patches(
            model, clip if te_requested else None, patch_dict, float(strength_model), strength_clip_used, meta or None)
        entries_after = lora.patch_entries_of(new_model)
        if entries_after - entries_before != mapping.model_entries:
            raise errors.UnsupportedFeature(_issue(
                "unsupported_feature", "lora_name",
                "add_patches added %d entries, the mapper accepted %d; this ComfyUI patches differently from the "
                "tested 0.42.0" % (entries_after - entries_before, mapping.model_entries),
                "update MossPhoton for this ComfyUI version"))

        prior_att = profile_in["attachment"] if profile_in is not None else None
        baseline = lora.provenance_baseline(prior_att, patches_uuid=model.patches_uuid, patch_entries=entries_before)
        att = lora.build_attachment(
            (prior_att["entries"] if prior_att is not None else []) + [lora.attachment_entry_of(entry)],
            patches_uuid=new_model.patches_uuid, patch_entries_total=entries_after, baseline_entries=baseline)
        new_model.set_attachments(consts.ATTACH_LORA_STACK, lora.attachment_json(att))
        new_profile = lora.build_identity_profile(prior_stack + [entry], attachment=att, model_identity=model_identity)

        _log.info("[MossPhoton] %s: role %s, strength %g, %d/%d modules mapped, %d patch entries, %s",
                  desc["basename"], role, float(strength_model), mapping.modules_mapped, mapping.modules_total,
                  mapping.model_entries, "text encoder on" if mapping.te_entries else "transformer only")
        _log.info("[MossPhoton] %s", lora.stack_journal(new_profile["stack"]))
        return (new_model, new_profile, new_clip if te_requested else clip)

    @staticmethod
    def _pass_through(model, lora_name, profile_in, clip):
        """``none`` or zero strength: the SAME model object, the incoming profile (or an empty one) and the clip."""
        if lora_name not in (None, "", "none"):
            _log.info("[MossPhoton] zero_strength_lora_skipped: %s has strength 0, nothing is applied or recorded",
                      lora_name)
        profile = profile_in if profile_in is not None else lora.empty_identity_profile(model)
        return (model, profile, clip)


NODE_CLASS_MAPPINGS = {NODE_ID: MossPhotonLoRAProfile}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "Moss Photon LoRA Profile"}
