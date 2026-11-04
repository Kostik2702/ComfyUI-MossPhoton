"""MossPhotonLoRAPhase (MP-84, v2): one LoRA, one phase, no patch.

The node sits on the MODEL line like an ordinary LoRA loader, validates the file (it exists, it maps to MOSS, its factors fit
and are finite, it is not the unsupported data-space DMD file) and then PATCHES NOTHING: it clones the MODEL and appends
``{lora_name, strength, phase, file identity}`` to the plan the clone carries in its ``model_options``
(``core/lora_plan.py``). MossPhotonSampler / MossPhotonSamplerAdvanced read the plan: a preset with Darkroom phases applies each
entry in the phases it names, any other preset applies every entry for the whole run (warning ``phase_plan_ignored`` for an
entry whose phase is not ``all`` or ``identity``). Chain one node per LoRA; order is apply order. A MODEL patched by an ordinary LoRA loader
is phase ``all`` and needs no node. Another sampler sees the MODEL without the LoRA (the plan is not a patch).
"""
import logging
import os

from ..core import adapter, consts, errors, lora, lora_plan

NODE_ID = consts.NODE_IDS_HELPERS[0]   # "MossPhotonLoRAPhase"
_log = logging.getLogger("MossPhoton")


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


class MossPhotonLoRAPhase:
    """Record one LoRA with a strength and the sampling phase it acts in; nothing is patched until a MossPhoton sampler runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The MOSS model. It is never modified: the node returns a clone that carries "
                                               "the plan."}),
                "lora_name": (_lora_choices(), {
                    "default": "none", "tooltip": "LoRA file from the loras folder; none = pass the model through."}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -lora_plan.STRENGTH_LIMIT, "max": lora_plan.STRENGTH_LIMIT, "step": 0.01,
                    "tooltip": "LoRA strength (the multiplier of the low-rank delta), as in an ordinary LoRA loader. 0 = "
                               "pass the model through."}),
                "phase": (list(lora_plan.PHASES), {
                    "default": "identity",
                    "tooltip": "Where in the sampling run the LoRA acts. Darkroom presets (turbo_darkroom_*) split the run by "
                               "sigma progress: identity = every phase from the first step (the character LoRA; on MOSS the "
                               "compose phase sets hair and silhouette) and it takes the commit weight boost, added to its "
                               "strength when that is positive (a negative strength keeps its value, with a warning); "
                               "composition+identity = compose, commit and hold, no boost (style and slider LoRAs; absent "
                               "in the finish); composition = the compose phase (layout, pose); texture = the finish "
                               "phase; all = every phase exactly like an ordinary loader, never boosted. A known distill "
                               "LoRA never takes the boost (put a distill LoRA in a stock loader before the LoRA Phase "
                               "nodes). Any other preset has no phases: every entry runs for the whole run, with a "
                               "warning when its phase is not all or identity."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("A clone of the model that carries this LoRA in its phase plan; chain the next LoRA Phase or connect a "
                       "MossPhoton sampler. Another sampler sees the model without the LoRA.",)
    FUNCTION = "apply"
    CATEGORY = consts.CAT_SETUP
    DESCRIPTION = ("Applies a LoRA only in one sampling phase (composition, identity, texture or both of the first two). "
                   "Patches nothing itself: the MossPhoton sampler applies it. Chain one node per LoRA.")
    SEARCH_ALIASES = ["lora by phase", "phase lora", "lora schedule", "moss lora"]

    @classmethod
    def IS_CHANGED(cls, lora_name="none", **kwargs):
        """Size and mtime of the LoRA file, so an edited file reruns the node."""
        if lora_name in (None, "", "none"):
            return "none"
        try:
            import folder_paths
            st = os.stat(folder_paths.get_full_path_or_raise("loras", lora_name))
            return "%d:%d" % (st.st_size, st.st_mtime_ns)
        except Exception:  # noqa: BLE001 - validation reports a missing file, not IS_CHANGED
            return "missing"

    def apply(self, model, lora_name, strength, phase="identity"):
        """Returns ``(model,)``: the SAME model for ``none`` or strength 0, else a clone with the entry appended.

        Raises only what cannot be worked around: ``unsupported_model`` and the other model errors of
        ``validate_model_facts``, ``FileNotFoundError`` from ComfyUI for a missing file, ``incompatible_lora`` (the
        unsupported SFT data-space DMD file, an architecture other than MOSS in the metadata, factors that do not fit in 5%
        of the found modules or more, non-finite factors, a file that maps less than half of its modules, a trainer flag
        ComfyUI does not apply). A smaller loss is a warning recorded on the entry and shown in the sampler's report.
        """
        if lora_name in (None, "", "none") or float(strength) == 0.0:
            if lora_name not in (None, "", "none"):
                _log.info("[MossPhoton] %s has strength 0: nothing is recorded", lora_name)
            return (model,)
        facts = adapter.probe_model(model)
        errors.raise_first_error(adapter.validate_model_facts(facts, node=NODE_ID, slg_requested=False))
        import folder_paths
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        desc = lora.describe_file(path, interrupt_check=_interrupt_check())
        # the unsupported SFT data-space DMD file stops before the tensors are loaded; role "other": only the known-file
        # table is asked, the role notes of MossPhotonLoRAProfile do not apply here
        errors.raise_first_error(lora.distill_issues("other", desc["sha256"], desc["basename"], node=NODE_ID))
        state, meta = lora.load_lora_state(path)
        state = lora.strip_trainer_flags(state, desc["basename"], node=NODE_ID)
        mapping, _patches = lora.map_lora_patches(model, state)
        warnings = lora.validate_mapping(desc, lora.parse_lora_metadata(meta), mapping, node=NODE_ID)
        entry = lora_plan.make_entry(lora_name, strength, phase, desc["basename"], desc["size"], desc["sha256"],
                                     warnings=lora.issues_to_dicts(warnings))
        for issue in warnings:
            _log.warning("%s", issue.format())
        _log.info("[MossPhoton] LoRA Phase: %s, strength %g, phase %s, %d/%d modules mapped", desc["basename"],
                  float(strength), phase, mapping.modules_mapped, mapping.modules_total)
        return (lora_plan.add_to_plan(model, entry),)


NODE_CLASS_MAPPINGS = {NODE_ID: MossPhotonLoRAPhase}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "Moss Photon LoRA Phase"}
