"""MossPhotonSchedule (design 4.2): the validated sigma schedule and its PHOTON_SCHEDULE_INFO.

Never allocates a GPU tensor; the schedule is validated on a Python list before it is returned.
"""
from ..core import adapter, consts, errors, sigma

NODE_ID = "MossPhotonSchedule"
SCHEDULER = "sgm_uniform"  # the default; the base presets use it
SCHEDULERS = ("sgm_uniform", "simple")  # simple: the turbo presets (docs/03-turbo.md 3.2)
MODES = ("reference", "shifted")
_SHIFT_EPS = 1e-9


class MossPhotonSchedule:
    """``reference``: the live model's native schedule for ``scheduler`` (sgm_uniform, or simple for the turbo
    presets). ``shifted``: recomputed for ``shift`` at multiplier 1.0 and marked not-reference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "The MOSS model; its live model sampling defines the schedule."}),
            "latent": ("LATENT", {"tooltip": "Checked for shape only."}),
            "steps": ("INT", {"default": 50, "min": 1, "max": sigma.MAX_STEPS,
                              "tooltip": "Number of sampling steps (the schedule has steps + 1 values)."}),
            "shift": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 100.0, "step": 0.01,
                                "tooltip": "Flow shift, used only in mode shifted (reference keeps the live shift)."}),
            "mode": (list(MODES), {"default": "reference",
                                   "tooltip": "reference = the model's own schedule; shifted = recomputed for shift."}),
            "scheduler": (list(SCHEDULERS), {
                "default": SCHEDULER,
                "tooltip": "Sigma spacing of the native KSampler scheduler. sgm_uniform = the base presets; "
                           "simple = the 8-step turbo presets (it differs from sgm_uniform, so pick the one "
                           "of your preset to get the reference schedule)."}),
        }}

    RETURN_TYPES = ("SIGMAS", "PHOTON_SCHEDULE_INFO")
    RETURN_NAMES = ("sigmas", "schedule_info")
    OUTPUT_TOOLTIPS = ("Validated sigma schedule (float32, CPU).", "Facts about the schedule and its validation.")
    FUNCTION = "build"
    CATEGORY = consts.CAT_SAMPLING
    DESCRIPTION = ("Builds and validates the MOSS sigma schedule. Reference mode equals the native KSampler "
                   "schedule of the chosen scheduler (sgm_uniform, or simple for the turbo presets) bit for bit.")

    def build(self, model, latent, steps, shift, mode, scheduler=SCHEDULER):
        """Raises the first error of model / latent / schedule validation (``invalid_schedule`` included)."""
        for param, value, allowed in (("mode", mode, MODES), ("scheduler", scheduler, SCHEDULERS)):
            if value not in allowed:
                raise errors.UnsupportedFeature(errors.make_issue(
                    "unsupported_feature", NODE_ID, param, "%s must be one of %s, got %r" % (param, list(allowed), value),
                    "pick one of the listed values"))
        facts = adapter.probe_model(model)
        errors.raise_first_error(adapter.validate_model_facts(facts, node=NODE_ID, slg_requested=False))
        if facts.shift is None:
            raise errors.InvalidModelSampling(errors.make_issue(
                "invalid_model_sampling", NODE_ID, "model", "the model sampling has no readable shift",
                "use the MOSS model as loaded, or ModelSamplingAuraFlow for another shift"))
        sigma_max = facts.sigma_max if facts.sigma_max is not None else 1.0  # as the sampler does
        fixed = adapter.fix_empty_latent(model, latent, node=NODE_ID)
        lat_issues = adapter.validate_latent_facts(adapter.probe_latent(latent, fixed),
                                                   channels=adapter.LATENT_CHANNELS, node=NODE_ID)
        errors.raise_first_error(lat_issues)
        warnings = [i for i in lat_issues if i.severity != "error"]

        live_shift = facts.shift
        if mode == "reference":
            sigmas = adapter.native_sigmas(model, scheduler, steps)
            shift_used = live_shift
            if abs(float(shift) - live_shift) > _SHIFT_EPS:
                warnings.append(errors.make_issue(
                    "shift_input_ignored", NODE_ID, "shift",
                    "mode reference keeps the live model shift %g, the shift input %g is ignored" % (live_shift, shift),
                    "use mode shifted to recompute the schedule for another shift"))
        else:
            sigmas = adapter.shifted_sigmas(model, scheduler, steps, shift)
            shift_used = float(shift)

        values = sigmas.tolist()
        issues = sigma.validate_sigmas(
            values, node=NODE_ID, param="steps", sigma_max=sigma_max, expect_steps=steps,
            ratio_is_error=(mode == "reference"))
        errors.raise_first_error(issues)
        warnings.extend(issues)

        native = adapter.native_sigmas(model, scheduler, steps)  # a second, fresh native call to compare with
        info = sigma.build_schedule_info(
            values, mode=mode, scheduler=scheduler, steps=steps, shift_model=live_shift, shift_used=shift_used,
            multiplier=facts.multiplier, native_reference=native.tolist(), warnings=warnings)
        return (sigmas, info)


NODE_CLASS_MAPPINGS = {NODE_ID: MossPhotonSchedule}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "Moss Photon Schedule"}
