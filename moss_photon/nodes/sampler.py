"""MossPhotonSampler and MossPhotonSamplerAdvanced (design 4.5, 4.6).

Thin nodes: they turn widgets into a ``solver.SampleRequest`` and return ``(latent, report)`` plus a text summary.
No comfy import at all; the sampling behaviour lives in ``core/solver.py``. The identity profile and the
constraints are plain dicts and are only passed through.

Darkroom presets (MP-71): a preset with a ``darkroom`` block (``turbo_darkroom_*``) runs the phase engine of
``core/darkroom.py`` in both nodes. On such a preset the ``cfg`` widget sets the FINISH guidance (the phase where the
negative prompt acts) and ``sampling.cfg`` stays the compose guidance; the Advanced node adds the phase knobs, each at a
sentinel that means "from the preset", and the optional ``finish_model``. The two things this module takes from
``core/lora`` are the boost loader the engine needs for a constant character profile (``lora.boost_patches``) and the loader
of the LoRA Phase plan a MODEL carries (``lora.plan_patches``, MP-84); the solver receives them as callables and never
imports ``lora`` (M4).
"""
from ..core import consts, darkroom, errors, guidance, lora, presets
from ..core import solver as core_solver

NODE_ID = "MossPhotonSampler"
ADVANCED_NODE_ID = "MossPhotonSamplerAdvanced"

FROM_PRESET = "from_preset"
SLG_CHOICES = (FROM_PRESET,) + guidance.SLG_MODES
ONOFF = (FROM_PRESET, "off", "on")
SCHEDULE_CHOICES = (FROM_PRESET,) + darkroom.SCHEDULES
FINGERPRINT_MODES = ("cached_or_compute", "cached_only", "off")
_SEED_MAX = 0xFFFFFFFFFFFFFFFF
_FALLBACK_DESCRIPTION = "Samples with a MossPhoton preset; every result carries a report with its real status."


def _preset_ids():
    try:
        return presets.selectable_ids()  # a broken preset file keeps its id: selecting it raises the typed error
    except errors.MossPhotonError:
        return list(presets.ALL_PRESET_IDS)


def _preset_tooltip():
    try:
        rows = presets.describe_catalog()
    except errors.MossPhotonError:
        return "Settings recipe. See the report for what was applied."
    return ("Settings recipe (SPEC 6.1). The four *_safe presets and night_photo are numerically equal to reference; "
            "differences are the input contract and diagnostics.\n"
            + "\n".join("%s: %s - %s" % (r["id"], r["status"], r["summary"]) for r in rows))


def _description():
    try:
        rows = presets.describe_catalog()
    except errors.MossPhotonError:
        return _FALLBACK_DESCRIPTION
    return (_FALLBACK_DESCRIPTION + " Presets: "
            + "; ".join("%s (%s)" % (r["id"], r["status"]) for r in rows) + ".")


def effective_cfg(preset, cfg, pin, node, resolved=None):
    """``(cfg, source)``: ``cfg > 0`` is the user's explicit value (``"user"``); ``0`` (the widget default) takes the
    preset's own cfg (``"preset"``). Raises what ``presets.resolve`` raises for an unknown preset or a bad pin
    (``resolved``: a ``ResolvedPreset`` the caller already has)."""
    if cfg > 0.0:
        return float(cfg), "user"
    resolved = resolved if resolved is not None else presets.resolve(preset, pin=pin, node=node)
    return float(resolved.effective["sampling"]["cfg"]), "preset"


def _fraction(value):
    """A float knob whose sentinel is a negative value (the fractions and weights of a preset are never negative)."""
    return None if value < 0.0 else float(value)


def _guidance(value):
    """A guidance knob whose sentinel is 0."""
    return None if value <= 0.0 else float(value)


def _flag(choice):
    """A from_preset / off / on combo -> ``None`` / ``False`` / ``True``."""
    return None if choice == FROM_PRESET else choice == "on"


def build_darkroom_overrides(schedule, compose_cfg, compose_slg, r_c, r_k, w_compose, w_commit, reexpose, sigma_r, r_f,
                             finish_cfg, eta, switch_model):
    """The Advanced phase knobs as the override dict of ``darkroom`` (``None`` = from the preset): every name of
    ``darkroom.OVERRIDE_NAMES``. A sentinel (``from_preset``, 0 for the guidance knobs, a negative number for the others)
    is ``None``."""
    return {
        "schedule": None if schedule == FROM_PRESET else schedule,
        "compose_cfg": _guidance(compose_cfg), "compose_slg": _flag(compose_slg),
        "r_c": _fraction(r_c), "r_k": _fraction(r_k), "w_compose": _fraction(w_compose),
        "w_commit": _fraction(w_commit), "reexpose": _flag(reexpose), "sigma_r": _fraction(sigma_r),
        "r_f": _fraction(r_f), "finish_cfg": _guidance(finish_cfg), "eta": _fraction(eta), "switch": _flag(switch_model),
    }


def build_overrides(steps, slg_mode, slg_layers, slg_start, slg_end, *, has_sigmas):
    """Advanced widgets -> preset overrides; a sentinel (0, ``from_preset``, empty, negative) means "from preset".

    ``steps`` is dropped when a ``sigmas`` input is connected (it is ignored there). ``slg_layers`` is parsed like
    the native node (every integer in the text).
    """
    overrides = {}
    if steps and not has_sigmas:
        overrides["steps"] = int(steps)
    if slg_mode != FROM_PRESET:
        overrides["slg_mode"] = slg_mode
    layers = guidance.parse_layers(slg_layers or "")
    if layers:
        overrides["slg_layers"] = layers
    if slg_start >= 0.0:
        overrides["slg_start"] = float(slg_start)
    if slg_end >= 0.0:
        overrides["slg_end"] = float(slg_end)
    return overrides


class MossPhotonSampler:
    """Samples with a preset on the stock CFGGuider + euler route and returns the latent and a PHOTON_REPORT."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "A MOSS-Image model (UNETLoader or the MossPhoton LoRA chain)."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning (an empty text is valid). "
                                                         "Not evaluated at CFG 1.0."}),
                "latent": ("LATENT", {"tooltip": "Empty latent (MossPhotonEmptyLatent); sets the image size in pixels/16."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _SEED_MAX, "control_after_generate": True,
                                 "tooltip": "Noise seed."}),
                "preset": (_preset_ids(), {"default": "reference", "tooltip": _preset_tooltip()}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "0 = preset value. A value above 0 is an explicit override of the "
                                             "preset (a different value marks the run customized). 1.0 runs one "
                                             "forward per step (no negative branch). On a darkroom preset "
                                             "(turbo_darkroom_*) it overrides the FINISH guidance, the phase where the "
                                             "negative prompt acts (darkroom.finish.cfg); the compose guidance stays "
                                             "the preset's cfg."}),
            },
            "optional": {
                "identity_profile": ("PHOTON_IDENTITY_PROFILE", {"tooltip": "From MossPhotonLoRAProfile (deprecated; "
                                                                            "optional). Ordinary LoRA loaders and "
                                                                            "MossPhotonLoRAPhase need no profile."}),
                "constraints": ("PHOTON_CONSTRAINTS", {"tooltip": "From MossPhotonPromptLayout; required by "
                                                                  "wardrobe_safe and scene_safe."}),
                "vae": ("VAE", {"tooltip": "Compatibility check only (128 channels, ratio 16)."}),
                "preview": ("BOOLEAN", {"default": True, "tooltip": "Show ComfyUI latent previews while sampling."}),
                "preset_pin": ("STRING", {"default": "", "tooltip": "<id>@<MAJOR.MINOR.PATCH>[#<sha8>]; filled "
                                                                    "automatically by the web extension. Empty runs the "
                                                                    "installed version with a warning; a pin that "
                                                                    "differs from it never stops the run."}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 1000,
                                  "tooltip": "0 = from preset. A positive value overrides the number of steps."}),
            },
        }

    RETURN_TYPES = ("LATENT", "PHOTON_REPORT")
    RETURN_NAMES = ("latent", "report")
    OUTPUT_TOOLTIPS = ("The denoised latent.", "Settings, hashes, forwards and label of this run.")
    FUNCTION = "sample"
    CATEGORY = consts.CAT_SAMPLING
    DESCRIPTION = _description()

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        try:
            return presets.catalog_sha256()
        except errors.MossPhotonError:
            return "invalid_catalog"

    def _run(self, node_id, model, positive, negative, latent, seed, preset, cfg, identity_profile, constraints,
             vae, preview, preset_pin, knobs=None, finish_model=None, **extra):
        """``knobs``: the Advanced phase knobs (``build_darkroom_overrides``; ``None`` values = from the preset). A
        preset without a ``darkroom`` block runs exactly as before; the solver warns about knobs and ``finish_model`` it
        ignores there (MP-71)."""
        pin = (preset_pin or "").strip()
        resolved = presets.resolve(preset, pin=pin, node=node_id)
        knobs = {k: v for k, v in (knobs or {}).items() if v is not None}
        if "darkroom" not in resolved.preset:
            value, source = effective_cfg(preset, cfg, pin, node_id, resolved=resolved)
            if knobs or finish_model is not None:
                extra["darkroom"] = darkroom.DarkroomRequest(overrides=knobs, finish_model=finish_model,
                                                             boost_loader=lora.boost_patches)
            result = core_solver.execute(core_solver.SampleRequest(
                node_id=node_id, model=model, positive=positive, negative=negative, latent=latent, seed=seed,
                preset_id=preset, preset_pin=pin, cfg=value, identity_profile=identity_profile,
                constraints=constraints, vae=vae, preview=preview, lora_loader=lora.plan_patches, **extra))
            summary = "%s\ncfg %g from the %s" % (result.summary, value,
                                                  "preset" if source == "preset" else "user (widget)")
            return {"ui": {"text": [summary]}, "result": (result.latent, result.report)}
        # a darkroom preset: sampling.cfg is the compose guidance (the compose_cfg knob overrides it); the cfg widget is the
        # finish guidance, unless the finish_cfg knob is set
        compose_knob = knobs.pop("compose_cfg", None)
        compose = float(resolved.effective["sampling"]["cfg"] if compose_knob is None else compose_knob)
        extra["allow_phased_lora"] = True  # the character adapter rides on the MODEL as a phased profile (MP-16)
        extra["darkroom"] = darkroom.DarkroomRequest(
            overrides=knobs, finish_model=finish_model, boost_loader=lora.boost_patches,
            cfg_widget=float(cfg) if cfg > 0.0 else None)
        result = core_solver.execute(core_solver.SampleRequest(
            node_id=node_id, model=model, positive=positive, negative=negative, latent=latent, seed=seed,
            preset_id=preset, preset_pin=pin, cfg=compose, identity_profile=identity_profile,
            constraints=constraints, vae=vae, preview=preview, lora_loader=lora.plan_patches, **extra))
        finish = result.report["darkroom"]["settings"]["finish"]["cfg"]
        finish_from = ("finish_cfg widget" if knobs.get("finish_cfg") is not None
                       else "cfg widget" if cfg > 0.0 else "preset")
        summary = ("%s\ncompose cfg %g from the %s\nfinish cfg %g from the %s (on a darkroom preset the cfg widget sets "
                   "the finish guidance)" % (result.summary, compose,
                                             "preset" if compose_knob is None else "user (compose_cfg widget)", finish,
                                             finish_from))
        return {"ui": {"text": [summary]}, "result": (result.latent, result.report)}

    def sample(self, model, positive, negative, latent, seed, preset, cfg, identity_profile=None, constraints=None,
               vae=None, preview=True, preset_pin="", steps=0):
        return self._run(NODE_ID, model, positive, negative, latent, seed, preset, cfg, identity_profile, constraints,
                         vae, preview, preset_pin, overrides={"steps": int(steps)} if steps else {})


class MossPhotonSamplerAdvanced(MossPhotonSampler):
    """Everything of MossPhotonSampler plus explicit overrides; every widget at its sentinel means "from preset"."""

    @classmethod
    def INPUT_TYPES(cls):
        spec = MossPhotonSampler.INPUT_TYPES()
        spec["optional"].pop("steps")  # Advanced keeps its existing required widget position.
        spec["required"].update({
            "steps": ("INT", {"default": 0, "min": 0, "max": 1000,
                              "tooltip": "0 = from preset. Ignored when sigmas is connected."}),
            "slg_mode": (list(SLG_CHOICES), {"default": FROM_PRESET, "tooltip": "Skip-layer guidance of the "
                                                                                "unconditional pass. On a turbo "
                                                                                "preset it is experimental and the "
                                                                                "run is not_reference."}),
            "slg_layers": ("STRING", {"default": "", "tooltip": "Empty = from preset. Layer indices separated by "
                                                                "space or comma."}),
            "slg_start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "SLG window start (fraction of the schedule); -1 = from preset."}),
            "slg_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "SLG window end (fraction of the schedule); -1 = from preset."}),
            "solver": (["euler"], {"default": "euler", "tooltip": "Reserved extension point; euler only."}),
            "cfg_profile": (["constant"], {"default": "constant", "tooltip": "Reserved extension point."}),
            "noise_policy": (["native", "seed_plus_i"], {"default": "native", "tooltip": "native = the stock KSampler "
                                                                                          "noise; seed_plus_i marks the run not_reference."}),
            "max_forwards": ("INT", {"default": 0, "min": 0, "max": 100000,
                                     "tooltip": "0 = unlimited. A plan above the limit is a warning in the report; the run "
                                                "goes on."}),
            "fingerprint": (list(FINGERPRINT_MODES), {"default": "cached_or_compute",
                                                      "tooltip": "How the checkpoint sha256 is obtained."}),
        })
        spec["optional"].update({
            "sigmas": ("SIGMAS", {"tooltip": "Replaces the native schedule after validation; not bit-equal to the "
                                             "native one marks the run not_reference."}),
            "report_step_times": ("BOOLEAN", {"default": False, "tooltip": "Add timing.step_s to the report."}),
            # MP-71: the darkroom phase knobs. Optional (and last) on purpose: the widget values of a saved workflow are
            # positional, so nothing before them moves. Every one at its sentinel means "from the preset"; on a preset
            # without a darkroom block a knob that is set is ignored with the warning darkroom_override_ignored.
            "finish_model": ("MODEL", {"tooltip": "Darkroom presets with switch_model on: the second model of the finish "
                                                  "phase (the same checkpoint and character without the distill adapter). "
                                                  "Ignored, with a warning, otherwise."}),
            "schedule": (list(SCHEDULE_CHOICES), {"default": FROM_PRESET,
                                                  "tooltip": "Darkroom presets. native = the preset's simple schedule. sa2_8 "
                                                             "= the face-band-dense SA2 knots of MP-53 (resampled when "
                                                             "steps is not 8; a connected sigmas input replaces it)."}),
            "compose_cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01,
                                      "tooltip": "Darkroom presets: guidance of the compose and commit phases "
                                                 "(sampling.cfg). 0 = from the preset; 1.0 runs one forward per step."}),
            "compose_slg": (list(ONOFF), {"default": FROM_PRESET,
                                          "tooltip": "Darkroom presets: skip-layer guidance (layer 22) in the compose and "
                                                     "commit phases; only acts when their guidance is above 1.0."}),
            "r_c": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                              "tooltip": "Darkroom presets: end of compose = start of commit, as sigma progress "
                                         "r = (sigma_0 - sigma) / (sigma_0 - sigma_last). -1 = from the preset."}),
            "r_k": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                              "tooltip": "Darkroom presets: end of commit (the character weight has reached w_commit); "
                                         "also where re-expose happens. -1 = from the preset."}),
            "w_compose": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10.0, "step": 0.05,
                                    "tooltip": "Darkroom presets: character LoRA weight before commit (the nominal weight "
                                               "of the profile's character LoRA). -1 = from the preset."}),
            "w_commit": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10.0, "step": 0.05,
                                   "tooltip": "Darkroom presets: character LoRA weight after commit (smoothstep over "
                                              "r_c..r_k). -1 = from the preset. With a constant profile the difference is "
                                              "an extra patch of the same LoRA in the commit phase. A LoRA Phase "
                                              "identity entry (positive strength) runs at strength + (weight - w_compose): "
                                              "the boost is additive, so with w_compose 0 and w_commit 1 a strength of 1 "
                                              "runs at 1.0 in compose and ends at 2.0."}),
            "reexpose": (list(ONOFF), {"default": FROM_PRESET,
                                       "tooltip": "Darkroom presets: at r_k take the x0 readout, re-noise it to sigma_r with "
                                                  "an independent seeded RNG and descend again. Off in the shipped "
                                                  "presets: it smoothed the result on MOSS (MP-54)."}),
            "sigma_r": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "Darkroom presets: re-expose noise level. -1 = from the preset."}),
            "r_f": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                              "tooltip": "Darkroom presets: start of finish (guidance, negative, noise, model switch). "
                                         "-1 = from the preset; at least r_k."}),
            "finish_cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01,
                                     "tooltip": "Darkroom presets: guidance of the finish phase, with the negative prompt. "
                                                "0 = from the preset, or from the cfg widget when that is set (this knob "
                                                "wins over it)."}),
            "eta": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                              "tooltip": "Darkroom presets: noise injection in the finish phase (rectified-flow euler "
                                         "ancestral), own seeded RNG. 0 = off, -1 = from the preset."}),
            "switch_model": (list(ONOFF), {"default": FROM_PRESET,
                                           "tooltip": "Darkroom presets: run the finish phase on finish_model. Needs "
                                                      "finish_model."}),
        })
        return spec

    def sample(self, model, positive, negative, latent, seed, preset, cfg, steps, slg_mode, slg_layers, slg_start,
               slg_end, solver, cfg_profile, noise_policy, max_forwards, fingerprint, identity_profile=None,
               constraints=None, vae=None, preview=True, preset_pin="", sigmas=None, report_step_times=False,
               finish_model=None, schedule=FROM_PRESET, compose_cfg=0.0, compose_slg=FROM_PRESET, r_c=-1.0, r_k=-1.0,
               w_compose=-1.0, w_commit=-1.0, reexpose=FROM_PRESET, sigma_r=-1.0, r_f=-1.0, finish_cfg=0.0, eta=-1.0,
               switch_model=FROM_PRESET):
        if solver != "euler":
            errors.raise_first_error([errors.make_issue(
                "unsupported_feature", ADVANCED_NODE_ID, "solver", "solver %r is not available in v1" % (solver,),
                "use solver euler", severity="error")])
        return self._run(
            ADVANCED_NODE_ID, model, positive, negative, latent, seed, preset, cfg, identity_profile, constraints, vae,
            preview, preset_pin, sigmas=sigmas, noise_policy=noise_policy, fingerprint_mode=fingerprint,
            max_forwards=max_forwards, report_step_times=report_step_times,
            allow_slg_override=True,  # MP-30 M2: the one place an SLG override may stand on a turbo preset
            allow_phased_lora=True,   # MP-16: the one place a phased (experimental) LoRA weight may run
            overrides=build_overrides(steps, slg_mode, slg_layers, slg_start, slg_end, has_sigmas=sigmas is not None),
            knobs=build_darkroom_overrides(schedule, compose_cfg, compose_slg, r_c, r_k, w_compose, w_commit, reexpose,
                                           sigma_r, r_f, finish_cfg, eta, switch_model),
            finish_model=finish_model)


NODE_CLASS_MAPPINGS = {NODE_ID: MossPhotonSampler, ADVANCED_NODE_ID: MossPhotonSamplerAdvanced}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "Moss Photon Sampler", ADVANCED_NODE_ID: "Moss Photon Sampler (Advanced)"}
