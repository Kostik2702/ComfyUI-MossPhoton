"""MossPhotonSampler and MossPhotonSamplerAdvanced (design 4.5, 4.6).

Thin nodes: they turn widgets into a ``solver.SampleRequest`` and return ``(latent, report)`` plus a text summary.
No comfy import at all; the sampling behaviour lives in ``core/solver.py``. The identity profile and the
constraints are plain dicts and are only passed through.
"""
from ..core import consts, errors, guidance, presets
from ..core import solver as core_solver

NODE_ID = "MossPhotonSampler"
ADVANCED_NODE_ID = "MossPhotonSamplerAdvanced"

FROM_PRESET = "from_preset"
SLG_CHOICES = (FROM_PRESET,) + guidance.SLG_MODES
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


def effective_cfg(preset, cfg, pin, node):
    """``(cfg, source)``: ``cfg > 0`` is the user's explicit value (``"user"``); ``0`` (the widget default) takes the
    preset's own cfg (``"preset"``). Raises what ``presets.resolve`` raises for an unknown preset or a bad pin."""
    if cfg > 0.0:
        return float(cfg), "user"
    return float(presets.resolve(preset, pin=pin, node=node).effective["sampling"]["cfg"]), "preset"


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
                                             "forward per step (no negative branch)."}),
            },
            "optional": {
                "identity_profile": ("PHOTON_IDENTITY_PROFILE", {"tooltip": "From MossPhotonLoRAProfile; required by "
                                                                            "identity_safe and the turbo presets."}),
                "constraints": ("PHOTON_CONSTRAINTS", {"tooltip": "From MossPhotonPromptLayout; required by "
                                                                  "wardrobe_safe and scene_safe."}),
                "vae": ("VAE", {"tooltip": "Compatibility check only (128 channels, ratio 16)."}),
                "preview": ("BOOLEAN", {"default": True, "tooltip": "Show ComfyUI latent previews while sampling."}),
                "preset_pin": ("STRING", {"default": "", "tooltip": "<id>@<MAJOR.MINOR.PATCH>[#<sha8>]; filled "
                                                                    "automatically by the web extension. Empty runs the "
                                                                    "installed version with a warning."}),
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
             vae, preview, preset_pin, **extra):
        pin = (preset_pin or "").strip()
        value, source = effective_cfg(preset, cfg, pin, node_id)
        result = core_solver.execute(core_solver.SampleRequest(
            node_id=node_id, model=model, positive=positive, negative=negative, latent=latent, seed=seed,
            preset_id=preset, preset_pin=pin, cfg=value, identity_profile=identity_profile,
            constraints=constraints, vae=vae, preview=preview, **extra))
        summary = "%s\ncfg %g from the %s" % (result.summary, value, "preset" if source == "preset" else "user (widget)")
        return {"ui": {"text": [summary]}, "result": (result.latent, result.report)}

    def sample(self, model, positive, negative, latent, seed, preset, cfg, identity_profile=None, constraints=None,
               vae=None, preview=True, preset_pin=""):
        return self._run(NODE_ID, model, positive, negative, latent, seed, preset, cfg, identity_profile, constraints,
                         vae, preview, preset_pin)


class MossPhotonSamplerAdvanced(MossPhotonSampler):
    """Everything of MossPhotonSampler plus explicit overrides; every widget at its sentinel means "from preset"."""

    @classmethod
    def INPUT_TYPES(cls):
        spec = MossPhotonSampler.INPUT_TYPES()
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
                                     "tooltip": "0 = unlimited. A plan above the limit stops before sampling."}),
            "fingerprint": (list(FINGERPRINT_MODES), {"default": "cached_or_compute",
                                                      "tooltip": "How the checkpoint sha256 is obtained."}),
        })
        spec["optional"].update({
            "sigmas": ("SIGMAS", {"tooltip": "Replaces the native schedule after validation; not bit-equal to the "
                                             "native one marks the run not_reference."}),
            "report_step_times": ("BOOLEAN", {"default": False, "tooltip": "Add timing.step_s to the report."}),
        })
        return spec

    def sample(self, model, positive, negative, latent, seed, preset, cfg, steps, slg_mode, slg_layers, slg_start,
               slg_end, solver, cfg_profile, noise_policy, max_forwards, fingerprint, identity_profile=None,
               constraints=None, vae=None, preview=True, preset_pin="", sigmas=None, report_step_times=False):
        if solver != "euler":
            errors.raise_first_error([errors.make_issue(
                "unsupported_feature", ADVANCED_NODE_ID, "solver", "solver %r is not available in v1" % (solver,),
                "use solver euler", severity="error")])
        return self._run(
            ADVANCED_NODE_ID, model, positive, negative, latent, seed, preset, cfg, identity_profile, constraints, vae,
            preview, preset_pin, sigmas=sigmas, noise_policy=noise_policy, fingerprint_mode=fingerprint,
            max_forwards=max_forwards, report_step_times=report_step_times,
            allow_slg_override=True,  # MP-30 M2: the one place an SLG override may stand on a turbo preset
            overrides=build_overrides(steps, slg_mode, slg_layers, slg_start, slg_end, has_sigmas=sigmas is not None))


NODE_CLASS_MAPPINGS = {NODE_ID: MossPhotonSampler, ADVANCED_NODE_ID: MossPhotonSamplerAdvanced}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "Moss Photon Sampler", ADVANCED_NODE_ID: "Moss Photon Sampler (Advanced)"}
