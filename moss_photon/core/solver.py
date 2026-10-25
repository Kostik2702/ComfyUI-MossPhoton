"""The sampling run: preflight, label, SLG clone, plain CFGGuider + euler, report (design 1.5, 4.5, 5.1).

``execute`` is the whole run. Everything that can be decided without a forward is decided first and raises a typed
``MossPhotonError`` before any clone, noise tensor or GPU allocation exists (preflight order of 5.1). The sampling
route is exactly the stock one: ``comfy.samplers.CFGGuider`` + ``sampler_object("euler")`` + ``guider.sample`` on ONE
``model.clone()`` that carries the SLG hook and a forward counter; the incoming MODEL is never written to.

The profile and the constraints arrive as plain dicts (PHOTON_IDENTITY_PROFILE, PHOTON_CONSTRAINTS) and are read by
key only: this module never imports ``constraints`` or ``lora`` (M4). Imports: stdlib and the MP-6 / MP-7 core
modules; comfy and torch only inside functions (R1).
"""
import dataclasses
import logging
import math
import time

from . import (adapter, canon, consts, errors, fingerprint, guard, guidance, lora_phase, phase, presets, report, rng,
               sigma)

__all__ = [
    "SampleRequest", "SampleResult", "RunPlan", "ProfileCheck", "ForwardStats", "StepTelemetry", "TURBO_PARITY_CLASS",
    "is_turbo", "plan_run", "check_layout_extent", "check_profile", "classify_label", "family_issues",
    "install_counter", "build_guider", "get_sampler", "make_callback", "execute",
]

_log = logging.getLogger("MossPhoton")

# parity_class of a run of a turbo preset: the reference is the native turbo graph (docs/03-turbo.md 3.1, Q14).
TURBO_PARITY_CLASS = "native_turbo"
_TURBO_POLICIES = ("require_distill_profile", "require_distill_and_character_profile")
_COUNTER_KEY = "moss_photon_counter"
_KEYS_SAMPLE = 20
_REFERENCE_ID = "reference"


# --------------------------------------------------------------------------- data
@dataclasses.dataclass
class SampleRequest:
    """Everything one run needs. ``cfg`` always wins over the preset; ``overrides`` holds the other widgets
    (``steps slg_mode slg_layers slg_start slg_end``) that differ from the preset. ``max_forwards`` 0 = unlimited.
    ``reference_dtypes`` ``None`` = ``consts.REFERENCE_WEIGHT_DTYPES`` (CPU tests pass ``("torch.float32",)``).
    ``allow_slg_override`` lets an SLG override stand on a turbo preset (experimental, label ``not_reference``); only
    MossPhotonSamplerAdvanced sets it (MP-30 M2). ``allow_phased_lora`` lets an identity profile with a phased LoRA
    weight run (experimental, label ``not_reference`` / ``lora_phased``); only MossPhotonSamplerAdvanced sets it (MP-16)."""

    node_id: str
    model: object
    positive: object
    negative: object
    latent: dict
    seed: int
    preset_id: str
    preset_pin: str = ""
    cfg: float = 3.0
    overrides: dict = dataclasses.field(default_factory=dict)
    sigmas: object = None
    noise_policy: str = "native"
    identity_profile: object = None
    constraints: object = None
    vae: object = None
    preview: bool = True
    fingerprint_mode: str = "cached_or_compute"
    max_forwards: int = 0
    reference_dtypes: object = None
    report_step_times: bool = False
    allow_slg_override: bool = False
    allow_phased_lora: bool = False


@dataclasses.dataclass
class SampleResult:
    latent: dict
    report: dict
    summary: str


@dataclasses.dataclass(frozen=True)
class RunPlan:
    """The exact forward count, known before the run: ``steps * branches + extra`` model forwards per image."""

    steps: int
    cfg: float
    branches: int
    extra: int
    batch: int
    expected_branch_forwards: int
    expected_sample_forwards: int
    limit: object  # smallest positive budget limit that applies, or None


@dataclasses.dataclass(frozen=True)
class ProfileCheck:
    """Result of comparing a PHOTON_IDENTITY_PROFILE with the MODEL (design 1.5)."""

    issues: tuple
    foreign: bool
    has_lora: bool
    telemetry: dict


class ForwardStats:
    """Observed model forwards: ``calls`` of ``apply_model``, ``elements`` = sum of batch sizes, ``skip_calls`` =
    calls that ran with a ``single_block`` replacement (the SLG unconditional pass)."""

    def __init__(self):
        self.calls = 0
        self.elements = 0
        self.skip_calls = 0


class StepTelemetry:
    """Seconds per step, measured between sampler callbacks; never touches a random generator."""

    def __init__(self, perf=time.perf_counter):
        self._perf = perf
        self._last = None
        self.steps = []

    def start(self):
        self._last = self._perf()

    def step(self, index):
        now = self._perf()
        if self._last is not None:
            self.steps.append(now - self._last)
        self._last = now

    def summary(self):
        if not self.steps:
            return {"step_s_mean": None, "step_s_max": None}
        return {"step_s_mean": round(sum(self.steps) / len(self.steps), 6), "step_s_max": round(max(self.steps), 6)}


# --------------------------------------------------------------------------- pure helpers
def is_turbo(preset):
    """True for a turbo preset: id ``turbo_*`` or a ``require_distill*`` LoRA policy (docs/03-turbo.md 5.2)."""
    return str(preset.get("id", "")).startswith("turbo_") or preset.get("lora_policy") in _TURBO_POLICIES


def plan_run(resolved, latent_facts, sigma_values, *, max_forwards, node="MossPhotonSampler"):
    """The forward budget of a run; pure. Raises ``budget_exceeded`` when the plan is over a limit.

    ``steps`` is ``len(sigma_values) - 1`` (custom sigmas replace the preset steps), ``cfg`` and ``extra_forwards``
    come from ``resolved.effective``. Limits: the preset's ``budget.max_forwards`` (``None`` = none) and the
    Advanced ``max_forwards`` (``0`` = none). The count is of model forwards per image; the sample count is that
    times the latent batch.
    """
    steps = len(sigma_values) - 1
    cfg = resolved.effective["sampling"]["cfg"]
    budget = resolved.effective["budget"]
    extra = budget["extra_forwards"]
    expected = sigma.expected_forwards(steps, cfg, extra)
    limits = [(v, name) for v, name in ((budget["max_forwards"], "the preset budget"),
                                         (max_forwards or None, "max_forwards")) if v]
    for value, name in limits:
        if expected > value:
            errors.raise_first_error([errors.make_issue(
                "budget_exceeded", node, "steps",
                "plan = %d forwards, limit = %d (%s)" % (expected, value, name),
                "lower steps or raise max_forwards", severity="error",
                detail={"planned": expected, "limit": value, "limit_source": name})])
    return RunPlan(steps=steps, cfg=cfg, branches=sigma.branches_per_step(cfg), extra=extra,
                   batch=latent_facts.batch, expected_branch_forwards=expected,
                   expected_sample_forwards=expected * latent_facts.batch,
                   limit=min([v for v, _ in limits]) if limits else None)


def check_layout_extent(constraints, pixel_width, pixel_height, *, node):
    """``invalid_layout`` issues when the layout boxes leave the canvas (A23); pure.

    ``constraints["layout"]["extent"] = [ymin, xmin, ymax, xmax]`` in pixels, abs_centralized: the centre of the
    canvas is the origin, so every y must lie in ``[-H/2, H/2]`` and every x in ``[-W/2, W/2]`` (edges included).
    No constraints, no layout, or no extent: nothing to check. ``H`` and ``W`` are the pixel size of the latent.
    """
    layout = (constraints or {}).get("layout")
    if not layout:
        return []
    extent = layout.get("extent")

    def bad(message, **detail):
        return [errors.make_issue(
            "invalid_layout", node, "constraints.layout.extent", message,
            "keep every layout box inside the canvas, or enlarge the latent", detail=detail)]

    if extent is None:
        return []
    if (not isinstance(extent, (list, tuple)) or len(extent) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in extent)):
        return bad("layout extent %r is not four finite numbers [ymin, xmin, ymax, xmax]" % (extent,))
    ymin, xmin, ymax, xmax = extent
    half_h, half_w = pixel_height / 2.0, pixel_width / 2.0
    out = []
    for name, value, low, high in (("ymin", ymin, -half_h, half_h), ("ymax", ymax, -half_h, half_h),
                                   ("xmin", xmin, -half_w, half_w), ("xmax", xmax, -half_w, half_w)):
        if not low <= value <= high:
            out.append(errors.make_issue(
                "invalid_layout", node, "constraints.layout.extent",
                "layout %s = %g is outside the %dx%d canvas (allowed %g..%g)"
                % (name, value, pixel_width, pixel_height, low, high),
                "keep every layout box inside the canvas, or enlarge the latent",
                detail={"edge": name, "value": value, "low": low, "high": high,
                        "pixel_width": pixel_width, "pixel_height": pixel_height}))
    return out


def _profile_shape_issue(profile, node):
    """An ``incompatible_lora`` / ``unsupported_feature`` issue for a profile that is not a known contract."""
    version = profile.get("schema_version")
    if isinstance(version, int) and not isinstance(version, bool) and version > consts.IDENTITY_PROFILE_SCHEMA:
        return errors.make_issue(
            "unsupported_feature", node, "identity_profile",
            "identity_profile schema_version = %d is newer than this MossPhoton knows (%d)"
            % (version, consts.IDENTITY_PROFILE_SCHEMA), "update the ComfyUI-MossPhoton pack", severity="error")
    missing = [k for k in ("kind", "stack", "attachment") if k not in profile]
    if profile.get("kind") != "photon_identity_profile" or missing or not isinstance(profile.get("stack"), list):
        return errors.make_issue(
            "incompatible_lora", node, "identity_profile",
            "the connected value is not a PHOTON_IDENTITY_PROFILE (kind %r, missing keys %s)"
            % (profile.get("kind"), missing or "none"),
            "connect the identity_profile output of MossPhotonLoRAProfile", severity="error")
    return None


def check_profile(profile, facts, preset_requires_lora, *, node):
    """Compare the identity profile with the MODEL; pure, reads dict fields and ``facts`` only (M4, A22).

    Order of design 1.5: (1) no profile, or an EMPTY one (``stack == []`` / ``attachment.entries == []``, what
    MossPhotonLoRAProfile returns for ``lora_name = none`` or strength 0; the MODEL then carries no attachment
    string): any patch entry on the MODEL is foreign;
    ``preset_requires_lora`` -> ``missing_lora``. (2) the MODEL's attachment string differs from
    ``canonical_json(profile["attachment"])`` -> ``incompatible_lora``. (3) strings equal but the live
    ``patches_uuid`` or entry count differ from the attachment's -> foreign patches added after the chain.
    (4) ``attachment.baseline_entries > 0`` -> foreign patches existed before the chain. Foreign patches are a
    warning ``foreign_patches`` plus ``telemetry``, not an error. ``telemetry`` is ``lora.foreign_patches`` without
    ``keys_sample`` (the caller fills it from the live model). ``facts``: ``lora_stack_attachment``,
    ``patches_uuid``, ``patch_entries``.
    """
    observed = int(facts.patch_entries)
    issues = []
    if profile is not None:
        shape = _profile_shape_issue(profile, node)
        if shape is not None:
            return ProfileCheck((shape,), False, False, _telemetry(0, observed, 0, True))
    stack = profile["stack"] if profile is not None else []
    if not stack or not profile["attachment"].get("entries"):
        if preset_requires_lora:
            issues.append(errors.make_issue(
                "missing_lora", node, "identity_profile",
                "the preset needs a LoRA profile and none with an adapter is connected",
                "add MossPhotonLoRAProfile (role=character) and connect its identity_profile output",
                severity="error"))
        foreign = observed > 0
        if foreign:
            issues.append(_foreign_issue(node, 0, observed, 0, False))
        return ProfileCheck(tuple(issues), foreign, False, _telemetry(0, observed, 0, not foreign))
    attachment = profile["attachment"]
    if facts.lora_stack_attachment is None or facts.lora_stack_attachment != canon.canonical_json(attachment):
        return ProfileCheck((errors.make_issue(
            "incompatible_lora", node, "identity_profile",
            "identity_profile does not belong to this MODEL: the MODEL was not produced by that LoRA chain",
            "connect the MODEL output of the same MossPhotonLoRAProfile chain", severity="error"),),
            False, True, _telemetry(attachment.get("patch_entries_total", 0), observed,
                                    attachment.get("baseline_entries", 0), False))
    expected = int(attachment["patch_entries_total"])
    baseline = int(attachment["baseline_entries"])
    uuid_match = facts.patches_uuid == attachment["patches_uuid"]
    foreign = (not uuid_match) or observed != expected or baseline > 0
    if foreign:
        issues.append(_foreign_issue(node, expected, observed, baseline, True))
    return ProfileCheck(tuple(issues), foreign, True, _telemetry(expected, observed, baseline, uuid_match))


def _telemetry(expected, observed, baseline, uuid_match):
    return {"entries_expected": int(expected), "entries_observed": int(observed), "baseline_entries": int(baseline),
            "uuid_match": bool(uuid_match), "extra_entries": max(0, int(observed) - int(expected)),
            "keys_sample": []}


def _foreign_issue(node, expected, observed, baseline, has_profile):
    what = ("the MODEL carries %d weight patch entries, the identity_profile explains %d (baseline before the chain: %d)"
            % (observed, expected, baseline)) if has_profile else (
        "the MODEL carries %d weight patch entries and no identity_profile explains them" % observed)
    return errors.make_issue(
        "foreign_patches", node, "model", what,
        "apply every LoRA through MossPhotonLoRAProfile and connect its MODEL; remove native LoRA loaders",
        detail={"entries_expected": int(expected), "entries_observed": int(observed),
                "baseline_entries": int(baseline)})


def _slg_signature(sampling):
    if sampling["slg_mode"] != "uncond_skip":
        return ("off",)
    return ("uncond_skip", tuple(sampling["slg_layers"]), float(sampling["slg_start"]), float(sampling["slg_end"]))


def _outside_verified(preset, steps_used, cfg):
    """True when ``steps_used`` or ``cfg`` is outside the verified range of the turbo preset's ``override_bounds``."""
    bounds = preset["sampling"].get("override_bounds") or {}
    for value, name in ((steps_used, "steps"), (cfg, "cfg")):
        b = bounds.get(name)
        if b is not None and not b["verified_min"] - 1e-9 <= value <= b["verified_max"] + 1e-9:
            return True
    return False


def classify_label(*, preset, effective, customized, reference_sampling, steps_used, cfg, live_shift,
                   sigmas_bit_equal_native, noise_policy, modifiers, foreign_patches, family_verified, has_lora,
                   slg_applied, calibrated, family_in_preset=True, phased=False):
    """The report block ``label`` (ladder of A21); pure.

    ``not_reference`` > ``reference_modified`` > ``reference_with_foreign_patches`` > ``reference_unverified`` >
    ``reference_with_lora`` > ``reference_exact``. A turbo preset (``is_turbo``) is compared with the native turbo
    graph instead: its steps, cfg and a LoRA stack are the recipe, so only custom sigmas, ``seed_plus_i``, another
    live shift, an applied SLG or a checkpoint family the preset does not list make it ``not_reference``, and a clean
    run is ``native_turbo``. A turbo run whose steps or cfg differ from the preset still equals the native graph with
    the same settings (lead decision, MP-30 M1): it stays ``native_turbo``, ``customized`` is true and
    ``not_reference_reasons`` lists ``steps_differ`` / ``cfg_differs`` and, outside the verified range of
    ``override_bounds``, ``turbo_setting_outside_verified`` (for turbo these three describe, they do not demote).
    ``effective`` is ``ResolvedPreset.effective``, ``reference_sampling`` the ``sampling`` block of the ``reference``
    preset, ``steps_used`` the number of sampler steps actually run. ``family_in_preset`` is false when the identified
    checkpoint family is not in the preset's ``model_families`` (M7): ``reference_unverified``, or for turbo
    ``not_reference`` with the reason ``model_family_unverified``. ``phased`` (a LoRA weight that moves with the
    sampling progress, MP-16) is always ``not_reference`` with the reason ``lora_phased``, turbo presets included.
    """
    sampling = effective["sampling"]
    reasons = []
    notes = []  # turbo only: what differs from the preset, without demoting the class
    turbo = is_turbo(preset)
    if not turbo:
        if steps_used != reference_sampling["steps"]:
            reasons.append("steps_differ")
        if not math.isclose(cfg, reference_sampling["cfg"], rel_tol=0.0, abs_tol=1e-9):
            reasons.append("cfg_differs")
        if _slg_signature(sampling) != _slg_signature(reference_sampling):
            reasons.append("slg_differs")
    else:
        preset_sampling = preset["sampling"]
        if steps_used != preset_sampling["steps"]:
            notes.append("steps_differ")
        if not math.isclose(cfg, preset_sampling["cfg"], rel_tol=0.0, abs_tol=1e-9):
            notes.append("cfg_differs")
        if _outside_verified(preset, steps_used, cfg):
            notes.append("turbo_setting_outside_verified")
        if slg_applied:
            reasons.append("slg_differs")
        if not family_in_preset:
            reasons.append("model_family_unverified")
    if not sigmas_bit_equal_native:
        reasons.append("custom_sigmas")
    if live_shift is not None and not math.isclose(live_shift, reference_sampling["shift"], rel_tol=0.0, abs_tol=1e-9):
        reasons.append("shift_differs")
    if noise_policy != "native":
        reasons.append("noise_seed_plus_i")
    if phased:
        reasons.append("lora_phased")
    if not turbo:
        if preset["status"] == "preview_only":
            reasons.append("draft_preset")
        if customized:
            reasons.append("customized")
    if reasons:
        parity = "not_reference"
    elif modifiers:
        parity = "reference_modified"
    elif foreign_patches:
        parity = "reference_with_foreign_patches"
    elif not family_verified or not family_in_preset:
        parity = "reference_unverified"
    elif turbo:
        parity = TURBO_PARITY_CLASS
    elif has_lora:
        parity = "reference_with_lora"
    else:
        parity = "reference_exact"
    ui = preset["ui"]
    return {
        "preset_status": preset["status"],
        "ui_label": ui["label"] + ("+customized" if customized else ""),
        "parity_class": parity,
        "not_reference_reasons": notes + reasons,
        "modifiers": list(modifiers),
        "calibrated": bool(calibrated),
        "customized": bool(customized),
        "recommended_final": bool(ui["recommended_final"]),
    }


def _trigger_issues(profile, constraints, node):
    """Warning ``trigger_not_in_prompt`` for every adapter whose trigger words are all missing from the compiled
    prompt. Needs the constraints input; the words themselves (private) are never put into the message."""
    text = (constraints or {}).get("compiled_positive")
    if not isinstance(text, str) or profile is None:
        return []
    lowered = text.lower()
    out = []
    for entry in profile.get("stack", []):
        hints = [h for h in entry.get("trigger_hints") or [] if isinstance(h, str) and h.strip()]
        if hints and not any(h.strip().lower() in lowered for h in hints):
            lora = entry.get("lora") or {}
            out.append(errors.make_issue(
                "trigger_not_in_prompt", node, "positive",
                "the %s LoRA %s has trigger words and none of them appears in the compiled prompt"
                % (entry.get("role"), lora.get("basename", "?")),
                "add the LoRA's trigger word to the prompt (MossPhotonPromptLayout identity block)"))
    return out


def family_issues(preset, family, node):
    """Warning ``model_family_unverified`` when the identified checkpoint ``family`` is not in the preset's
    ``model_families`` (SPEC 6.1 "допустимые модели", MP-30 M7); pure.

    An unidentified checkpoint (``unknown-moss``) is skipped: ``unverified_checkpoint`` / ``fingerprint_unavailable``
    already say so, and the label is ``reference_unverified``. ``checkpoint_policy`` is deliberately not read: its only
    value in use, ``native-moss-only``, is what the run already does (any MOSS-architecture checkpoint, an unknown hash
    is the warning ``unverified_checkpoint``); ``native-moss-verified-only`` is reserved by the design and no preset
    uses it.
    """
    families = list(preset["model_families"])
    if family == "unknown-moss" or family in families:
        return []
    return [errors.make_issue(
        "model_family_unverified", node, "model",
        "the checkpoint is %s, preset %s lists %s" % (family, preset["id"], ", ".join(families)),
        "load a checkpoint of the listed family, or choose a preset that lists %s" % family,
        detail={"family": family, "preset_families": families})]


# --------------------------------------------------------------------------- comfy-facing helpers (lazy imports)
def install_counter(model_clone, stats):
    """Install the ``APPLY_MODEL`` forward counter on ``model_clone`` (a clone, never the incoming MODEL).

    The wrapper only counts and calls through, so it is bit-transparent (A5, check C2). ``stats`` is a
    ``ForwardStats``.
    """
    import comfy.patcher_extension as pe

    def counter(executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        stats.calls += 1
        stats.elements += int(x.shape[0])
        replaced = (transformer_options.get("patches_replace") or {}).get("dit") or {}
        if any(key[0] == guidance.SKIP_BLOCK_NAME for key in replaced):
            stats.skip_calls += 1
        return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    pe.add_wrapper_with_key(pe.WrappersMP.APPLY_MODEL, _COUNTER_KEY, counter, model_clone.model_options,
                            is_model_options=True)


def build_guider(patcher, positive, negative, cfg):
    """The plain ``comfy.samplers.CFGGuider`` of the stock path (A1)."""
    import comfy.samplers
    guider = comfy.samplers.CFGGuider(patcher)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    return guider


def get_sampler(name):
    """``comfy.samplers.sampler_object(name)``; only ``euler`` exists in v1 (``unsupported_feature`` otherwise)."""
    if name != "euler":
        errors.raise_first_error([errors.make_issue(
            "unsupported_feature", "MossPhotonSamplerAdvanced", "solver", "solver %r is not available in v1" % (name,),
            "use solver euler", severity="error")])
    import comfy.samplers
    return comfy.samplers.sampler_object(name)


def _poll_interrupt():
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


def make_callback(patcher, total_steps, *, preview, telemetry, finite_check, node="MossPhotonSampler"):
    """The per-step callback ``(step, x0, x, total)`` for the sampler; it never draws a random number.

    In order: step timing, finite check of the denoised prediction (``non_finite_output`` at that step), the
    preview/progress callback, an interrupt poll. ``preview`` wraps ComfyUI's ``latent_preview.prepare_callback``
    (progress bar plus preview images); without it a plain progress bar is updated. A preview that cannot be built
    is logged and skipped. The returned function has the attribute ``preview_active``.
    """
    base = None
    if preview:
        try:
            import latent_preview
            base = latent_preview.prepare_callback(patcher, total_steps)
        except Exception as exc:  # noqa: BLE001 - a missing previewer must not stop a run
            _log.warning("[MossPhoton] preview unavailable, continuing without it: %s", exc)
    pbar = None
    if base is None:
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(total_steps)
        except Exception as exc:  # noqa: BLE001
            _log.debug("[MossPhoton] no progress bar: %s", exc)

    def callback(step, x0, x, total):
        telemetry.step(step)
        if finite_check:
            guard.check_finite(x0, node=node, where="step %d" % step)
        if base is not None:
            base(step, x0, x, total)
        elif pbar is not None:
            pbar.update_absolute(step + 1, total)
        _poll_interrupt()

    callback.preview_active = base is not None
    return callback


def _device_info(device_name):
    return {"measured": False, "peak_vram_bytes": None, "device": str(device_name)}


def _memory_begin(device_name):
    if str(device_name).startswith("cuda"):
        import torch
        torch.cuda.reset_peak_memory_stats()


def _memory_end(device_name):
    if str(device_name).startswith("cuda"):
        import torch
        return {"measured": True, "peak_vram_bytes": int(torch.cuda.max_memory_allocated()), "device": str(device_name)}
    return _device_info(device_name)


def _patch_keys_sample(model):
    keys = []
    for key in getattr(model, "patches", None) or {}:
        keys.append(str(key if isinstance(key, str) else key[0]))
        if len(keys) >= _KEYS_SAMPLE:
            break
    return keys


def _lora_stack(profile):
    stack = []
    for entry in (profile or {}).get("stack", []):
        lora, coverage = entry["lora"], entry.get("coverage") or {}
        stack.append({
            "role": entry["role"], "basename": lora["basename"], "size": lora["size"], "sha256": lora.get("sha256"),
            "strength_model": entry["strength_model"], "strength_clip": entry["strength_clip"],
            "applied_to": dict(entry["applied_to"]), "coverage_ratio": coverage.get("ratio", 0.0),
            "layers_missing": list(coverage.get("layers_missing") or []),
            "keys_patched": entry["keys_patched"], "patch_entries": entry["patch_entries"],
            "trigger_present": bool(entry.get("trigger_present")),
        })
    return stack


def _cond_block(facts):
    return {"entries": facts.entries, "tokens": list(facts.token_counts), "dim": facts.dim, "sha256": facts.sha256}


def _summary(rep):
    slg = rep["sampling"]["slg"]
    line = "slg %s" % slg["mode"]
    if slg["mode"] == "uncond_skip":
        line += " layers %s window %s..%s" % (slg["layers"], slg["start"], slg["end"])
    return report.format_summary(rep) + "\n" + line


# --------------------------------------------------------------------------- the run
def execute(req):
    """One sampler run; returns a ``SampleResult``. Every typed failure is raised before the first forward.

    Preflight (nothing is moved to the GPU, nothing is mutated), in order: preset, pin and overrides; model facts;
    latent (after ComfyUI's own empty-latent repair), conditioning, VAE; requirements of the preset; trigger words;
    profile vs MODEL; phase plan; required guards; SLG; layout extent; sigmas; budget (all cheap, before the checkpoint
    is hashed); checkpoint identity; family gate; label.
    Execution: one clone with the SLG hook and the counter, noise, plain ``CFGGuider`` + euler inside
    ``torch.inference_mode()``, ``to_intermediate``, finite check, report. A phased LoRA weight (MP-16, Advanced only)
    runs the same guider once per weight segment, each on its own direct clone (``lora_phase.run_segments``). Interrupts and errors propagate; the
    references to the clone, the guider and the noise are dropped in ``finally``.
    """
    node = req.node_id
    rb = report.ReportBuilder(node)

    def note(issues):
        for issue in issues:
            if issue.severity != "error":
                rb.warn(issue)

    def check(issues):
        errors.raise_first_error(issues)
        note(issues)

    # 1. preset, pin, overrides: cheap and pure, before the model is looked at
    other = {k: v for k, v in (req.overrides or {}).items() if k != "cfg"}
    cfg = float(req.cfg)
    allow_slg = req.allow_slg_override
    first = presets.resolve(req.preset_id, overrides=other or None, pin=req.preset_pin, node=node,
                            allow_slg_override=allow_slg)
    overrides = dict(other)
    if not math.isclose(cfg, first.effective["sampling"]["cfg"], rel_tol=0.0, abs_tol=1e-9):
        overrides["cfg"] = cfg
        first = presets.resolve(req.preset_id, overrides=overrides, pin=req.preset_pin, node=node,
                                allow_slg_override=allow_slg)
    smp = first.effective["sampling"]
    note(presets.catalog_issues())  # m22: a broken preset file disables only itself; say so

    # 2. model
    facts = adapter.probe_model(req.model, reference_dtypes=req.reference_dtypes)
    sigma_max = facts.sigma_max if facts.sigma_max is not None else 1.0
    branches = sigma.branches_per_step(cfg)
    slg_wanted = smp["slg_mode"] == "uncond_skip" and bool(smp["slg_layers"])
    slg_applied = slg_wanted and branches == 2
    check(adapter.validate_model_facts(facts, node=node, slg_requested=slg_applied))

    # 3. inputs, after the native empty-latent repair (M8)
    fixed = adapter.fix_empty_latent(req.model, req.latent, node=node)
    lfacts = adapter.probe_latent(req.latent, fixed)
    check(adapter.validate_latent_facts(lfacts, channels=adapter.LATENT_CHANNELS, node=node))
    pos_facts = adapter.probe_conditioning(req.positive, "positive")
    neg_facts = adapter.probe_conditioning(req.negative, "negative")
    check(adapter.validate_cond_facts(pos_facts, name="positive", text_dim=facts.text_dim, node=node)
          + adapter.validate_cond_facts(neg_facts, name="negative", text_dim=facts.text_dim, node=node))
    if req.vae is not None:
        check(adapter.validate_vae_facts(adapter.probe_vae(req.vae), channels=adapter.LATENT_CHANNELS,
                                         ratio=adapter.SPATIAL_RATIO, node=node))
    if req.noise_policy not in rng.NOISE_POLICIES:
        check([errors.make_issue("unsupported_feature", node, "noise_policy",
                                 "noise policy %r is not one of %s" % (req.noise_policy, ", ".join(rng.NOISE_POLICIES)),
                                 "use noise policy native or seed_plus_i", severity="error")])
    if req.noise_policy == "seed_plus_i" and lfacts.has_batch_index:
        check([errors.make_issue("unsupported_feature", node, "noise_policy",
                                 "noise policy seed_plus_i does not support a latent with batch_index",
                                 "use noise policy native for this latent", severity="error")])
    note(rng.seed_warnings(req.seed, lfacts.batch, req.noise_policy, node=node))

    # 4. requirements of the preset, profile vs MODEL, phase plan, guards: cheap, and the preset's constraints do not
    # depend on the checkpoint family, so a run these refuse never pays for the checkpoint hash (m1)
    profile, constraints = req.identity_profile, req.constraints
    preset = first.preset
    check(presets.evaluate_requirements(first, {
        "lora_roles": [s.get("role") for s in (profile or {}).get("stack", [])],
        "lora_stack": [{"basename": (s.get("lora") or {}).get("basename"), "role": s.get("role"),
                        "sha256": (s.get("lora") or {}).get("sha256")} for s in (profile or {}).get("stack", [])],
        "blocks_present": (constraints or {}).get("blocks_present") or {},
        "latent_is_empty": lfacts.is_empty, "has_noise_mask": lfacts.has_noise_mask, "reference_photos": False,
    }, node))
    note(_trigger_issues(profile, constraints, node))
    for w in (constraints or {}).get("warnings") or []:
        rb.warn(w)
    requires_lora = any(c["kind"] == "requires_lora_role" for c in preset["constraints"])
    pcheck = check_profile(profile, facts, requires_lora, node=node)
    check(list(pcheck.issues))
    phase_plan = phase.check_phase_plan(profile, node=node, allowed=req.allow_phased_lora)
    for i, entry in enumerate((profile or {}).get("stack", [])):
        for w in entry.get("warnings") or []:
            item = w.to_dict() if hasattr(w, "to_dict") else dict(w)
            # m8 / NEW-5: which adapter it belongs to, under one key (check_stack's detail.index is the same number;
            # the profile keeps it, the stack-rule tests and the LoRA node read it there)
            detail = {k: v for k, v in (item.get("detail") or {}).items() if k != "index"}
            item["detail"] = dict(detail, stack_index=i)
            rb.warn(item, scope="lora[%d]" % i)
    required = guard.required_guards(preset)
    if required:
        check([errors.make_issue(
            "missing_guard_dependency", node, "guard", "the preset requires guard(s) %s that v1 cannot run"
            % ", ".join(required), "choose a preset whose guards are optional", severity="error")])
    items = guard.guard_items(preset)
    note([errors.make_issue("guard_not_run", node, g["id"], "guard %s did not run (%s)" % (g["id"], g["reason"]),
                            "none needed; detectors arrive with the second queue") for g in items])

    # 5. SLG, layout, sigmas, forward budget: pure arithmetic on the first resolve, before the checkpoint is hashed (m1)
    check(guidance.validate_slg(smp["slg_mode"], smp["slg_layers"], smp["slg_start"], smp["slg_end"],
                                num_layers=facts.num_layers, node=node))
    check(check_layout_extent(constraints, lfacts.pixel_width, lfacts.pixel_height, node=node))
    scheduler = smp["scheduler"]
    if req.sigmas is None:
        native = adapter.native_sigmas(req.model, scheduler, smp["steps"])
        values = native.tolist()
        check(sigma.validate_sigmas(values, node=node, param="steps", sigma_max=sigma_max,
                                    expect_steps=smp["steps"]))
        sigmas_source, bit_equal = "native_scheduler", True
    else:
        values = req.sigmas.tolist() if hasattr(req.sigmas, "tolist") else list(req.sigmas)
        check(sigma.validate_sigmas(values, node=node, param="sigmas", sigma_max=sigma_max,
                                    ratio_is_error=False))
        fresh = adapter.native_sigmas(req.model, scheduler, len(values) - 1).tolist()
        bit_equal = sigma.compare_sigmas(values, fresh)["bit_equal"]
        sigmas_source = "input"
        if not bit_equal:
            note([errors.make_issue("custom_sigmas", node, "sigmas",
                                    "the connected sigmas are not the native %s schedule for %d steps"
                                    % (scheduler, len(values) - 1),
                                    "use MossPhotonSchedule(mode=reference) for the verified schedule")])
    if not lfacts.is_empty:
        if values[0] >= 1.0:
            note([errors.make_issue("latent_content_ignored", node, "latent",
                                    "first sigma is 1.0: the latent content is multiplied by (1 - sigma) = 0",
                                    "none needed; feed MossPhotonEmptyLatent")])
        else:
            note([errors.make_issue("img2img_not_verified", node, "latent",
                                    "a non-empty latent with first sigma %g runs image-to-image, which is outside "
                                    "the verified scope" % values[0], "use sigma 1.0 and an empty latent")])
    phased_run = None
    if phase_plan is not None:  # MP-16, EXPERIMENTAL: the weights are known from the sigmas alone, before any forward
        adapter_on_model = lora_phase.adapter_of(req.model)
        check(lora_phase.check_adapter(adapter_on_model, profile["stack"][phase_plan.stack_index], phase_plan,
                                       node=node))
        schedule = phase.build_schedule(values, phase_plan)
        phased_run = (adapter_on_model, schedule)
        note([errors.make_issue(
            "phase_lora_experimental", node, "identity_profile",
            "stack[%d] runs with a LoRA weight that moves %g -> %g over the sampling progress %g..%g (%d steps, %d "
            "weight segments, one model clone each)" % (phase_plan.stack_index, phase_plan.w0, phase_plan.w1,
                                                        phase_plan.a, phase_plan.b, len(schedule.applied),
                                                        len(schedule.segments)),
            "experimental and outside every verification; use a constant weight for a reference result",
            detail={"stack_index": phase_plan.stack_index, "levels": phase_plan.levels})])
    plan = plan_run(first, lfacts, values, max_forwards=req.max_forwards, node=node)
    steps_used = plan.steps

    # 6. checkpoint identity (may hash the file once; interruptible)
    ident = fingerprint.identify_checkpoint(req.model, mode=req.fingerprint_mode, node=node,
                                            interrupt_check=_poll_interrupt)
    note(ident["warnings"])
    family = ident["family"]
    identity = ident["identity"]

    # 7. settings with the family (calibration manifest) and the profile
    resolved = presets.resolve(req.preset_id, overrides=overrides or None, pin=req.preset_pin, node=node,
                               model_family=family, lora_profile=profile, num_layers=facts.num_layers,
                               allow_slg_override=allow_slg)
    preset = resolved.preset
    note(resolved.warnings)
    if any(w.code == "manifest_missing" for w in resolved.warnings):
        rb.fallback("manifest_missing", "reference configuration used")
    family_gate = family_issues(preset, family, node)
    note(family_gate)
    if resolved.effective != first.effective:  # a calibration manifest changed the settings: plan again with them
        plan = plan_run(resolved, lfacts, values, max_forwards=req.max_forwards, node=node)
        steps_used = plan.steps

    # 8. label and report sections
    live_shift = facts.shift if facts.shift is not None else smp["shift"]
    diff = list(resolved.diff)
    shift_changed = not math.isclose(live_shift, smp["shift"], rel_tol=0.0, abs_tol=1e-9)
    if shift_changed:
        diff = sorted(diff + [{"path": "sampling.shift", "base": smp["shift"], "effective": live_shift}],
                      key=lambda d: d["path"])
        note([errors.make_issue("shift_differs_from_preset", node, "model",
                                "the live model shift is %g, the preset says %g" % (live_shift, smp["shift"]),
                                "remove ModelSamplingAuraFlow for the verified shift, or accept the change")])
    customized = resolved.customized or shift_changed
    if branches == 1:
        note([errors.make_issue("negative_unused_cfg1", node, "negative",
                                "the negative prompt is not evaluated at CFG 1.0",
                                "raise CFG above 1.0 to use the negative prompt")])
        if slg_wanted:
            note([errors.make_issue("slg_ineffective_cfg1", node, "slg_mode",
                                    "SLG is skipped at CFG 1.0 (there is no unconditional pass); forwards unchanged",
                                    "raise CFG above 1.0 to use SLG, or set slg_mode off")])
    if preset["status"] == "preview_only":
        note([errors.make_issue("preview_only_preset", node, "preset",
                                "preset %s is a draft" % preset["id"],
                                "re-run the chosen seed with a final preset")])
    foreign_block = dict(pcheck.telemetry)
    if pcheck.foreign:
        foreign_block["keys_sample"] = _patch_keys_sample(req.model)
    reference_sampling = presets.catalog()[_REFERENCE_ID]["sampling"]
    calibrated = resolved.manifest_sha256 is not None and identity["classified_by"] == "known_table"
    cond_modifiers = {adapter.COND_MODIFIER} if pos_facts.has_area_or_hooks or neg_facts.has_area_or_hooks else set()
    modifiers = tuple(sorted(set(facts.foreign_modifiers) | cond_modifiers))
    label = classify_label(
        preset=preset, effective=resolved.effective, customized=customized, reference_sampling=reference_sampling,
        steps_used=steps_used, cfg=cfg, live_shift=live_shift, sigmas_bit_equal_native=bit_equal,
        noise_policy=req.noise_policy, modifiers=modifiers, foreign_patches=pcheck.foreign,
        family_verified=identity["verified_matrix"] is True, has_lora=pcheck.has_lora, slg_applied=slg_applied,
        calibrated=calibrated, family_in_preset=not family_gate, phased=phase_plan is not None)
    rb.section("label", **label)

    ms = req.model.get_model_object("model_sampling")
    window = guidance.slg_sigma_window(ms, smp["slg_start"], smp["slg_end"])
    env = adapter.env_info(req.model)
    rb.section("env", **env)
    rb.section("preset", id=preset["id"], version=preset["preset_version"], sha256=resolved.preset_sha256,
               pin=resolved.pin, pinned=resolved.pinned, manifest_sha256=resolved.manifest_sha256,
               layers_applied=list(resolved.layers_applied), diff=diff)
    rb.section("model", architecture="moss_image", family=family, num_layers=facts.num_layers, identity=identity,
               model_sampling={"class": facts.model_sampling_class, "shift": live_shift,
                               "multiplier": facts.multiplier})
    if constraints and constraints.get("text_encoder") is not None:
        rb.section("text_encoder", **constraints["text_encoder"])
    rb.section("conditioning", positive=_cond_block(pos_facts), negative=_cond_block(neg_facts))
    rb.section("inputs", seed=int(req.seed), batch=lfacts.batch, width=lfacts.pixel_width,
               height=lfacts.pixel_height, latent_shape=[lfacts.batch, lfacts.channels, lfacts.height, lfacts.width],
               latent_dtype=lfacts.dtype, latent_was_empty=lfacts.is_empty,
               latent_raw_shape=list(lfacts.raw_shape), latent_rescaled_by_native_fix=lfacts.rescaled_by_native_fix,
               noise_policy=req.noise_policy, subseeds=rng.subseeds(req.seed, lfacts.batch, req.noise_policy))
    # n3: input sigmas that differ from the native schedule were not produced by the preset's scheduler
    rb.section("sampling", steps=steps_used, sampler="euler", scheduler=scheduler if bit_equal else "custom",
               shift=live_shift, cfg=cfg,
               slg=guidance.describe_slg(smp["slg_mode"], smp["slg_layers"], smp["slg_start"], smp["slg_end"], window),
               restart=False, ancestral_noise=False,
               sigmas={"source": sigmas_source, "n": len(values), "sha256": sigma.sigma_hash(values),
                       "first": float(values[0]), "last": float(values[-1]), "equals_native": bool(bit_equal)})
    rb.section("lora", stack=_lora_stack(profile), foreign_patches=foreign_block,
               te_applied=bool(((profile or {}).get("summary") or {}).get("te_applied", False)),
               profile_digest=(profile or {}).get("digest"))
    if phased_run is not None:
        rb.section("lora", phase=phase.describe_schedule(phased_run[1], clones_built=0))
    rb.section("forwards", expected_branch_forwards=plan.expected_branch_forwards,
               expected_sample_forwards=plan.expected_sample_forwards, extra=plan.extra)
    rb.section("preview", enabled=bool(req.preview))
    rb.mark_phase("preflight_done")

    # 9. execution: ONE clone (SLG hook + counter), plain CFGGuider + euler
    stats, tele = ForwardStats(), StepTelemetry()
    clone = guider = noise = callback = sigma_tensor = sampler = None
    clones_built = 0
    try:
        if phased_run is None:
            clone = guidance.apply_slg(req.model, layers=smp["slg_layers"], start=smp["slg_start"],
                                       end=smp["slg_end"]) if slg_applied else req.model
            if clone is req.model:
                clone = req.model.clone()
            install_counter(clone, stats)
        import torch
        noise = rng.make_noise(fixed, req.seed, req.noise_policy, req.latent.get("batch_index"), node=node)
        rb.section("inputs", noise_sha256=rng.tensor_sha256(noise))
        sampler = get_sampler("euler")
        # a phased run builds its clones segment by segment (lora_phase); the preview only needs the latent format
        callback = make_callback(clone if phased_run is None else req.model, steps_used, preview=req.preview,
                                 telemetry=tele, finite_check=True, node=node)
        sigma_tensor = torch.tensor(values, dtype=torch.float32)
        import comfy.utils
        _memory_begin(env["device"])
        tele.start()
        with torch.inference_mode():
            if phased_run is None:
                guider = build_guider(clone, req.positive, req.negative, cfg)
                samples = guider.sample(noise, fixed.clone(), sampler, sigma_tensor, callback=callback,
                                        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=req.seed)
            else:
                def run_clone(level):
                    run = guidance.apply_slg(level, layers=smp["slg_layers"], start=smp["slg_start"],
                                             end=smp["slg_end"]) if slg_applied else level
                    install_counter(run, stats)
                    return run

                samples, clones_built = lora_phase.run_segments(
                    phased_run[1], req.model, phased_run[0], make_clone=run_clone,
                    make_guider=lambda run: build_guider(run, req.positive, req.negative, cfg), sampler=sampler,
                    noise=noise, latent_image=fixed, sigmas=sigma_tensor, callback=callback,
                    disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=req.seed, poll=_poll_interrupt)
        rb.mark_phase("sampling_done")
        samples = adapter.to_intermediate(samples)
        guard.check_finite(samples, node=node, where="final latent")
        memory = _memory_end(env["device"])
        preview_active = bool(callback.preview_active)
    finally:
        clone = guider = noise = callback = sigma_tensor = sampler = None

    # 10. result
    out = req.latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    match = stats.elements == plan.expected_sample_forwards
    if not match:
        rb.warn(errors.make_issue(
            "forward_count_mismatch", node, "steps",
            "observed %d sample forwards, planned %d" % (stats.elements, plan.expected_sample_forwards),
            "report the recipe_hash; the forward count is part of the parity check",
            detail={"observed": stats.elements, "expected": plan.expected_sample_forwards}))
    timing = dict(tele.summary())
    if req.report_step_times:
        timing["step_s"] = [round(s, 6) for s in tele.steps]
    rb.section("timing", **timing)
    if phased_run is not None:
        rb.section("lora", phase=phase.describe_schedule(phased_run[1], clones_built=clones_built))
    rb.section("forwards", model_calls=stats.calls, sample_forwards=stats.elements, slg_skip_calls=stats.skip_calls,
               match=match)
    rb.section("memory", **memory)
    rb.section("preview", enabled=preview_active)
    rb.section("guards", finite_check="passed", items=items, composition_limit={"enabled": False},
               guard_passed=False)
    rb.section("output", latent_shape=[int(d) for d in samples.shape], latent_sha256=rng.tensor_sha256(samples),
               finite=True)
    final = rb.finalize()
    return SampleResult(latent=out, report=final, summary=_summary(final))
