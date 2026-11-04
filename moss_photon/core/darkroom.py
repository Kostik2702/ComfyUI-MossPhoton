"""MossPhoton darkroom phase engine (MP-57, EXPERIMENTAL; in the main samplers since MP-71): the phase plan (pure) and the
segment engine (comfy glue).

A preset that carries a ``darkroom`` block (``presets/turbo_darkroom_*.json``) runs through this engine in MossPhotonSampler /
MossPhotonSamplerAdvanced; the block is the single source of the phase values (``resolve_settings``), the steps are the preset's
``sampling.steps``. A preset without the block never reaches this module.

The run is split by sigma progress ``r = (sigma_0 - sigma) / (sigma_0 - sigma_last)`` (``1 - sigma`` on the native 1 -> 0
schedules) into ordered phases, each with its own settings. A step belongs to the phase its START ``r`` falls in:

* ``compose``  [0, r_c)    guidance ``compose_cfg``, SLG on/off;
* ``commit``   [r_c, r_k)  the character LoRA weight moves ``w_compose -> w_commit`` (smoothstep, six levels: the clone-per-
  level mechanism of ``lora_phase``, character adapter only); compose guidance stays;
* ``hold``     [r_k, r_f)  only when ``r_k < r_f``: compose guidance, the committed weight;
* ``finish``   [r_f, 1]    ``finish_cfg`` with the user's negative prompt, optional noise injection ``eta`` (euler ancestral,
  rectified-flow form, own seeded RNG), optional switch to a second MODEL (the same checkpoint and character without the
  distill adapter, connected by the user).

Optional ``re-expose`` at ``r_k``: the x0 readout of the last step before ``r_k`` is re-noised to ``sigma_r`` with an
INDEPENDENT seeded RNG (``seed + REEXPOSE_SEED_OFFSET``) and the schedule below ``sigma_r`` is walked again.

Plan (``resolve_settings`` / ``build_plan``) is pure arithmetic: every invalid order or bound raises a typed error before the
first forward (and before any tensor exists). ``run_plan`` is the comfy glue; it receives the clone / guider factories from
``solver`` (no import of ``solver`` here), runs ONE ``CFGGuider.sample`` per segment on its own direct clone, carries the raw
latent ``x`` over exactly (``lora_phase.Carrier``) and keeps at most one segment clone alive. Nothing here draws a random
number from the main noise stream or from torch's global generator.

LoRAs (MP-84): none is required. A model patched by ordinary loaders is phase ``all`` (it never leaves the clone the engine
starts from). The entries of a LoRA Phase plan (``lora_plan``) are added to each segment clone by the darkroom phase they
name (``lora_plan.PHASE_SEGMENTS``), and the commit boost ``w_compose -> w_commit`` is added to the strength of the
``identity`` entries (``lora_plan.takes_boost``); a run with no entry that can take it skips the boost and says so. Every setting a run cannot use
degrades: ``resolve_settings`` moves a value to the nearest usable one and returns the notes, the plan skips what has no
step; only a schedule that cannot be walked at all is refused.

An all-neutral block (one segment, the stock sampler object, the stock guider, no plan) is bit-equal to the stock turbo run
with the same inputs. Imports: stdlib, ``errors`` / ``guidance`` / ``lora_phase`` / ``lora_plan`` / ``phase`` / ``rng`` /
``sigma``; torch and comfy only inside functions (R1).
"""
import dataclasses
import math

from . import consts, errors, guidance, lora_phase, lora_plan, phase, rng, sigma

__all__ = [
    "VERSION", "LABEL", "REASON", "SCHEDULES", "SA2_8", "OVERRIDE_NAMES",
    "REEXPOSE_SEED_OFFSET", "NOISE_SEED_OFFSET", "PHASE_NAMES", "Settings", "DarkroomRun", "Step", "Segment",
    "DarkroomPlan", "ConstantBoost", "BOOST_PHASED", "BOOST_EXTRA", "resolve_settings", "sa2_values", "reexpose_seed",
    "noise_seed", "build_plan", "describe", "renoise", "run_plan",
]

NODE = "MossPhotonSampler"        # the node named in an issue when the caller gives none
VERSION = 1                       # the report block and the plan format
LABEL = "darkroom_experimental"   # report.darkroom.label
REASON = "darkroom_experimental"  # label.not_reference_reasons
PROGRESS = "sigma_progress"

REEXPOSE_SEED_OFFSET = 1000003    # research/turbo_refine/darkroom.py SEED_R_OFFSET (MP-54)
NOISE_SEED_OFFSET = 2000003       # the finish-phase noise injection (own constant, same idea)
CFG_MAX = consts.CFG_MAX
SIGMA_R_MIN = 0.01
_KNOT_EPS = 1e-4                  # research/turbo_refine/darkroom.py tail_below

SCHEDULES = ("native", "sa2_8")
# MP-53 arm SA2 (research/turbo_phases/arms.py): steps at r = .00 .04 .08 .12 .17 .24 .375 .583 of an 8-step run. Measured for
# eight steps only; another step count resamples these knots (``sa2_values``), a spacing nobody measured.
SA2_8 = (1.0, 0.96, 0.92, 0.88, 0.83, 0.76, 0.625, 0.4166666567325592, 0.0)

PHASE_NAMES = ("compose", "commit", "hold", "finish")
BOOST_PHASED, BOOST_EXTRA = "phased_adapter", "extra_patch"  # how the commit boost reaches the character LoRA
MODEL_MAIN, MODEL_FINISH = "main", "finish"
SAMPLER_EULER, SAMPLER_ANCESTRAL = "euler", "euler_ancestral_rf"

# The phase knobs a caller may override (the Advanced sampler's widgets, ``None`` = from the preset's ``darkroom`` block).
# ``steps`` is not one of them: the steps of a run are ``sampling.steps`` of the preset (or its ``steps`` override).
OVERRIDE_NAMES = ("schedule", "compose_cfg", "compose_slg", "r_c", "r_k", "w_compose", "w_commit", "reexpose", "sigma_r",
                  "r_f", "finish_cfg", "eta", "switch")

_SLG_LAYERS = (22,)               # used only when compose SLG is switched on (the reference preset's layer)
_SLG_START, _SLG_END = 0.0, 1.0


def reexpose_seed(seed):
    """The independent seed of the re-expose noise: ``(seed + REEXPOSE_SEED_OFFSET) & SEED_MASK``."""
    return (int(seed) + REEXPOSE_SEED_OFFSET) & rng.SEED_MASK


def noise_seed(seed):
    """The independent seed of the finish noise injection: ``(seed + NOISE_SEED_OFFSET) & SEED_MASK``."""
    return (int(seed) + NOISE_SEED_OFFSET) & rng.SEED_MASK


def sa2_values(steps=8):
    """The SA2 schedule as a list of ``steps + 1`` floats: the eight-step knots as measured for ``steps == 8``, else those knots
    resampled linearly (position ``i * 8 / steps``), still strictly descending from 1.0 to 0.0."""
    if steps == len(SA2_8) - 1:
        return list(SA2_8)
    knots = len(SA2_8) - 1
    values = []
    for i in range(steps):
        pos = i * knots / steps
        index = int(pos)
        values.append(SA2_8[index] + (SA2_8[index + 1] - SA2_8[index]) * (pos - index))
    return values + [0.0]


# --------------------------------------------------------------------------- settings
@dataclasses.dataclass(frozen=True)
class Settings:
    """The ``darkroom`` block of a preset with the caller's overrides applied and validated; ``overrides`` names the knobs the
    caller set (the report records them)."""

    preset_id: str
    preset_version: str
    schedule: str
    compose_cfg: float
    compose_slg: bool
    slg_layers: tuple
    slg_start: float
    slg_end: float
    r_c: float
    r_k: float
    w_compose: float
    w_commit: float
    reexpose: bool
    sigma_r: float
    r_f: float
    finish_cfg: float
    eta: float
    switch: bool
    overrides: tuple
    finish_cfg_source: str = "preset"   # "preset", "finish_cfg" (the Advanced knob) or "cfg_widget" (the sampler's cfg widget)
    notes: tuple = ()                   # ``(param, message, fix)`` of every value moved to a usable one (``setting_adjusted``)

    @property
    def ref(self):
        """``<preset id>@<preset_version>``: what the report records as the recipe."""
        return "%s@%s" % (self.preset_id, self.preset_version)

    def to_dict(self):
        """The ``darkroom.settings`` report object."""
        return {
            "compose": {"cfg": self.compose_cfg, "slg": self.compose_slg, "slg_layers": list(self.slg_layers),
                        "slg_start": self.slg_start, "slg_end": self.slg_end},
            "commit": {"r_c": self.r_c, "r_k": self.r_k, "w_compose": self.w_compose, "w_commit": self.w_commit},
            "reexpose": {"enabled": self.reexpose, "sigma_r": self.sigma_r},
            "finish": {"r_f": self.r_f, "cfg": self.finish_cfg, "eta": self.eta, "switch": self.switch},
        }


@dataclasses.dataclass(frozen=True)
class DarkroomRequest:
    """What ``solver.SampleRequest.darkroom`` carries: the caller's phase knobs (``overrides``: names of
    ``OVERRIDE_NAMES`` -> value, ``None`` = from the preset), the optional second MODEL of the finish phase, ``split_at`` (a
    test seam: extra step indices where a segment is forced to end, never set by a node) and ``boost_loader`` (``(model,
    character_entry) -> (patch_dict, metadata, reason)``; see ``ConstantBoost``). The solver turns it into a ``DarkroomRun``
    when the preset has a ``darkroom`` block, and into warnings when it has none. ``compose_cfg`` is not carried here: the
    compose guidance of a run is the effective ``sampling.cfg``. ``cfg_widget`` is the value of the ``cfg`` widget (``None`` =
    unset) a node passes for a preset WITH a block: there it is the finish guidance, and the ``finish_cfg`` knob wins over it."""

    overrides: dict = dataclasses.field(default_factory=dict)
    finish_model: object = None
    split_at: tuple = ()
    boost_loader: object = None
    cfg_widget: object = None


@dataclasses.dataclass(frozen=True)
class DarkroomRun:
    """A ``DarkroomRequest`` resolved against a preset's block: the validated settings, the optional second MODEL of the
    finish phase, ``split_at`` and ``boost_loader`` (same meaning as in ``DarkroomRequest``)."""

    settings: Settings
    finish_model: object = None
    split_at: tuple = ()
    boost_loader: object = None


@dataclasses.dataclass(frozen=True)
class ConstantBoost:
    """The character LoRA of a profile whose weight is CONSTANT (the default of MossPhotonLoRAProfile), as the commit boost
    needs it: the commit weight above the constant one is an EXTRA patch of the same LoRA (additive LoRA math: constant 1.0
    plus an extra 0.2 is 1.2) applied on the commit-segment clones only. ``reason`` is ``None`` when ``patch_dict`` is the
    LoRA's mapped patch dict (read from the cached state of the file the profile names), else why the boost is skipped
    (``file_not_found`` / ``size_differs`` / ``sha_differs`` / ``mapping_differs`` / ``load_error``): the run then goes on
    without the boost and says so. ``stack_index``, ``basename``, ``size``, ``sha256`` and ``strength`` are the profile
    entry's."""

    stack_index: int
    basename: str
    size: int
    sha256: object
    strength: float
    patch_dict: object = None
    metadata: object = None
    patch_entries: int = 0
    reason: object = None

    def adapter(self):
        """The ``lora_phase.PhaseAdapter`` the engine builds its extra-patch clones from."""
        return lora_phase.PhaseAdapter(self.stack_index, self.basename, self.size, self.sha256, self.patch_dict,
                                       self.metadata, self.patch_entries)


def _bad(node, param, message, fix, **detail):
    errors.raise_first_error([errors.make_issue("invalid_override", node, param, message, fix, severity="error",
                                                detail=detail or None)])


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _number(node, name, v, low, high, notes):
    """``v`` as a float within ``low..high``: a finite number outside it is moved to the nearest bound and noted (a setting
    conflict degrades, MP-84); a value that is not a finite number at all is refused (only an API caller makes one)."""
    if not _is_number(v):
        _bad(node, name, "%s = %r is not a finite number in %g..%g" % (name, v, low, high),
             "set %s between %g and %g" % (name, low, high), value=repr(v))
    used = min(max(float(v), low), high)
    if used != v:
        notes.append((name, "%s = %g is outside %g..%g: %g was used" % (name, v, low, high, used),
                      "set %s between %g and %g to drop this note" % (name, low, high)))
    return used


def _flag(node, name, v):
    if not isinstance(v, bool):
        _bad(node, name, "%s = %r is not a boolean" % (name, v), "use on or off")
    return v


def resolve_settings(preset, overrides=None, *, node=NODE):
    """The validated ``Settings`` of the ``darkroom`` block of ``preset`` (a catalog preset dict) with ``overrides`` (a dict of
    ``OVERRIDE_NAMES``; ``None`` values = from the block).

    Pure. A setting conflict degrades (MP-84): a number outside its bounds is moved to the nearest bound, phases out of order
    are sorted (``r_c <= r_k <= r_f``), a weight change over an empty commit window keeps ``w_compose`` (no boost); each is a
    ``(param, message, fix)`` in ``Settings.notes``, which the solver turns into ``setting_adjusted`` warnings. The ``sa2_8``
    schedule has no step count to match: the solver resamples its knots (``sa2_values``). Raises ``invalid_override`` (issue
    node ``node``) only for a preset without a block, an unknown override name, a value of the wrong type (not a finite
    number, not a boolean, not a listed schedule): what only an API caller can pass.
    """
    block = preset.get("darkroom")
    if block is None:
        _bad(node, "preset", "preset %r has no darkroom block" % (preset.get("id"),),
             "choose a preset with a darkroom block (turbo_darkroom_portrait, turbo_darkroom_fullbody)")
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    unknown = sorted(set(overrides) - set(OVERRIDE_NAMES))
    if unknown:
        _bad(node, "overrides", "unknown darkroom override(s) %s" % ", ".join(unknown),
             "use only %s" % ", ".join(OVERRIDE_NAMES))
    schedule = overrides.get("schedule", block["schedule"])
    if schedule not in SCHEDULES:
        _bad(node, "schedule", "schedule = %r, expected one of %s" % (schedule, ", ".join(SCHEDULES)),
             "choose native or sa2_8")
    notes = []
    r_c = _number(node, "r_c", overrides.get("r_c", block["commit"]["r_c"]), 0.0, 1.0, notes)
    r_k = _number(node, "r_k", overrides.get("r_k", block["commit"]["r_k"]), 0.0, 1.0, notes)
    r_f = _number(node, "r_f", overrides.get("r_f", block["finish"]["r_f"]), 0.0, 1.0, notes)
    if not r_c <= r_k <= r_f:
        ordered = sorted((r_c, r_k, r_f))
        notes.append(("r_c", "the phases are out of order: r_c = %g, r_k = %g, r_f = %g (need 0 <= r_c <= r_k <= r_f <= 1): "
                             "they were sorted to %g, %g, %g" % ((r_c, r_k, r_f) + tuple(ordered)),
                      "compose ends at r_c, commit at r_k, finish starts at r_f"))
        r_c, r_k, r_f = ordered
    w_compose = _number(node, "w_compose", overrides.get("w_compose", block["commit"]["w_compose"]), 0.0,
                        phase.WEIGHT_LIMIT, notes)
    w_commit = _number(node, "w_commit", overrides.get("w_commit", block["commit"]["w_commit"]), 0.0, phase.WEIGHT_LIMIT,
                       notes)
    if w_compose != w_commit and not r_c < r_k:
        notes.append(("r_k", "the commit weight moves %g -> %g over an empty commit window (r_c = r_k = %g): there is no "
                             "boost, w_commit = %g was used" % (w_compose, w_commit, r_c, w_compose),
                      "raise r_k above r_c, or set w_commit equal to w_compose"))
        w_commit = w_compose
    compose_cfg = _number(node, "compose_cfg", overrides.get("compose_cfg", block["compose"]["cfg"]), 1.0, CFG_MAX, notes)
    finish_cfg = _number(node, "finish_cfg", overrides.get("finish_cfg", block["finish"]["cfg"]), 1.0, CFG_MAX, notes)
    eta = _number(node, "eta", overrides.get("eta", block["finish"]["eta"]), 0.0, 1.0, notes)
    sigma_r = _number(node, "sigma_r", overrides.get("sigma_r", block["reexpose"]["sigma_r"]), SIGMA_R_MIN, 1.0, notes)
    return Settings(
        preset_id=preset["id"], preset_version=preset["preset_version"], schedule=schedule, compose_cfg=compose_cfg,
        compose_slg=_flag(node, "compose_slg", overrides.get("compose_slg", block["compose"]["slg"])),
        slg_layers=_SLG_LAYERS, slg_start=_SLG_START, slg_end=_SLG_END, r_c=r_c, r_k=r_k, w_compose=w_compose,
        w_commit=w_commit, reexpose=_flag(node, "reexpose", overrides.get("reexpose", block["reexpose"]["enabled"])),
        sigma_r=sigma_r, r_f=r_f, finish_cfg=finish_cfg, eta=eta,
        switch=_flag(node, "switch", overrides.get("switch", block["finish"]["switch"])),
        overrides=tuple(sorted(overrides)), notes=tuple(notes))


# --------------------------------------------------------------------------- the plan
@dataclasses.dataclass(frozen=True)
class Step:
    """One model step of the run (``index`` counts the effective steps; ``path`` is the sigma path it walks)."""

    index: int
    path: int
    local: int
    sigma: float
    sigma_next: float
    r: float
    phase: str
    cfg: float
    slg: bool
    weight: object           # applied character weight (phased adapter) or None
    model: str
    sampler: str


@dataclasses.dataclass(frozen=True)
class Segment:
    """Steps ``first .. stop - 1`` share every setting and one ``CFGGuider.sample``; ``sigmas[path][local:local_stop + 1]``."""

    first: int
    stop: int
    path: int
    local: int
    local_stop: int
    cfg: float
    slg: bool
    weight: object
    model: str
    sampler: str
    eta: float
    reexpose_before: bool

    @property
    def steps(self):
        return self.stop - self.first


@dataclasses.dataclass(frozen=True)
class DarkroomPlan:
    settings: Settings
    seed: int
    paths: tuple             # sigma paths (tuples of floats); one, or two with a re-expose
    steps: tuple
    segments: tuple
    total_steps: int
    expected_branch_forwards: int
    boost_mode: object       # BOOST_PHASED (the profile's phased adapter), BOOST_EXTRA (extra patch) or None
    character: dict          # the report's ``darkroom.character`` object
    reexpose_step: object    # effective index of the first step after the re-noise, or None
    sigma_first: float
    sigma_last: float
    lora_entries: tuple = ()          # the resolved LoRA Phase entries of the main MODEL (``lora_plan.Resolved``)
    finish_lora_entries: tuple = ()   # the same for the finish MODEL
    plan_boost: bool = False          # the identity entries of ``lora_entries`` take the commit boost
    notes: tuple = ()                 # ``(code, param, message, fix)``: what the plan could not use and left out

    @property
    def neutral(self):
        return (len(self.segments) == 1 and self.boost_mode is None and not self.lora_entries
                and not self.finish_lora_entries and self.segments[0].sampler == SAMPLER_EULER)

    @property
    def uses_adapter(self):
        """True when every segment clone carries the character LoRA through ``PhaseAdapter`` weights (phased profile)."""
        return self.boost_mode == BOOST_PHASED

    @property
    def uses_negative(self):
        return any(s.cfg != 1.0 for s in self.steps)


def _progress(sigma_value, first, last):
    return (first - sigma_value) / (first - last)


def _phase_of(r, st):
    if r < st.r_c:
        return "compose"
    if r < st.r_k:
        return "commit"
    if r < st.r_f:
        return "hold"
    return "finish"


def _level_weight(w0, w1, levels, j):
    """The weight of level ``j`` (the same arithmetic as ``phase`` uses for its discretised schedule)."""
    if j <= 0:
        return w0
    if j >= levels - 1:
        return w1
    return w0 + (w1 - w0) * (j / (levels - 1))


def _applied_weight(st, r):
    if st.w_compose == st.w_commit:
        return st.w_compose
    s = phase.smoothstep((r - st.r_c) / (st.r_k - st.r_c))
    return _level_weight(st.w_compose, st.w_commit, phase.PHASE_LEVELS, int(round(s * (phase.PHASE_LEVELS - 1))))


def build_plan(settings, sigma_values, *, seed, num_layers=None, phased=None, constant_boost=None, has_finish_model=False,
               split_at=(), node=NODE, lora_entries=(), finish_lora_entries=()):
    """The ``DarkroomPlan`` for the schedule ``sigma_values``; pure. It raises a typed error only for a schedule that cannot
    be walked (no step, not descending) and for SLG layers the model does not have, before any forward (the issues name
    ``node``); every other setting it cannot use degrades (MP-84) and is named in ``DarkroomPlan.notes``.

    ``phased`` is ``None`` for a profile with only constant weights, else ``{"stack_index", "role"}`` of the profile's single
    phased adapter, which the caller guarantees to be the character (the solver runs any other phased adapter as a constant
    plan entry): the character weight is then applied per segment by level clones and the recipe's ``w_compose`` /
    ``w_commit`` replace the weights of the profile's own phase plan. ``num_layers`` bounds the SLG layers. ``split_at``
    forces segment ends at the given effective step indices (test seam). For a constant profile ``constant_boost`` (a
    ``ConstantBoost``) carries the character LoRA of the profile: with ``reason`` ``None`` the commit weights are applied as
    extra patches of it, otherwise the boost is skipped (no weights, ``character.boost_skipped`` = the reason).
    ``lora_entries`` / ``finish_lora_entries`` (``lora_plan.Resolved``) are the LoRA Phase entries of the main / finish MODEL:
    a segment ends where the set of entries acting in it changes, and the entries of ``lora_entries`` in a boosted phase
    (``lora_plan.takes_boost``: identity; no known distill) take the commit boost and are named in
    ``character.boosted_entries``. No character adapter and no such entry: the boost is skipped (``character.boost_skipped`` =
    ``no_identity_lora``). A finish switch without a finish model, a re-expose point outside the run and a finish phase with
    no step are left out (``notes``), the returned plan carries the settings it really runs.
    """
    st = settings
    notes = []
    values = [float(v) for v in (sigma_values.tolist() if hasattr(sigma_values, "tolist") else sigma_values)]
    n = len(values) - 1
    if n < 1:
        _bad(node, "sigmas", "a darkroom run needs at least one step, got %d sigma values" % len(values), "connect a schedule")
    first, last = values[0], values[-1]
    if not first > last:
        _bad(node, "sigmas", "the schedule does not descend (%g -> %g)" % (first, last), "use a descending schedule")
    boost = st.w_compose != st.w_commit
    plan_boost = boost and bool(lora_plan.boost_indices(lora_entries))
    if phased is not None:
        boost_mode = BOOST_PHASED
    elif boost and constant_boost is not None and constant_boost.reason is None:
        boost_mode = BOOST_EXTRA
    else:
        boost_mode = None
    weighted = boost_mode is not None or plan_boost
    if boost and not weighted:
        skipped = constant_boost.reason if constant_boost is not None else "no_identity_lora"
    else:
        skipped = None
    mode = {BOOST_PHASED: "phased_adapter", BOOST_EXTRA: "constant_profile_extra_patch"}.get(
        boost_mode, "lora_plan_extra_patch" if plan_boost else "constant_profile")
    if boost_mode == BOOST_PHASED and plan_boost:  # both are boosted: the report names both
        mode = "phased_adapter+lora_plan_extra_patch"
    character = {
        "mode": mode,
        "stack_index": phased["stack_index"] if phased is not None else (
            constant_boost.stack_index if constant_boost is not None else None),
        "profile_phase_plan_replaced": phased is not None,
        "base_strength": constant_boost.strength if (constant_boost is not None and phased is None) else None,
        "boost_skipped": skipped}
    if plan_boost:   # MP-86: the entries that take the boost, by file name (their phases are in ``lora.plan``)
        character["boosted_entries"] = [lora_entries[i].basename for i in lora_plan.boost_indices(lora_entries)]
    if st.switch and not has_finish_model:
        notes.append(("darkroom_override_ignored", "switch",
                      "switch_model is on and no finish_model is connected: the finish phase runs on the main model",
                      "connect the finish_model (the same checkpoint without the distill adapter) or turn switch_model off"))
        st = dataclasses.replace(st, switch=False)
    if st.compose_slg:
        errors.raise_first_error(guidance.validate_slg("uncond_skip", st.slg_layers, st.slg_start, st.slg_end,
                                                       num_layers=num_layers, node=node))

    # the sigma paths: the base schedule, and below sigma_r again after a re-expose
    reexpose_at = None
    if st.reexpose:
        progress = [_progress(v, first, last) for v in values[:n]]
        reexpose_at = next((i for i, r in enumerate(progress) if r >= st.r_k), None)
        why = None
        if reexpose_at is None or reexpose_at < 1:
            why = "re-expose at r_k = %g falls %s" % (
                st.r_k, "before the first step" if reexpose_at == 0 else "after the last step")
        elif st.sigma_r > first:
            why = "sigma_r = %g is above the first sigma %g" % (st.sigma_r, first)
        if why is not None:
            notes.append(("darkroom_override_ignored", "reexpose", "%s: re-expose is off for this run" % why,
                          "choose r_k between the first and the last step and sigma_r below the first sigma, or turn "
                          "reexpose off"))
            st = dataclasses.replace(st, reexpose=False)
            reexpose_at = None
    if st.reexpose:
        tail = [st.sigma_r] + [v for v in values[1:] if v < st.sigma_r - _KNOT_EPS]
        paths = (tuple(values[:reexpose_at + 1]), tuple(tail))
    else:
        paths = (tuple(values),)

    steps = []
    for pi, path in enumerate(paths):
        for local in range(len(path) - 1):
            sg = path[local]
            r = _progress(sg, first, last)
            name = _phase_of(r, st)
            cfg = st.finish_cfg if name == "finish" else st.compose_cfg
            switched = name == "finish" and st.switch
            steps.append(Step(
                index=len(steps), path=pi, local=local, sigma=sg, sigma_next=path[local + 1], r=r, phase=name, cfg=cfg,
                slg=bool(st.compose_slg and name != "finish" and cfg != 1.0),
                weight=_applied_weight(st, r) if weighted else None,
                model=MODEL_FINISH if switched else MODEL_MAIN,
                sampler=SAMPLER_ANCESTRAL if (name == "finish" and st.eta > 0.0) else SAMPLER_EULER))
    steps = tuple(steps)
    if not any(s.phase == "finish" for s in steps) and (st.finish_cfg != st.compose_cfg or st.eta > 0.0 or st.switch):
        notes.append(("darkroom_override_ignored", "r_f",
                      "the finish phase (r >= %g) holds no step of this %d-step run: its settings do nothing" % (
                          st.r_f, len(steps)), "lower r_f or use more steps to include the finish phase"))

    forced = {int(i) for i in split_at}
    segments, begin = [], 0
    for i in range(1, len(steps) + 1):
        if i < len(steps):
            a, b = steps[begin], steps[i]
            same = (a.path == b.path and a.cfg == b.cfg and a.slg == b.slg and a.weight == b.weight
                    and a.model == b.model and a.sampler == b.sampler and i not in forced)
            entries = finish_lora_entries if a.model == MODEL_FINISH else lora_entries
            same = same and lora_plan.active(entries, a.phase) == lora_plan.active(entries, b.phase)
            if same:
                continue
        a = steps[begin]
        segments.append(Segment(
            first=begin, stop=i, path=a.path, local=a.local, local_stop=steps[i - 1].local + 1, cfg=a.cfg, slg=a.slg,
            weight=a.weight, model=a.model, sampler=a.sampler, eta=st.eta if a.sampler == SAMPLER_ANCESTRAL else 0.0,
            reexpose_before=bool(begin > 0 and steps[begin].path != steps[begin - 1].path)))
        begin = i
    expected = sum(sigma.branches_per_step(s.cfg) for s in steps)
    return DarkroomPlan(
        settings=st, seed=int(seed), paths=paths, steps=steps, segments=tuple(segments), total_steps=len(steps),
        expected_branch_forwards=expected, boost_mode=boost_mode, character=character,
        reexpose_step=None if reexpose_at is None else next(s.index for s in steps if s.path == 1),
        sigma_first=first, sigma_last=last, lora_entries=tuple(lora_entries),
        finish_lora_entries=tuple(finish_lora_entries), plan_boost=plan_boost, notes=tuple(notes))


def _phase_bounds(st):
    return {"compose": (0.0, st.r_c), "commit": (st.r_c, st.r_k), "hold": (st.r_k, st.r_f), "finish": (st.r_f, 1.0)}


def describe(plan, *, batch=1, observed=None, clones_built=None):
    """The ``darkroom`` report block (JSON-safe). ``observed`` = the sample forwards counted per segment after the run
    (``None`` before it); ``batch`` turns the per-image branch forwards into sample forwards."""
    st = plan.settings
    span = plan.sigma_first - plan.sigma_last
    obs_by_phase = {}
    seg_rows = []
    for i, seg in enumerate(plan.segments):
        seg_obs = None if observed is None else int(observed[i])
        row_steps = plan.steps[seg.first:seg.stop]
        seg_rows.append({
            "first_step": seg.first, "last_step": seg.stop - 1, "path": seg.path, "cfg": seg.cfg, "slg": seg.slg,
            "weight": seg.weight, "model": seg.model, "sampler": seg.sampler, "eta": seg.eta,
            "reexpose_before": seg.reexpose_before, "sigma_start": plan.paths[seg.path][seg.local],
            "sigma_end": plan.paths[seg.path][seg.local_stop],
            "expected_sample_forwards": batch * sum(sigma.branches_per_step(s.cfg) for s in row_steps),
            "observed_sample_forwards": seg_obs})
    # observed forwards per phase: a segment may hold steps of two phases only when their settings are equal, so the
    # per-step split of a segment's count is exact (every step of a segment costs the same number of branches)
    if observed is not None:
        for seg, count in zip(plan.segments, observed):
            per_step = int(count) / seg.steps
            for s in plan.steps[seg.first:seg.stop]:
                obs_by_phase[s.phase] = obs_by_phase.get(s.phase, 0.0) + per_step
    phases = []
    bounds = _phase_bounds(st)
    for name in PHASE_NAMES:
        if name == "hold" and not st.r_k < st.r_f:
            continue
        r0, r1 = bounds[name]
        mine = [s for s in plan.steps if s.phase == name]
        weights = [s.weight for s in mine if s.weight is not None]
        phases.append({
            "name": name, "r_start": r0, "r_end": r1, "sigma_start": plan.sigma_first - r0 * span,
            "sigma_end": plan.sigma_first - r1 * span, "steps": len(mine),
            "first_step": mine[0].index if mine else None, "last_step": mine[-1].index if mine else None,
            "cfg": st.finish_cfg if name == "finish" else st.compose_cfg,
            "slg": bool(st.compose_slg and name != "finish" and st.compose_cfg != 1.0),
            "weight_first": weights[0] if weights else None, "weight_last": weights[-1] if weights else None,
            "model": MODEL_FINISH if (name == "finish" and st.switch) else MODEL_MAIN,
            "sampler": SAMPLER_ANCESTRAL if (name == "finish" and st.eta > 0.0) else SAMPLER_EULER,
            "expected_sample_forwards": batch * sum(sigma.branches_per_step(s.cfg) for s in mine),
            "observed_sample_forwards": None if observed is None else int(round(obs_by_phase.get(name, 0.0)))})
    return {
        "version": VERSION, "recipe": st.ref, "label": LABEL, "neutral": plan.neutral, "overrides": list(st.overrides),
        "progress": PROGRESS, "schedule": st.schedule, "steps": plan.total_steps,
        "paths": [{"steps": len(p) - 1, "sigmas_sha256": sigma.sigma_hash(list(p))} for p in plan.paths],
        "settings": st.to_dict(), "phases": phases, "segments": seg_rows,
        "rng": {
            "main_seed": plan.seed,
            "reexpose": {"enabled": st.reexpose, "offset": REEXPOSE_SEED_OFFSET,
                         "seed": reexpose_seed(plan.seed) if st.reexpose else None},
            "noise": {"enabled": st.eta > 0.0, "offset": NOISE_SEED_OFFSET,
                      "seed": noise_seed(plan.seed) if st.eta > 0.0 else None},
        },
        "character": dict(plan.character),
        "clones_built": None if clones_built is None else int(clones_built),
    }


# --------------------------------------------------------------------------- comfy glue
def renoise(x0, sigma_r, seed):
    """``sigma_r * N(seed) + (1 - sigma_r) * x0``: the re-expose start. ``N`` is float32 gaussian noise from a PRIVATE CPU
    generator seeded ``seed`` (shape of ``x0``), cast to ``x0``'s device and dtype; torch's global generator and the main
    noise stream are not touched."""
    import torch
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    noise = torch.randn(tuple(x0.shape), dtype=torch.float32, generator=gen, device="cpu").to(device=x0.device,
                                                                                                dtype=x0.dtype)
    return sigma_r * noise + (1.0 - sigma_r) * x0


def _ancestral_sampler(eta, gen):
    """A KSAMPLER: ComfyUI's rectified-flow euler ancestral with the noise of the PRIVATE generator ``gen``."""
    import torch
    import comfy.k_diffusion.sampling as kd
    import comfy.samplers

    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
        def noise_sampler(_sigma, _sigma_next):
            return torch.randn(tuple(x.shape), dtype=torch.float32, generator=gen, device="cpu").to(
                device=x.device, dtype=x.dtype)

        return kd.sample_euler_ancestral_RF(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                                            eta=eta, s_noise=1.0, noise_sampler=noise_sampler)

    return comfy.samplers.KSAMPLER(sample, {}, {})


def run_plan(plan, *, model, adapter, finish_model, make_clone, make_guider, sampler, noise, latent_image, callback,
             disable_pbar, seed, poll, forwards):
    """Sample the plan segment by segment; returns ``(samples, clones_built, observed)``.

    ``make_clone(model, slg)`` turns a direct clone into the model of the segment (SLG hook when ``slg``, forward counter),
    on that clone itself (it must return it: the sampling patcher is a direct clone of the incoming MODEL, never a clone of
    a throwaway); the SLG hook is taken off again when the segment ends, because a neighbour may share the clone.
    ``make_guider(clone, cfg)`` builds the plain CFGGuider, ``forwards()`` returns the sample forwards counted so far
    (``observed[i]`` is the count of segment ``i``). ``adapter`` is the ``PhaseAdapter`` that carries the character patches
    (``plan.boost_mode``: the profile's phased adapter, or the ``ConstantBoost`` adapter) or ``None``. Every segment runs on its own direct clone, dropped when the segment ends (at most one is alive); the
    incoming MODELs are never written to. ``poll`` raises on an interrupt and runs between segments. Exceptions propagate
    after the segment's own cleanup.

    LoRA Phase entries (``plan.lora_entries`` on the main MODEL, ``plan.finish_lora_entries`` on the finish MODEL) are added to
    the DIRECT clone of that MODEL by the darkroom phase the segment starts in; two neighbouring segments that need the same
    clone (the same MODEL, the same entries, the same boost weight) share it, so ComfyUI patches the weights once. A plan
    without entries and without a plan boost takes the old path untouched (``model.clone()`` per segment).
    """
    import torch
    if plan.boost_mode is not None and adapter is None:
        raise RuntimeError("the darkroom plan needs the character adapter (phased or extra patch) and none was given")
    tensors = [torch.tensor(list(p), dtype=torch.float32) for p in plan.paths]
    carrier = lora_phase.Carrier()
    box = {}
    noise_gen = [None]
    observed, built, samples = [], 0, None
    shared = [None]   # (key, level clone) of the last segment that used the LoRA Phase entries, kept while the next needs it

    def stock_level(seg):
        if seg.model == MODEL_FINISH:
            return finish_model.clone()
        if plan.boost_mode == BOOST_PHASED:
            return lora_phase.make_level_model(model, adapter, seg.weight)
        if plan.boost_mode == BOOST_EXTRA and seg.weight != plan.settings.w_compose:
            # the profile's constant weight stays on the MODEL; the rest of the commit weight is added on top
            return lora_phase.make_level_model(model, adapter, seg.weight - plan.settings.w_compose)
        return model.clone()

    def level_for(seg):
        """The DIRECT clone the segment's model clone is made from: the stock one, plus the LoRA Phase entries that act in
        the segment's darkroom phase."""
        finish = seg.model == MODEL_FINISH
        entries = plan.finish_lora_entries if finish else plan.lora_entries
        step_phase = plan.steps[seg.first].phase
        acting = lora_plan.active(entries, step_phase)
        if not acting:   # no entry acts here: the path of a run without a plan, untouched
            shared[0] = None
            return stock_level(seg)
        key = (seg.model, acting, seg.weight)
        if shared[0] is not None and shared[0][0] == key:
            return shared[0][1]
        shared[0] = None
        boost = seg.weight - plan.settings.w_compose if (plan.plan_boost and not finish) else 0.0
        level = lora_plan.apply_to(stock_level(seg), entries, step_phase, boost)
        shared[0] = (key, level)
        return level

    def record(step, x0, x, total):
        box["x0"] = x0
        if callback is not None:
            callback(step, x0, x, total)

    try:
        for seg in plan.segments:
            poll()
            if seg.reexpose_before:
                carrier.x = renoise(box["x0"], plan.settings.sigma_r, reexpose_seed(plan.seed))
                box.clear()
            if seg.sampler == SAMPLER_ANCESTRAL:
                if noise_gen[0] is None:
                    noise_gen[0] = torch.Generator(device="cpu")
                    noise_gen[0].manual_seed(noise_seed(plan.seed))
                real = _ancestral_sampler(seg.eta, noise_gen[0])
            else:
                real = sampler
            use = real if len(plan.segments) == 1 else lora_phase.resume_sampler(real, carrier)
            clone = guider = slg_level = None
            hook_before = None
            before = forwards()
            try:
                level = level_for(seg)
                if seg.slg:
                    # ``make_clone`` puts the SLG hook on this very clone (a clone of it would have a throwaway parent), and
                    # the next segment may share it: the slot is put back when this segment ends
                    slg_level, hook_before = level, guidance.cond_hook_slot(level)
                clone = make_clone(level, seg.slg)
                level = None  # at most one segment clone stays alive
                built += 1
                guider = make_guider(clone, seg.cfg)
                samples = guider.sample(
                    noise, latent_image.clone(), use, tensors[seg.path][seg.local:seg.local_stop + 1],
                    callback=lora_phase.global_steps(record, seg.first, plan.total_steps),
                    disable_pbar=disable_pbar, seed=seed)
            finally:
                if slg_level is not None:
                    guidance.restore_cond_hook_slot(slg_level, hook_before)
                level = clone = guider = slg_level = None
            observed.append(forwards() - before)
    finally:
        carrier.x = None
        box.clear()
        noise_gen[0] = None
        shared[0] = None
    return samples, built, observed
