"""Phased LoRA weight by sampling progress (SPEC 5 "phases and LoRA", MP-16, EXPERIMENTAL): the math, plan and its gate.

Progress is never a step number: ``r = cumulative(|d sigma|) / total(|d sigma|)`` over the sigma array of the run, taken
at the START of each step (the weight a step uses is the one at the sigma its model forward sees). The weight is
``w(r) = w0 + (w1 - w0) * smoothstep((r - a) / (b - a))`` with ``smoothstep(t) = 3t^2 - 2t^3`` on ``[0, 1]`` (clamped).
A constant weight (``w0 == w1``) is the reference: ``w(r) == w0`` exactly, one level, one segment.

The sampler applies the weight per *segment* (``lora_phase.py``): the exact per-step weights are discretised to ``levels``
equally spaced values between ``w0`` and ``w1`` (the clone count is bounded by ``levels``) and a run of steps with the same
value is one segment. ``planned`` weights are the exact smoothstep values, ``applied`` the discretised ones.

Imports: stdlib and ``errors`` only; pure arithmetic, no torch, no comfy.
"""
import dataclasses
import hashlib
import math
import struct

from . import errors

__all__ = [
    "PHASE_LEVELS", "MAX_PHASE_LEVELS", "PLAN_VERSION", "PLAN_KIND", "PROGRESS", "MECHANISM", "APPLY_ORDER",
    "WEIGHT_LIMIT", "ADVANCED_NODE", "PhasePlan", "Segment", "StepSchedule", "smoothstep", "progress_fractions",
    "lora_weight", "make_plan", "plan_to_dict", "plan_from_dict", "weight_profile_of", "plan_step_weights",
    "build_schedule", "describe_schedule", "check_phase_plan",
]

PHASE_LEVELS = 6        # discrete weight levels = the most clones one run builds
MAX_PHASE_LEVELS = 16
PLAN_VERSION = 1
PLAN_KIND = "smoothstep_by_sigma_progress"
PROGRESS = "cumulative_abs_dsigma"
MECHANISM = "clone_per_level"
APPLY_ORDER = "after_constant_adapters"
WEIGHT_LIMIT = 10.0     # the bound of the strength_model widget
ADVANCED_NODE = "MossPhotonSamplerAdvanced"

_PLAN_KEYS = ("version", "kind", "progress", "mechanism", "apply_order", "levels", "stack_index", "w0", "w1", "a", "b")


@dataclasses.dataclass(frozen=True)
class PhasePlan:
    """One phased adapter: ``stack_index`` in the identity profile, the two weights and the smoothstep window."""

    stack_index: int
    w0: float
    w1: float
    a: float
    b: float
    levels: int = PHASE_LEVELS


@dataclasses.dataclass(frozen=True)
class Segment:
    """Steps ``first .. stop - 1`` run with the LoRA at ``weight``."""

    first: int
    stop: int
    weight: float


@dataclasses.dataclass(frozen=True)
class StepSchedule:
    plan: PhasePlan
    planned: tuple     # exact smoothstep weight of every step
    applied: tuple     # the discretised weight every step actually runs with
    segments: tuple    # run-length form of ``applied``


def _issue(node, param, message, fix, **detail):
    detail.setdefault("feature", "phase_plan")
    return errors.make_issue("unsupported_feature", node, param, message, fix, severity="error", detail=detail)


def _refuse(node, param, message, fix, **detail):
    errors.raise_first_error([_issue(node, param, message, fix, **detail)])


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# --------------------------------------------------------------------------- the math
def smoothstep(t):
    """``3t^2 - 2t^3`` with ``t`` clamped to ``[0, 1]``."""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def progress_fractions(sigmas):
    """``r`` at the start of every step, one value per step (``len(sigmas) - 1``), from ``cumulative(|d sigma|)``.

    ``r[0] == 0.0`` and ``r[-1] < 1``; the weight of a step is the one at the sigma its forward sees. Raises
    ``unsupported_feature`` for fewer than two values, a non-finite value or an array that never moves.
    """
    values = [float(s) for s in sigmas]
    if len(values) < 2:
        _refuse("(graph)", "sigmas", "a phase plan needs at least one sampling step, got %d sigma values" % len(values),
                "connect a schedule with at least two sigmas")
    if not all(math.isfinite(v) for v in values):
        _refuse("(graph)", "sigmas", "the sigma array holds a non-finite value", "connect a valid schedule")
    deltas = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    total = math.fsum(deltas)
    if not total > 0.0:
        _refuse("(graph)", "sigmas", "the sigma array has no extent (total |d sigma| = 0), r is undefined",
                "connect a schedule that moves from the first to the last sigma")
    out, running = [], 0.0
    for delta in deltas:
        out.append(running / total)
        running += delta
    return tuple(out)


def lora_weight(r, w0, w1, a, b):
    """``w0 + (w1 - w0) * smoothstep((r - a) / (b - a))``; ``w0 == w1`` gives ``w0`` exactly."""
    return w0 + (w1 - w0) * smoothstep((r - a) / (b - a))


# --------------------------------------------------------------------------- the plan
def make_plan(stack_index, w0, w1, a, b, *, levels=PHASE_LEVELS, node="MossPhotonLoRAProfile"):
    """Validated ``PhasePlan``. Raises ``unsupported_feature`` naming the bad value (weights within +-10, ``0 <= a < b <= 1``)."""
    for name, value in (("w0", w0), ("w1", w1)):
        if not _is_number(value) or abs(value) > WEIGHT_LIMIT:
            _refuse(node, name, "%s = %r is not a finite number within +-%g" % (name, value, WEIGHT_LIMIT),
                    "set %s between -%g and %g" % (name, WEIGHT_LIMIT, WEIGHT_LIMIT), value=value)
    for name, value in (("a", a), ("b", b)):
        if not _is_number(value) or not 0.0 <= value <= 1.0:
            _refuse(node, name, "%s = %r is not a fraction of the sampling progress in 0..1" % (name, value),
                    "set %s between 0 and 1" % name, value=value)
    if not a < b:
        _refuse(node, "b", "the transition window needs a < b, got a = %r, b = %r" % (a, b),
                "raise b above a (a hard switch is not a smoothstep)", a=a, b=b)
    if isinstance(levels, bool) or not isinstance(levels, int) or not 2 <= levels <= MAX_PHASE_LEVELS:
        _refuse(node, "levels", "levels = %r is not an integer in 2..%d" % (levels, MAX_PHASE_LEVELS),
                "use a level count between 2 and %d" % MAX_PHASE_LEVELS, value=levels)
    if isinstance(stack_index, bool) or not isinstance(stack_index, int) or stack_index < 0:
        _refuse(node, "profile_in", "stack_index %r is not a non-negative integer" % (stack_index,),
                "rebuild the LoRA chain")
    return PhasePlan(int(stack_index), float(w0), float(w1), float(a), float(b), int(levels))


def plan_to_dict(plan):
    """The ``PHOTON_IDENTITY_PROFILE.phase_plan`` object (JSON-safe)."""
    return {"version": PLAN_VERSION, "kind": PLAN_KIND, "progress": PROGRESS, "mechanism": MECHANISM,
            "apply_order": APPLY_ORDER, "levels": plan.levels, "stack_index": plan.stack_index,
            "w0": plan.w0, "w1": plan.w1, "a": plan.a, "b": plan.b}


def plan_from_dict(obj, *, node):
    """Parse ``phase_plan``; any other key set, version, kind, progress, mechanism or order is refused."""
    if not isinstance(obj, dict) or set(obj) != set(_PLAN_KEYS):
        _refuse(node, "identity_profile", "the phase_plan is not an object with exactly the keys %s" % list(_PLAN_KEYS),
                "rebuild the profile with MossPhotonLoRAProfile profile=phased")
    fixed = {"version": PLAN_VERSION, "kind": PLAN_KIND, "progress": PROGRESS, "mechanism": MECHANISM,
             "apply_order": APPLY_ORDER}
    for key, expected in fixed.items():
        if obj[key] != expected or type(obj[key]) is not type(expected):
            _refuse(node, "identity_profile", "phase_plan.%s is %r, this MossPhoton knows %r" % (key, obj[key], expected),
                    "update the ComfyUI-MossPhoton pack or rebuild the profile")
    return make_plan(obj["stack_index"], obj["w0"], obj["w1"], obj["a"], obj["b"], levels=obj["levels"], node=node)


def weight_profile_of(plan):
    """The ``stack[].weight_profile`` object of the phased entry."""
    return {"kind": "phased", "w0": plan.w0, "w1": plan.w1, "a": plan.a, "b": plan.b}


def check_phase_plan(profile, *, node, allowed):
    """The ``PhasePlan`` of an identity profile, or ``None`` when no adapter is phased.

    ``allowed`` is True for MossPhotonSamplerAdvanced only: any other node raises ``unsupported_feature`` naming it.
    Also refused: a ``phase_plan`` without exactly one phased stack entry (and the reverse), an entry whose
    ``weight_profile`` differs from the plan, a ``weight_profile.kind`` this version does not know, a malformed plan.
    """
    if profile is None:
        return None
    stack = profile.get("stack") if isinstance(profile.get("stack"), list) else []
    kinds = [((e.get("weight_profile") or {}).get("kind") if isinstance(e, dict) else None) for e in stack]
    unknown = [k for k in kinds if k not in ("constant", "phased")]
    if unknown:
        _refuse(node, "identity_profile", "the profile has weight_profile.kind %r, which this version does not know"
                % (unknown[0],), "update the ComfyUI-MossPhoton pack", kind=unknown[0])
    raw = profile.get("phase_plan")
    phased = [i for i, k in enumerate(kinds) if k == "phased"]
    if raw is None and not phased:
        return None
    if not allowed:
        _refuse(node, "identity_profile",
                "the identity profile carries a phased LoRA weight (phase_plan), which only %s runs" % ADVANCED_NODE,
                "connect the profile to %s, or set MossPhotonLoRAProfile profile=constant" % ADVANCED_NODE,
                node_required=ADVANCED_NODE)
    if raw is None or len(phased) != 1:
        _refuse(node, "identity_profile", "a phased profile needs a phase_plan and exactly one phased stack entry "
                "(plan: %s, phased entries: %d)" % ("set" if raw is not None else "missing", len(phased)),
                "rebuild the chain with MossPhotonLoRAProfile")
    plan = plan_from_dict(raw, node=node)
    if plan.stack_index != phased[0]:
        _refuse(node, "identity_profile", "phase_plan.stack_index = %d but stack[%d] is the phased entry"
                % (plan.stack_index, phased[0]), "rebuild the chain with MossPhotonLoRAProfile")
    if stack[plan.stack_index].get("weight_profile") != weight_profile_of(plan):
        _refuse(node, "identity_profile", "stack[%d].weight_profile differs from the phase_plan" % plan.stack_index,
                "rebuild the chain with MossPhotonLoRAProfile")
    return plan


# --------------------------------------------------------------------------- the schedule
def plan_step_weights(sigmas, plan):
    """The exact smoothstep weight of every step: ``lora_weight(r_k)`` for the progress at the start of step ``k``."""
    return tuple(lora_weight(r, plan.w0, plan.w1, plan.a, plan.b) for r in progress_fractions(sigmas))


def _level_weight(plan, j):
    if j <= 0:
        return plan.w0
    if j >= plan.levels - 1:
        return plan.w1
    return plan.w0 + (plan.w1 - plan.w0) * (j / (plan.levels - 1))


def build_schedule(sigmas, plan):
    """``StepSchedule``: exact weights, the discretised weights the run applies, and the segments of equal weight.

    Each step's smoothstep value ``s`` in ``[0, 1]`` is rounded to the nearest of ``plan.levels`` equally spaced
    fractions; level 0 is ``w0`` and the last level is ``w1`` exactly. ``w0 == w1`` gives one level, one segment.
    ``|applied - planned| <= |w1 - w0| / (2 * (levels - 1))``.
    """
    fractions = progress_fractions(sigmas)
    planned, applied = [], []
    for r in fractions:
        s = smoothstep((r - plan.a) / (plan.b - plan.a))
        planned.append(plan.w0 + (plan.w1 - plan.w0) * s)
        applied.append(_level_weight(plan, int(round(s * (plan.levels - 1)))))
    segments, start = [], 0
    for k in range(1, len(applied) + 1):
        if k == len(applied) or applied[k] != applied[start]:
            segments.append(Segment(start, k, applied[start]))
            start = k
    return StepSchedule(plan, tuple(planned), tuple(applied), tuple(segments))


def _digest(values):
    return hashlib.sha256(struct.pack("<%dd" % len(values), *values)).hexdigest()


def describe_schedule(schedule, *, clones_built):
    """The report block ``lora.phase``: the plan, hashes of the exact and applied per-step weights, the run-length segments."""
    plan = schedule.plan
    return {
        "mechanism": MECHANISM, "kind": PLAN_KIND, "progress": PROGRESS, "apply_order": APPLY_ORDER,
        "stack_index": plan.stack_index, "w0": plan.w0, "w1": plan.w1, "a": plan.a, "b": plan.b, "levels": plan.levels,
        "steps": len(schedule.applied), "planned_sha256": _digest(schedule.planned),
        "weights_sha256": _digest(schedule.applied),
        "segments": [{"first_step": s.first, "last_step": s.stop - 1, "weight": s.weight} for s in schedule.segments],
        "clones_built": int(clones_built),
    }
