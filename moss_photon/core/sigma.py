"""Pure schedule math: validation, hashing, comparison, forward counts, PHOTON_SCHEDULE_INFO (design 1.2, 2.6).

Imports: stdlib and ``errors`` / ``canon`` only. Never imports torch or comfy: every check runs on a Python list, so a
bad schedule is rejected before any tensor or GPU allocation exists (D4, SPEC 5). A tensor is accepted anywhere an
array is taken, through its ``tolist()``.
"""
import math
import struct

from . import canon, errors

__all__ = [
    "JUMP_RATIO_MAX", "MAX_STEPS", "VALIDATION_CHECKS", "sigma_hash", "validate_sigmas", "observed_jump_ratio",
    "compare_sigmas", "branches_per_step", "expected_forwards", "build_schedule_info",
]

# Largest allowed ratio between consecutive step sizes (design 1.2). The native 50-step sgm_uniform/shift-5 array
# stays within [1.033, 1.145]; the worst native ratio over steps 1..1000 is 2.13 (3 steps).
JUMP_RATIO_MAX = 3.0
MAX_STEPS = 1000

# Names of the checks, in the order validate_sigmas runs them (schedule_info.validation.checks).
VALIDATION_CHECKS = (
    "finite", "one_dimensional", "strictly_decreasing", "ends_at_zero", "no_duplicates", "within_model_range",
    "step_ratio",
)


def _as_list(values):
    """A plain list from a list / tuple / 1-D tensor (anything with ``tolist``)."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def _f32_bytes(values):
    return struct.pack("<%df" % len(values), *values)


def sigma_hash(values):
    """sha256 hex of the little-endian float32 bytes of ``values`` (list, tuple or 1-D tensor).

    Values are rounded to float32 first, so a float32 tensor and its ``tolist()`` hash the same. The native
    50-step shift-5 array gives ``consts.REFERENCE_SIGMAS_SHA256``.
    """
    return canon.sha256_hex(_f32_bytes(_as_list(values)))


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def observed_jump_ratio(values):
    """Largest ``max(r, 1/r)`` of consecutive step-size ratios ``r = d[i+1] / d[i]``, last step to zero excluded.

    ``1.0`` when there are fewer than two comparable steps. Meaningful only for a strictly decreasing array.
    """
    vals = _as_list(values)
    deltas = [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]
    worst = 1.0
    for i in range(len(deltas) - 2):  # the last delta (to zero) is structurally larger and is excluded
        if deltas[i] > 0 and deltas[i + 1] > 0:
            r = deltas[i + 1] / deltas[i]
            worst = max(worst, r, 1.0 / r)
    return worst


def validate_sigmas(values, *, node, param="sigmas", sigma_max=1.0, expect_steps=None, ratio_is_error=True):
    """Validate a sigma schedule; returns ``list[Issue]`` (reason ``invalid_schedule``, never raises for bad data).

    Checks in order, stopping at the first structural failure (later checks assume the earlier ones): 1-D and
    numeric; ``2 <= n <= MAX_STEPS + 1``; all finite; ``0 < sigma[0] <= sigma_max``; ``sigma[-1] == 0.0`` exactly;
    all ``>= 0``; strictly decreasing (a repeated value is reported as a duplicate); ``n == expect_steps + 1`` when
    ``expect_steps`` is given; step ratio (``JUMP_RATIO_MAX``). A failed step-ratio check is an error when
    ``ratio_is_error`` (``reference`` mode) and the warning ``sigma_step_ratio`` otherwise. Issue messages name the
    index. ``sigma_max`` is in sigma units (1.0 for MOSS).
    """
    def err(msg, fix, **detail):
        return [errors.make_issue("invalid_schedule", node, param, msg, fix, detail=detail or None)]

    fix_default = "regenerate with MossPhotonSchedule(mode=reference)"
    try:
        vals = _as_list(values)
    except TypeError:
        return err("sigmas is not an array (%s)" % type(values).__name__, fix_default)
    if any(isinstance(x, (list, tuple)) or hasattr(x, "tolist") for x in vals):
        return err("sigmas is not one-dimensional", fix_default, check="one_dimensional")
    bad = next((i for i, x in enumerate(vals) if not _is_number(x)), None)
    if bad is not None:
        return err("sigmas[%d] = %r is not a number" % (bad, vals[bad]), fix_default, check="one_dimensional", index=bad)
    n = len(vals)
    if n < 2 or n > MAX_STEPS + 1:
        return err("sigmas has %d values, expected 2..%d" % (n, MAX_STEPS + 1), fix_default, check="length", n=n)
    bad = next((i for i, x in enumerate(vals) if not math.isfinite(x)), None)
    if bad is not None:
        return err("sigmas[%d] = %r is not finite" % (bad, vals[bad]), fix_default, check="finite", index=bad)
    if not (0.0 < vals[0] <= sigma_max):
        return err("sigmas[0] = %r, expected 0 < sigma <= %r" % (vals[0], sigma_max), fix_default,
                   check="within_model_range", index=0)
    if vals[-1] != 0.0:
        return err("sigmas[%d] = %r, the schedule must end at exactly 0.0" % (n - 1, vals[-1]), fix_default,
                   check="ends_at_zero", index=n - 1)
    bad = next((i for i, x in enumerate(vals) if x < 0.0), None)
    if bad is not None:
        return err("sigmas[%d] = %r is negative" % (bad, vals[bad]), fix_default, check="within_model_range", index=bad)
    for i in range(n - 1):
        if vals[i + 1] == vals[i]:
            return err("duplicate value %r at index %d" % (vals[i], i + 1), fix_default, check="no_duplicates",
                       index=i + 1)
        if vals[i + 1] > vals[i]:
            return err("not strictly decreasing at index %d (%r -> %r)" % (i + 1, vals[i], vals[i + 1]), fix_default,
                       check="strictly_decreasing", index=i + 1)
    if expect_steps is not None and n != expect_steps + 1:
        return err("sigmas has %d values, expected steps + 1 = %d" % (n, expect_steps + 1), fix_default,
                   check="length", n=n, expected=expect_steps + 1)
    deltas = [vals[i] - vals[i + 1] for i in range(n - 1)]
    for i in range(n - 3):  # pairs (d[i], d[i+1]) without the last step to zero
        r = deltas[i + 1] / deltas[i]
        if r > JUMP_RATIO_MAX or r < 1.0 / JUMP_RATIO_MAX:
            msg = ("step size changes by x%.3g between steps %d and %d (limit x%g)"
                   % (r, i, i + 1, JUMP_RATIO_MAX))
            if ratio_is_error:
                return err(msg, fix_default, check="step_ratio", index=i + 1, ratio=r)
            return [errors.make_issue("sigma_step_ratio", node, param, msg,
                                      "use MossPhotonSchedule(mode=reference) for the verified schedule",
                                      detail={"check": "step_ratio", "index": i + 1, "ratio": r})]
    return []


def compare_sigmas(a, b):
    """Compare two schedules after rounding to float32.

    ``{"equal", "bit_equal", "max_abs_diff", "n_differ", "same_length"}``: ``equal`` = same length and every value
    ``==``; ``bit_equal`` = identical float32 bytes (differs from ``equal`` only for ``-0.0`` / NaN); ``n_differ``
    counts differing positions plus the length difference; ``max_abs_diff`` is taken over the common prefix.
    """
    va, vb = _as_list(a), _as_list(b)
    fa = struct.unpack("<%df" % len(va), _f32_bytes(va))
    fb = struct.unpack("<%df" % len(vb), _f32_bytes(vb))
    common = min(len(fa), len(fb))
    diffs = [abs(x - y) for x, y in zip(fa[:common], fb[:common])]
    n_differ = sum(1 for x, y in zip(fa[:common], fb[:common]) if not x == y) + abs(len(fa) - len(fb))
    return {
        "equal": len(fa) == len(fb) and n_differ == 0,
        "bit_equal": _f32_bytes(va) == _f32_bytes(vb),
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "n_differ": n_differ,
        "same_length": len(fa) == len(fb),
    }


def branches_per_step(cfg):
    """Model branches per step: 1 if ``cfg`` is 1.0 (the cond-only shortcut, CUI/comfy/samplers.py:610-612) else 2."""
    return 1 if math.isclose(cfg, 1.0) else 2


def expected_forwards(steps, cfg, extra=0):
    """``steps * branches_per_step(cfg) + extra`` (the exact forward count before the run)."""
    return steps * branches_per_step(cfg) + extra


def build_schedule_info(sigmas, *, mode, scheduler, steps, shift_model, shift_used, multiplier,
                        native_reference=None, computed_on="cpu", warnings=()):
    """PHOTON_SCHEDULE_INFO dict of design 2.6 for an already validated ``sigmas`` array.

    ``native_reference`` is a fresh native schedule (array) to compare against; it is required (the contract
    always carries the comparison; ``ValueError`` otherwise, programmer error). ``is_native_reference`` is true iff
    ``mode == "reference"`` and the arrays are bit-equal. ``warnings`` are ``Issue`` objects, stored as dicts.
    """
    if native_reference is None:
        raise ValueError("build_schedule_info needs native_reference (a fresh native schedule)")
    vals = _as_list(sigmas)
    cmp = compare_sigmas(vals, native_reference)
    return {
        "schema_version": 1,
        "kind": "photon_schedule_info",
        "mode": mode,
        "scheduler": scheduler,
        "steps": int(steps),
        "n_sigmas": len(vals),
        "dtype": "float32",
        "computed_on": computed_on,
        "shift_model": float(shift_model),
        "shift_used": float(shift_used),
        "multiplier": float(multiplier),
        "sigma_first": float(vals[0]),
        "sigma_second": float(vals[1]),
        "sigma_penultimate": float(vals[-2]),
        "sigma_last": float(vals[-1]),
        "sigmas_sha256": sigma_hash(vals),
        "is_native_reference": mode == "reference" and cmp["bit_equal"],
        "native_reference": {
            "sha256": sigma_hash(native_reference),
            "equal": cmp["equal"],
            "bit_equal": cmp["bit_equal"],
            "max_abs_diff": cmp["max_abs_diff"],
            "n_differ": cmp["n_differ"],
        },
        "model_calls_per_step": {"cfg_ne_1": 2, "cfg_eq_1": 1},
        "validation": {
            "ok": True,
            "jump_ratio_max": JUMP_RATIO_MAX,
            "observed_jump_ratio_max": observed_jump_ratio(vals),
            "checks": list(VALIDATION_CHECKS),
        },
        "warnings": [w.to_dict() if hasattr(w, "to_dict") else dict(w) for w in warnings],
    }
