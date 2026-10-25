"""PHOTON_REPORT builder, validator and helpers (design 2.3).

Imports: stdlib and ``canon`` / ``consts`` / ``errors`` only; pure, no torch, no comfy.

The builder never copies private data (R5): a whole object that carries ``private_paths`` (PHOTON_CONSTRAINTS, the
identity profile) is refused, the caller copies the allowed fields only. Reports are plain JSON-safe dicts.
"""
import copy
import math
import re
import time

from . import canon, consts, errors

__all__ = [
    "ReportBuilder", "RECIPE_HASH_PATHS", "recipe_hash", "validate_report", "format_summary", "with_diagnostics",
    "redact",
]

_PHASES = ("preflight_done", "sampling_done")

# Leaf paths that make up the recipe hash (design 2.3). ``[*]`` = every element of the list, in order. The two
# ``model.identity`` paths below count only when ``model.identity.sha256`` is null (m3), and the two
# ``lora.stack[*]`` basename / size paths only when that entry's sha256 is null.
RECIPE_HASH_PATHS = (
    "pack.version",
    "env.comfyui_version", "env.torch", "env.cuda", "env.device_name", "env.attention", "env.inference_dtype",
    "env.weights_streaming", "env.deterministic_flag",
    "preset.id", "preset.version", "preset.sha256", "preset.manifest_sha256",
    "label.modifiers",
    "model.family", "model.identity.sha256", "model.identity.basename", "model.identity.size",
    "sampling",
    "inputs.seed", "inputs.batch", "inputs.width", "inputs.height", "inputs.noise_policy", "inputs.noise_sha256",
    "conditioning.positive.sha256", "conditioning.negative.sha256",
    "lora.stack[*].sha256", "lora.stack[*].basename", "lora.stack[*].size",
    "lora.stack[*].strength_model", "lora.stack[*].strength_clip", "lora.stack[*].applied_to",
    "lora.foreign_patches.entries_observed", "lora.foreign_patches.uuid_match",
)
_IDENTITY_FALLBACK = ("model.identity.basename", "model.identity.size")
_LORA_FALLBACK = ("basename", "size")


def _get(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _recipe_material(report):
    """The exact JSON-safe structure whose canonical hash is ``recipe_hash``."""
    material = {}
    identity_sha = _get(report, "model.identity.sha256")
    stack = _get(report, "lora.stack")
    stack = stack if isinstance(stack, list) else []
    for path in RECIPE_HASH_PATHS:
        if path.startswith("lora.stack[*]."):
            field = path.split(".")[-1]
            values = []
            for item in stack:
                item = item if isinstance(item, dict) else {}
                if field in _LORA_FALLBACK and item.get("sha256") is not None:
                    values.append(None)
                else:
                    values.append(item.get(field))
            material[path] = values
        elif path in _IDENTITY_FALLBACK and identity_sha is not None:
            material[path] = None
        else:
            material[path] = _get(report, path)
    return material


def recipe_hash(report):
    """sha256 of the canonical JSON of the recipe material (excludes timing, memory, preview, warnings, time)."""
    return canon.sha256_hex(canon.canonical_json(_recipe_material(report)))


def _utc(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _find_private(obj, path="report"):
    """First path of a dict that carries a ``private_paths`` key, else None."""
    if isinstance(obj, dict):
        if "private_paths" in obj:
            return path
        for k, v in obj.items():
            hit = _find_private(v, "%s.%s" % (path, k))
            if hit:
                return hit
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hit = _find_private(v, "%s[%d]" % (path, i))
            if hit:
                return hit
    return None


def _check_known_version(report):
    version = report.get("schema_version") if isinstance(report, dict) else None
    if isinstance(version, int) and not isinstance(version, bool) and version > consts.REPORT_SCHEMA:
        errors.raise_first_error([errors.make_issue(
            "unsupported_feature", "report", "schema_version",
            "report schema_version = %d is newer than this MossPhoton knows (%d)" % (version, consts.REPORT_SCHEMA),
            "update the ComfyUI-MossPhoton pack", severity="error")])


class ReportBuilder:
    """Collects the sections of one run and finalizes a PHOTON_REPORT dict.

    ``clock`` (seconds since the epoch) and ``perf`` (monotonic seconds) are injectable for tests.
    """

    def __init__(self, node_id, *, clock=time.time, perf=time.perf_counter):
        self._clock = clock
        self._perf = perf
        self._t0 = perf()
        self._marks = {}
        self._report = {
            "schema_version": consts.REPORT_SCHEMA,
            "kind": "photon_report",
            "node": node_id,
            "pack": {"name": consts.PACK_NAME, "version": consts.PACK_VERSION},
            "text_encoder": None,
            "warnings": [],
            "fallbacks": [],
            "privacy": {"prompts_logged": False, "reference_photos_logged": False},
        }
        self._warned = set()

    def section(self, name, **fields):
        """Shallow-merge ``fields`` into ``report[name]`` (created as ``{}`` when absent or null).

        Values are deep-copied (inputs are never aliased). A value that is, or contains, an object with
        ``private_paths`` raises ``ValueError``: copy the allowed fields only (R5).
        """
        if name in ("schema_version", "kind", "node", "recipe_hash", "created_utc", "warnings", "fallbacks"):
            raise ValueError("section name %r is owned by the builder" % (name,))
        hit = _find_private(fields)
        if hit:
            raise ValueError("section %r would copy a private object (%s carries private_paths)" % (name, hit))
        current = self._report.get(name)
        if not isinstance(current, dict):
            current = {}
            self._report[name] = current
        current.update(copy.deepcopy(fields))

    def warn(self, issue, scope=None):
        """Append an ``Issue`` (or its dict) to ``warnings``; one entry per (code, param, scope). Errors are refused.

        ``scope`` separates warnings that share a code and a param but belong to different things, such as the same
        note about two LoRAs of one stack (``param`` is ``lora_name`` for both); it is not written to the report.
        """
        item = issue.to_dict() if hasattr(issue, "to_dict") else copy.deepcopy(dict(issue))
        if item.get("severity") == "error":
            raise ValueError("an error issue (%s) is raised, not recorded as a warning" % item.get("code"))
        key = (item.get("code"), item.get("param"), scope)
        if key in self._warned:
            return
        self._warned.add(key)
        self._report["warnings"].append(item)

    def fallback(self, reason, action):
        """Record a recoverable fallback to the reference configuration."""
        self._report["fallbacks"].append({"reason": reason, "action": action})

    def mark_phase(self, name):
        """Record the end of ``preflight_done`` or ``sampling_done`` (feeds ``timing``)."""
        if name not in _PHASES:
            raise ValueError("phase must be one of %s, got %r" % (_PHASES, name))
        self._marks[name] = self._perf()

    def finalize(self):
        """Fill ``timing``, ``created_utc`` and ``recipe_hash``, validate, return a deep-copied plain dict.

        Raises ``ValueError`` listing the problems when the report does not validate (the producer is wrong, not
        the user).
        """
        now = self._perf()
        report = copy.deepcopy(self._report)
        pre = self._marks.get("preflight_done")
        samp = self._marks.get("sampling_done")
        timing = report.get("timing") if isinstance(report.get("timing"), dict) else {}
        timing.update({
            "total_s": round(now - self._t0, 6),
            "preflight_s": round(pre - self._t0, 6) if pre is not None else None,
            "sampling_s": round(samp - pre, 6) if pre is not None and samp is not None else None,
            "estimate_s": None,  # always null in v1 (LD10)
        })
        timing.setdefault("step_s_mean", None)
        timing.setdefault("step_s_max", None)
        report["timing"] = timing
        report["created_utc"] = _utc(self._clock())
        report["recipe_hash"] = recipe_hash(report)
        problems = validate_report(report)
        hit = _find_private(report)
        if hit:
            problems.append("%s carries private_paths" % hit)
        if problems:
            raise ValueError("invalid PHOTON_REPORT: " + "; ".join(problems[:10]))
        return report


# --------------------------------------------------------------------------- structural validation
# A compact transcription of the field tables of design 2.3 (the same contract as the strict JSON schema in
# tests/fixtures/contracts/schemas/report.schema.json, which a unit test cross-checks): O(required, optional),
# A(item), E(*values), and type names joined by "|": int num str bool null sha. Unknown keys are errors.

class _O:
    def __init__(self, req, opt=None):
        self.req, self.opt = req, opt or {}


class _A:
    def __init__(self, item, min_items=0, max_items=None):
        self.item, self.min_items, self.max_items = item, min_items, max_items


class _E:
    def __init__(self, *values, prefix=None):
        self.values, self.prefix = values, prefix


_SHA = re.compile(r"^[0-9a-f]{64}$")
_SEMVER = re.compile(r"^[0-9]+[.][0-9]+[.][0-9]+$")
_PIN = re.compile(r"^[a-z0-9_]+@[0-9]+[.][0-9]+[.][0-9]+#[0-9a-f]{8}$")
_BASENAME = re.compile(r"^[^/\\]+$")
_CODE = re.compile(r"^[a-z][a-z0-9_]*$")
_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?Z$")

_ISSUE = _O({"code": "code", "severity": _E("error", "warning", "info"), "node": "str", "param": "str",
             "message": "str", "fix": "str"}, {"detail": "obj"})
_COND = _O({"entries": "int", "tokens": _A("int"), "dim": "int", "sha256": "sha"})

_SPEC = _O({
    "schema_version": _E(1),
    "kind": _E("photon_report"),
    "node": "str",
    "pack": _O({"name": "str", "version": "semver"}),
    "env": _O({
        "comfyui_version": "str", "comfyui_commit": "str|null", "python": "str", "torch": "str", "cuda": "str|null",
        "device": "str", "device_name": "str|null",
        "attention": _O({"global": _E("sage", "flash", "xformers", "pytorch", "split", "sub_quad", "comfy_kitchen",
                                      prefix="unknown:"),  # m15: a backend this table does not know, by function name
                         "model_override": "str|null"}),
        "deterministic_flag": "bool", "weights_streaming": "bool", "inference_dtype": "str"}),
    "label": _O({
        "preset_status": _E("stable", "candidate", "experimental", "preview_only"), "ui_label": "str",
        "parity_class": _E("not_reference", "reference_modified", "reference_with_foreign_patches",
                           "reference_unverified", "reference_with_lora", "reference_exact", "native_turbo"),
        "not_reference_reasons": _A("str"), "modifiers": _A("str"), "calibrated": "bool", "customized": "bool",
        "recommended_final": "bool"}),
    "preset": _O({
        "id": _E(*consts.PRESET_IDS, *consts.TURBO_PRESET_IDS), "version": "semver", "sha256": "sha", "pin": "pin", "pinned": "bool",
        "manifest_sha256": "sha|null", "layers_applied": _A("str"),
        "diff": _A(_O({"path": "str", "base": "any", "effective": "any"}))}),
    "model": _O({
        "architecture": _E("moss_image"), "family": _E("moss-image-sft", "moss-image-base", "unknown-moss"),
        "num_layers": "int",
        "identity": _O({"sha256": "sha|null", "size": "int", "basename": "basename", "hash_source":
                        _E("computed", "cache", "unavailable"), "classified_by": _E("known_table", "none"),
                        "official": "bool|null", "variant": "str|null", "verified_matrix": "bool|null"}),
        "model_sampling": _O({"class": "str", "shift": "num", "multiplier": "num"})}),
    "text_encoder": "obj|null",
    "conditioning": _O({"positive": _COND, "negative": _COND}),
    "inputs": _O({
        "seed": "int", "batch": "int", "width": "int", "height": "int", "latent_shape": _A("int", 4, 4),
        "latent_dtype": "str", "latent_was_empty": "bool", "latent_raw_shape": _A("int"),
        "latent_rescaled_by_native_fix": "bool", "noise_policy": _E("native", "seed_plus_i"),
        "subseeds": _A("int", 1), "noise_sha256": "sha"}),
    "sampling": _O({
        "steps": "int", "sampler": "str", "scheduler": "str", "shift": "num", "cfg": "num",
        "slg": _O({"mode": "str", "layers": _A("int"), "start": "num", "end": "num", "sigma_window": _A("num", 2, 2)}),
        "restart": "bool", "ancestral_noise": "bool",
        "sigmas": _O({"source": _E("native_scheduler", "input"), "n": "int", "sha256": "sha", "first": "num",
                      "last": "num", "equals_native": "bool|null"})}),
    "lora": _O({
        "stack": _A(_O({
            "role": _E("character", "style", "wardrobe", "scene", "other", "distill"), "basename": "str",
            "size": "int", "sha256": "sha|null",
            "strength_model": "num", "strength_clip": "num", "applied_to": "obj", "coverage_ratio": "num",
            "layers_missing": _A("int"), "keys_patched": "int", "patch_entries": "int", "trigger_present": "bool"})),
        "foreign_patches": _O({
            "entries_expected": "int", "entries_observed": "int", "baseline_entries": "int", "uuid_match": "bool",
            "extra_entries": "int", "keys_sample": _A("str", 0, 20)}),
        "te_applied": "bool", "profile_digest": "sha|null"}),
    "guards": _O({
        "finite_check": _E("passed", "failed", "skipped"),
        "items": _A(_O({"id": "str", "status": _E("not_run"),
                        "reason": _E("missing_guard_dependency", "not_implemented_v1")})),
        "composition_limit": _O({"enabled": _E(False)}), "guard_passed": _E(False)}),
    "forwards": _O({k: "int" for k in ("expected_branch_forwards", "expected_sample_forwards", "model_calls",
                                       "sample_forwards", "slg_skip_calls", "extra")} | {"match": "bool"}),
    "timing": _O({"total_s": "num|null", "preflight_s": "num|null", "sampling_s": "num|null",
                  "step_s_mean": "num|null", "step_s_max": "num|null", "estimate_s": "null"},
                 {"step_s": _A("num")}),
    "memory": _O({"measured": "bool", "peak_vram_bytes": "int|null", "device": "str"}),
    "preview": _O({"enabled": "bool"}),
    "output": _O({"latent_shape": _A("int"), "latent_sha256": "sha", "finite": "bool"}),
    "warnings": _A(_ISSUE),
    "fallbacks": _A(_O({"reason": "str", "action": "str"})),
    "privacy": _O({"prompts_logged": _E(False), "reference_photos_logged": _E(False)}),
    "recipe_hash": "sha",
    "created_utc": "str",
}, {"diagnostics": "obj"})

# Numeric lower bounds of the schema (path of the leaf -> minimum).
_MINIMUMS = {
    "model.num_layers": 1, "model.identity.size": 0, "inputs.seed": 0, "inputs.batch": 1, "inputs.width": 16,
    "inputs.height": 16, "sampling.steps": 1, "sampling.sigmas.n": 2, "conditioning.positive.entries": 1,
    "conditioning.negative.entries": 1, "conditioning.positive.dim": 1, "conditioning.negative.dim": 1,
}


_PATTERNS = {"sha": _SHA, "semver": _SEMVER, "pin": _PIN, "basename": _BASENAME, "code": _CODE}


def _type_ok(name, value):
    if name == "any":
        return True
    if name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "num":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if name == "str":
        return isinstance(value, str)
    if name == "bool":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "obj":
        return isinstance(value, dict)
    if name in _PATTERNS:
        return isinstance(value, str) and bool(_PATTERNS[name].match(value))
    raise AssertionError("unknown type name " + name)


def _check(spec, value, path, out):
    if isinstance(spec, str):
        if not any(_type_ok(t, value) for t in spec.split("|")):
            out.append("%s: expected %s, got %s" % (path, spec, _describe(value)))
    elif isinstance(spec, _E):
        named = isinstance(value, str) and spec.prefix is not None and len(value) > len(spec.prefix) \
            and value.startswith(spec.prefix)
        if not named and not any(value == v and type(value) is type(v) for v in spec.values):
            out.append("%s: expected one of %s, got %s" % (path, list(spec.values), _describe(value)))
    elif isinstance(spec, _A):
        if not isinstance(value, list):
            out.append("%s: expected array, got %s" % (path, _describe(value)))
            return
        if len(value) < spec.min_items or (spec.max_items is not None and len(value) > spec.max_items):
            out.append("%s: expected %s..%s items, got %d" % (path, spec.min_items, spec.max_items, len(value)))
        for i, item in enumerate(value):
            _check(spec.item, item, "%s[%d]" % (path, i), out)
    else:
        if not isinstance(value, dict):
            out.append("%s: expected object, got %s" % (path, _describe(value)))
            return
        for key in spec.req:
            if key not in value:
                out.append("%s.%s: required key is missing" % (path, key))
        for key in value:
            if key not in spec.req and key not in spec.opt:
                out.append("%s.%s: unexpected key" % (path, key))
        for key, sub in list(spec.req.items()) + list(spec.opt.items()):
            if key in value:
                _check(sub, value[key], "%s.%s" % (path, key), out)


def _describe(value):
    return "%s %r" % (type(value).__name__, value if not isinstance(value, (dict, list)) else "...")


def validate_report(report):
    """Structural problems of a PHOTON_REPORT as ``["report.path: message", ...]``; ``[]`` = valid.

    Checks required keys, unexpected keys, JSON types, enums, sha256 shape, array sizes and a few lower bounds
    (the strict contract of design 2.3). A ``schema_version`` newer than ``consts.REPORT_SCHEMA`` is reported as
    a problem here and raises ``unsupported_feature`` in ``with_diagnostics`` / ``format_summary`` / ``redact``.
    """
    out = []
    if not isinstance(report, dict):
        return ["report: expected object, got %s" % _describe(report)]
    _check(_SPEC, report, "report", out)
    for path, minimum in _MINIMUMS.items():
        value = _get(report, path)
        if isinstance(value, int) and not isinstance(value, bool) and value < minimum:
            out.append("report.%s: %d is below the minimum %d" % (path, value, minimum))
    for path in ("inputs.width", "inputs.height"):
        value = _get(report, path)
        if isinstance(value, int) and not isinstance(value, bool) and value % 16:
            out.append("report.%s: %d is not a multiple of 16" % (path, value))
    created = report.get("created_utc")
    if isinstance(created, str) and not _UTC.match(created):
        out.append("report.created_utc: %r is not an ISO 8601 UTC timestamp" % created)
    return out


# --------------------------------------------------------------------------- summary, diagnostics, redaction
def _short(h, n=8):
    return h[:n] if isinstance(h, str) else "-"


def format_summary(report):
    """One block of text for ``ui.text`` and logs. Facts only; no prompt text, no paths (R5)."""
    _check_known_version(report)
    def g(path, default="-"):
        value = _get(report, path)
        return default if value is None else value

    lines = [
        "MossPhoton %s: preset %s@%s (%s), label %s" % (
            g("node"), g("preset.id"), g("preset.version"), g("label.ui_label"), g("label.parity_class")),
        "model %s (%s, sha %s), %sx%s batch %s, seed %s (%s)" % (
            g("model.family"), g("model.identity.basename"), _short(_get(report, "model.identity.sha256")),
            g("inputs.width"), g("inputs.height"), g("inputs.batch"), g("inputs.seed"), g("inputs.noise_policy")),
        "sampling %s steps, %s/%s, cfg %s, shift %s, sigmas %s (native: %s)" % (
            g("sampling.steps"), g("sampling.sampler"), g("sampling.scheduler"), g("sampling.cfg"),
            g("sampling.shift"), _short(_get(report, "sampling.sigmas.sha256")), g("sampling.sigmas.equals_native")),
        "forwards %s/%s (match %s), total %s s" % (
            g("forwards.sample_forwards"), g("forwards.expected_sample_forwards"), g("forwards.match"),
            g("timing.total_s")),
    ]
    modifiers = _get(report, "label.modifiers")
    if modifiers:
        lines.append("modifiers: " + ", ".join(str(m) for m in modifiers))
    stack = _get(report, "lora.stack")
    if stack:
        lines.append("lora: %d adapter(s)" % len(stack))
    warns = report.get("warnings") or []
    if warns:
        lines.append("warnings: " + ", ".join(str(w.get("code")) for w in warns if isinstance(w, dict)))
    lines.append("recipe %s" % _short(report.get("recipe_hash"), 12))
    return "\n".join(lines)


def with_diagnostics(report, diagnostics):
    """Deep copy of ``report`` with ``diagnostics`` attached; ``recipe_hash`` is kept (parent recipe).

    The attached block gets ``parent_recipe_hash`` = the report's ``recipe_hash`` unless it sets one itself
    (ASSUMED reading of "parent_recipe_hash kept"). The original is not mutated. A newer ``schema_version``
    raises ``unsupported_feature``.
    """
    _check_known_version(report)
    out = copy.deepcopy(report)
    block = copy.deepcopy(diagnostics)
    if isinstance(block, dict):
        block.setdefault("parent_recipe_hash", report.get("recipe_hash"))
    out["diagnostics"] = block
    return out


def _drop_path(obj, path):
    """Remove ``a.b[*].c`` style ``path`` from ``obj`` in place; missing parts are ignored."""
    parts = path.split(".")
    cur = [obj]
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        star = part.endswith("[*]")
        key = part[:-3] if star else part
        nxt = []
        for node in cur:
            if not isinstance(node, dict) or key not in node:
                continue
            if last and not star:
                del node[key]
            elif star:
                if isinstance(node[key], list):
                    if last:
                        node[key] = []
                    else:
                        nxt.extend(node[key])
            else:
                nxt.append(node[key])
        cur = nxt


def _redact_privates(obj):
    if isinstance(obj, dict):
        for key in list(obj):
            _redact_privates(obj[key])
        paths = obj.get("private_paths")
        if isinstance(paths, list):
            for p in paths:
                if isinstance(p, str):
                    _drop_path(obj, p)
            obj.pop("private_paths", None)
    elif isinstance(obj, list):
        for item in obj:
            _redact_privates(item)


def redact(report, *, keep_basenames=True):
    """Deep copy safe to share: drops every ``private_paths`` listed path (and the key), optionally basenames.

    ``keep_basenames=False`` replaces the basenames of the checkpoint, the LoRAs and the text encoder with
    ``"redacted"`` (ASSUMED marker; the schema requires a string). The original is not mutated.
    """
    _check_known_version(report)
    out = copy.deepcopy(report)
    _redact_privates(out)
    if not keep_basenames:
        ident = _get(out, "model.identity")
        if isinstance(ident, dict) and "basename" in ident:
            ident["basename"] = "redacted"
        stack = _get(out, "lora.stack")
        for item in stack if isinstance(stack, list) else []:
            if isinstance(item, dict) and "basename" in item:
                item["basename"] = "redacted"
        te = out.get("text_encoder")
        if isinstance(te, dict) and "basename" in te:
            te["basename"] = "redacted"
    return out
