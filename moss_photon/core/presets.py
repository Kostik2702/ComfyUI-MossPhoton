"""Preset catalog, settings resolver and the pure "effective view" (design 1.3, 4.9, 2.1, 2.2).

Imports: stdlib + ``canon``, ``errors``, ``schema_lite``, ``consts`` only. No comfy, no torch.

Presets are versioned recipes of sampling numbers and input contracts. They never carry or change text, seed, size,
source image, LoRA strength or checkpoint type: ``ResolvedPreset.effective`` has exactly the keys of the preset
``sampling`` and ``budget`` sections, and an override naming anything else is ``invalid_override``.

Resolution order (SPEC 6.2): preset version -> model-specific calibration manifest -> LoRA profile -> explicit override.
"""
import copy
import dataclasses
import json
import logging
import math
import os
import re

from . import canon, consts, errors, schema_lite

__all__ = [
    "OVERRIDE_KEYS", "ResolvedPreset", "preset_dir", "load_schema", "validate_preset", "catalog", "catalog_ids",
    "catalog_issues", "selectable_ids", "describe_catalog", "catalog_sha256", "preset_sha256", "pin_of", "parse_pin", "load_manifest",
    "resolve", "diff_settings", "evaluate_requirements", "effective_view", "has_darkroom", "darkroom_diff", "SLG_KEYS", "TURBO_PRESET_IDS",
    "ALL_PRESET_IDS",
]

_log = logging.getLogger("MossPhoton")
_logged_broken = set()

# Turbo presets (docs/03-turbo.md 5), the darkroom ones (MP-71) included. ``consts.PRESET_IDS`` stays the eight of SPEC 6.1;
# the catalog order is those eight followed by the turbo ones. The list is ``consts.TURBO_PRESET_IDS``: the report schema, the
# preset schema and the fixture schema are cross-checked against that one literal.
TURBO_PRESET_IDS = consts.TURBO_PRESET_IDS
ALL_PRESET_IDS = tuple(consts.PRESET_IDS) + TURBO_PRESET_IDS
_DISTILL_POLICIES = ("require_distill_profile", "require_distill_and_character_profile")

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCHEMA_FILE = "schema.json"

# Override names (SPEC 6.2 / design 1.3); the "sampling." prefix is implied.
OVERRIDE_KEYS = ("steps", "cfg", "slg_mode", "slg_layers", "slg_start", "slg_end", "shift")
_INT_KEYS = ("steps",)
_FLOAT_KEYS = ("cfg", "shift", "slg_start", "slg_end")

_PIN_RE = re.compile(r"([a-z0-9_]+)@([0-9]+[.][0-9]+[.][0-9]+)(?:#([0-9a-f]{8}))?")
_FAMILY_RE = re.compile(r"[A-Za-z0-9_.-]+")

_NOT_KNOWN = ["live model shift", "checkpoint family", "LoRA stack", "time"]


def preset_dir():
    """Absolute path of ``<pkg>/presets``."""
    return os.path.join(_PKG_ROOT, "presets")


def _calibration_dir():
    return os.path.join(_PKG_ROOT, "moss_photon", "calibration")


def _issue(code, node, param, message, fix, **kw):
    return errors.make_issue(code, node, param, message, fix, **kw)


def _read_json(path, label):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise errors.InvalidPresetFile(_issue(
            "invalid_preset_file", "presets", label, "file is missing", "restore the file from the package or reinstall"))
    except (OSError, ValueError) as exc:
        raise errors.InvalidPresetFile(_issue(
            "invalid_preset_file", "presets", label, "cannot be parsed: %s" % exc,
            "restore the file from the package or reinstall"))


def load_schema(directory=None):
    """The parsed ``presets/schema.json``. Raises ``invalid_preset_file`` when missing, unparseable or unusable."""
    schema = _read_json(os.path.join(directory or preset_dir(), _SCHEMA_FILE), "presets/" + _SCHEMA_FILE)
    try:
        schema_lite.Validator(schema)
    except schema_lite.SchemaDefinitionError as exc:
        raise errors.InvalidPresetFile(_issue(
            "invalid_preset_file", "presets", "presets/" + _SCHEMA_FILE, str(exc),
            "restore the file from the package or reinstall"))
    return schema


def _numeric_leaves(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if prefix == "sampling" and key == "override_bounds":
                continue  # the bounds carry their own ``source`` objects
            out += _numeric_leaves(value, _join(prefix, key))
    elif isinstance(obj, list):
        if obj and all(isinstance(x, int) and not isinstance(x, bool) for x in obj):
            out.append(prefix)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(prefix)
    return out


def _join(prefix, key):
    return "%s.%s" % (prefix, key) if prefix else key


def validate_preset(obj, stem=None, schema=None):
    """Schema errors plus the cross-field rules of design 1.3, as ``"path: message"`` strings (empty = valid).

    ``stem`` (file name without ``.json``), when given, must equal ``id``. ``schema`` defaults to ``load_schema()``.
    Cross-field rules run only when the schema check passed: ``budget.nominal_steps == sampling.steps``;
    ``slg_start <= slg_end``; ``slg_mode == "uncond_skip"`` needs ``slg_layers``; ``require_explicit_user_profile``
    needs a ``requires_lora_role`` constraint; a constraint with severity ``error`` needs a reason that is a raisable
    code; the turbo rules of ``_turbo_problems`` (distill policies need their ``requires_lora_role`` constraints, SLG off
    and ``recommended_final`` false; ``override_bounds`` ordered, default inside verified, ``max_forwards`` at least the
    bounds corner); every numeric leaf of ``sampling`` / ``budget`` (null ``max_forwards`` excluded, ``bool`` is not a number)
    has a ``sources`` entry and every ``sources`` key names such a leaf (``darkroom.*`` numbers included); ``stable``
    implies ``calibration.state == official``; ``experimental`` / ``preview_only`` imply ``ui.recommended_final == false``;
    the rules of ``_darkroom_problems`` for the optional ``darkroom`` block.
    """
    found = schema_lite.validate(obj, schema if schema is not None else load_schema())
    problems = ["%s: %s" % (e.path or "(root)", e.message) for e in found]
    if problems:
        return problems
    if stem is not None and obj["id"] != stem:
        problems.append("id: %r does not match the file name %r" % (obj["id"], stem))
    smp, bud = obj["sampling"], obj["budget"]
    if bud["nominal_steps"] != smp["steps"]:
        problems.append("budget.nominal_steps: %r must equal sampling.steps %r" % (bud["nominal_steps"], smp["steps"]))
    if smp["slg_start"] > smp["slg_end"]:
        problems.append("sampling.slg_start: %r must be <= slg_end %r" % (smp["slg_start"], smp["slg_end"]))
    if smp["slg_mode"] == "uncond_skip" and not smp["slg_layers"]:
        problems.append("sampling.slg_layers: must not be empty when slg_mode is uncond_skip")
    if obj["lora_policy"] == "require_explicit_user_profile" \
            and not any(c["kind"] == "requires_lora_role" for c in obj["constraints"]):
        problems.append("constraints: lora_policy require_explicit_user_profile needs a requires_lora_role constraint")
    for i, c in enumerate(obj["constraints"]):
        info = errors.REASONS.get(c["reason"])
        if c["severity"] == "error" and (info is None or info.exc is None):
            problems.append("constraints[%d].reason: %r cannot be raised with severity error" % (i, c["reason"]))
    sourced = {"sampling": smp, "budget": bud}
    if "darkroom" in obj:  # MP-71: the phase values are numbers of the recipe like the sampling ones
        sourced["darkroom"] = obj["darkroom"]
    leaves = set(_numeric_leaves(sourced))
    if bud["max_forwards"] is None:
        leaves.discard("budget.max_forwards")
    missing, stale = sorted(leaves - set(obj["sources"])), sorted(set(obj["sources"]) - leaves)
    if missing:
        problems.append("sources: no entry for numeric leaf %s" % ", ".join(missing))
    if stale:
        problems.append("sources: entry names no numeric leaf: %s" % ", ".join(stale))
    if obj["status"] == "stable" and obj["calibration"]["state"] != "official":
        problems.append("status: stable requires calibration.state official")
    if obj["status"] in ("experimental", "preview_only") and obj["ui"]["recommended_final"]:
        problems.append("ui.recommended_final: must be false for status %s" % obj["status"])
    return problems + _turbo_problems(obj) + _darkroom_problems(obj)


def _darkroom_problems(obj):
    """Cross-field rules of the optional ``darkroom`` block (MP-71); empty for a preset without one.

    The run would refuse the same things (``core/darkroom.py``) before the first forward; here the catalog refuses them
    when the file is read, so a shipped preset can not be a plan that never runs. No LoRA is required (MP-84: the commit
    weight boosts the ``identity`` entries of a LoRA Phase plan when there are some, else it is skipped with a warning),
    ``compose.cfg`` is ``sampling.cfg`` (the one number the sampler resolves; the block repeats it so the block reads as a
    whole), ``0 <= r_c <= r_k <= r_f <= 1`` and a weight change needs an open commit window. ``sa2_8`` with another step
    count is not a problem: the run resamples its knots (``darkroom.sa2_values``).
    """
    block = obj.get("darkroom")
    if block is None:
        return []
    problems = []
    if block["compose"]["cfg"] != obj["sampling"]["cfg"]:
        problems.append("darkroom.compose.cfg: %r must equal sampling.cfg %r" % (
            block["compose"]["cfg"], obj["sampling"]["cfg"]))
    c = block["commit"]
    if not 0.0 <= c["r_c"] <= c["r_k"] <= block["finish"]["r_f"] <= 1.0:
        problems.append("darkroom.commit: needs 0 <= r_c <= r_k <= finish.r_f <= 1, got %r, %r, %r" % (
            c["r_c"], c["r_k"], block["finish"]["r_f"]))
    elif c["w_compose"] != c["w_commit"] and not c["r_c"] < c["r_k"]:
        problems.append("darkroom.commit: the weight moves %r -> %r over an empty commit window" % (
            c["w_compose"], c["w_commit"]))
    return problems


def _turbo_problems(obj):
    """Cross-field rules of the turbo additions (docs/03-turbo.md 5.2); empty for presets that use none of them."""
    problems = []
    smp, bud, cons = obj["sampling"], obj["budget"], obj["constraints"]
    needs = {"require_distill_profile": ("distill",), "require_distill_and_character_profile": ("distill", "character")}
    for role in needs.get(obj["lora_policy"], ()):
        if not any(c["kind"] == "requires_lora_role" and c.get("role") == role for c in cons):
            problems.append("constraints: lora_policy %s needs a requires_lora_role constraint for role %s" % (
                obj["lora_policy"], role))
    if obj["lora_policy"] in _DISTILL_POLICIES:
        if smp["slg_mode"] != "off":
            problems.append("sampling.slg_mode: must be off for a distill preset (SLG is an Advanced-only override)")
        # ASSUMED (decision Q9): no distill preset is recommended until the turbo gate passes; the lead lifts this then.
        if obj["ui"]["recommended_final"]:
            problems.append("ui.recommended_final: must be false for a distill preset until the turbo gate passes")
    for i, c in enumerate(cons):
        if c["kind"] == "limits_lora_roles" and "roles" not in c:
            problems.append("constraints[%d].roles: limits_lora_roles needs roles" % i)
    bounds = smp.get("override_bounds")
    if bounds is None:
        return problems
    for name in ("steps", "cfg"):
        b = bounds[name]
        if not b["min"] <= b["verified_min"] <= b["verified_max"] <= b["max"]:
            problems.append("sampling.override_bounds.%s: needs min <= verified_min <= verified_max <= max" % name)
        elif not b["verified_min"] <= smp[name] <= b["verified_max"]:
            problems.append("sampling.override_bounds.%s: the preset default %r must be inside the verified range" % (
                name, smp[name]))
    if bud["max_forwards"] is None:
        problems.append("budget.max_forwards: must be set when override_bounds exist")
    else:
        corner = bounds["steps"]["max"] * _branches_per_step(bounds["cfg"]["max"]) + bud["extra_forwards"]
        if bud["max_forwards"] < corner:
            problems.append("budget.max_forwards: %r is below the forwards %r of the override_bounds corner" % (
                bud["max_forwards"], corner))
    return problems


def _scan_catalog(directory):
    """``(found, broken)``: validated ``{id: preset}`` in ``ALL_PRESET_IDS`` order and ``{stem: error Issue}`` for every
    ``<stem>.json`` that cannot be used (unparseable, schema violation, broken cross-field rule, id differs from the
    stem). One broken preset file never takes the others down (m22); a broken schema or an unreadable directory does,
    because nothing can be validated then."""
    directory = directory or preset_dir()
    schema = load_schema(directory)
    try:
        names = sorted(n for n in os.listdir(directory) if n.endswith(".json") and n != _SCHEMA_FILE)
    except OSError as exc:
        raise errors.InvalidPresetFile(_issue(
            "invalid_preset_file", "presets", "presets", "cannot list the directory: %s" % exc,
            "restore the files from the package or reinstall"))
    found, broken = {}, {}
    for name in names:
        label, stem = "presets/" + name, name[:-len(".json")]
        try:
            obj = _read_json(os.path.join(directory, name), label)
            problems = validate_preset(obj, stem=stem, schema=schema)
        except errors.InvalidPresetFile as exc:
            broken[stem] = exc.issue
            continue
        if problems:
            broken[stem] = _issue(
                "invalid_preset_file", "presets", label, "; ".join(problems[:3]),
                "restore the file from the package or reinstall", detail={"problems": problems})
            continue
        found[obj["id"]] = obj
    for stem, issue in broken.items():
        key = (directory, issue.param, issue.message)
        if key not in _logged_broken:  # the catalog is scanned several times per run: log each problem once
            _logged_broken.add(key)
            _log.warning("[MossPhoton] preset file %s is disabled: %s", issue.param, issue.message)
    return {pid: found[pid] for pid in ALL_PRESET_IDS if pid in found}, broken


def _load_catalog(directory):
    """Validated ``{id: preset}`` for every usable ``<id>.json`` of the directory, in ``ALL_PRESET_IDS`` order."""
    return _scan_catalog(directory)[0]


def catalog(directory=None):
    """``{id: validated preset}`` in ``PRESET_IDS`` order, deep-copied on every call.

    ``directory`` (default ``preset_dir()``) exists for tests. A preset file that fails validation is left out (see
    ``catalog_issues``) and ``resolve`` of its id raises ``invalid_preset_file`` naming the file. Raises
    ``invalid_preset_file`` for a missing/unparseable/unusable schema or an unreadable directory.
    """
    return copy.deepcopy(_load_catalog(directory))


def catalog_issues(directory=None):
    """Warnings ``invalid_preset_file`` (one per unusable preset file), in file order; empty for a healthy catalog.

    The fix text says the preset is disabled. A sampler run records them in its report so a broken or user-added file
    is seen, not silently ignored.
    """
    return [dataclasses.replace(issue, severity="warning",
                                fix="the preset %s is disabled until the file is restored from the package; "
                                    "the other presets are not affected" % stem)
            for stem, issue in _scan_catalog(directory)[1].items()]


def catalog_ids(directory=None):
    """Preset ids present in the directory, in ``ALL_PRESET_IDS`` order (``consts.PRESET_IDS`` then the turbo ones)."""
    return list(_load_catalog(directory))


def selectable_ids(directory=None):
    """Ids a node combo lists: the usable presets plus the ids whose own file is broken, in ``ALL_PRESET_IDS`` order.

    A broken id stays selectable so a saved workflow reaches the typed ``invalid_preset_file`` error naming the file
    instead of ComfyUI's "value not in list" (NEW-2); an id whose file is missing is not listed.
    """
    found, broken = _scan_catalog(directory)
    return [pid for pid in ALL_PRESET_IDS if pid in found or pid in broken]


def describe_catalog(directory=None):
    """``[{"id","status","label","group","recommended_final","summary"}]`` in catalog order (for UI lists)."""
    return [_describe(p) for p in _load_catalog(directory).values()]


def _describe(p):
    ui = p["ui"]
    return {"id": p["id"], "status": p["status"], "label": ui["label"], "group": ui["group"],
            "recommended_final": ui["recommended_final"], "summary": ui["summary"]}


def preset_sha256(preset):
    """``sha256(canonical_json(preset))``, lowercase hex."""
    return canon.sha256_hex(canon.canonical_json(preset))


def catalog_sha256(directory=None):
    """Hash over the schema and every preset of the directory (changes when any content changes; IS_CHANGED key)."""
    directory = directory or preset_dir()
    parts = {pid: preset_sha256(p) for pid, p in _load_catalog(directory).items()}
    parts[_SCHEMA_FILE] = canon.sha256_hex(canon.canonical_json(load_schema(directory)))
    return canon.sha256_hex(canon.canonical_json(parts))


def pin_of(preset):
    """``"<id>@<preset_version>#<sha8>"`` of an installed catalog entry (sha8 = first 8 hex digits of its hash)."""
    return "%s@%s#%s" % (preset["id"], preset["preset_version"], canon.short(preset_sha256(preset), 8))


def _mismatch(message, fix, node="MossPhotonSampler"):
    return errors.PresetVersionMismatch(_issue("preset_version_mismatch", node, "preset_pin", message, fix))


def parse_pin(pin, node="MossPhotonSampler"):
    """``""`` -> ``None``; ``"<id>@<MAJOR.MINOR.PATCH>[#<sha8>]"`` -> ``{"id", "version", "sha8" or None}``.

    Pure; id, version and sha8 are exact (lowercase hex), there is no prefix matching: ``reference@1.0`` and
    ``reference@1.0.0#zz`` are malformed. A malformed pin or a non-string raises ``preset_version_mismatch``.
    """
    if pin == "":
        return None
    m = _PIN_RE.fullmatch(pin) if isinstance(pin, str) else None
    if m is None:
        raise _mismatch("preset_pin %r is malformed" % (pin,),
                        "set preset_pin to <id>@<MAJOR.MINOR.PATCH>[#<sha8>], for example reference@1.0.0", node)
    return {"id": m.group(1), "version": m.group(2), "sha8": m.group(3)}


def load_manifest(model_family, preset_id):
    """Calibration manifest ``moss_photon/calibration/<family>/<preset_id>.json`` parsed, or ``None`` when absent.

    ``None`` for an unknown family (``None``), a name that is not a plain file-name component, or a missing file.
    A present but unparseable file raises ``invalid_preset_file``. ASSUMED manifest format (the design defines none):
    ``{"schema_version": 1, "overrides": {<override name>: value}}``.
    """
    if not isinstance(model_family, str) or not isinstance(preset_id, str):
        return None
    if not _FAMILY_RE.fullmatch(model_family) or not _FAMILY_RE.fullmatch(preset_id) \
            or model_family.startswith(".") or preset_id.startswith("."):
        return None
    path = os.path.join(_calibration_dir(), model_family, preset_id + ".json")
    if not os.path.isfile(path):
        return None
    return _read_json(path, "calibration/%s/%s.json" % (model_family, preset_id))


@dataclasses.dataclass(frozen=True)
class ResolvedPreset:
    """Result of ``resolve``. ``preset`` and ``effective`` are private deep copies (the catalog is never touched).

    ``effective`` has the keys ``sampling`` and ``budget`` of the preset (``sampling`` without ``override_bounds``). ``diff`` lists the changes made by the
    explicit overrides (``canon.deep_diff`` items). ``pinned`` is true when a non-empty pin matched.
    """

    preset: dict
    effective: dict
    layers_applied: tuple
    customized: bool
    diff: tuple
    warnings: tuple
    preset_sha256: str
    manifest_sha256: object
    pin: str
    pinned: bool = False

    def to_dict(self):
        """JSON-safe dict with id, version and both hashes."""
        return {
            "id": self.preset["id"], "version": self.preset["preset_version"], "pin": self.pin, "pinned": self.pinned,
            "preset_sha256": self.preset_sha256, "manifest_sha256": self.manifest_sha256,
            "layers_applied": list(self.layers_applied), "customized": self.customized,
            "diff": copy.deepcopy(list(self.diff)), "effective": copy.deepcopy(self.effective),
            "warnings": [w.to_dict() for w in self.warnings], "preset": copy.deepcopy(self.preset),
        }


def diff_settings(base, effective):
    """``canon.deep_diff`` of two ``{"sampling", "budget"}`` dicts: ``[{"path","base","effective"}]`` sorted by path."""
    return canon.deep_diff(base, effective)


def _branches_per_step(cfg):
    # ASSUMED duplicate of sigma.branches_per_step (design 1.2): presets must not import the unfinished sigma module
    # in wave 1. test_effective_view compares both once sigma.py exists.
    return 1 if math.isclose(cfg, 1.0) else 2


def _apply_settings(effective, settings, num_layers):
    """Apply override-style settings to a copy of ``effective``. Returns ``(new_effective, [(param, message)])``."""
    problems = []
    new = copy.deepcopy(effective)
    clean = {}
    for key, value in settings.items():
        if key not in OVERRIDE_KEYS:
            problems.append((str(key), "is not an overridable setting; allowed: %s" % ", ".join(OVERRIDE_KEYS)))
        elif value is not None:
            clean[key] = value
    if problems:
        return effective, problems
    trial = dict(new["sampling"])
    trial.update(clean)
    schema = load_schema()
    wrapper = {"$defs": schema["$defs"], "$ref": "#/$defs/sampling"}
    for err in schema_lite.validate(trial, wrapper):
        head = err.path.split(".")[0].split("[")[0] if err.path else "sampling"
        if head in clean or err.path == "":
            problems.append((head or "sampling", err.message))
    if problems:
        return effective, problems
    for key in clean:
        if key in _INT_KEYS:
            trial[key] = int(trial[key])
        elif key in _FLOAT_KEYS:
            trial[key] = float(trial[key])
        elif key == "slg_layers":
            trial[key] = [int(x) for x in trial[key]]
    if trial["slg_start"] > trial["slg_end"]:
        problems.append(("slg_start", "slg_start %r must be <= slg_end %r" % (trial["slg_start"], trial["slg_end"])))
    if trial["slg_mode"] == "uncond_skip" and not trial["slg_layers"]:
        problems.append(("slg_layers", "must not be empty when slg_mode is uncond_skip"))
    if num_layers is not None and any(x >= num_layers for x in trial["slg_layers"]):
        problems.append(("slg_layers", "every layer must be < num_layers %d, got %r" % (num_layers, trial["slg_layers"])))
    if problems:
        return effective, problems
    new["sampling"] = trial
    if "steps" in clean:
        new["budget"]["nominal_steps"] = trial["steps"]
    return new, []


_EPS = 1e-9
_SLG_KEYS = ("slg_mode", "slg_layers", "slg_start", "slg_end")
SLG_KEYS = _SLG_KEYS  # the sampling SLG overrides (the solver drops them on a darkroom preset)
_ADVANCED_NODE = "MossPhotonSamplerAdvanced"


def _override_fix(param, preset, allow_slg):
    """The fix text of an ``invalid_override``: names the bounds of a turbo preset and the preset-value route (m6)."""
    bounds = preset["sampling"].get("override_bounds") if preset else None
    if param in _SLG_KEYS and not allow_slg and preset and preset["lora_policy"] in _DISTILL_POLICIES:
        return ("SLG on a turbo preset is an experimental override of %s only; use that node or leave %s at the "
                "preset value" % (_ADVANCED_NODE, param))
    if bounds and param in ("steps", "cfg"):
        return ("stay inside the preset bounds: steps [%s, %s], cfg [%s, %s]; set cfg to 0 (the widget default) to use "
                "the preset value %s" % (bounds["steps"]["min"], bounds["steps"]["max"], bounds["cfg"]["min"],
                                         bounds["cfg"]["max"], preset["sampling"]["cfg"]))
    return "%s must be inside the preset bounds (see presets/schema.json); remove the override or correct the value" % param


def _override_error(problems, node, source="override", preset=None, allow_slg=False):
    param, message = problems[0]
    more = "" if len(problems) == 1 else " (+%d more)" % (len(problems) - 1)
    return errors.InvalidOverride(_issue(
        "invalid_override", node, param, "%s: %s%s" % (param, message, more), _override_fix(param, preset, allow_slg),
        detail={"problems": [{"param": p, "message": m} for p, m in problems], "source": source}))


def _given(overrides):
    return {k for k, v in overrides.items() if v is not None}


def _slg_changed(preset, base_smp, new_smp, overrides):
    """The SLG settings a caller changed on a distill preset (empty for any other preset)."""
    if preset["lora_policy"] not in _DISTILL_POLICIES:
        return []
    given = _given(overrides)
    return [key for key in _SLG_KEYS if key in given and new_smp[key] != base_smp[key]]


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _tame_overrides(preset, base_smp, overrides, num_layers, allow_slg_override, node):
    """``(overrides, [Issue])``: the overrides a run can use, each value that is outside what the engine can run moved to the
    nearest one it can, with the warning ``setting_adjusted`` (MP-84: a setting conflict degrades, it does not stop).

    ``cfg`` into ``[1, CFG_MAX]``, ``steps`` into ``[1, 1000]``, ``slg_start`` / ``slg_end`` into ``[0, 1]`` and swapped when
    start > end, ``slg_layers`` without the layers the model does not have (``num_layers`` when known), duplicates and past
    the sixteenth, ``slg_mode`` uncond_skip with no layer left becomes off, and SLG overrides on a distill preset outside the
    Advanced node are dropped. A value of the wrong TYPE is not touched: the schema refuses it (only an API caller makes one).
    Pure; ``overrides`` is not modified.
    """
    out = {k: v for k, v in overrides.items() if v is not None}
    notes = []

    def adjust(param, message, fix):
        notes.append(_issue("setting_adjusted", node, param, message, fix, detail={"param": param}))

    def clamp(name, low, high, why):
        value = out.get(name)
        if _is_number(value) and not low <= value <= high:
            used = min(max(value, low), high)
            out[name] = int(used) if name == "steps" else float(used)
            adjust(name, "%s = %g is outside %g..%g (%s): %g was used" % (name, value, low, high, why, used),
                   "set %s between %g and %g to drop this note" % (name, low, high))

    clamp("cfg", 1.0, consts.CFG_MAX, "the guidance scale of this sampler is defined from 1")
    steps = out.get("steps")
    if _is_number(steps) and float(steps) == int(steps):
        out["steps"] = int(steps)
        clamp("steps", 1, 1000, "a run needs at least one step")
    clamp("slg_start", 0.0, 1.0, "a fraction of the schedule")
    clamp("slg_end", 0.0, 1.0, "a fraction of the schedule")
    start, end = out.get("slg_start", base_smp["slg_start"]), out.get("slg_end", base_smp["slg_end"])
    if _is_number(start) and _is_number(end) and start > end and ("slg_start" in out or "slg_end" in out):
        out["slg_start"], out["slg_end"] = float(end), float(start)
        adjust("slg_start", "slg_start %g is above slg_end %g: the two were swapped" % (start, end),
               "set slg_start below slg_end to drop this note")
    layers = out.get("slg_layers")
    if isinstance(layers, (list, tuple)) and all(isinstance(x, int) and not isinstance(x, bool) for x in layers):
        keep, seen = [], set()
        for x in layers:
            if 0 <= x <= 255 and (num_layers is None or x < num_layers) and x not in seen and len(keep) < 16:
                keep.append(x)
                seen.add(x)
        if keep != list(layers):
            adjust("slg_layers", "slg_layers %s holds layers the model does not have, repeats or more than 16: %s were used"
                   % (list(layers), keep), "name each existing layer once to drop this note")
        out["slg_layers"] = keep
    if out.get("slg_mode", base_smp["slg_mode"]) == "uncond_skip" and not out.get("slg_layers", base_smp["slg_layers"]):
        out["slg_mode"] = "off"
        adjust("slg_mode", "slg_mode uncond_skip needs at least one layer and none is left: SLG is off",
               "enter the layer indices to skip, or set slg_mode off")
    if not allow_slg_override and preset["lora_policy"] in _DISTILL_POLICIES:
        for key in _SLG_KEYS:
            if key in out and out[key] != base_smp[key]:
                del out[key]
                adjust(key, "SLG is off in turbo presets and only the %s node may override it (experimental): %s is ignored"
                       % (_ADVANCED_NODE, key), "use %s, or leave %s at the preset value" % (_ADVANCED_NODE, key))
    return out, notes


def _bounds_notes(preset, new_smp, overrides, node):
    """Warnings ``override_outside_bounds`` for ``steps`` / ``cfg`` overrides outside ``sampling.override_bounds`` (a turbo
    preset): the value is used as given (MP-84; it was ``invalid_override`` before)."""
    notes = []
    given = _given(overrides)
    bounds = preset["sampling"].get("override_bounds")
    for name in ("steps", "cfg") if bounds else ():
        b = bounds[name]
        if name in given and not b["min"] - _EPS <= new_smp[name] <= b["max"] + _EPS:
            notes.append(_issue(
                "override_outside_bounds", node, name,
                "%s %s is outside the preset bounds [%s, %s] (verified [%s, %s]): it is used as given, the result is not "
                "covered by any verification" % (name, new_smp[name], b["min"], b["max"], b["verified_min"], b["verified_max"]),
                "use %s inside [%s, %s] for a verified run" % (name, b["verified_min"], b["verified_max"]),
                detail={"param": name, "value": new_smp[name], "bounds": [b["min"], b["max"]],
                        "verified": [b["verified_min"], b["verified_max"]], "source": copy.deepcopy(b["source"])}))
    return notes


def _override_notes(preset, smp, overrides, node, slg_changed=()):
    """Warnings for overrides that are legal but outside what was verified (R-T7), or inert (R-T9)."""
    notes = []
    given = _given(overrides)
    bounds = preset["sampling"].get("override_bounds")
    if bounds:
        for name in ("steps", "cfg"):
            b, value = bounds[name], smp[name]
            inside = b["min"] - _EPS <= value <= b["max"] + _EPS  # outside the bounds: override_outside_bounds says it
            if name in given and inside and not b["verified_min"] - _EPS <= value <= b["verified_max"] + _EPS:
                verified = [b["verified_min"], b["verified_max"]]
                notes.append(_issue(
                    "turbo_setting_outside_verified", node, name,
                    "%s %s is inside the preset bounds [%s, %s] but outside the verified range [%s, %s]" % (
                        name, value, b["min"], b["max"], verified[0], verified[1]),
                    "use %s inside [%s, %s] for a verified run; the run is recorded as customized" % (
                        name, verified[0], verified[1]),
                    detail={"param": name, "value": value, "verified": verified, "bounds": [b["min"], b["max"]],
                            "source": copy.deepcopy(b["source"])}))
    if slg_changed:
        notes.append(_issue(
            "turbo_slg_experimental", node, "slg_mode",
            "SLG on a turbo preset (%s changed) is experimental: no verification covers a distilled model with a "
            "skipped layer; above cfg 1.0 the run is not the native turbo graph" % ", ".join(slg_changed),
            "use it to explore only; remove the SLG overrides for the verified turbo run",
            detail={"changed": list(slg_changed)}))
    if given & {"cfg", "slg_mode"} and smp["slg_mode"] == "uncond_skip" and _branches_per_step(smp["cfg"]) == 1:
        notes.append(_issue(
            "slg_ineffective_cfg1", node, "slg_mode",
            "SLG is skipped at cfg 1.0 (there is no negative branch to skip a layer in); forwards are unchanged",
            "raise cfg above 1.0 to use SLG, or set slg_mode off"))
    return notes


def resolve(preset_id, *, overrides=None, pin="", model_family=None, lora_profile=None, manifest_loader=load_manifest,
            directory=None, num_layers=None, requires_calibration=(), node="MossPhotonSampler",
            allow_slg_override=False, _check_pin=True):
    """Resolve a preset into effective settings (design 1.3). Never mutates the catalog.

    Layers in order: (1) preset; (2) calibration manifest, only when ``calibration.manifest`` is not null: loaded with
    ``manifest_loader(model_family, preset_id)``; absent -> warning ``manifest_missing`` and the reference
    configuration stays, unless the preset has ``fallback == "error_on_incompatibility"`` or the caller passes
    ``requires_calibration`` (names of requested features that need calibration, ASSUMED because the design gives
    no parameter for it) -> ``calibration_required``; (3) LoRA profile: recorded in ``layers_applied``, no parameter
    change in v1; (4) ``overrides`` (names in ``OVERRIDE_KEYS``; ``None`` values are treated as absent): validated
    against the schema bounds, integer settings cast to ``int``; any override whose value differs from the layers
    below gives ``customized`` plus a ``diff`` (``budget.nominal_steps`` follows ``steps``). ``num_layers`` (model
    layers, optional) bounds ``slg_layers``. MP-84: a value the engine cannot run is moved to the nearest one it can with the
    warning ``setting_adjusted`` (``_tame_overrides``). Turbo presets (docs/03-turbo.md 5.3): ``steps`` / ``cfg`` outside
    ``sampling.override_bounds`` -> warning ``override_outside_bounds``, the value is used (it was ``invalid_override``,
    R-T6); inside the bounds but outside the verified range -> warning ``turbo_setting_outside_verified`` with ``detail``
    (R-T7); an SLG override that changes a distill preset is dropped with ``setting_adjusted`` unless ``allow_slg_override``
    (only MossPhotonSamplerAdvanced passes it; MP-30 M2: "Advanced" of Q6 is that node), then the warning
    ``turbo_slg_experimental``; ``slg_ineffective_cfg1`` info when SLG is requested at cfg 1 (R-T9).
    ``effective["sampling"]`` never carries ``override_bounds`` (it stays in ``preset``). A preset whose own file is unusable
    raises ``invalid_preset_file`` naming it (m22).

    ``pin``: ``""`` -> warning ``preset_unpinned``, ``pinned = False``; else exact match of id, version and (when
    given) sha8 against the installed entry; a pin that differs only in sha8 runs the
    installed entry with the warning ``preset_changed_same_version`` (``pinned`` stays true); a pin of another id or
    version, or a malformed one, runs the installed entry with ``preset_pin_mismatch`` (MP-84; ``preset_version_mismatch``
    before). Raises ``invalid_preset`` for an unknown id, ``invalid_override`` (an unknown name or a value of the wrong type:
    only an API caller makes one), ``calibration_required``. ``node`` names the node in the issues.
    """
    cat, broken = _scan_catalog(directory)
    if preset_id in broken:  # m22: the file of THIS preset is unusable; the others still resolve
        raise errors.InvalidPresetFile(broken[preset_id])
    if preset_id not in cat:
        raise errors.UnknownPreset(_issue(
            "invalid_preset", node, "preset", "unknown preset %r" % (preset_id,),
            "choose one of: %s" % ", ".join(cat)))
    preset = copy.deepcopy(cat[preset_id])
    sha = preset_sha256(preset)
    installed_pin = pin_of(preset)
    warnings = []
    pinned = False
    if _check_pin:
        try:
            parsed = parse_pin(pin, node)
            malformed = False
        except errors.PresetVersionMismatch:  # MP-84: a pin is advisory, the installed preset always runs
            parsed, malformed = None, True
        if malformed:
            warnings.append(_issue(
                "preset_pin_mismatch", node, "preset_pin",
                "preset_pin %r is malformed: the installed preset %s was used" % (pin, installed_pin),
                "set preset_pin to %s (or clear it) to record the current recipe" % installed_pin,
                detail={"workflow_pin": pin if isinstance(pin, str) else repr(pin), "installed_pin": installed_pin}))
        elif parsed is None:
            warnings.append(_issue(
                "preset_unpinned", node, "preset_pin",
                "preset_pin is empty: installed %s was used" % installed_pin,
                "set preset_pin to %s so a later package update cannot change this workflow" % installed_pin))
        else:
            bad = [name for name, ok in (
                ("id", parsed["id"] == preset["id"]), ("version", parsed["version"] == preset["preset_version"]),
                ("sha8", parsed["sha8"] is None or parsed["sha8"] == canon.short(sha, 8))) if not ok]
            if bad == ["sha8"]:  # MP-76: same id and version, other content: run the installed entry and say so
                warnings.append(_issue(
                    "preset_changed_same_version", node, "preset_pin",
                    "workflow pin %s and installed pin %s have the same version and another sha8: the installed preset "
                    "was used, results may differ, see CHANGELOG" % (pin, installed_pin),
                    "set preset_pin to the installed value %s to silence this warning" % installed_pin,
                    detail={"workflow_pin": pin, "installed_pin": installed_pin}))
            elif bad:
                warnings.append(_issue(
                    "preset_pin_mismatch", node, "preset_pin",
                    "workflow was made with %s; installed is %s (differs in %s): the installed preset was used, results "
                    "may differ, see CHANGELOG" % (pin, installed_pin, ", ".join(bad)),
                    "set preset_pin to the installed value %s to record the current recipe, or reinstall the MossPhoton "
                    "version that carries %s" % (installed_pin, pin),
                    detail={"workflow_pin": pin, "installed_pin": installed_pin, "differs_in": bad}))
            pinned = not bad or bad == ["sha8"]
    effective = {"sampling": {k: copy.deepcopy(v) for k, v in preset["sampling"].items() if k != "override_bounds"},
                 "budget": copy.deepcopy(preset["budget"])}
    layers = ["preset"]
    manifest_sha = None
    if preset["calibration"]["manifest"] is not None:
        manifest = manifest_loader(model_family, preset_id)
        if manifest is None:
            if requires_calibration or preset["fallback"] == "error_on_incompatibility":
                raise errors.CalibrationRequired(_issue(
                    "calibration_required", node, "preset",
                    "preset %s names calibration manifest %r which is not installed%s" % (
                        preset_id, preset["calibration"]["manifest"],
                        " (requested: %s)" % ", ".join(requires_calibration) if requires_calibration else ""),
                    "install the calibration manifest or choose a preset without it"))
            warnings.append(_issue(
                "manifest_missing", node, "preset",
                "preset %s names calibration manifest %r which is not installed; the reference configuration is used" % (
                    preset_id, preset["calibration"]["manifest"]),
                "install the calibration manifest or ignore this warning"))
        else:
            settings = manifest.get("overrides", {}) if isinstance(manifest, dict) else {}
            effective, problems = _apply_settings(effective, settings if isinstance(settings, dict) else {"overrides": 0},
                                                  num_layers)
            if problems:
                raise errors.InvalidPresetFile(_issue(
                    "invalid_preset_file", node, "calibration.manifest",
                    "manifest of %s is invalid: %s: %s" % (preset_id, problems[0][0], problems[0][1]),
                    "restore the manifest from the package or reinstall"))
            layers.append("manifest")
            manifest_sha = canon.sha256_hex(canon.canonical_json(manifest))
    if lora_profile is not None:
        layers.append("lora_profile")
    base = copy.deepcopy(effective)
    diff = []
    if overrides:
        overrides, adjusted = _tame_overrides(preset, base["sampling"], dict(overrides), num_layers, allow_slg_override, node)
        warnings += adjusted
        effective, problems = _apply_settings(effective, overrides, num_layers)
        if problems:
            raise _override_error(problems, node, preset=preset, allow_slg=allow_slg_override)
        layers.append("override")
        diff = diff_settings(base, effective)
        if diff:
            warnings.append(_issue(
                "customized_run", node, "overrides",
                "%d setting(s) differ from the preset: %s" % (len(diff), ", ".join(d["path"] for d in diff)),
                "none needed; the diff is recorded in the report"))
        warnings += _bounds_notes(preset, effective["sampling"], overrides, node)
        warnings += _override_notes(preset, effective["sampling"], overrides, node,
                                    _slg_changed(preset, base["sampling"], effective["sampling"], overrides))
    return ResolvedPreset(
        preset=preset, effective=effective, layers_applied=tuple(layers), customized=bool(diff), diff=tuple(diff),
        warnings=tuple(warnings), preset_sha256=sha, manifest_sha256=manifest_sha, pin=installed_pin, pinned=pinned)


def _known_under_another_role(stack, role):
    """Stack items whose sha256 is a known, usable distill LoRA of ``role`` but whose own role differs (MP-50).

    Only ``role == "distill"`` has a known table; an ``incompatible`` file is not offered for re-roling."""
    out = []
    for item in stack or ():
        row = consts.KNOWN_DISTILL_LORAS.get(item.get("sha256") or "")
        if row is not None and row["status"] != "incompatible" and row["role"] == role and item.get("role") != role:
            out.append(item)
    return out


def _effective_role(item):
    """The role a requirement rule uses for a stack item (MP-60, audit F03): ``distill`` for a file whose sha256 is in
    ``consts.KNOWN_DISTILL_LORAS`` whatever role the user set, else the user's role. Nothing is rewritten."""
    sha = item.get("sha256")
    return "distill" if isinstance(sha, str) and sha in consts.KNOWN_DISTILL_LORAS else item.get("role")


def evaluate_requirements(resolved, ctx, node="MossPhotonSampler"):
    """Issues for every ``constraints[]`` entry of the preset that fails against ``ctx``.

    ``ctx`` keys (missing ones default to the permissive value): ``lora_roles`` list, ``blocks_present`` dict,
    ``latent_is_empty`` bool (default true), ``has_noise_mask`` bool, ``reference_photos`` bool, ``lora_stack`` list of
    ``{"basename", "role", "sha256"}`` (optional; names the files in a ``requires_lora_role`` error, MP-50). Severity comes
    from the entry, the code from its ``reason``. A ``recommends_input`` entry has reason ``none`` (the schema lists
    raisable codes only); its issue carries the registered status code ``reference_photos_recommended`` (info, MP-30
    M6) and its fix points to MossPhotonDiagnostics, the node that takes reference photos. Turbo additions (docs/03-turbo.md 5.3): a ``limits_lora_roles`` entry gives the status code
    ``preset_lora_roles_exceeded`` (the constraint reason stays ``none``, ASSUMED: the schema lists raisable codes only);
    a ``distill`` role in a preset without a distill policy gives the warning ``distill_lora_non_turbo`` (an error outside
    ``advanced`` before MP-84, lead decision Q4). A ``requires_lora_role`` entry is never an error (MP-84): a role that is not
    recorded gives the warning ``lora_not_recorded`` (an info for the ``distill`` role: a merged distill checkpoint is
    assumed, unless a known distill file sits in the stack under another role), and nothing when ``ctx["lora_present"]`` (the
    MODEL carries LoRA patches or a LoRA Phase plan that no profile explains).

    MP-60 (audit F03): with ``lora_stack`` given, a file whose sha256 is a known distill LoRA is a distill for every
    rule here, whatever role the user set. It counts as no other role: it never satisfies ``requires_lora_role``
    ``character`` (nor ``distill`` unless the user also set that role, MP-50: the error names the file), it is not an
    extra role of ``limits_lora_roles``, and in a preset without a distill policy it is the distill error, naming the
    file and the role it was given. An unknown hash keeps the user's role. ``blocks_unverified`` true (the constraints
    do not describe the conditioning that arrived) skips the ``requires_block`` entries: neither passed nor failed.
    """
    lora_roles = list(ctx.get("lora_roles") or [])
    blocks = ctx.get("blocks_present") or {}
    stack = ctx.get("lora_stack")
    if stack is None:
        provided = effective = lora_roles
    else:
        effective = [_effective_role(s) for s in stack]
        provided = [s.get("role") for s in stack if _effective_role(s) == s.get("role")]
    out = []
    for c in resolved.preset["constraints"]:
        kind, severity, reason = c["kind"], c["severity"], c["reason"]
        detail = None
        if kind == "requires_lora_role":
            # MP-84: a role is never required. Ordinary loaders, a LoRA Phase plan and merged checkpoints are not recorded as
            # roles, so a role that is not recorded is a note (and none when something else carries LoRA patches).
            if c["role"] in provided or ctx.get("lora_present"):
                continue
            reason, severity = "lora_not_recorded", "warning"
            param = "identity_profile"
            if stack is None:
                present = "roles present: %s" % (lora_roles or "none")
            else:
                present = "present: %s" % (", ".join("%s:%s" % (s.get("basename"), s.get("role")) for s in stack)
                                           or "none")
            message = ("preset %s expects a LoRA with role %r and none is recorded (%s); the run goes on, which is fine "
                       "when the checkpoint already carries it" % (resolved.preset["id"], c["role"], present))
            fix = "optional: add MossPhotonLoRAProfile (role=%s), an ordinary LoRA loader or MossPhotonLoRAPhase" % c["role"]
            wrong = _known_under_another_role(stack, c["role"])
            if wrong:
                # never re-role it for the user (SPEC: no silent changes): name the file and the role it was given
                message = "preset %s expects a LoRA with role %r, but %s; %s" % (
                    resolved.preset["id"], c["role"],
                    "; ".join("%s is a known %s LoRA but is loaded with role %r" % (s["basename"], c["role"], s["role"])
                              for s in wrong), present)
                fix = "; ".join("set role=%s on the MossPhotonLoRAProfile that loads %s" % (c["role"], s["basename"])
                                for s in wrong)
                detail = {"known_distill": [{"basename": s["basename"], "role": s["role"]} for s in wrong]}
            claimed = [s for s in stack or () if c["role"] != "distill" and s.get("role") == c["role"]
                       and _effective_role(s) == "distill"]
            if claimed:
                # the fix is the part of the one-line note that is never shortened, so it carries file and role
                fix = ("; ".join("%s has role %r but is a known distill file, so it is not a %s LoRA: set role=distill "
                                 "and use turbo_8step / turbo_identity, or remove it"
                                 % (s.get("basename"), s.get("role"), c["role"]) for s in claimed)
                       + "; add a real %s LoRA with MossPhotonLoRAProfile (role=%s)" % (c["role"], c["role"]))
                detail = dict(detail or {}, known_distill_as_role=[
                    {"basename": s.get("basename"), "role": s.get("role")} for s in claimed])
            if c["role"] == "distill" and not wrong and not claimed:
                # MP-84 (owner 2026-11-02): turbo comes from the preset and a merged distill checkpoint is legitimate, so a
                # distill role nobody recorded is an info (a distill file loaded under another role stays a warning)
                severity = "info"
                message = ("preset %s expects a distill LoRA and none is recorded (%s); a merged distill checkpoint is "
                           "assumed and the run goes on" % (resolved.preset["id"], present))
                fix = ("none needed for a merged distill checkpoint; otherwise add an ordinary LoRA loader, MossPhotonLoRAPhase or "
                       "MossPhotonLoRAProfile (role=distill)")
        elif kind == "requires_block":
            if blocks.get(c["block"]) or ctx.get("blocks_unverified"):
                continue
            param = "constraints"
            message = "preset %s needs a non-empty %r block" % (resolved.preset["id"], c["block"])
            fix = "fill the %s field of MossPhotonPromptLayout and connect its constraints output" % c["block"]
        elif kind == "forbids_source_latent":
            if ctx.get("latent_is_empty", True) and not ctx.get("has_noise_mask", False):
                continue
            param = "latent"
            message = "preset %s accepts only an empty latent without a noise mask" % resolved.preset["id"]
            fix = "connect MossPhotonEmptyLatent instead of an image-derived latent"
        elif kind == "limits_lora_roles":
            extra = [r for r in effective if r not in c["roles"]]
            if not extra:
                continue
            # ASSUMED: the constraint reason is "none" (the schema lists raisable codes only); the code is fixed by the kind.
            reason = "preset_lora_roles_exceeded"
            param = "identity_profile"
            message = "preset %s is built for LoRA roles %s only; also present: %s" % (
                resolved.preset["id"], ", ".join(c["roles"]), ", ".join(extra))
            fix = "use turbo_identity for a distill plus character LoRA stack, or remove the %s LoRA" % ", ".join(extra)
        else:  # recommends_input
            if ctx.get(c["input"]):
                continue
            # The constraint reason stays "none" (the schema lists raisable codes only); the issue gets a status code.
            reason = "reference_photos_recommended"
            param = c["input"]
            message = "preset %s works best with %s; v1 runs no face check, so none was made" % (
                resolved.preset["id"], c["input"].replace("_", " "))
            fix = ("connect the reference photos to MossPhotonDiagnostics (reference_images, with "
                   "consent_reference_photos on); the sampler has no such input, and v1 only counts the photos")
        out.append(_issue(reason, node, param, message, fix, severity=severity, detail=detail))
    if "distill" in effective and resolved.preset["lora_policy"] not in _DISTILL_POLICIES:
        # R-T3, lead decision Q4: an error in the other presets, a warning in advanced.
        mislabeled = [s for s in stack or () if _effective_role(s) == "distill" and s.get("role") != "distill"]
        who = "".join(" %s is a known distill LoRA loaded with role %r." % (s.get("basename"), s.get("role"))
                      for s in mislabeled)
        how = "".join("; %s: set role=distill and use a turbo preset, or remove it" % s.get("basename")
                      for s in mislabeled)
        detail = ({"known_distill": [{"basename": s.get("basename"), "role": s.get("role")} for s in mislabeled]}
                  if mislabeled else None)
        # MP-84: a warning in every preset (it was an error outside ``advanced``): the preset decides turbo or not, the run
        # goes on with the settings the preset has
        out.append(_issue(
            "distill_lora_non_turbo", node, "identity_profile",
            "a distill LoRA is built for 8 steps at CFG 1, preset %s runs its own settings." % resolved.preset["id"] + who,
            "use turbo_8step / turbo_identity, or set steps 8 and cfg 1.0 yourself" + how, detail=detail))
    return out


_KNOB_PATHS = {
    "schedule": ("schedule",), "compose_cfg": ("compose", "cfg"), "compose_slg": ("compose", "slg"),
    "r_c": ("commit", "r_c"), "r_k": ("commit", "r_k"), "w_compose": ("commit", "w_compose"),
    "w_commit": ("commit", "w_commit"), "reexpose": ("reexpose", "enabled"), "sigma_r": ("reexpose", "sigma_r"),
    "r_f": ("finish", "r_f"), "finish_cfg": ("finish", "cfg"), "eta": ("finish", "eta"), "switch": ("finish", "switch"),
}


def has_darkroom(preset_id, directory=None):
    """True when the usable catalog entry ``preset_id`` has a ``darkroom`` block (a broken or unknown id: False; ``resolve``
    reports it)."""
    return "darkroom" in (_load_catalog(directory).get(preset_id) or {})


def darkroom_diff(base, new):
    """``[{"path","base","effective"}]`` (sorted) of the leaves of the ``darkroom`` block ``base`` whose value in ``new``
    (a dict of the same shape, may have more keys) differs; paths are ``darkroom.<group>.<name>``. ``compose.cfg`` is left
    out: it is ``sampling.cfg``, which the sampling diff already carries."""
    out = []

    def walk(b, n, prefix):
        for key, value in b.items():
            path = "%s.%s" % (prefix, key)
            if isinstance(value, dict):
                walk(value, n[key], path)
            elif n[key] != value and path != "darkroom.compose.cfg":
                out.append({"path": path, "base": value, "effective": n[key]})

    walk(base, new, "darkroom")
    return sorted(out, key=lambda d: d["path"])


def _patched_block(preset, sampling, knobs, node):
    """The ``darkroom`` block of ``preset`` with ``knobs`` (engine names) applied, checked by the schema and by the
    cross-field rules of ``_darkroom_problems``; ``invalid_override`` (issue node ``node``) otherwise."""
    block = copy.deepcopy(preset["darkroom"])
    for name, value in knobs.items():
        path = _KNOB_PATHS[name]
        target = block
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
    block["compose"]["cfg"] = sampling["cfg"]  # the compose guidance of a run is the effective sampling.cfg
    schema = load_schema()
    wrapper = {"$defs": schema["$defs"], "$ref": "#/$defs/darkroom"}
    problems = [(e.path or "darkroom", e.message) for e in schema_lite.validate(block, wrapper)]
    if not problems:
        trial = dict(preset, darkroom=block,
                     sampling=dict(preset["sampling"], cfg=sampling["cfg"], steps=sampling["steps"]))
        problems = [("darkroom", m) for m in _darkroom_problems(trial)]
    if problems:
        raise _override_error(problems, node, preset=None)
    return block


def effective_view(preset_id, overrides=None, directory=None, allow_slg_override=False, darkroom_knobs=None,
                   node="MossPhotonSampler"):
    """Payload of ``GET /mossphoton/presets/{id}/resolved`` (design 4.9). Pure: no model, no GPU, only the catalog.

    Raises ``invalid_preset`` (unknown id) and ``invalid_override`` (issue node ``node``); ``allow_slg_override`` is the
    Advanced node's reading of ``resolve`` (NEW-1). ``forwards.expected_branch_forwards`` is
    ``steps * branches_per_step(cfg) + budget.extra_forwards`` (100 for reference, 50 for draft_fast at cfg 3).

    A preset with a ``darkroom`` block (MP-71) adds ``payload["darkroom"]``, the block as the run will use it. The ``cfg``
    override of such a preset is the finish guidance (the phase where the negative acts; ``sampling.cfg`` stays the
    preset's compose guidance), so it lands in ``darkroom.finish.cfg`` unless the ``finish_cfg`` knob is given.
    ``darkroom_knobs`` (the Advanced phase knobs by engine name, ``None`` = from the preset; ``compose_cfg`` becomes the
    ``sampling.cfg`` override) are applied to the block, and a block value that differs from the preset's makes the view
    ``customized`` with a ``darkroom.*`` diff entry, like the run. The sampling SLG overrides of such a preset are dropped
    with a warning, as the run drops them. The forward count shown is the compose-guidance count (the finish steps run two
    branches: the report has the exact number).
    """
    overrides = dict(overrides) if overrides else None
    knobs = {k: v for k, v in (darkroom_knobs or {}).items() if v is not None}
    smp_overrides, extra_warnings = overrides, []
    if has_darkroom(preset_id, directory):
        smp_overrides = {k: v for k, v in (overrides or {}).items() if k != "cfg" and k not in _SLG_KEYS}
        ignored = sorted(k for k in (overrides or {}) if k in _SLG_KEYS)
        if ignored:
            extra_warnings.append(_issue(
                "darkroom_override_ignored", node, ignored[0],
                "on a darkroom preset the sampling SLG settings %s are ignored; the darkroom SLG is compose_slg"
                % ", ".join(ignored), "use compose_slg", detail={"fields": ignored, "preset": preset_id}))
        finish_cfg = (overrides or {}).get("cfg")
        if finish_cfg is not None:
            if not _is_number(finish_cfg):
                raise _override_error([("cfg", "the finish guidance must be a number in [1.0, %s], got %r"
                                        % (consts.CFG_MAX, finish_cfg))], node, preset=None)
            if not 1.0 <= finish_cfg <= consts.CFG_MAX:  # MP-84: the run moves it to the nearest guidance it can run
                used = min(max(float(finish_cfg), 1.0), consts.CFG_MAX)
                extra_warnings.append(_issue(
                    "setting_adjusted", node, "cfg",
                    "cfg = %g: on a darkroom preset the cfg widget is the finish guidance, defined from 1 to %g: %g is used"
                    % (finish_cfg, consts.CFG_MAX, used), "set cfg to 0 (the preset's value) or between 1 and %g"
                    % consts.CFG_MAX))
                finish_cfg = used
            knobs.setdefault("finish_cfg", float(finish_cfg))  # the finish_cfg knob wins over the cfg widget
        if "compose_cfg" in knobs:
            smp_overrides["cfg"] = knobs["compose_cfg"]
    elif knobs:
        names = sorted(knobs)
        extra_warnings.append(_issue(
            "darkroom_override_ignored", node, names[0],
            "preset %s has no darkroom block: the darkroom setting(s) %s are ignored" % (preset_id, ", ".join(names)),
            "choose a preset with a darkroom block", detail={"fields": names, "preset": preset_id}))
    r = resolve(preset_id, overrides=smp_overrides or None, directory=directory, allow_slg_override=allow_slg_override,
                _check_pin=False)
    smp, bud = r.effective["sampling"], r.effective["budget"]
    branches = _branches_per_step(smp["cfg"])
    p = r.preset
    payload = {
        "schema_version": consts.PRESET_SCHEMA,
        "preset": dict(_describe(p), version=p["preset_version"], pin=r.pin),
        "effective": copy.deepcopy(r.effective),
        "customized": r.customized,
        "diff": copy.deepcopy(list(r.diff)),
        "forwards": {"branches_per_step": branches,
                     "expected_branch_forwards": smp["steps"] * branches + bud["extra_forwards"],
                     "note": "exact for the steps shown; custom sigmas replace steps"},
        "requirements": copy.deepcopy(p["constraints"]),
        "guards": copy.deepcopy(p["guards"]),
        "warnings": [w.to_dict() for w in list(r.warnings) + extra_warnings],
        "not_known_before_the_run": list(_NOT_KNOWN),
    }
    if "override_bounds" in p["sampling"]:  # turbo presets only: the UI limits its steps / cfg inputs with it
        payload["override_bounds"] = copy.deepcopy(p["sampling"]["override_bounds"])
    if "darkroom" in p:
        try:
            block = _patched_block(p, smp, {k: v for k, v in knobs.items() if k != "compose_cfg"}, node)
        except errors.InvalidOverride as exc:
            # MP-84: a run moves such a knob to the nearest value it can use (``darkroom.resolve_settings``); the panel shows
            # the preset's own block and says so, it does not stop
            payload["warnings"].append(_issue(
                "setting_adjusted", node, exc.issue.param,
                "the darkroom settings do not run as set (%s): the run moves them to the nearest usable values; the panel "
                "shows the preset's block" % exc.issue.message, "correct the setting to drop this note").to_dict())
            block = _patched_block(p, smp, {}, node)
        block["compose"]["cfg"] = smp["cfg"]
        changed = darkroom_diff(p["darkroom"], block)
        if changed:
            payload["diff"] = sorted(payload["diff"] + changed, key=lambda d: d["path"])
            payload["customized"] = True
        payload["darkroom"] = block
        payload["forwards"]["note"] = ("compose guidance only: the finish phase (r >= darkroom.finish.r_f) runs two branches "
                                       "per step at cfg above 1; the report has the exact count")
    return payload
