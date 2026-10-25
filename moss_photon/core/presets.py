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
    "resolve", "diff_settings", "evaluate_requirements", "effective_view", "TURBO_PRESET_IDS", "ALL_PRESET_IDS",
]

_log = logging.getLogger("MossPhoton")
_logged_broken = set()

# Turbo presets (docs/03-turbo.md 5). ``consts.PRESET_IDS`` stays the eight of SPEC 6.1; the catalog order is those eight
# followed by the turbo ones.
TURBO_PRESET_IDS = ("turbo_8step", "turbo_identity")
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
    has a ``sources`` entry and every ``sources`` key names such a leaf; ``stable`` implies ``calibration.state ==
    official``; ``experimental`` / ``preview_only`` imply ``ui.recommended_final == false``.
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
    leaves = set(_numeric_leaves({"sampling": smp, "budget": bud}))
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
    return problems + _turbo_problems(obj)


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


def _bounds_problems(preset, base_smp, new_smp, overrides, allow_slg_override):
    """``[(param, message)]`` for overrides outside ``sampling.override_bounds`` and, unless ``allow_slg_override``, for
    SLG overrides on a distill preset."""
    problems = []
    given = _given(overrides)
    bounds = preset["sampling"].get("override_bounds")
    if bounds:
        for name in ("steps", "cfg"):
            b = bounds[name]
            if name in given and not b["min"] - _EPS <= new_smp[name] <= b["max"] + _EPS:
                problems.append((name, "%s must be in [%s, %s], got %s" % (name, b["min"], b["max"], new_smp[name])))
    if not allow_slg_override:
        for key in _slg_changed(preset, base_smp, new_smp, overrides):
            problems.append((key, "SLG is off in turbo presets; override %s only in the %s node (experimental)"
                             % (key, _ADVANCED_NODE)))
    return problems


def _override_notes(preset, smp, overrides, node, slg_changed=()):
    """Warnings for overrides that are legal but outside what was verified (R-T7), or inert (R-T9)."""
    notes = []
    given = _given(overrides)
    bounds = preset["sampling"].get("override_bounds")
    if bounds:
        for name in ("steps", "cfg"):
            b, value = bounds[name], smp[name]
            if name in given and not b["verified_min"] - _EPS <= value <= b["verified_max"] + _EPS:
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
    layers, optional) bounds ``slg_layers``. Turbo presets (docs/03-turbo.md 5.3): ``steps`` / ``cfg`` outside
    ``sampling.override_bounds`` -> ``invalid_override`` (R-T6); inside the bounds but outside the verified range ->
    warning ``turbo_setting_outside_verified`` with ``detail`` (R-T7); an SLG override that changes a distill preset ->
    ``invalid_override`` (owner Q6), unless ``allow_slg_override`` (only MossPhotonSamplerAdvanced passes it; MP-30 M2:
    "Advanced" of Q6 is that node), then the warning ``turbo_slg_experimental``; ``slg_ineffective_cfg1`` info when
    SLG is requested at cfg 1 (R-T9). ``effective["sampling"]`` never carries ``override_bounds`` (it stays in
    ``preset``). A preset whose own file is unusable raises ``invalid_preset_file`` naming it (m22).

    ``pin``: ``""`` -> warning ``preset_unpinned``, ``pinned = False``; else exact match of id, version and (when
    given) sha8 against the installed entry or ``preset_version_mismatch``. Raises ``invalid_preset`` for an unknown
    id, ``invalid_override``, ``calibration_required``. ``node`` names the node in the issues.
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
        parsed = parse_pin(pin, node)
        if parsed is None:
            warnings.append(_issue(
                "preset_unpinned", node, "preset_pin",
                "preset_pin is empty: installed %s was used" % installed_pin,
                "set preset_pin to %s so a later package update cannot change this workflow" % installed_pin))
        else:
            bad = [name for name, ok in (
                ("id", parsed["id"] == preset["id"]), ("version", parsed["version"] == preset["preset_version"]),
                ("sha8", parsed["sha8"] is None or parsed["sha8"] == canon.short(sha, 8))) if not ok]
            if bad:
                raise _mismatch(
                    "workflow was made with %s; installed is %s (differs in %s)" % (pin, installed_pin, ", ".join(bad)),
                    "set preset_pin to the installed value %s to accept the change (results may differ, see CHANGELOG), "
                    "or reinstall the MossPhoton version that carries %s" % (installed_pin, pin), node)
            pinned = True
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
        effective, problems = _apply_settings(effective, dict(overrides), num_layers)
        if not problems:
            problems = _bounds_problems(preset, base["sampling"], effective["sampling"], overrides, allow_slg_override)
        if problems:
            raise _override_error(problems, node, preset=preset, allow_slg=allow_slg_override)
        layers.append("override")
        diff = diff_settings(base, effective)
        if diff:
            warnings.append(_issue(
                "customized_run", node, "overrides",
                "%d setting(s) differ from the preset: %s" % (len(diff), ", ".join(d["path"] for d in diff)),
                "none needed; the diff is recorded in the report"))
        warnings += _override_notes(preset, effective["sampling"], overrides, node,
                                    _slg_changed(preset, base["sampling"], effective["sampling"], overrides))
    return ResolvedPreset(
        preset=preset, effective=effective, layers_applied=tuple(layers), customized=bool(diff), diff=tuple(diff),
        warnings=tuple(warnings), preset_sha256=sha, manifest_sha256=manifest_sha, pin=installed_pin, pinned=pinned)


def evaluate_requirements(resolved, ctx, node="MossPhotonSampler"):
    """Issues for every ``constraints[]`` entry of the preset that fails against ``ctx``.

    ``ctx`` keys (missing ones default to the permissive value): ``lora_roles`` list, ``blocks_present`` dict,
    ``latent_is_empty`` bool (default true), ``has_noise_mask`` bool, ``reference_photos`` bool. Severity comes
    from the entry, the code from its ``reason``. A ``recommends_input`` entry has reason ``none`` (the schema lists
    raisable codes only); its issue carries the registered status code ``reference_photos_recommended`` (info, MP-30
    M6) and its fix points to MossPhotonDiagnostics, the node that takes reference photos. Turbo additions (docs/03-turbo.md 5.3): a ``limits_lora_roles`` entry gives the status code
    ``preset_lora_roles_exceeded`` (the constraint reason stays ``none``, ASSUMED: the schema lists raisable codes only);
    a ``distill`` role in a preset without a distill policy gives ``incompatible_lora`` (error), in ``advanced`` the
    warning ``distill_lora_non_turbo`` (lead decision Q4).
    """
    lora_roles = list(ctx.get("lora_roles") or [])
    blocks = ctx.get("blocks_present") or {}
    out = []
    for c in resolved.preset["constraints"]:
        kind, severity, reason = c["kind"], c["severity"], c["reason"]
        if kind == "requires_lora_role":
            if c["role"] in lora_roles:
                continue
            param = "identity_profile"
            message = "preset %s needs a LoRA with role %r; roles present: %s" % (
                resolved.preset["id"], c["role"], lora_roles or "none")
            fix = "add MossPhotonLoRAProfile (role=%s) and connect its identity_profile output" % c["role"]
        elif kind == "requires_block":
            if blocks.get(c["block"]):
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
            extra = [r for r in lora_roles if r not in c["roles"]]
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
        out.append(_issue(reason, node, param, message, fix, severity=severity))
    if "distill" in lora_roles and resolved.preset["lora_policy"] not in _DISTILL_POLICIES:
        # R-T3, lead decision Q4: an error in the other presets, a warning in advanced.
        if resolved.preset["id"] == "advanced":
            out.append(_issue(
                "distill_lora_non_turbo", node, "identity_profile",
                "a distill LoRA is built for 8 steps at CFG 1, preset advanced runs its own settings",
                "use turbo_8step / turbo_identity, or set steps 8 and cfg 1.0 yourself"))
        else:
            out.append(_issue(
                "incompatible_lora", node, "identity_profile",
                "preset %s cannot run a distill LoRA (built for 8 steps at CFG 1)" % resolved.preset["id"],
                "use turbo_8step / turbo_identity or remove the distill LoRA"))
    return out


def effective_view(preset_id, overrides=None, directory=None, allow_slg_override=False):
    """Payload of ``GET /mossphoton/presets/{id}/resolved`` (design 4.9). Pure: no model, no GPU, only the catalog.

    Raises ``invalid_preset`` (unknown id) and ``invalid_override``; ``allow_slg_override`` is the Advanced node's
    reading of ``resolve`` (NEW-1). ``forwards.expected_branch_forwards`` is
    ``steps * branches_per_step(cfg) + budget.extra_forwards`` (100 for reference, 50 for draft_fast at cfg 3).
    """
    r = resolve(preset_id, overrides=overrides, directory=directory, allow_slg_override=allow_slg_override,
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
        "warnings": [w.to_dict() for w in r.warnings],
        "not_known_before_the_run": list(_NOT_KNOWN),
    }
    if "override_bounds" in p["sampling"]:  # turbo presets only: the UI limits its steps / cfg inputs with it
        payload["override_bounds"] = copy.deepcopy(p["sampling"]["override_bounds"])
    return payload
