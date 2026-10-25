"""LoRA profile core (design 1.6, 1.9, 2.4, 4.4; docs/03-turbo.md 5.4): grouping, coverage, stack rules, the
PHOTON_IDENTITY_PROFILE builder and the comfy-facing mapper / applier.

Imports: stdlib and ``errors`` / ``canon`` / ``consts`` / ``fingerprint`` (design 7.3). comfy and torch are imported
inside the functions that need them (R1): ``load_lora_state``, ``map_lora_patches``, ``apply_patches``. Everything else
is pure and runs on plain dicts.

Coverage is counted PER LORA FILE from the native mapper's own patch dict (B1, LD1). Nothing in this module reads
``model.patches`` to decide coverage, so a file maps the same on a clean MODEL and on one that already carries other
LoRAs; ``patch_entries_of`` exists only for the provenance arithmetic of the attachment (M1).
"""
import collections
import copy
import dataclasses
import json
import math
import os
import re
import threading

from . import canon, consts, errors, fingerprint

__all__ = [
    "NODE", "COVERAGE_OK", "COVERAGE_MIN", "WEIGHT_SUFFIXES", "SCALE_SUFFIXES", "ROLES", "DEFAULT_ROLE",
    "WEIGHT_PROFILE_KINDS", "KEY_FORMATS", "MappingResult", "group_lora_modules", "weight_modules",
    "classify_key_format", "te_prefixed_modules", "coverage_verdict", "parse_lora_metadata", "sidecar_path",
    "read_sidecar", "describe_file", "distill_entry", "distill_issues", "base_match_of", "check_stack",
    "summarize_mapping", "coverage_block", "sidecar_block", "patch_entries_of", "provenance_baseline",
    "build_attachment", "attachment_json", "build_stack_entry", "build_identity_profile", "empty_identity_profile",
    "validate_identity_profile", "profile_digest", "check_profile_in", "stack_journal", "issues_to_dicts",
    "load_lora_state", "map_lora", "map_lora_patches", "apply_patches", "StateCache", "STATE_CACHE_SLOTS",
]

NODE = consts.NODE_IDS[3]  # "MossPhotonLoRAProfile"

COVERAGE_OK = 0.95   # ratio >= this: ok (design 1.9)
COVERAGE_MIN = 0.50  # COVERAGE_MIN <= ratio < COVERAGE_OK: warning; below: error unless allow_low

WEIGHT_SUFFIXES = (".lora_A.weight", ".lora_B.weight", ".lora_A", ".lora_B", ".lora_down.weight", ".lora_up.weight",
                   ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2",
                   ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b", ".hada_t1", ".hada_t2",
                   ".diff", ".diff_b", ".set_weight", ".w_norm", ".b_norm", ".lora.up.weight", ".lora.down.weight")
SCALE_SUFFIXES = (".alpha", ".dora_scale")

# Longest suffix first, so ".lora_A.weight" wins over ".lora_A" and ".diff_b" over ".diff".
_SUFFIXES_LONGEST_FIRST = tuple(sorted(WEIGHT_SUFFIXES + SCALE_SUFFIXES, key=len, reverse=True))
_WEIGHT_SUFFIX_SET = frozenset(WEIGHT_SUFFIXES)

ROLES = ("character", "style", "wardrobe", "scene", "other", "distill")
DEFAULT_ROLE = "character"
WEIGHT_PROFILE_KINDS = ("constant",)  # MP-16 adds "phased"; the node refuses anything else
KEY_FORMATS = ("peft_split_qkv", "peft_fused_qkv", "kohya", "lokr", "unknown")

# Prefixes of text-encoder modules in LoRA files: the generic ComfyUI form and the diffusers / kohya forms.
_TE_PREFIXES = ("text_encoders.", "text_encoder.", "lora_te")
_LOKR_SUFFIXES = frozenset(s for s in WEIGHT_SUFFIXES if s.startswith(".lokr_"))
_KOHYA_SUFFIXES = frozenset((".lora_down.weight", ".lora_up.weight"))
_PEFT_SUFFIXES = frozenset((".lora_A.weight", ".lora_B.weight", ".lora_A", ".lora_B"))
_LAYER_RE = re.compile(r"^diffusion_model\.layers\.(\d+)\.")

_METADATA_KEYS = ("ss_base_model_version", "ss_sd_model_name", "software", "ss_network_dim", "modelspec.title")
# Metadata keys that name the architecture a LoRA was trained for (ai-toolkit, kohya, the MOSS trainer).
_ARCHITECTURE_KEYS = ("ss_base_model_version", "ss_architecture", "modelspec.architecture")
_MAX_UNMAPPED_EXAMPLES = 5
_STRENGTH_EPS = 1e-9


# --------------------------------------------------------------------------- grouping and format
def group_lora_modules(tensor_keys):
    """Module name -> ``frozenset`` of the suffixes present, over LoRA tensor key names.

    The longest known suffix is stripped; a key with no known suffix becomes its own module with suffix ``""``.
    """
    grouped = {}
    for key in tensor_keys:
        for suffix in _SUFFIXES_LONGEST_FIRST:
            if key.endswith(suffix):
                grouped.setdefault(key[:-len(suffix)], set()).add(suffix)
                break
        else:
            grouped.setdefault(key, set()).add("")
    return {module: frozenset(suffixes) for module, suffixes in grouped.items()}


def weight_modules(grouped):
    """Modules with at least one WEIGHT suffix. Alpha-only modules (stray ``.alpha`` scalars) are not modules."""
    return {module: suffixes for module, suffixes in grouped.items() if suffixes & _WEIGHT_SUFFIX_SET}


def classify_key_format(tensor_keys):
    """``lokr`` | ``kohya`` | ``peft_fused_qkv`` | ``peft_split_qkv`` | ``unknown`` (the weight-suffix family first).

    ASSUMED (docs/03-turbo.md 5.4 leaves the definition to MP-10): ``kohya`` is the ``lora_down`` / ``lora_up``
    suffix family or a ``lora_unet_`` key prefix, whatever the module prefix is, so the 8-step distill files
    (``diffusion_model.<module>.lora_down.weight`` with split q/k/v) are ``kohya``. ``peft_*`` is the ``lora_A`` /
    ``lora_B`` family; it is ``fused`` when a ``.qkv`` module exists. Hada files are ``unknown`` (no enum value).
    """
    keys = list(tensor_keys)
    grouped = group_lora_modules(keys)
    suffixes = set()
    for found in grouped.values():
        suffixes |= found
    if suffixes & _LOKR_SUFFIXES:
        return "lokr"
    if suffixes & _KOHYA_SUFFIXES or any(k.startswith("lora_unet_") for k in keys):
        return "kohya"
    if suffixes & _PEFT_SUFFIXES:
        return "peft_fused_qkv" if any(m.endswith(".qkv") for m in grouped) else "peft_split_qkv"
    return "unknown"


def te_prefixed_modules(modules):
    """Sorted list of the module names that belong to a text encoder (generic, diffusers and kohya prefixes)."""
    return sorted(m for m in modules if m.startswith(_TE_PREFIXES))


# --------------------------------------------------------------------------- coverage
def coverage_verdict(modules_total, modules_mapped, *, allow_low=False):
    """``{"ratio", "verdict": ok|warn|error, "code", "forced"}`` for one file (design 1.6, 1.9).

    ``modules_total <= 0`` or ``modules_mapped <= 0`` is always ``error`` / ``incompatible_lora``, also with
    ``allow_low``. ``ratio >= 0.95`` ok; ``0.50 <= ratio < 0.95`` warn ``partial_lora_coverage``; below 0.50 error
    ``incompatible_lora``, or warn ``lora_low_coverage_forced`` (``forced`` true) with ``allow_low``. ``ratio`` is
    capped at 1.0 (a module can map to more than one patch entry).
    """
    if modules_total <= 0 or modules_mapped <= 0:
        return {"ratio": 0.0, "verdict": "error", "code": "incompatible_lora", "forced": False}
    ratio = min(1.0, modules_mapped / modules_total)
    if ratio >= COVERAGE_OK:
        return {"ratio": ratio, "verdict": "ok", "code": None, "forced": False}
    if ratio >= COVERAGE_MIN:
        return {"ratio": ratio, "verdict": "warn", "code": "partial_lora_coverage", "forced": False}
    if allow_low:
        return {"ratio": ratio, "verdict": "warn", "code": "lora_low_coverage_forced", "forced": True}
    return {"ratio": ratio, "verdict": "error", "code": "incompatible_lora", "forced": False}


@dataclasses.dataclass(frozen=True)
class MappingResult:
    """What the native mapper does with ONE LoRA file (B1); never derived from ``model.patches``."""

    tensors_total: int
    modules_total: int        # weight-bearing modules of THIS file (text-encoder modules only when te_active)
    modules_mapped: int       # accepted patch entries: model entries + (te entries when te_active)
    model_entries: int        # accepted entries that land on the MODEL (what add_patches appends)
    keys_patched: int         # distinct model weight keys among them (fused q/k/v count once)
    te_entries: int           # accepted entries that land on the CLIP (0 unless te_active)
    te_modules_ignored: int   # text-encoder modules in the file while te_active is false
    te_modules_in_file: int
    te_active: bool
    layers_total: int
    layers_covered: int
    layers_missing: tuple
    unmapped_examples: tuple
    key_format: str


def _key_name(patch_key):
    return patch_key if isinstance(patch_key, str) else patch_key[0]


def summarize_mapping(tensor_keys, *, accepted_model, accepted_te, mapper_names, num_layers, te_active):
    """Pure core of ``map_lora``: build a ``MappingResult`` from key names only.

    ``accepted_model`` / ``accepted_te``: the native patch-dict keys (str or ``(key, offset, ...)`` tuples) whose
    weight key exists in the model / clip state dict (the ``add_patches`` filter). ``mapper_names``: the module
    names the native key maps know. ``num_layers``: model layers or ``None``. ``tensor_keys``: the (converted)
    LoRA keys.
    """
    tensor_keys = list(tensor_keys)
    modules = weight_modules(group_lora_modules(tensor_keys))
    te_mods = te_prefixed_modules(modules)
    te_set = set(te_mods)
    unet_mods = sorted(m for m in modules if m not in te_set)
    accepted_model = list(accepted_model)
    accepted_te = list(accepted_te) if te_active else []
    covered = sorted({int(m.group(1)) for m in (_LAYER_RE.match(_key_name(k)) for k in accepted_model) if m})
    layers_total = int(num_layers) if num_layers else 0
    covered_set = set(covered)
    missing = tuple(i for i in range(layers_total) if i not in covered_set)
    candidates = unet_mods + (te_mods if te_active else [])
    unmapped = tuple([m for m in candidates if m not in mapper_names][:_MAX_UNMAPPED_EXAMPLES])
    return MappingResult(
        tensors_total=len(tensor_keys),
        modules_total=len(unet_mods) + (len(te_mods) if te_active else 0),
        modules_mapped=len(accepted_model) + len(accepted_te),
        model_entries=len(accepted_model),
        keys_patched=len({_key_name(k) for k in accepted_model}),
        te_entries=len(accepted_te),
        te_modules_ignored=0 if te_active else len(te_mods),
        te_modules_in_file=len(te_mods),
        te_active=bool(te_active),
        layers_total=layers_total,
        layers_covered=len(covered),
        layers_missing=missing,
        unmapped_examples=unmapped,
        key_format=classify_key_format(tensor_keys),
    )


def coverage_block(mapping, verdict):
    """The ``stack[].coverage`` object (design 2.4). ``verdict`` is a ``coverage_verdict`` result that is not an error."""
    return {
        "tensors_total": mapping.tensors_total,
        "modules_total": mapping.modules_total,
        "modules_mapped": mapping.modules_mapped,
        "ratio": verdict["ratio"],
        "verdict": "error-forced" if verdict["forced"] else verdict["verdict"],
        "layers_total": mapping.layers_total,
        "layers_covered": mapping.layers_covered,
        "layers_missing": list(mapping.layers_missing),
        "te_modules_ignored": mapping.te_modules_ignored,
        "unmapped_examples": list(mapping.unmapped_examples),
    }


# --------------------------------------------------------------------------- metadata, sidecar, files
def _text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _software(value):
    """``ai-toolkit 0.13.34`` from the JSON string some trainers write; else the trimmed text."""
    text = _text(value)
    if text is None:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict) and data.get("name"):
        return ("%s %s" % (data["name"], data.get("version", ""))).strip()[:80]
    return text[:80]


def _model_name(value):
    """Basename of a model reference (a path or an ``urn:air`` string), never a full path."""
    text = _text(value)
    if text is None:
        return None
    return re.split(r"[\\/]", text)[-1][:120] or None


_WORD_BASE = re.compile(r"(?<![a-z])base(?![a-z])")
_WORD_SFT = re.compile(r"(?<![a-z])sft(?![a-z])")


def _trained_on(model_name):
    if not model_name:
        return "unknown"
    low = model_name.lower()
    has_base, has_sft = bool(_WORD_BASE.search(low)), bool(_WORD_SFT.search(low))
    if has_base == has_sft:
        return "unknown"
    return "base" if has_base else "sft"


def parse_lora_metadata(meta):
    """Whitelisted metadata plus the two derived verdicts (design 2.4, 6.5).

    Returns ``{"whitelist": {...only the five allowed keys, never a full path...}, "trained_on": base|sft|unknown,
    "architecture": moss|other|unknown, "architecture_value": str|None}``. ``trained_on`` comes from the words
    ``base`` / ``sft`` in ``ss_sd_model_name`` (both or neither: ``unknown``). ``architecture`` reads
    ``ss_base_model_version``, ``ss_architecture`` and ``modelspec.architecture``: any value containing ``moss``
    is ``moss``, otherwise the first value found is ``other``, no value is ``unknown``.
    """
    meta = meta or {}
    white = {
        "ss_base_model_version": _text(meta.get("ss_base_model_version")),
        "ss_sd_model_name": _model_name(meta.get("ss_sd_model_name")),
        "software": _software(meta.get("software")),
        "ss_network_dim": _text(meta.get("ss_network_dim")),
        "modelspec.title": _text(meta.get("modelspec.title")),
    }
    values = [v for v in (_text(meta.get(k)) for k in _ARCHITECTURE_KEYS) if v]
    if not values:
        architecture, arch_value = "unknown", None
    elif any("moss" in v.lower() for v in values):
        architecture, arch_value = "moss", next(v for v in values if "moss" in v.lower())
    else:
        architecture, arch_value = "other", values[0]
    return {
        "whitelist": {k: v for k, v in white.items() if v is not None},
        "trained_on": _trained_on(white["ss_sd_model_name"]),
        "architecture": architecture,
        "architecture_value": arch_value,
    }


def sidecar_path(lora_path):
    """``<lora stem>.photon.json`` next to the LoRA file."""
    return os.path.splitext(lora_path)[0] + ".photon.json"


_SIDECAR_KEYS = frozenset(("schema_version", "verified_strength", "dataset", "recommended_prompt", "trained_on"))


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_verified_strength(vs):
    if not isinstance(vs, dict) or set(vs) != {"min", "max", "recommended"}:
        raise ValueError("verified_strength must be an object with exactly min, max and recommended")
    if not all(_number(vs[k]) for k in vs):
        raise ValueError("verified_strength values must be finite numbers")
    if not vs["min"] <= vs["recommended"] <= vs["max"]:
        raise ValueError("verified_strength needs min <= recommended <= max")
    return {"min": float(vs["min"]), "max": float(vs["max"]), "recommended": float(vs["recommended"])}


def read_sidecar(lora_path):
    """Validated ``<stem>.photon.json`` or ``None`` when there is no such file. ASSUMED format (design leaves it open).

    Allowed keys: ``schema_version`` (1), ``verified_strength`` ``{min, max, recommended}``, ``dataset`` (object or
    null), ``recommended_prompt`` (string or null), ``trained_on`` (``base`` or ``sft``). Anything else, or a file
    that is not valid UTF-8 JSON, raises ``ValueError`` naming the problem (the node logs it and ignores the file).
    Returns ``{"found": True, "curated": True, "verified_strength", "dataset", "recommended_prompt", "trained_on"}``.
    """
    path = sidecar_path(lora_path)
    if not os.path.isfile(path):
        return None
    name = os.path.basename(path)
    try:
        with open(path, "rb") as f:
            data = json.loads(f.read().decode("utf-8-sig"))
        return _parse_sidecar(data)
    except (OSError, ValueError) as exc:
        raise ValueError("sidecar %s: %s" % (name, exc)) from None


def _parse_sidecar(data):
    if not isinstance(data, dict):
        raise ValueError("must be a JSON object")
    unknown = sorted(set(data) - _SIDECAR_KEYS)
    if unknown:
        raise ValueError("unknown keys %s" % unknown)
    if data.get("schema_version", 1) != 1:
        raise ValueError("schema_version must be 1")
    vs = data.get("verified_strength")
    dataset = data.get("dataset")
    prompt = data.get("recommended_prompt")
    trained = data.get("trained_on")
    if dataset is not None and not isinstance(dataset, dict):
        raise ValueError("dataset must be an object or null")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError("recommended_prompt must be a string or null")
    if trained is not None and trained not in ("base", "sft"):
        raise ValueError("trained_on must be base or sft")
    return {"found": True, "curated": True, "verified_strength": None if vs is None else _check_verified_strength(vs),
            "dataset": dataset, "recommended_prompt": prompt, "trained_on": trained}


def sidecar_block(sidecar, known_distill=None):
    """The ``stack[].sidecar`` object. A found sidecar wins; a known distill hash supplies ``verified_strength``."""
    if sidecar is not None:
        return {k: sidecar[k] for k in ("found", "curated", "verified_strength", "dataset", "recommended_prompt")}
    if known_distill is not None:
        s = float(known_distill["recipe"]["strength"])
        return {"found": False, "curated": True, "verified_strength": {"min": s, "max": s, "recommended": s},
                "dataset": None, "recommended_prompt": None}
    return {"found": False, "curated": False, "verified_strength": None, "dataset": None, "recommended_prompt": None}


def describe_file(path, *, mode="cached_or_compute", cache=None, interrupt_check=None):
    """``{"basename", "size", "sha256"}`` of a LoRA file; ``sha256`` is ``None`` for mode ``off`` or a cache miss in
    ``cached_only``. Never a full path. Raises ``OSError`` for a missing file."""
    info = fingerprint.file_basic_info(path)
    sha = fingerprint.sha256_file(path, cache=cache, mode=mode, interrupt_check=interrupt_check)
    return {"basename": info["basename"], "size": info["size"], "sha256": sha}


# --------------------------------------------------------------------------- distill (docs/03-turbo.md 5.4)
def distill_entry(sha256):
    """Deep copy of the ``consts.KNOWN_DISTILL_LORAS`` row for ``sha256`` or ``None``."""
    if not sha256:
        return None
    row = consts.KNOWN_DISTILL_LORAS.get(sha256)
    return copy.deepcopy(row) if row is not None else None


def distill_issues(role, sha256, basename, *, node=NODE):
    """Issues from the known-distill table for one file.

    A known ``incompatible`` hash is an ``incompatible_lora`` error in ANY role (the SFT data-space DMD file needs
    its own sampler). A ``distill``-role file whose hash is unknown (or not computed) gets the warning
    ``unverified_distill_lora``. A known ``verified_recipe`` file has no issue here; the Base-trained note
    (``lora_base_mismatch`` as info) is produced by the node from ``base_match``.
    """
    row = distill_entry(sha256)
    if row is not None and row["status"] == "incompatible":
        return [errors.make_issue(
            row.get("reason_code", "incompatible_lora"), node, "lora_name",
            "%s (%s...) is not supported: %s" % (row["name"], sha256[:8], row.get("reason", "incompatible distill LoRA")),
            "use the 8-step DMD2 LoRA (lodestones-8-steps-dmd2-lora) with the turbo presets",
            detail={"sha256": sha256, "name": row["name"]})]
    if role == "distill" and row is None:
        why = "its sha256 is unknown" if sha256 else "its sha256 was not computed (fingerprint mode)"
        return [errors.make_issue(
            "unverified_distill_lora", node, "lora_name",
            "%s is declared as a distill LoRA but %s; the 8-step recipe is not verified for it" % (basename, why),
            "use lodestones-8-steps-dmd2-lora (verified recipe) or accept an unverified result",
            detail={"sha256": sha256})]
    return []


def base_match_of(trained_on, family):
    """``True`` / ``False`` when both the LoRA's training base and the model family are known, else ``None``."""
    families = {"moss-image-sft": "sft", "moss-image-base": "base"}
    model_base = families.get(family)
    if trained_on not in ("base", "sft") or model_base is None:
        return None
    return trained_on == model_base


# --------------------------------------------------------------------------- stack rules
def _identity_key(entry):
    lora = entry["lora"]
    return lora["sha256"] or "%s:%s" % (lora["basename"], lora["size"])


def check_stack(entries, *, node=NODE):
    """Stack sanity over profile ``stack`` entries in order; each issue carries ``detail["index"]`` of the entry it is about.

    ``duplicate_lora`` (same sha256, or basename+size without one), ``multiple_character_loras`` (warning, second
    and later), a second ``distill`` role (``incompatible_lora`` error: "use one at a time", docs/03-turbo.md R-T5),
    and ``lora_strength_outside_verified_range`` for ``strength_model`` outside the entry's sidecar
    ``verified_strength``.
    """
    mk = errors.make_issue
    out, seen, characters, distills = [], {}, [], []
    for position, entry in enumerate(entries):
        idx = entry.get("index", position)
        name = entry["lora"]["basename"]
        key = _identity_key(entry)
        if key in seen:
            out.append(mk("duplicate_lora", node, "lora_name",
                          "%s is already in the stack at index %d (it is applied twice)" % (name, seen[key]),
                          "remove one of the two MossPhotonLoRAProfile nodes or use a different file",
                          detail={"index": idx, "same_as": seen[key]}))
        else:
            seen[key] = idx
        if entry["role"] == "character":
            if characters:
                out.append(mk("multiple_character_loras", node, "role",
                              "%s is a second character LoRA (first at index %d)" % (name, characters[0]),
                              "keep one character LoRA or set the other role to style / wardrobe / scene / other",
                              detail={"index": idx, "first": characters[0]}))
            characters.append(idx)
        if entry["role"] == "distill":
            if distills:
                out.append(mk("incompatible_lora", node, "role",
                              "%s is a second distill LoRA (first at index %d); distill LoRAs are used one at a time"
                              % (name, distills[0]),
                              "keep one distill LoRA in the stack",
                              detail={"index": idx, "first": distills[0]}))
            distills.append(idx)
        vs = (entry.get("sidecar") or {}).get("verified_strength")
        strength = entry["strength_model"]
        if vs and not (vs["min"] - _STRENGTH_EPS <= strength <= vs["max"] + _STRENGTH_EPS):
            out.append(mk("lora_strength_outside_verified_range", node, "strength_model",
                          "strength %g is outside the verified range [%g, %g] for %s"
                          % (strength, vs["min"], vs["max"], name),
                          "set strength_model to %g (recommended) or accept an unverified result" % vs["recommended"],
                          detail={"index": idx, "strength_model": strength, "verified": [vs["min"], vs["max"]]}))
    return out


def issues_to_dicts(issues):
    """JSON-safe dicts of ``Issue`` objects, in order."""
    return [i.to_dict() for i in issues]


def stack_journal(stack):
    """One log line with the apply order of the stack (bit-relevant, docs/03-turbo.md 4.1). No paths, no triggers."""
    parts = ["%d %s %s (%g)" % (e["index"], e["role"], e["lora"]["basename"], e["strength_model"]) for e in stack]
    line = "LoRA stack, apply order is bit-relevant: " + " -> ".join(parts)
    roles = [e["role"] for e in stack]
    if "distill" in roles and "character" in roles and roles.index("character") < roles.index("distill"):
        line += " [character before distill: not the canonical turbo order, recorded as is]"
    return line


# --------------------------------------------------------------------------- attachment and profile
def patch_entries_of(patcher):
    """``sum(len(v) for v in patcher.patches.values())`` (the unit of the attachment, design 2.4)."""
    return sum(len(v) for v in patcher.patches.values())


def provenance_baseline(prior_attachment, *, patches_uuid, patch_entries):
    """``baseline_entries`` for the attachment a chain node produces (design 4.4 step 8).

    First node (no prior attachment): every entry on the input MODEL is baseline. Later node: the prior baseline,
    plus the entries a foreign loader added between the nodes (only when the live ``patches_uuid`` differs from the
    one recorded by the previous node).
    """
    if prior_attachment is None:
        return int(patch_entries)
    baseline = int(prior_attachment["baseline_entries"])
    if str(patches_uuid) != prior_attachment["patches_uuid"]:
        baseline += max(0, int(patch_entries) - int(prior_attachment["patch_entries_total"]))
    return baseline


_ATTACH_ENTRY_KEYS = ("basename", "size", "sha256", "strength_model", "strength_clip", "te", "patch_entries")


def build_attachment(entries, *, patches_uuid, patch_entries_total, baseline_entries):
    """The profile's ``attachment`` object (design 2.4). ``entries``: dicts with the seven entry keys, in apply order."""
    out = []
    for entry in entries:
        missing = [k for k in _ATTACH_ENTRY_KEYS if k not in entry]
        if missing:
            raise ValueError("attachment entry lacks %s" % missing)
        out.append({k: entry[k] for k in _ATTACH_ENTRY_KEYS})
    return {"version": 1, "patches_uuid": str(patches_uuid), "patch_entries_total": int(patch_entries_total),
            "baseline_entries": int(baseline_entries), "entries": out}


def attachment_json(att):
    """The string stored on the MODEL under ``consts.ATTACH_LORA_STACK``: canonical JSON of the attachment."""
    return canon.canonical_json(att)


def attachment_entry_of(stack_entry):
    """The attachment entry for one stack entry."""
    lora = stack_entry["lora"]
    return {"basename": lora["basename"], "size": lora["size"], "sha256": lora["sha256"],
            "strength_model": stack_entry["strength_model"], "strength_clip": stack_entry["strength_clip"],
            "te": stack_entry["applied_to"]["text_encoder"], "patch_entries": stack_entry["patch_entries"]}


def build_stack_entry(*, index, role, lora, strength_model, strength_clip, mapping, verdict, trained_on, base_match,
                      sidecar, trigger_hints, warnings):
    """One ``stack[]`` item (design 2.4). ``lora``: ``{basename, size, sha256, metadata}``; ``warnings``: ``Issue`` list."""
    hints = [h for h in trigger_hints if h]
    return {
        "index": index,
        "role": role,
        "lora": lora,
        "strength_model": float(strength_model),
        "strength_clip": float(strength_clip),
        "keys_patched": mapping.keys_patched,
        "patch_entries": mapping.model_entries,
        "applied_to": {"transformer": mapping.model_entries > 0, "text_encoder": mapping.te_entries > 0},
        "weight_profile": {"kind": "constant", "w": float(strength_model)},
        "key_format": mapping.key_format,
        "coverage": coverage_block(mapping, verdict),
        "trained_on": trained_on,
        "base_match": base_match,
        "sidecar": sidecar,
        "trigger_present": bool(hints),
        "trigger_hints": hints,
        "warnings": issues_to_dicts(warnings),
    }


def profile_digest(profile):
    """sha256 of the canonical JSON of ``profile["stack"]`` without the private ``trigger_hints`` (identification only)."""
    stack = [{k: v for k, v in entry.items() if k != "trigger_hints"} for entry in profile["stack"]]
    return canon.sha256_hex(canon.canonical_json(stack))


def build_identity_profile(stack, *, attachment, model_identity):
    """The PHOTON_IDENTITY_PROFILE dict (design 2.4) for ``stack`` (entries in apply order) and its ``attachment``."""
    roles = [e["role"] for e in stack]
    profile = {
        "schema_version": consts.IDENTITY_PROFILE_SCHEMA,
        "kind": "photon_identity_profile",
        "stack": list(stack),
        "summary": {
            "count": len(stack),
            "roles": roles,
            "has_character": "character" in roles,
            "has_distill": "distill" in roles,
            "te_applied": any(e["applied_to"]["text_encoder"] for e in stack),
            "patch_entries_total": attachment["patch_entries_total"],
        },
        "attachment": attachment,
        "private_paths": ["stack[*].trigger_hints"],
        "model_identity": model_identity,
        "phase_plan": None,
        "digest": "0" * 64,
    }
    profile["digest"] = profile_digest(profile)
    return profile


def empty_identity_profile(model):
    """Profile of a pass-through node that starts a chain: no stack; every patch entry on ``model`` is baseline."""
    total = patch_entries_of(model)
    att = build_attachment([], patches_uuid=model.patches_uuid, patch_entries_total=total, baseline_entries=total)
    return build_identity_profile([], attachment=att, model_identity=None)


_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_identity_profile(p):
    """Structural and arithmetic problems of an identity profile as readable strings (empty list = valid).

    Mirrors ``tests/fixtures/contracts/schemas/identity_profile.schema.json`` for the fields consumers read and
    adds the cross-field rules: summary vs stack, attachment entries vs stack (same order), sequential indices,
    ``total == baseline + sum(entry patch_entries)``, digest equals ``profile_digest``.
    """
    problems = []
    if not isinstance(p, dict):
        return ["profile is not an object"]
    if p.get("kind") != "photon_identity_profile":
        problems.append("kind must be photon_identity_profile")
    if p.get("schema_version") != consts.IDENTITY_PROFILE_SCHEMA:
        problems.append("schema_version must be %d" % consts.IDENTITY_PROFILE_SCHEMA)
    for key in ("stack", "summary", "attachment", "private_paths", "model_identity", "phase_plan", "digest"):
        if key not in p:
            problems.append("missing key %s" % key)
    if problems:
        return problems
    stack, att, summary = p["stack"], p["attachment"], p["summary"]
    if not isinstance(stack, list) or not isinstance(att, dict) or not isinstance(summary, dict):
        return ["stack must be a list, attachment and summary objects"]
    if p["private_paths"] != ["stack[*].trigger_hints"]:
        problems.append("private_paths must be ['stack[*].trigger_hints']")
    if p["phase_plan"] is not None:
        problems.append("phase_plan must be null in v1")
    if not isinstance(p["digest"], str) or not _SHA_RE.match(p["digest"]):
        problems.append("digest must be a 64-hex sha256")
    elif p["digest"] != profile_digest(p):
        problems.append("digest does not match the stack")
    required_entry = ("index", "role", "lora", "strength_model", "strength_clip", "keys_patched", "patch_entries",
                      "applied_to", "weight_profile", "key_format", "coverage", "trained_on", "base_match", "sidecar",
                      "trigger_present", "trigger_hints", "warnings")
    for i, e in enumerate(stack):
        if not isinstance(e, dict):
            problems.append("stack[%d] is not an object" % i)
            continue
        lacking = [k for k in required_entry if k not in e]
        if lacking:
            problems.append("stack[%d] lacks %s" % (i, lacking))
            continue
        if e["index"] != i:
            problems.append("stack[%d].index is %r" % (i, e["index"]))
        if e["role"] not in ROLES:
            problems.append("stack[%d].role %r is not one of %s" % (i, e["role"], list(ROLES)))
        if e["key_format"] not in KEY_FORMATS:
            problems.append("stack[%d].key_format %r unknown" % (i, e["key_format"]))
        if e["weight_profile"].get("kind") not in WEIGHT_PROFILE_KINDS:
            problems.append("stack[%d].weight_profile.kind %r is not supported in v1" % (i, e["weight_profile"].get("kind")))
        if e["trained_on"] not in ("base", "sft", "unknown"):
            problems.append("stack[%d].trained_on %r unknown" % (i, e["trained_on"]))
        if e["coverage"].get("verdict") not in ("ok", "warn", "error-forced"):
            problems.append("stack[%d].coverage.verdict %r unknown" % (i, e["coverage"].get("verdict")))
        if e["coverage"].get("modules_mapped", -1) < e["patch_entries"]:
            problems.append("stack[%d].coverage.modules_mapped is below patch_entries" % i)
    roles = [e.get("role") for e in stack if isinstance(e, dict)]
    expected_summary = {"count": len(stack), "roles": roles, "has_character": "character" in roles,
                        "has_distill": "distill" in roles, "te_applied": any(
                            isinstance(e, dict) and isinstance(e.get("applied_to"), dict)
                            and e["applied_to"].get("text_encoder") for e in stack),
                        "patch_entries_total": att.get("patch_entries_total")}
    for key, value in expected_summary.items():
        if key == "has_distill" and key not in summary:  # optional in the schema: profiles of v1 predate it
            continue
        if summary.get(key) != value:
            problems.append("summary.%s is %r, expected %r" % (key, summary.get(key), value))
    for key in ("version", "patches_uuid", "patch_entries_total", "baseline_entries", "entries"):
        if key not in att:
            problems.append("attachment lacks %s" % key)
    if "entries" in att:
        entries = att["entries"]
        if len(entries) != len(stack):
            problems.append("attachment has %d entries for a stack of %d" % (len(entries), len(stack)))
        else:
            for i, (a, e) in enumerate(zip(entries, stack)):
                if isinstance(e, dict) and "lora" in e and (
                        a.get("basename") != e["lora"]["basename"] or a.get("patch_entries") != e.get("patch_entries")):
                    problems.append("attachment.entries[%d] does not describe stack[%d]" % (i, i))
            if all(isinstance(a, dict) and "patch_entries" in a for a in entries) and "baseline_entries" in att \
                    and "patch_entries_total" in att:
                got = att["baseline_entries"] + sum(a["patch_entries"] for a in entries)
                if got != att["patch_entries_total"]:
                    problems.append("attachment.patch_entries_total is %d, baseline + entries is %d"
                                    % (att["patch_entries_total"], got))
    return problems


def check_profile_in(profile_in, model_attachment, *, node=NODE):
    """Validate the chain input ``profile_in`` against the MODEL it arrives with. Raises, returns ``None``.

    ``unsupported_feature``: newer ``schema_version``, a non-null ``phase_plan`` or a ``weight_profile.kind`` other
    than ``constant`` (MP-16). ``incompatible_lora``: not an identity profile, or its attachment is not the string
    stored on the MODEL (the profile belongs to another chain, so the new profile would describe LoRAs that are not
    on this MODEL). An empty profile (pass-through start of a chain) matches any MODEL.
    """
    mk = errors.make_issue
    if not isinstance(profile_in, dict) or profile_in.get("kind") != "photon_identity_profile":
        raise errors.IncompatibleLora(mk(
            "incompatible_lora", node, "profile_in", "profile_in is not a PHOTON_IDENTITY_PROFILE",
            "connect the identity_profile output of another MossPhotonLoRAProfile"))
    version = profile_in.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version > consts.IDENTITY_PROFILE_SCHEMA:
        raise errors.UnsupportedFeature(mk(
            "unsupported_feature", node, "profile_in",
            "profile_in schema_version %r is newer than this MossPhoton knows (%d)"
            % (version, consts.IDENTITY_PROFILE_SCHEMA), "update MossPhoton or rebuild the chain"))
    if profile_in.get("phase_plan") is not None:
        raise errors.UnsupportedFeature(mk(
            "unsupported_feature", node, "profile_in",
            "profile_in carries a phase_plan; phased weights are not available in v1",
            "remove the phase plan (MP-16 adds it after v1)"))
    stack = profile_in.get("stack")
    for e in stack if isinstance(stack, list) else []:
        wp = e.get("weight_profile") if isinstance(e, dict) else None
        if isinstance(wp, dict) and wp.get("kind") not in WEIGHT_PROFILE_KINDS:
            raise errors.UnsupportedFeature(mk(
                "unsupported_feature", node, "profile_in",
                "profile_in has weight_profile.kind %r; only constant is available in v1" % (wp.get("kind"),),
                "use constant weights"))
    problems = validate_identity_profile(profile_in)
    if problems:
        raise errors.IncompatibleLora(mk(
            "incompatible_lora", node, "profile_in", "profile_in is malformed: %s" % problems[0],
            "connect the unchanged identity_profile output of another MossPhotonLoRAProfile",
            detail={"problems": problems[:5]}))
    if profile_in["attachment"]["entries"] and model_attachment != attachment_json(profile_in["attachment"]):
        raise errors.IncompatibleLora(mk(
            "incompatible_lora", node, "profile_in",
            "profile_in does not belong to this MODEL: the MODEL was not produced by that LoRA chain",
            "connect the MODEL and identity_profile outputs of the same MossPhotonLoRAProfile node"))


# --------------------------------------------------------------------------- comfy part (lazy imports, R1)
# Slots of the loaded-file cache. The stock LoraLoader keeps one file per node; here one cache serves every node, so a
# chain re-run must fit all its files: turbo_identity is distill (1.2 GB) + character (about 150 MB) = 2, and 4 also
# holds a style and a scene/wardrobe adapter on top. The worst case is four files of CPU RAM (a few GB with the
# distill LoRA), a bounded price for no re-read of a 1.2 GB file on every queue press.
STATE_CACHE_SLOTS = 4


class StateCache:
    """A small thread-safe LRU of ``key -> (state_dict, metadata)``; ``get`` loads on a miss.

    The state dict is shared and must not be modified; the metadata is returned as a fresh dict. A failing loader
    caches nothing. The loader runs outside the lock (a 1.2 GB read must not block other nodes).
    """

    def __init__(self, slots):
        if slots < 1:
            raise ValueError("a cache needs at least one slot, got %r" % (slots,))
        self._slots = slots
        self._items = collections.OrderedDict()
        self._lock = threading.Lock()

    def __len__(self):
        with self._lock:
            return len(self._items)

    def get(self, key, loader):
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
                return hit[0], dict(hit[1])
        sd, meta = loader()
        meta = dict(meta or {})
        with self._lock:
            self._items[key] = (sd, meta)
            self._items.move_to_end(key)
            while len(self._items) > self._slots:
                self._items.popitem(last=False)
        return sd, dict(meta)


_states = StateCache(STATE_CACHE_SLOTS)


def _state_key(path):
    """``(realpath, mtime_ns, size)``: a file edited in place, or the same name in another folder, is a new key."""
    st = os.stat(path)
    return (os.path.normcase(os.path.realpath(path)), int(st.st_mtime_ns), int(st.st_size))


def load_lora_state(path):
    """``(state_dict, metadata)`` of a LoRA file, loaded as the native LoraLoader does (safe load, CPU).

    The last ``STATE_CACHE_SLOTS`` files stay in memory (LRU) keyed by path + mtime + size, so re-running a chain of
    LoRA nodes does not re-read a 1.2 GB distill LoRA (m17). The state dict is shared and must not be modified;
    ``metadata`` is a fresh plain dict (empty when the file carries none). Raises whatever ComfyUI raises for an
    unreadable file; no network, no ``pickle`` (safetensors only is the caller's file choice, ``safe_load=True``).
    """
    import comfy.utils
    return _states.get(_state_key(path), lambda: comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True))


def map_lora_patches(model, lora_sd, *, clip=None):
    """``(MappingResult, patch_dict)`` of ONE LoRA file from ONE run of the native mapper (B1, LD1). Reads, never patches.

    ``key map`` = ``model_lora_keys_unet`` (+ ``model_lora_keys_clip`` when ``clip`` is given), ``convert_lora``,
    ``load_lora(..., log_missing=False)``; an entry is accepted when its weight key is in the model (or clip) state
    dict, exactly the filter of ``ModelPatcher.add_patches``. ``model`` is a ``ModelPatcher`` (only ``.model`` is
    used), so the result is the same for a clean MODEL and one that already carries LoRAs. The patch dict is what
    ``apply_patches`` applies, so the mapper is not run a second time.
    """
    import comfy.lora
    import comfy.lora_convert
    converted = comfy.lora_convert.convert_lora(lora_sd)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    patch_dict = comfy.lora.load_lora(converted, key_map, log_missing=False)
    model_keys = set(model.model.state_dict().keys())
    clip_keys = set(clip.cond_stage_model.state_dict().keys()) if clip is not None else set()
    accepted_model = [k for k in patch_dict if _key_name(k) in model_keys]
    accepted_te = [k for k in patch_dict if _key_name(k) in clip_keys]
    layers = getattr(getattr(model.model, "diffusion_model", None), "layers", None)
    mapping = summarize_mapping(
        converted.keys(), accepted_model=accepted_model, accepted_te=accepted_te, mapper_names=set(key_map),
        num_layers=len(layers) if layers is not None else None, te_active=clip is not None)
    return mapping, patch_dict


def map_lora(model, lora_sd, *, clip=None):
    """The ``MappingResult`` of ``map_lora_patches`` alone (coverage without applying)."""
    return map_lora_patches(model, lora_sd, clip=clip)[0]


def apply_patches(model, clip, patch_dict, strength_model, strength_clip, metadata):
    """Apply a mapped LoRA the way ``comfy.sd.load_lora_for_models`` does after its own mapping step.

    Returns ``(new_model, new_clip)``; both are clones, the inputs are untouched. ``clip`` is ``None`` for the
    default transformer-only path (``patch_dict`` then holds no text-encoder entry and the returned clip is ``None``).
    ``tests/integration/test_lora_apply_equals_native.py`` pins the equality with the native function.
    """
    new_model = model.clone()
    new_model.add_patches(patch_dict, strength_model)
    if metadata:
        new_model.set_attachments("lora_metadata", metadata)
    new_clip = None
    if clip is not None:
        new_clip = clip.clone()
        new_clip.add_patches(patch_dict, strength_clip)
        if metadata:
            new_clip.patcher.set_attachments("lora_metadata", metadata)
    return new_model, new_clip
