"""Prompt compiler, layout validator, conflict detector and PHOTON_CONSTRAINTS builder (design 1.4, 2.5, 4.3).

Pure module: stdlib + ``errors``, ``canon``, ``consts`` only (design 7.3). No comfy, no torch.

Decisions this module carries:

* OD1 (owner): ``forbidden_changes`` is stored in the constraints object only. It is never compiled into any prompt.
* The user text is emitted first and verbatim. Nothing is ever invented: no default adjectives, jewellery, make-up,
  objects, locations or text.
* The layout extent check against the canvas is NOT here (A23): the canvas size lives in the latent, the sampler
  checks ``layout.extent``.
* The SPEC 6.3 requirements "wardrobe_safe needs a wardrobe block, scene_safe a scene block" are carried as data
  (``blocks_present``) for the sampler to enforce; nothing is enforced here.
"""
import json
import re
from typing import NamedTuple, Optional

from .canon import sha256_hex
from .consts import COMPILER_VERSION, CONSTRAINTS_SCHEMA
from .errors import InvalidLayout, make_issue

__all__ = [
    "NODE", "BLOCK_NAMES", "COMPILED_BLOCKS", "BLOCK_LABELS", "PRIVATE_PATHS", "LAYOUT_LABELS",
    "COLOR_WORDS", "ATTRIBUTE_NOUNS", "LayoutResult", "compile_prompt", "parse_layout", "layout_prompt",
    "detect_conflicts", "conflict_issues", "build_constraints", "validate_constraints",
]

NODE = "MossPhotonPromptLayout"

BLOCK_NAMES = ("identity", "wardrobe", "scene", "pose", "camera", "lighting", "forbidden_changes")
# OD1: forbidden_changes is NOT compiled.
COMPILED_BLOCKS = ("identity", "wardrobe", "scene", "pose", "camera", "lighting")
BLOCK_LABELS = {"identity": "Subject", "wardrobe": "Clothing", "scene": "Scene", "pose": "Pose",
                "camera": "Camera", "lighting": "Lighting"}

# Keys of the constraints object that hold user prose; report.redact drops exactly these (design 2.5, R5).
PRIVATE_PATHS = ("fields", "user_text", "compiled_positive", "negative_text")

LAYOUT_LABELS = ("text", "character", "object")
_COORD_SPACE = "abs_centralized"
_BBOX_FORMAT = "yxyx"

# Conflict lexicons. Changing them changes the output of detect_conflicts, so a change bumps COMPILER_VERSION.
COLOR_WORDS = frozenset((
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "grey",
    "blonde", "blond", "auburn", "ginger", "silver", "golden", "beige", "violet", "turquoise", "burgundy",
))
_COLOR_ALIASES = {"grey": "gray", "blond": "blonde"}
# surface form -> canonical attribute name
ATTRIBUTE_NOUNS = {
    "dress": "dress", "gown": "dress", "shirt": "shirt", "blouse": "blouse", "sweater": "sweater",
    "jumper": "sweater", "hoodie": "hoodie", "jacket": "jacket", "coat": "coat", "skirt": "skirt",
    "pants": "pants", "trousers": "pants", "jeans": "jeans", "shorts": "shorts", "top": "top", "suit": "suit",
    "hat": "hat", "cap": "cap", "scarf": "scarf", "gloves": "gloves", "shoes": "shoes", "boots": "boots",
    "sneakers": "sneakers", "tie": "tie",
    "hair": "hair", "eye": "eyes", "eyes": "eyes",
}
_SOURCE_ORDER = ("text",) + COMPILED_BLOCKS
_WORD_WINDOW = 2

_TOKEN_RE = re.compile(r"[a-z]+|[^a-z\s]+")
_WORD_RE = re.compile(r"[a-z]+")


def _has_content(text):
    return bool(text) and bool(text.strip())


def compile_prompt(user_text, blocks):
    """Compile the positive prompt (design 4.3 step 1).

    ``user_text`` comes first, verbatim (no strip, no case change). When at least one of ``COMPILED_BLOCKS`` has a
    non-whitespace character, one line per non-empty block follows in the fixed order of ``COMPILED_BLOCKS``,
    ``"<Label>: <block text trimmed at both ends>"``, separated from the user text by one blank line (nothing when the
    user text is blank). ``forbidden_changes`` and unknown keys are ignored. Inner characters and line breaks of a
    block are untouched. With no blocks the result is ``user_text`` unchanged. Never raises.
    """
    lines = []
    for name in COMPILED_BLOCKS:
        text = blocks.get(name) or ""
        if _has_content(text):
            lines.append("%s: %s" % (BLOCK_LABELS[name], text.strip()))
    if not lines:
        return user_text
    sep = "\n\n" if _has_content(user_text) else ""
    return user_text + sep + "\n".join(lines)


class LayoutResult(NamedTuple):
    """Outcome of ``parse_layout``.

    ``text``: the string for the text encoder (compact layout JSON, or ``compiled_text`` unchanged when the layout
    has zero items). ``document``: the validated document in native key order, or ``None`` for zero items.
    ``layout``: the ``layout`` object of PHOTON_CONSTRAINTS (design 2.5), or ``None`` for zero items.
    """

    text: str
    document: Optional[dict]
    layout: Optional[dict]


def _layout_error(path, message):
    full = "%s: %s" % (path, message)
    return InvalidLayout(make_issue(
        "invalid_layout", NODE, "layout_json", full,
        "use abs_centralized / yxyx integer boxes; put the prose either in 'positive' or in summary.caption, "
        "not both", detail={"path": path}))


def _reject_constant(token):
    raise ValueError("%s is not valid JSON" % token)


def _check_keys(obj, allowed, path):
    for key in obj:
        if key not in allowed:
            raise _layout_error(("%s.%s" % (path, key)) if path else str(key),
                                "unknown key (allowed: %s)" % ", ".join(allowed))


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_item(item, index):
    path = "items[%d]" % index
    if not isinstance(item, dict):
        raise _layout_error(path, "expected an object, got %s" % type(item).__name__)
    _check_keys(item, ("label", "caption", "text", "palette", "bbox"), path)
    label = item.get("label")
    if label not in LAYOUT_LABELS:
        raise _layout_error(path + ".label", "expected one of %s, got %r" % ("|".join(LAYOUT_LABELS), label))
    caption = item.get("caption")
    if not isinstance(caption, str) or not caption:
        raise _layout_error(path + ".caption", "expected a non-empty string, got %r" % (caption,))
    out = {"label": label, "caption": caption}
    if label == "text":
        if "text" not in item:
            raise _layout_error(path + ".text", "required for label 'text'")
        if not isinstance(item["text"], str):
            raise _layout_error(path + ".text", "expected a string, got %s" % type(item["text"]).__name__)
        out["text"] = item["text"]
    elif "text" in item:
        raise _layout_error(path + ".text", "only allowed for label 'text', this item is %r" % label)
    if "palette" in item:
        palette = item["palette"]
        if not isinstance(palette, list):
            raise _layout_error(path + ".palette", "expected a list of strings, got %s" % type(palette).__name__)
        for j, color in enumerate(palette):
            if not isinstance(color, str):
                raise _layout_error("%s.palette[%d]" % (path, j), "expected a string, got %r" % (color,))
        if palette:  # ASSUMED: an empty palette is dropped like the native node does (it tests truthiness)
            out["palette"] = palette
    bbox = item.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise _layout_error(path + ".bbox", "expected 4 integers [y0, x0, y1, x1], got %r" % (bbox,))
    for j, value in enumerate(bbox):
        if not _is_int(value):
            raise _layout_error("%s.bbox[%d]" % (path, j), "expected an integer, got %r" % (value,))
    y0, x0, y1, x1 = bbox
    if y0 >= y1:
        raise _layout_error(path + ".bbox[2]", "y1 (%d) must be greater than y0 (%d)" % (y1, y0))
    if x0 >= x1:
        raise _layout_error(path + ".bbox[3]", "x1 (%d) must be greater than x0 (%d)" % (x1, x0))
    out["bbox"] = list(bbox)
    return out


def layout_prompt(document):
    """The string the text encoder sees for a layout document: compact JSON, same separators as the native node
    (``json.dumps(layout, ensure_ascii=False, separators=(",", ":"))``, ``nodes_moss_image.py:46``)."""
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


def parse_layout(layout_json, *, compiled_text):
    """Validate ``layout_json`` against the native MOSS layout schema (abs_centralized, yxyx) and build the prompt.

    ``compiled_text``: the compiled positive prompt; its ``.strip()`` is the layout caption (like the native node,
    A24). Rules: design 4.3 step 3. Raises ``InvalidLayout`` (code ``invalid_layout``) naming the first violating
    field in the message and in ``issue.detail["path"]`` (``$`` = the whole text, ``items[2].bbox[1]`` ...).
    Zero items -> ``LayoutResult(compiled_text, None, None)`` (the native node returns the prompt unchanged).
    Coordinates are NOT checked against a canvas here (MP-9, A23).
    """
    try:
        data = json.loads(layout_json, parse_constant=_reject_constant)
    except ValueError as exc:  # JSONDecodeError is a ValueError
        raise _layout_error("$", "not valid JSON (%s)" % exc)
    if not isinstance(data, dict):
        raise _layout_error("$", "expected a JSON object, got %s" % type(data).__name__)
    _check_keys(data, ("summary", "items", "bbox_coord_space", "bbox_format"), "")

    for key, expected in (("bbox_coord_space", _COORD_SPACE), ("bbox_format", _BBOX_FORMAT)):
        if key in data and data[key] != expected:
            raise _layout_error(key, "expected %r, got %r" % (expected, data[key]))

    items = data.get("items")
    if not isinstance(items, list):
        raise _layout_error("items", "required, expected a list, got %s" % type(items).__name__)

    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        raise _layout_error("summary", "expected an object, got %s" % type(summary).__name__)
    _check_keys(summary, ("caption", "meta", "canvas"), "summary")
    caption = summary.get("caption", "")
    if not isinstance(caption, str):
        raise _layout_error("summary.caption", "expected a string, got %s" % type(caption).__name__)
    if "meta" in summary and not isinstance(summary["meta"], list):
        raise _layout_error("summary.meta", "expected a list, got %s" % type(summary["meta"]).__name__)
    if "canvas" in summary and not isinstance(summary["canvas"], str):
        raise _layout_error("summary.canvas", "expected a string, got %s" % type(summary["canvas"]).__name__)

    parsed = [_parse_item(item, i) for i, item in enumerate(items)]
    if not parsed:
        return LayoutResult(compiled_text, None, None)

    wanted = compiled_text.strip()
    if caption and wanted and caption != wanted:
        raise _layout_error("summary.caption", "the caption is given twice and differs from the 'positive' text; "
                                               "leave summary.caption empty or equal to it")
    out_summary = {"caption": caption or wanted}
    for key in ("meta", "canvas"):
        if key in summary:
            out_summary[key] = summary[key]

    document = {"summary": out_summary, "items": parsed, "bbox_coord_space": _COORD_SPACE, "bbox_format": _BBOX_FORMAT}
    bboxes = [item["bbox"] for item in parsed]
    layout = {
        "items": len(parsed),
        "bbox_coord_space": _COORD_SPACE,
        "bbox_format": _BBOX_FORMAT,
        "bboxes": bboxes,
        "extent": [min(b[0] for b in bboxes), min(b[1] for b in bboxes),
                   max(b[2] for b in bboxes), max(b[3] for b in bboxes)],
        "experimental": True,
    }
    return LayoutResult(layout_prompt(document), document, layout)


def _source_key(field):
    return (_SOURCE_ORDER.index(field), "") if field in _SOURCE_ORDER else (len(_SOURCE_ORDER), field)


def _colors_by_attribute(text):
    """``{attribute: set(colors)}`` for one source: colour words within 2 words before an attribute noun, never
    crossing punctuation or another attribute noun."""
    tokens = _TOKEN_RE.findall(text.lower())
    found = {}
    for i, token in enumerate(tokens):
        attribute = ATTRIBUTE_NOUNS.get(token)
        if attribute is None:
            continue
        seen = 0
        j = i - 1
        while j >= 0 and seen < _WORD_WINDOW:
            prev = tokens[j]
            if not _WORD_RE.fullmatch(prev) or prev in ATTRIBUTE_NOUNS:
                break
            if prev in COLOR_WORDS:
                found.setdefault(attribute, set()).add(_COLOR_ALIASES.get(prev, prev))
            seen += 1
            j -= 1
    return found


def detect_conflicts(sources):
    """Conflicting colours of the same garment/hair/eye noun between different sources (design 4.3 step 2).

    ``sources``: ``{field: text}``, normally ``text`` plus each non-empty compiled block (the caller leaves
    ``forbidden_changes`` out). Heuristic and deterministic: lowercase ASCII words; a colour word within the 2 words
    before an attribute noun forms a (colour, noun) pair; two different sources that give the same noun disjoint
    colour sets are one conflict. ``red dress`` vs ``blue dress`` -> 1 conflict, ``red dress`` vs ``a red dress`` ->
    none. Returns ``[{attribute, kind, sources: [{field, value}], message}]`` sorted by attribute; sources are in the
    fixed field order, independent of dict order. Never edits text, never raises.
    """
    per_source = {}
    for field in sorted(sources, key=_source_key):
        text = sources[field]
        if isinstance(text, str) and text:
            per_source[field] = _colors_by_attribute(text)
    attributes = sorted({a for found in per_source.values() for a in found})
    conflicts = []
    for attribute in attributes:
        holders = [(field, found[attribute]) for field, found in per_source.items() if attribute in found]
        if any(a[1].isdisjoint(b[1]) for k, a in enumerate(holders) for b in holders[k + 1:]):
            values = [{"field": field, "value": ", ".join(sorted(colors))} for field, colors in holders]
            conflicts.append({
                "attribute": attribute,
                "kind": "color_mismatch",
                "sources": values,
                "message": "%s colour differs: %s" % (
                    attribute, "; ".join("%s says %s" % (v["field"], v["value"]) for v in values)),
            })
    return conflicts


def conflict_issues(conflicts):
    """One ``attribute_conflict`` warning (Issue dict) per conflict, in conflict order.

    The warning reaches the report, the PNG chunk and the sidecar, so it names the attribute and the FIELDS only;
    the colour words stay in ``conflict["message"]`` / ``sources[].value`` of the private constraints (owner
    2026-10-25, MP-30 m10).
    """
    out = []
    for conflict in conflicts:
        fields = [s["field"] for s in conflict["sources"]]
        issue = make_issue(
            "attribute_conflict", NODE, "positive",
            "free text and structured fields disagree: %s colour conflict between %s" % (
                conflict["attribute"], " and ".join(fields)),
            "make the text and the structured fields agree, or remove the attribute from one of them",
            detail={"attribute": conflict["attribute"], "sources": fields})
        out.append(issue.to_dict())
    return out


def _flags(blocks):
    return {name: _has_content(blocks.get(name) or "") for name in BLOCK_NAMES}


def build_constraints(user_text, negative_text, blocks, layout, conflicts, text_encoder=None):
    """Build the PHOTON_CONSTRAINTS dict (design 2.5; validates against ``tests/fixtures/contracts/schemas``).

    ``layout``: a ``LayoutResult`` or ``None``. ``compiled_positive`` is the exact text sent to the encoder: the
    layout JSON when ``layout`` has items, else ``compile_prompt(user_text, blocks)``. ``blocks_present[name]`` is
    true iff the block has a non-whitespace character (the data behind the SPEC 6.3 empty-wardrobe/scene errors,
    which the sampler raises). ``fields`` keeps the 7 blocks as typed. ``warnings`` carry one ``attribute_conflict``
    per conflict and ``layout_experimental`` when a layout is used. ``text_encoder``: dict or ``None``, stored as is.
    """
    fields = {name: (blocks.get(name) or "") for name in BLOCK_NAMES}
    compiled = compile_prompt(user_text, blocks)
    if layout is not None and layout.document is not None:
        positive, layout_info = layout.text, layout.layout
    else:
        positive, layout_info = compiled, None
    warnings = conflict_issues(conflicts)
    if layout_info is not None:
        warnings.append(make_issue(
            "layout_experimental", NODE, "layout_json",
            "layout prompting is marked experimental by the model vendor",
            "compare a run with and without the layout before relying on it").to_dict())
    return {
        "schema_version": CONSTRAINTS_SCHEMA,
        "kind": "photon_constraints",
        "compiler_version": COMPILER_VERSION,
        "private_paths": list(PRIVATE_PATHS),
        "user_text": user_text,
        "negative_text": negative_text,
        "compiled_positive": positive,
        "fields": fields,
        "blocks_present": _flags(fields),
        "prompt_sha256": {"positive": sha256_hex(positive), "negative": sha256_hex(negative_text)},
        "layout": layout_info,
        "conflicts": [dict(c) for c in conflicts],
        "warnings": warnings,
        "text_encoder": text_encoder,
    }


_REQUIRED_KEYS = ("schema_version", "kind", "compiler_version", "private_paths", "user_text", "negative_text",
                  "compiled_positive", "fields", "blocks_present", "prompt_sha256", "layout", "conflicts",
                  "warnings", "text_encoder")


def validate_constraints(c):
    """Structural and consistency problems of a constraints dict as readable strings (empty list = valid).

    Checks keys and types the sampler relies on (not a full JSON Schema; the tests cross-check with jsonschema):
    version/kind, every ``private_paths`` entry exists, the 7 block names in ``fields`` and ``blocks_present``,
    ``blocks_present`` equals "has non-whitespace text", prompt hashes match the texts, layout shape.
    """
    if not isinstance(c, dict):
        return ["constraints is not an object"]
    problems = ["missing key %s" % k for k in _REQUIRED_KEYS if k not in c]
    if problems:
        return problems
    if c["schema_version"] != CONSTRAINTS_SCHEMA:
        problems.append("schema_version %r, expected %d" % (c["schema_version"], CONSTRAINTS_SCHEMA))
    if c["kind"] != "photon_constraints":
        problems.append("kind %r, expected 'photon_constraints'" % (c["kind"],))
    if not _is_int(c["compiler_version"]) or c["compiler_version"] < 1:
        problems.append("compiler_version must be an integer >= 1")
    if not isinstance(c["private_paths"], list) or not c["private_paths"]:
        problems.append("private_paths must be a non-empty list")
    else:
        problems += ["private path %r is not a key of the object" % p for p in c["private_paths"] if p not in c]
    for key in ("fields", "blocks_present"):
        if not isinstance(c[key], dict) or set(c[key]) != set(BLOCK_NAMES):
            problems.append("%s must have exactly the keys %s" % (key, ", ".join(BLOCK_NAMES)))
    if not problems:
        for name in BLOCK_NAMES:
            want = _has_content(c["fields"][name])
            if c["blocks_present"][name] is not want:
                problems.append("blocks_present.%s is %r but the field text says %r" % (name, c["blocks_present"][name], want))
    hashes = c["prompt_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != {"positive", "negative"}:
        problems.append("prompt_sha256 must have exactly the keys positive, negative")
    else:
        if hashes["positive"] != sha256_hex(c["compiled_positive"]):
            problems.append("prompt_sha256.positive does not match compiled_positive")
        if hashes["negative"] != sha256_hex(c["negative_text"]):
            problems.append("prompt_sha256.negative does not match negative_text")
    layout = c["layout"]
    if layout is not None:
        if not isinstance(layout, dict):
            problems.append("layout must be null or an object")
        else:
            if layout.get("bbox_coord_space") != _COORD_SPACE or layout.get("bbox_format") != _BBOX_FORMAT:
                problems.append("layout must be abs_centralized / yxyx")
            bboxes = layout.get("bboxes")
            if not isinstance(bboxes, list) or layout.get("items") != len(bboxes):
                problems.append("layout.items must equal len(layout.bboxes)")
            extent = layout.get("extent")
            if not (isinstance(extent, list) and len(extent) == 4 and all(_is_int(v) for v in extent)):
                problems.append("layout.extent must be 4 integers")
    return problems
