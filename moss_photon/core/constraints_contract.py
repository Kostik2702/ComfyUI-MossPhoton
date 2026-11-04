"""The PHOTON_CONSTRAINTS contract shared by its producer (``constraints``, ``nodes/prompt``) and its consumer
(``solver``): the block names, the structural validator and the binding of the constraints to the conditioning (MP-60,
audit F02).

Why a module of its own: the sampler reads the constraints as a dict and imports neither ``constraints`` nor ``lora``
(design M4 / LD5, guarded by ``test_no_forbidden_imports``). Validating the incoming object and comparing its
conditioning fingerprint are the two things the sampler must do with it, so both live here, in a leaf both sides may
import. Imports: stdlib and ``errors`` / ``canon`` / ``consts``; no torch (tensors are only duck-typed).

The binding: ``MossPhotonPromptLayout`` encodes the prompt and stores ``conditioning`` = the fingerprints of the
positive and the negative conditioning it OUTPUT. The sampler fingerprints the positive and the negative it actually
RECEIVED. A difference means the constraints come from another branch of the graph, or the conditioning was changed
after the prompt node (concat, area, timestep range, a control or hook): then the block flags, the layout extent and
the trigger words describe something else than what is sampled, and those checks are not run.
"""
import hashlib

from .canon import canonical_json, sha256_hex
from .consts import CONSTRAINTS_SCHEMA

__all__ = [
    "BLOCK_NAMES", "COORD_SPACE", "BBOX_FORMAT", "REQUIRED_KEYS", "SKIPPED_CHECKS", "has_content", "is_int",
    "validate_constraints", "conditioning_fingerprint", "conditioning_binding", "bind_conditioning", "binding_state",
]

BLOCK_NAMES = ("identity", "wardrobe", "scene", "pose", "camera", "lighting", "forbidden_changes")
COORD_SPACE = "abs_centralized"
BBOX_FORMAT = "yxyx"

REQUIRED_KEYS = ("schema_version", "kind", "compiler_version", "private_paths", "user_text", "negative_text",
                 "compiled_positive", "fields", "blocks_present", "prompt_sha256", "layout", "conflicts",
                 "warnings", "text_encoder", "conditioning")

# The sampler checks that depend on the constraints; all of them are skipped (not passed) when the binding fails.
SKIPPED_CHECKS = ("blocks_present", "layout_extent", "trigger_words", "constraint_warnings", "text_encoder")

_HEX = frozenset("0123456789abcdef")
_MAX_DEPTH = 6


def has_content(text):
    return bool(text) and bool(text.strip())


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def validate_constraints(c):
    """Structural and consistency problems of a constraints dict as readable strings (empty list = valid).

    Checks keys and types the sampler relies on (not a full JSON Schema; the tests cross-check with jsonschema):
    version/kind, every ``private_paths`` entry exists, the 7 block names in ``fields`` and ``blocks_present``,
    ``blocks_present`` equals "has non-whitespace text", prompt hashes match the texts, layout shape, and the
    ``conditioning`` binding (``None`` = built without conditioning, else the two sha256 fingerprints).
    """
    if not isinstance(c, dict):
        return ["constraints is not an object"]
    problems = ["missing key %s" % k for k in REQUIRED_KEYS if k not in c]
    if problems:
        return problems
    if c["schema_version"] != CONSTRAINTS_SCHEMA:
        problems.append("schema_version %r, expected %d" % (c["schema_version"], CONSTRAINTS_SCHEMA))
    if c["kind"] != "photon_constraints":
        problems.append("kind %r, expected 'photon_constraints'" % (c["kind"],))
    if not is_int(c["compiler_version"]) or c["compiler_version"] < 1:
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
            want = has_content(c["fields"][name])
            if c["blocks_present"][name] is not want:
                problems.append("blocks_present.%s is %r but the field text says %r" % (name, c["blocks_present"][name], want))
    hashes = c["prompt_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != {"positive", "negative"}:
        problems.append("prompt_sha256 must have exactly the keys positive, negative")
    elif not isinstance(c["compiled_positive"], str) or not isinstance(c["negative_text"], str):
        problems.append("compiled_positive and negative_text must be strings")
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
            if layout.get("bbox_coord_space") != COORD_SPACE or layout.get("bbox_format") != BBOX_FORMAT:
                problems.append("layout must be abs_centralized / yxyx")
            bboxes = layout.get("bboxes")
            if not isinstance(bboxes, list) or layout.get("items") != len(bboxes):
                problems.append("layout.items must equal len(layout.bboxes)")
            extent = layout.get("extent")
            if not (isinstance(extent, list) and len(extent) == 4 and all(is_int(v) for v in extent)):
                problems.append("layout.extent must be 4 integers")
    binding = c["conditioning"]
    if binding is not None and (not isinstance(binding, dict) or set(binding) != {"positive_sha256", "negative_sha256"}
                                or not all(_is_sha256(v) for v in binding.values())):
        problems.append("conditioning must be null or an object with the sha256 keys positive_sha256, negative_sha256")
    return problems


# --------------------------------------------------------------------------- the conditioning fingerprint
def _describe(value, depth=0):
    """A JSON-safe, deterministic description of one value inside a conditioning entry.

    Tensors by shape, dtype and the sha256 of their float32 bytes; dicts and lists recursively; numbers and strings as
    they are; any other object (a control net, a hook group) by its type name only, because its content cannot be
    read reliably: the fingerprint then proves the same kind of object, not the same weights.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return repr(value)
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "detach"):
        flat = value.detach().cpu().float().contiguous()
        return {"tensor": [[int(d) for d in flat.shape], str(value.dtype),
                           hashlib.sha256(flat.numpy().tobytes()).hexdigest()]}
    if depth >= _MAX_DEPTH:
        return {"object": type(value).__qualname__}
    if isinstance(value, dict):
        return {str(k): _describe(v, depth + 1) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_describe(v, depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(repr(v) for v in value)
    return {"object": type(value).__qualname__}


def conditioning_fingerprint(cond):
    """sha256 of the tensors AND the metadata dicts of a CONDITIONING list ``[[tensor, {...}], ...]``.

    Unlike the sampler's report hash (embedding bytes only) this covers every key of the entry dict: pooled output,
    area, mask, strength, ``start_percent`` / ``end_percent``, hooks. Anything that is not a list gets the fingerprint
    of its type name (it can never equal a real conditioning's).
    """
    if not isinstance(cond, (list, tuple)):
        return sha256_hex(canonical_json({"not_a_conditioning": type(cond).__qualname__}))
    return sha256_hex(canonical_json([_describe(entry) for entry in cond]))


def conditioning_binding(positive, negative):
    """The ``conditioning`` object of the constraints for the two conditionings the prompt node outputs."""
    return {"positive_sha256": conditioning_fingerprint(positive), "negative_sha256": conditioning_fingerprint(negative)}


def bind_conditioning(constraints, positive, negative):
    """A copy of ``constraints`` bound to ``positive`` / ``negative`` (the input is not modified)."""
    bound = dict(constraints)
    bound["conditioning"] = conditioning_binding(positive, negative)
    return bound


def binding_state(constraints, positive, negative):
    """``None`` when the constraints describe the conditioning that arrived, else ``{"reason", "branches"}``.

    ``reason``: ``unbound`` (the constraints carry no fingerprint) or ``mismatch``; ``branches``: which of
    ``positive`` / ``negative`` differ (both for ``unbound``). Pure; ``constraints`` must have passed
    ``validate_constraints``.
    """
    binding = constraints.get("conditioning")
    if binding is None:
        return {"reason": "unbound", "branches": ["positive", "negative"]}
    got = conditioning_binding(positive, negative)
    differ = [name for name in ("positive", "negative") if binding[name + "_sha256"] != got[name + "_sha256"]]
    return {"reason": "mismatch", "branches": differ} if differ else None
