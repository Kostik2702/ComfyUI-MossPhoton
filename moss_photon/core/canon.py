"""Canonical JSON, hashing helpers and dotted deep diff (design 1.1, Wave 0).

Leaf module: stdlib only, no comfy, no torch (design 7.3).
"""
import hashlib
import json

__all__ = ["canonical_json", "sha256_hex", "short", "deep_diff"]


def canonical_json(obj):
    """Byte-exact canonical form: sorted keys, no whitespace, UTF-8 kept (not escaped), NaN/Inf rejected.

    Floats use Python's shortest round-trip ``repr`` (``0.1``, ``1.0``, ``1e-07``); ``3`` and ``3.0`` stay
    distinct, tuples serialise like lists. Raises ``ValueError`` for NaN/Inf and ``TypeError`` for a value that is
    not JSON-serialisable.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_hex(data):
    """Lowercase hex sha256 of ``bytes``, or of a ``str`` encoded as UTF-8. Other types raise ``TypeError``."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif not isinstance(data, bytes):
        raise TypeError("sha256_hex needs bytes or str, got %s" % type(data).__name__)
    return hashlib.sha256(data).hexdigest()


def short(h, n=8):
    """First ``n`` (>= 1) characters of a hash string."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("short(): n must be an int >= 1, got %r" % (n,))
    return h[:n]


def _equal(a, b):
    """Equality where bool is not a number (``True != 1``) and ``3 == 3.0``; recursive for dicts and lists."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def deep_diff(a, b, prefix=""):
    """Differences of two JSON-like values as ``[{"path", "base", "effective"}]`` sorted by path.

    Dict keys are recursed with dotted paths (``sampling.cfg``); lists and scalars are compared whole. A key
    present on one side only reports ``None`` on the other side (ASSUMED: the design does not define the absent
    marker). Inputs are not mutated.
    """
    out = []
    _diff_into(a, b, prefix, out)
    out.sort(key=lambda item: item["path"])
    return out


def _diff_into(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in set(a) | set(b):
            sub = "%s.%s" % (path, key) if path else str(key)
            if key not in a:
                if b[key] is not None:
                    out.append({"path": sub, "base": None, "effective": b[key]})
            elif key not in b:
                if a[key] is not None:
                    out.append({"path": sub, "base": a[key], "effective": None})
            else:
                _diff_into(a[key], b[key], sub, out)
    elif not _equal(a, b):
        out.append({"path": path, "base": a, "effective": b})
