"""Pure-stdlib JSON Schema subset validator (design 1.3, amendment).

Leaf module: no imports from the package (design 7.3), no comfy, no torch.

It implements exactly the keyword subset the preset schema uses, with the semantics of ``jsonschema`` Draft 2020-12
on those keywords (a unit test cross-checks both on the eight presets and a mutation corpus):

* ``type`` is a name or a list of names; ``bool`` is never an ``integer`` or ``number``; a float with an integral
  value (``50.0``) is an ``integer``, ``50.5`` is not.
* ``const`` / ``enum`` / ``uniqueItems`` compare with ``bool`` distinct from numbers (``0 != False``) and ``1 == 1.0``.
* ``$ref`` is local ``#/$defs/<name>`` only.
* Keys starting with ``x-`` are annotations and ignored. Any other keyword outside ``SUPPORTED_KEYWORDS`` raises
  ``UnsupportedSchemaKeyword`` when the ``Validator`` is constructed: it is never silently ignored.
"""
import re
from typing import NamedTuple

__all__ = ["SUPPORTED_KEYWORDS", "SchemaError", "SchemaDefinitionError", "UnsupportedSchemaKeyword", "Validator",
           "validate"]

SUPPORTED_KEYWORDS = frozenset((
    "$schema", "$id", "$defs", "$ref", "title", "description", "type", "enum", "const", "properties", "required",
    "additionalProperties", "items", "minItems", "maxItems", "uniqueItems", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "pattern",
))

_TYPE_NAMES = frozenset(("object", "array", "string", "integer", "number", "boolean", "null"))
_REF_PREFIX = "#/$defs/"


class SchemaError(NamedTuple):
    """One violation. ``path`` is dotted/indexed (``sampling.steps``, ``constraints[0].kind``; ``""`` = the root).

    For ``required`` and ``additionalProperties`` the path is the object that has the problem (as ``jsonschema``
    reports it) and the message names the keys.
    """

    path: str
    keyword: str
    message: str


class SchemaDefinitionError(ValueError):
    """The schema itself is malformed (raised at ``Validator`` construction)."""


class UnsupportedSchemaKeyword(SchemaDefinitionError):
    """The schema uses a keyword (or a ``$ref`` form) outside the supported subset."""

    def __init__(self, keyword, path=""):
        super().__init__("unsupported schema keyword %r at %s" % (keyword, path or "(root)"))
        self.keyword = keyword
        self.path = path


def _is_bool(v):
    return isinstance(v, bool)


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_type(value, name):
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return _is_bool(value)
    if name == "null":
        return value is None
    if name == "number":
        return _is_number(value)
    if name == "integer":
        return isinstance(value, int) and not _is_bool(value) or isinstance(value, float) and value.is_integer()
    return False


def _equal(a, b):
    """JSON equality: bool is not a number, ``1 == 1.0``, recursive for lists and dicts."""
    if _is_bool(a) or _is_bool(b):
        return _is_bool(a) and _is_bool(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _join(path, name):
    return "%s.%s" % (path, name) if path else str(name)


def _show(value):
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


class Validator:
    """Validate instances against one schema. Construction checks the schema keywords (raises, never ignores)."""

    def __init__(self, schema):
        if not isinstance(schema, dict):
            raise SchemaDefinitionError("schema must be an object, got %s" % type(schema).__name__)
        self._root = schema
        self._patterns = {}
        defs = schema.get("$defs", {})
        if not isinstance(defs, dict):
            raise SchemaDefinitionError("$defs must be an object")
        self._check_schema(schema, "")

    # ------------------------------------------------------------------ schema check
    def _check_schema(self, node, path):
        if _is_bool(node):
            return
        if not isinstance(node, dict):
            raise SchemaDefinitionError("schema at %s must be an object or boolean" % (path or "(root)"))
        for key in node:
            if key.startswith("x-"):
                continue
            if key not in SUPPORTED_KEYWORDS:
                raise UnsupportedSchemaKeyword(key, path)
        if "$ref" in node:
            ref = node["$ref"]
            name = ref[len(_REF_PREFIX):] if isinstance(ref, str) and ref.startswith(_REF_PREFIX) else None
            if not name or "/" in name or name not in self._root.get("$defs", {}):
                raise UnsupportedSchemaKeyword("$ref", path)
        if "type" in node:
            names = node["type"] if isinstance(node["type"], list) else [node["type"]]
            if not names or any(n not in _TYPE_NAMES for n in names if isinstance(n, str)) \
                    or any(not isinstance(n, str) for n in names):
                raise SchemaDefinitionError("bad type at %s: %r" % (path or "(root)", node["type"]))
        if "enum" in node and not isinstance(node["enum"], list):
            raise SchemaDefinitionError("enum at %s must be a list" % (path or "(root)"))
        if "required" in node and not (isinstance(node["required"], list)
                                       and all(isinstance(k, str) for k in node["required"])):
            raise SchemaDefinitionError("required at %s must be a list of strings" % (path or "(root)"))
        if "pattern" in node:
            try:
                self._patterns[node["pattern"]] = re.compile(node["pattern"])
            except (re.error, TypeError) as exc:
                raise SchemaDefinitionError("bad pattern at %s: %s" % (path or "(root)", exc))
        for key in ("minItems", "maxItems", "minLength", "maxLength"):
            if key in node and not (isinstance(node[key], int) and not _is_bool(node[key]) and node[key] >= 0):
                raise SchemaDefinitionError("%s at %s must be an int >= 0" % (key, path or "(root)"))
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if key in node and not _is_number(node[key]):
                raise SchemaDefinitionError("%s at %s must be a number" % (key, path or "(root)"))
        if "uniqueItems" in node and not _is_bool(node["uniqueItems"]):
            raise SchemaDefinitionError("uniqueItems at %s must be a boolean" % (path or "(root)"))
        for key in ("properties", "$defs"):
            if key in node:
                if not isinstance(node[key], dict):
                    raise SchemaDefinitionError("%s at %s must be an object" % (key, path or "(root)"))
                for name, sub in node[key].items():
                    self._check_schema(sub, _join(path, name) if key == "properties" else _join(path, "$defs." + name))
        if "items" in node:
            if not isinstance(node["items"], dict):
                raise UnsupportedSchemaKeyword("items", path)
            self._check_schema(node["items"], path + "[]")
        if "additionalProperties" in node:
            self._check_schema(node["additionalProperties"], _join(path, "*"))

    # ------------------------------------------------------------------ validation
    def validate(self, instance):
        """Return the list of ``SchemaError`` (empty = valid), in traversal order. Never raises on bad instances."""
        out = []
        self._visit(instance, self._root, "", out)
        return out

    def _visit(self, value, schema, path, out):
        if schema is True or schema == {}:
            return
        if schema is False:
            out.append(SchemaError(path, "false", "no value is allowed here"))
            return
        if "$ref" in schema:
            self._visit(value, self._root["$defs"][schema["$ref"][len(_REF_PREFIX):]], path, out)
        if "type" in schema:
            names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
            if not any(_is_type(value, n) for n in names):
                out.append(SchemaError(path, "type", "%s is not of type %s" % (_show(value), ", ".join(names))))
        if "const" in schema and not _equal(value, schema["const"]):
            out.append(SchemaError(path, "const", "%s was expected" % _show(schema["const"])))
        if "enum" in schema and not any(_equal(value, e) for e in schema["enum"]):
            out.append(SchemaError(path, "enum", "%s is not one of %s" % (_show(value), _show(schema["enum"]))))
        if _is_number(value):
            self._visit_number(value, schema, path, out)
        if isinstance(value, str):
            self._visit_string(value, schema, path, out)
        if isinstance(value, list):
            self._visit_array(value, schema, path, out)
        if isinstance(value, dict):
            self._visit_object(value, schema, path, out)

    @staticmethod
    def _visit_number(value, schema, path, out):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(SchemaError(path, "minimum", "%s is less than the minimum of %s" % (value, schema["minimum"])))
        if "maximum" in schema and value > schema["maximum"]:
            out.append(SchemaError(path, "maximum", "%s is greater than the maximum of %s" % (value, schema["maximum"])))
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            out.append(SchemaError(path, "exclusiveMinimum",
                                   "%s is less than or equal to the minimum of %s" % (value, schema["exclusiveMinimum"])))
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            out.append(SchemaError(path, "exclusiveMaximum",
                                   "%s is greater than or equal to the maximum of %s" % (value, schema["exclusiveMaximum"])))

    def _visit_string(self, value, schema, path, out):
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append(SchemaError(path, "minLength", "%s is too short (minimum %d)" % (_show(value), schema["minLength"])))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append(SchemaError(path, "maxLength", "%s is too long (maximum %d)" % (_show(value), schema["maxLength"])))
        if "pattern" in schema and not self._patterns[schema["pattern"]].search(value):
            out.append(SchemaError(path, "pattern", "%s does not match %r" % (_show(value), schema["pattern"])))

    def _visit_array(self, value, schema, path, out):
        if "minItems" in schema and len(value) < schema["minItems"]:
            out.append(SchemaError(path, "minItems", "%s has fewer than %d items" % (_show(value), schema["minItems"])))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            out.append(SchemaError(path, "maxItems", "%s has more than %d items" % (_show(value), schema["maxItems"])))
        if schema.get("uniqueItems") is True:
            if any(_equal(value[i], value[j]) for i in range(len(value)) for j in range(i + 1, len(value))):
                out.append(SchemaError(path, "uniqueItems", "%s has non-unique elements" % _show(value)))
        if "items" in schema:
            for i, item in enumerate(value):
                self._visit(item, schema["items"], "%s[%d]" % (path, i), out)

    def _visit_object(self, value, schema, path, out):
        props = schema.get("properties", {})
        missing = [k for k in schema.get("required", ()) if k not in value]
        if missing:
            out.append(SchemaError(path, "required", "missing required propert%s %s" % (
                "y" if len(missing) == 1 else "ies", ", ".join(repr(k) for k in missing))))
        for name, sub in props.items():
            if name in value:
                self._visit(value[name], sub, _join(path, name), out)
        if "additionalProperties" in schema:
            extra = [k for k in value if k not in props]
            rule = schema["additionalProperties"]
            if rule is False:
                if extra:
                    out.append(SchemaError(path, "additionalProperties", "additional propert%s %s not allowed" % (
                        "y" if len(extra) == 1 else "ies", ", ".join(repr(k) for k in extra))))
            elif rule is not True:
                for k in extra:
                    self._visit(value[k], rule, _join(path, k), out)


def validate(instance, schema):
    """``Validator(schema).validate(instance)``. Raises ``UnsupportedSchemaKeyword`` for a keyword outside the subset."""
    return Validator(schema).validate(instance)
