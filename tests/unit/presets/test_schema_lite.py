"""MP-7: core/schema_lite.py keyword by keyword (design 1.3, 6.2)."""
import unittest

from moss_photon.core import schema_lite
from moss_photon.core.schema_lite import UnsupportedSchemaKeyword, validate


def keywords(instance, schema):
    return sorted((e.path, e.keyword) for e in validate(instance, schema))


class KeywordTest(unittest.TestCase):
    def test_type_single_and_list(self):
        self.assertEqual(validate("a", {"type": "string"}), [])
        self.assertEqual(keywords(1, {"type": "string"}), [("", "type")])
        self.assertEqual(validate(None, {"type": ["integer", "null"]}), [])
        self.assertEqual(validate(3, {"type": ["integer", "null"]}), [])
        self.assertEqual(keywords("x", {"type": ["integer", "null"]}), [("", "type")])

    def test_bool_is_not_integer(self):
        for name in ("integer", "number"):
            self.assertEqual(keywords(True, {"type": name}), [("", "type")])
        self.assertEqual(validate(True, {"type": "boolean"}), [])
        self.assertEqual(keywords(1, {"type": "boolean"}), [("", "type")])

    def test_integral_float_is_integer(self):
        self.assertEqual(validate(50.0, {"type": "integer"}), [])
        self.assertEqual(keywords(50.5, {"type": "integer"}), [("", "type")])
        self.assertEqual(validate(2, {"type": "number"}), [])

    def test_enum_and_const_keep_bool_apart(self):
        self.assertEqual(validate("a", {"enum": ["a", "b"]}), [])
        self.assertEqual(keywords("c", {"enum": ["a", "b"]}), [("", "enum")])
        self.assertEqual(keywords(0, {"const": False}), [("", "const")])
        self.assertEqual(keywords(False, {"const": 0}), [("", "const")])
        self.assertEqual(validate(False, {"const": False}), [])
        self.assertEqual(keywords(1, {"enum": [True]}), [("", "enum")])
        self.assertEqual(validate(1.0, {"const": 1}), [])

    def test_properties_required_additional_false(self):
        schema = {"type": "object", "required": ["a"], "additionalProperties": False,
                  "properties": {"a": {"type": "integer"}}}
        self.assertEqual(validate({"a": 1}, schema), [])
        self.assertEqual(keywords({}, schema), [("", "required")])
        self.assertEqual(keywords({"a": 1, "b": 2}, schema), [("", "additionalProperties")])
        self.assertEqual(keywords({"a": "x"}, schema), [("a", "type")])

    def test_additional_properties_schema(self):
        schema = {"additionalProperties": {"type": "integer"}}
        self.assertEqual(validate({"x": 1}, schema), [])
        self.assertEqual(keywords({"x": "s"}, schema), [("x", "type")])

    def test_items_and_item_counts(self):
        schema = {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "integer"}}
        self.assertEqual(validate([1], schema), [])
        self.assertEqual(keywords([], schema), [("", "minItems")])
        self.assertEqual(keywords([1, 2, 3], schema), [("", "maxItems")])
        self.assertEqual(keywords([1, "a"], schema), [("[1]", "type")])

    def test_unique_items(self):
        schema = {"uniqueItems": True}
        self.assertEqual(validate([1, 2], schema), [])
        self.assertEqual(keywords([1, 1.0], schema), [("", "uniqueItems")])
        self.assertEqual(validate([0, False], schema), [])
        self.assertEqual(validate([1, 2], {"uniqueItems": False}), [])

    def test_numeric_bounds(self):
        s = {"minimum": 1, "maximum": 3, "exclusiveMinimum": 0, "exclusiveMaximum": 4}
        self.assertEqual(validate(2, s), [])
        self.assertEqual(keywords(0, s), [("", "exclusiveMinimum"), ("", "minimum")])
        self.assertEqual(keywords(3.5, s), [("", "maximum")])
        self.assertEqual(keywords(4, {"exclusiveMaximum": 4}), [("", "exclusiveMaximum")])
        self.assertEqual(validate("text", s), [])   # bounds only apply to numbers
        self.assertEqual(validate(True, {"minimum": 5}), [])

    def test_string_length_and_pattern(self):
        s = {"type": "string", "minLength": 2, "maxLength": 3, "pattern": "^[a-z]+$"}
        self.assertEqual(validate("ab", s), [])
        self.assertEqual(keywords("a", s), [("", "minLength")])
        self.assertEqual(keywords("abcd", s), [("", "maxLength")])
        self.assertEqual(keywords("aB", s), [("", "pattern")])

    def test_local_ref(self):
        schema = {"$defs": {"n": {"type": "integer"}}, "properties": {"a": {"$ref": "#/$defs/n"}}}
        self.assertEqual(validate({"a": 1}, schema), [])
        self.assertEqual(keywords({"a": "x"}, schema), [("a", "type")])

    def test_error_paths(self):
        schema = {"properties": {"sampling": {"properties": {"steps": {"type": "integer"}}},
                                 "constraints": {"items": {"properties": {"kind": {"enum": ["k"]}}}}}}
        self.assertEqual(keywords({"sampling": {"steps": "x"}, "constraints": [{"kind": "k"}, {"kind": "z"}]}, schema),
                         [("constraints[1].kind", "enum"), ("sampling.steps", "type")])

    def test_annotations_ignored(self):
        self.assertEqual(validate(1, {"type": "integer", "x-unit": "steps", "title": "t", "description": "d"}), [])


class UnsupportedKeywordTest(unittest.TestCase):
    def test_unsupported_keyword_raises(self):
        for kw in ("oneOf", "anyOf", "allOf", "not", "if", "patternProperties", "format", "multipleOf", "dependentRequired",
                   "propertyNames", "contains", "minProperties", "prefixItems", "$anchor", "default"):
            with self.assertRaises(UnsupportedSchemaKeyword, msg=kw):
                validate({}, {kw: 1})

    def test_unsupported_nested_keyword_raises(self):
        nested = [
            {"properties": {"a": {"oneOf": []}}},
            {"$defs": {"d": {"format": "date"}}},
            {"items": {"anyOf": []}},
            {"additionalProperties": {"not": {}}},
            {"properties": {"a": {"properties": {"b": {"multipleOf": 2}}}}},
        ]
        for schema in nested:
            with self.assertRaises(UnsupportedSchemaKeyword, msg=str(schema)):
                schema_lite.Validator(schema)

    def test_property_named_like_a_keyword_is_fine(self):
        schema = {"properties": {"oneOf": {"type": "integer"}, "format": {"type": "string"}}}
        self.assertEqual(validate({"oneOf": 1, "format": "x"}, schema), [])

    def test_ref_forms_outside_subset_raise(self):
        for ref in ("http://x/y", "#/properties/a", "#/$defs/missing", "#", 5):
            with self.assertRaises(UnsupportedSchemaKeyword, msg=str(ref)):
                schema_lite.Validator({"$defs": {"n": {}}, "properties": {"a": {"$ref": ref}}})

    def test_error_names_keyword_and_path(self):
        with self.assertRaises(UnsupportedSchemaKeyword) as cm:
            schema_lite.Validator({"properties": {"a": {"oneOf": []}}})
        self.assertEqual((cm.exception.keyword, cm.exception.path), ("oneOf", "a"))

    def test_supported_set_is_the_design_list(self):
        self.assertEqual(len(schema_lite.SUPPORTED_KEYWORDS), 23)
        self.assertNotIn("oneOf", schema_lite.SUPPORTED_KEYWORDS)

    def test_bad_pattern_is_a_definition_error(self):
        with self.assertRaises(schema_lite.SchemaDefinitionError):
            schema_lite.Validator({"pattern": "("})


if __name__ == "__main__":
    unittest.main()
