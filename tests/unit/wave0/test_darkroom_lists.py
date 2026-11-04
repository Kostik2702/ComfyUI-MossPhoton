"""MP-71: the literal lists the darkroom presets touch are ONE list in four places (this pack drifted three times on that):
``consts.TURBO_PRESET_IDS`` is the source; ``presets.ALL_PRESET_IDS``, ``presets/schema.json`` (``id`` enum), the report validator
(``report._SPEC``) and the fixture schema (``report.schema.json``) are compared with it here. The reason codes of the darkroom
(MP-57 / MP-67 / MP-71) are compared between the registry, the validator's code pattern and the fixture schema's.
"""
import importlib
import json
import os
import re
import unittest

from moss_photon.core import consts, errors, presets, report

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_SCHEMA = os.path.join(TESTS_DIR, "fixtures", "contracts", "schemas", "report.schema.json")
DARKROOM_CODES = ("darkroom_experimental", "darkroom_boost_skipped", "finish_model_ignored", "darkroom_override_ignored")


def load(path):
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


class PresetIdListsTest(unittest.TestCase):
    EXPECTED = tuple(consts.PRESET_IDS) + tuple(consts.TURBO_PRESET_IDS)

    def test_the_source_has_the_darkroom_presets(self):
        self.assertEqual(consts.TURBO_PRESET_IDS[-2:], ("turbo_darkroom_portrait", "turbo_darkroom_fullbody"))
        self.assertEqual(len(set(consts.TURBO_PRESET_IDS)), len(consts.TURBO_PRESET_IDS))

    def test_presets_module_uses_the_source(self):
        self.assertIs(presets.TURBO_PRESET_IDS, consts.TURBO_PRESET_IDS)
        self.assertEqual(presets.ALL_PRESET_IDS, self.EXPECTED)

    def test_the_preset_schema_enum_is_the_same_list(self):
        self.assertEqual(tuple(presets.load_schema()["properties"]["id"]["enum"]), self.EXPECTED)

    def test_the_report_validator_enum_is_the_same_list(self):
        ids = report._SPEC.req["preset"].req["id"]
        self.assertEqual(tuple(ids.values), self.EXPECTED)

    def test_the_fixture_schema_enum_is_the_same_list(self):
        schema = load(REPORT_SCHEMA)
        self.assertEqual(tuple(schema["properties"]["preset"]["properties"]["id"]["enum"]), self.EXPECTED)

    def test_every_listed_preset_has_its_file_and_only_those_have(self):
        names = sorted(n[:-5] for n in os.listdir(presets.preset_dir()) if n.endswith(".json") and n != "schema.json")
        self.assertEqual(names, sorted(self.EXPECTED))


class CodeListsTest(unittest.TestCase):
    def test_the_darkroom_codes_are_registered_status_codes(self):
        for code in DARKROOM_CODES:
            with self.subTest(code=code):
                self.assertEqual(errors.REASONS[code].severity, "warning")
                self.assertIsNone(errors.REASONS[code].exc)

    def test_the_issue_code_pattern_of_both_contracts_takes_them(self):
        pattern = load(REPORT_SCHEMA)["$defs"]["issue"]["properties"]["code"]["pattern"]
        self.assertEqual(pattern, report._CODE.pattern)
        for code in DARKROOM_CODES:
            self.assertTrue(re.match(pattern, code), code)

    def test_the_recipe_pattern_is_the_same_in_both_contracts(self):
        schema = load(REPORT_SCHEMA)
        self.assertEqual(schema["properties"]["darkroom"]["properties"]["recipe"]["pattern"], report._RECIPE.pattern)


class DarkroomNodeIsGoneTest(unittest.TestCase):
    def test_no_node_module_no_node_id(self):
        with self.assertRaises(ImportError):
            importlib.import_module("moss_photon.nodes.darkroom")
        self.assertNotIn("MossPhotonDarkroomSampler", consts.NODE_IDS + consts.NODE_IDS_SECOND_QUEUE)
        with open(os.path.join(os.path.dirname(TESTS_DIR), "__init__.py"), "r", encoding="utf-8") as f:
            self.assertNotIn("nodes.darkroom", f.read())


if __name__ == "__main__":
    unittest.main()
