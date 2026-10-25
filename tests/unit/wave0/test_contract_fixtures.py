"""Wave 0: the four contract fixtures are valid against the schemas of design section 2 (jsonschema cross-check).

Fixtures (copies of research/design_mp4/contracts/examples/) live in tests/fixtures/contracts/*.json, the schemas
transcribed from the field tables of 2.3-2.6 in tests/fixtures/contracts/schemas/*.schema.json (ASSUMED: the design
defines these types by field tables and examples, not by a JSON Schema file; the transcription is strict, i.e.
additionalProperties false on every enumerated object). Producer tests validate their output against the fixture
keys, consumer tests feed the fixtures in.
"""
import copy
import json
import os
import unittest

from jsonschema import Draft202012Validator

from moss_photon.core import consts

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTRACTS = os.path.join(TESTS_DIR, "fixtures", "contracts")
SCHEMAS = os.path.join(CONTRACTS, "schemas")

# fixture file -> (schema file, consts schema-version name)
TYPES = {
    "constraints": ("constraints.schema.json", "CONSTRAINTS_SCHEMA"),
    "identity_profile": ("identity_profile.schema.json", "IDENTITY_PROFILE_SCHEMA"),
    "report": ("report.schema.json", "REPORT_SCHEMA"),
    "schedule_info": ("schedule_info.schema.json", "SCHEDULE_INFO_SCHEMA"),
}


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8")), raw


def fixture(name):
    return load(os.path.join(CONTRACTS, name + ".json"))[0]


def validator(name):
    schema = load(os.path.join(SCHEMAS, TYPES[name][0]))[0]
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


class FixtureInventoryTest(unittest.TestCase):
    def test_exactly_the_four_fixtures_and_schemas(self):
        files = sorted(f for f in os.listdir(CONTRACTS) if f.endswith(".json"))
        # identity_profile_turbo.json (MP-24) is validated by test_turbo_contracts.py
        self.assertEqual(files, sorted([n + ".json" for n in TYPES] + ["identity_profile_turbo.json"]))
        schemas = sorted(os.listdir(SCHEMAS))
        self.assertEqual(schemas, sorted(s for s, _ in TYPES.values()))

    def test_files_are_lf_utf8_without_bom_with_final_newline(self):
        paths = [os.path.join(CONTRACTS, n + ".json") for n in TYPES]
        paths += [os.path.join(SCHEMAS, s) for s, _ in TYPES.values()]
        for path in paths:
            with self.subTest(path=os.path.basename(path)):
                raw = load(path)[1]
                self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
                self.assertNotIn(b"\r", raw)
                self.assertTrue(raw.endswith(b"\n"))
                raw.decode("utf-8")


class FixturesValidateTest(unittest.TestCase):
    def test_fixtures_validate_against_design_schemas(self):
        for name in TYPES:
            with self.subTest(fixture=name):
                errors = sorted(validator(name).iter_errors(fixture(name)), key=lambda e: list(e.path))
                self.assertEqual([e.message + " @ " + "/".join(map(str, e.path)) for e in errors], [])

    def test_schema_version_equals_consts(self):
        for name, (_, const_name) in TYPES.items():
            with self.subTest(fixture=name):
                self.assertEqual(fixture(name)["schema_version"], getattr(consts, const_name))

    def test_report_pack_and_sigma_hash_equal_consts(self):
        rep = fixture("report")
        self.assertEqual(rep["pack"], {"name": consts.PACK_NAME, "version": consts.PACK_VERSION})
        self.assertEqual(rep["sampling"]["sigmas"]["sha256"], consts.REFERENCE_SIGMAS_SHA256)
        self.assertEqual(rep["model"]["identity"]["sha256"], consts.SFT_BF16_SHA256)
        self.assertIn(rep["node"], consts.NODE_IDS)
        self.assertIn(rep["preset"]["id"], consts.PRESET_IDS)

    def test_schedule_info_native_hash_and_counts(self):
        si = fixture("schedule_info")
        self.assertEqual(si["sigmas_sha256"], consts.REFERENCE_SIGMAS_SHA256)
        self.assertEqual(si["native_reference"]["sha256"], consts.REFERENCE_SIGMAS_SHA256)
        self.assertEqual(si["n_sigmas"], si["steps"] + 1)

    def test_constraints_compiler_version_and_private_paths_exist(self):
        c = fixture("constraints")
        self.assertEqual(c["compiler_version"], consts.COMPILER_VERSION)
        for path in c["private_paths"]:
            self.assertIn(path, c)
        # OD1: forbidden_changes is never compiled into the prompt
        self.assertNotIn("forbidden", c["compiled_positive"].lower())
        for key, present in c["blocks_present"].items():
            self.assertEqual(present, bool(c["fields"][key].strip()), key)

    def test_identity_profile_entry_arithmetic(self):
        ip = fixture("identity_profile")
        att, stack = ip["attachment"], ip["stack"]
        self.assertEqual(att["patch_entries_total"],
                         att["baseline_entries"] + sum(e["patch_entries"] for e in att["entries"]))
        self.assertEqual(att["patch_entries_total"], ip["summary"]["patch_entries_total"])
        self.assertEqual(ip["summary"]["patch_entries_total"], sum(s["patch_entries"] for s in stack))
        self.assertEqual(ip["summary"]["count"], len(stack))
        for s in stack:
            self.assertEqual(s["coverage"]["modules_mapped"], s["patch_entries"])

    def test_report_forward_counts_and_private_data_absent(self):
        rep = fixture("report")
        fw = rep["forwards"]
        self.assertEqual(fw["match"], fw["sample_forwards"] == fw["expected_sample_forwards"])
        self.assertIsNone(rep["timing"]["estimate_s"])
        self.assertEqual(rep["privacy"], {"prompts_logged": False, "reference_photos_logged": False})
        for sentinel in ("Anna", "red wool sweater"):
            self.assertNotIn(sentinel, json.dumps(rep))


def _drop(path):
    def mutate(doc):
        node = doc
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
    return mutate


def _set(path, value):
    def mutate(doc):
        node = doc
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return mutate


MUTATIONS = {
    "report": {
        "unknown top-level key": _set(["surprise"], 1),
        "bad kind": _set(["kind"], "photon_other"),
        "future schema_version": _set(["schema_version"], 2),
        "bad parity_class": _set(["label", "parity_class"], "reference_perfect"),
        "bad hash case": _set(["recipe_hash"], "A" * 64),
        "short hash": _set(["inputs", "noise_sha256"], "0" * 63),
        "prompts_logged true": _set(["privacy", "prompts_logged"], True),
        "guard_passed true": _set(["guards", "guard_passed"], True),
        "estimate_s not null": _set(["timing", "estimate_s"], 1.5),
        "path in basename": _set(["model", "identity", "basename"], "C:\\models\\a.safetensors"),
        "width not multiple of 16": _set(["inputs", "width"], 1000),
        "missing label.modifiers": _drop(["label", "modifiers"]),
        "missing lora.foreign_patches": _drop(["lora", "foreign_patches"]),
        "missing preset.pinned": _drop(["preset", "pinned"]),
        "bad created_utc": _set(["created_utc"], "2026-10-25 12:00"),
        "issue without fix": _set(["warnings", 0], {"code": "x", "severity": "info", "node": "n", "param": "p", "message": "m"}),
        "bad noise policy": _set(["inputs", "noise_policy"], "random"),
    },
    "constraints": {
        "unknown key": _set(["surprise"], 1),
        "missing blocks_present": _drop(["blocks_present"]),
        "bad bbox_format": _set(["layout"], {"items": 1, "bbox_coord_space": "abs_centralized", "bbox_format": "xyxy",
                                              "bboxes": [[0, 0, 1, 1]], "extent": [0, 0, 1, 1], "experimental": True}),
        "missing forbidden_changes block": _drop(["fields", "forbidden_changes"]),
        "text_encoder path": _set(["text_encoder", "basename"], "models/a.safetensors"),
        "empty private_paths": _set(["private_paths"], []),
    },
    "identity_profile": {
        "unknown key": _set(["surprise"], 1),
        "phase_plan set": _set(["phase_plan"], {"kind": "phased"}),
        "bad role": _set(["stack", 0, "role"], "villain"),
        "bad key_format": _set(["stack", 0, "key_format"], "other"),
        "ratio above 1": _set(["stack", 0, "coverage", "ratio"], 1.5),
        "missing attachment": _drop(["attachment"]),
        "missing patches_uuid": _drop(["attachment", "patches_uuid"]),
        "wrong private_paths": _set(["private_paths"], ["stack[*].lora"]),
        "lora full path": _set(["stack", 0, "lora", "basename"], "D:/x/a.safetensors"),
    },
    "schedule_info": {
        "unknown key": _set(["surprise"], 1),
        "bad mode": _set(["mode"], "custom"),
        "bad scheduler": _set(["scheduler"], "karras"),
        "validation failed": _set(["validation", "ok"], False),
        "missing native_reference": _drop(["native_reference"]),
        "bad sigmas hash": _set(["sigmas_sha256"], "xyz"),
        "bad check name": _set(["validation", "checks"], ["finite", "magic"]),
    },
}


class SchemasAreNotVacuousTest(unittest.TestCase):
    def test_every_mutation_is_rejected(self):
        for name, muts in MUTATIONS.items():
            for label, mutate in muts.items():
                with self.subTest(fixture=name, mutation=label):
                    doc = copy.deepcopy(fixture(name))
                    mutate(doc)
                    self.assertFalse(validator(name).is_valid(doc), "mutation was accepted")

    def test_deleting_any_top_level_key_is_rejected(self):
        for name in TYPES:
            base = fixture(name)
            for key in base:
                with self.subTest(fixture=name, dropped=key):
                    doc = copy.deepcopy(base)
                    del doc[key]
                    self.assertFalse(validator(name).is_valid(doc))

    def test_optional_report_diagnostics_is_accepted(self):
        doc = copy.deepcopy(fixture("report"))
        doc["diagnostics"] = {"anything": 1}
        self.assertTrue(validator("report").is_valid(doc))

    def test_shifted_schedule_variant_is_accepted(self):
        doc = copy.deepcopy(fixture("schedule_info"))
        doc.update({"mode": "shifted", "shift_used": 3.0, "is_native_reference": False})
        doc["native_reference"].update({"equal": False, "bit_equal": False, "max_abs_diff": 0.2, "n_differ": 50})
        self.assertTrue(validator("schedule_info").is_valid(doc))

    def test_control_unmutated_fixtures_are_valid(self):
        """The mutation tests above mean something only if the unmutated documents pass."""
        for name in TYPES:
            self.assertTrue(validator(name).is_valid(fixture(name)), name)


if __name__ == "__main__":
    unittest.main()
