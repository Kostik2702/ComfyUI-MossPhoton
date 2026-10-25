"""MP-16: the additive report block lora.phase: the structural validator and the JSON schema agree, the recipe hash sees it
only when it is there."""
import copy
import json
import os
import unittest

from jsonschema import Draft202012Validator

from moss_photon.core import report as report_mod
from moss_photon.core import phase

CONTRACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "fixtures", "contracts")


def load(*parts):
    with open(os.path.join(CONTRACTS, *parts), "rb") as f:
        return json.loads(f.read().decode("utf-8"))


SCHEMA = Draft202012Validator(load("schemas", "report.schema.json"))
SIGMAS = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]


def with_phase(clones_built=3):
    r = copy.deepcopy(load("report.json"))
    sch = phase.build_schedule(SIGMAS, phase.make_plan(0, 0.3, 0.9, 0.2, 0.6))
    r["lora"]["phase"] = phase.describe_schedule(sch, clones_built=clones_built)
    return r


class ReportPhaseTest(unittest.TestCase):
    def test_a_phase_block_validates_in_both_contracts(self):
        r = with_phase()
        self.assertEqual(report_mod.validate_report(r), [])
        self.assertEqual([e.message for e in SCHEMA.iter_errors(r)], [])

    def test_the_fixture_without_a_block_still_validates(self):
        r = load("report.json")
        self.assertNotIn("phase", r["lora"])
        self.assertEqual(report_mod.validate_report(r), [])
        self.assertEqual([e.message for e in SCHEMA.iter_errors(r)], [])

    def test_both_contracts_reject_the_same_broken_blocks(self):
        mutations = {
            "extra key": lambda b: b.update(extra=1),
            "missing key": lambda b: b.pop("weights_sha256"),
            "bad mechanism": lambda b: b.update(mechanism="hooks"),
            "bad hash": lambda b: b.update(planned_sha256="xyz"),
            "no segments": lambda b: b.update(segments=[]),
            "segment key": lambda b: b["segments"][0].pop("weight"),
            "string weight": lambda b: b.update(w0="0.3"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name):
                r = with_phase()
                mutate(r["lora"]["phase"])
                self.assertTrue(report_mod.validate_report(r), "validator: " + name)
                self.assertTrue(list(SCHEMA.iter_errors(r)), "schema: " + name)

    def test_the_recipe_hash_is_unchanged_without_a_block_and_sees_the_plan_with_one(self):
        plain = load("report.json")
        material = report_mod._recipe_material(plain)
        self.assertNotIn("lora.phase", material)
        a, b = with_phase(), with_phase()
        self.assertEqual(report_mod.recipe_hash(a), report_mod.recipe_hash(b))
        self.assertNotEqual(report_mod.recipe_hash(a), report_mod.recipe_hash(plain))
        b["lora"]["phase"]["w1"] = 0.8
        self.assertNotEqual(report_mod.recipe_hash(a), report_mod.recipe_hash(b))
        c = with_phase()
        c["lora"]["phase"]["segments"][0]["weight"] = 0.31
        self.assertNotEqual(report_mod.recipe_hash(a), report_mod.recipe_hash(c))

    def test_the_summary_mentions_a_phased_run(self):
        self.assertIn("phased lora (experimental)", report_mod.format_summary(with_phase()))
        self.assertNotIn("phased", report_mod.format_summary(load("report.json")))


if __name__ == "__main__":
    unittest.main()
