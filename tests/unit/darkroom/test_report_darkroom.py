"""MP-57: the additive report block ``darkroom``. The structural validator and the JSON schema agree, the names equal
``core/darkroom.py``, the recipe hash sees the plan (and not what was observed), the summary names the run.

Guarantee (d), the report side: ``phases[]`` with sigma per phase, weights, RNG seeds, the recipe id@version and the label.
"""
import copy
import json
import os
import unittest

from jsonschema import Draft202012Validator

from moss_photon.core import darkroom
from moss_photon.core import report as report_mod
from tests.unit.darkroom.support import OFF
from tests.unit.darkroom.test_plan import CHARACTER, S12, plan_of

CONTRACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "fixtures", "contracts")


def load(*parts):
    with open(os.path.join(CONTRACTS, *parts), "rb") as f:
        return json.loads(f.read().decode("utf-8"))


SCHEMA_DOC = load("schemas", "report.schema.json")
SCHEMA = Draft202012Validator(SCHEMA_DOC)
DK_SCHEMA = SCHEMA_DOC["properties"]["darkroom"]["properties"]


def with_block(plan=None, observed=True, **kw):
    r = copy.deepcopy(load("report.json"))
    plan = plan or plan_of(**kw)
    seen = [sum(1 if s.cfg == 1.0 else 2 for s in plan.steps[seg.first:seg.stop]) for seg in plan.segments]
    r["darkroom"] = darkroom.describe(plan, observed=seen if observed else None, clones_built=len(plan.segments))
    return r


class BlockTest(unittest.TestCase):
    def test_a_block_validates_in_both_contracts(self):
        for kwargs in ({}, {"overrides": {"eta": 0.3}}, {"overrides": {"reexpose": True, "r_k": 0.3, "r_f": 0.3}},
                       {"overrides": {"r_f": 0.4}}, {"recipe": OFF, "phased": None}):
            for observed in (True, False):
                with self.subTest(kwargs=kwargs, observed=observed):
                    r = with_block(observed=observed, **kwargs)
                    self.assertEqual(report_mod.validate_report(r), [])
                    self.assertEqual([e.message for e in SCHEMA.iter_errors(r)], [])

    def test_the_fixture_without_a_block_still_validates(self):
        r = load("report.json")
        self.assertNotIn("darkroom", r)
        self.assertEqual(report_mod.validate_report(r), [])
        self.assertEqual([e.message for e in SCHEMA.iter_errors(r)], [])

    def test_both_contracts_reject_the_same_broken_blocks(self):
        mutations = {
            "extra key": lambda b: b.update(extra=1),
            "missing key": lambda b: b.pop("rng"),
            "bad label": lambda b: b.update(label="stable"),
            "bad recipe": lambda b: b.update(recipe="turbo_darkroom_portrait"),
            "recipe without a semver": lambda b: b.update(recipe="turbo_darkroom_portrait@1"),
            "bad hash": lambda b: b["paths"][0].update(sigmas_sha256="xyz"),
            "no paths": lambda b: b.update(paths=[]),
            "three paths": lambda b: b.update(paths=b["paths"] * 3),
            "no segments": lambda b: b.update(segments=[]),
            "phase name": lambda b: b["phases"][0].update(name="develop"),
            "phase key": lambda b: b["phases"][0].pop("cfg"),
            "string cfg": lambda b: b["phases"][0].update(cfg="1.0"),
            "segment model": lambda b: b["segments"][0].update(model="third"),
            "segment sampler": lambda b: b["segments"][0].update(sampler="heun"),
            "rng key": lambda b: b["rng"]["noise"].pop("seed"),
            "character mode": lambda b: b["character"].update(mode="hooks"),
            "settings key": lambda b: b["settings"]["finish"].pop("eta"),
            "bool switch": lambda b: b["settings"]["finish"].update(switch=1),
            "progress": lambda b: b.update(progress="cumulative_abs_dsigma"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name):
                r = with_block()
                mutate(r["darkroom"])
                self.assertTrue(report_mod.validate_report(r), "validator: " + name)
                self.assertTrue(list(SCHEMA.iter_errors(r)), "schema: " + name)

    def test_the_enumerations_equal_the_module_constants(self):
        phases = DK_SCHEMA["phases"]["items"]["properties"]
        self.assertEqual(tuple(phases["name"]["enum"]), darkroom.PHASE_NAMES)
        self.assertEqual(sorted(phases["model"]["enum"]), sorted((darkroom.MODEL_MAIN, darkroom.MODEL_FINISH)))
        self.assertEqual(sorted(phases["sampler"]["enum"]), sorted((darkroom.SAMPLER_EULER, darkroom.SAMPLER_ANCESTRAL)))
        self.assertEqual(DK_SCHEMA["label"]["const"], darkroom.LABEL)
        self.assertEqual(DK_SCHEMA["version"]["const"], darkroom.VERSION)
        self.assertEqual(DK_SCHEMA["progress"]["const"], darkroom.PROGRESS)
        self.assertEqual(sorted(DK_SCHEMA["schedule"]["enum"]), sorted(darkroom.SCHEDULES))
        block = darkroom.describe(plan_of())
        self.assertEqual((block["label"], block["version"], block["progress"]),
                         (darkroom.LABEL, darkroom.VERSION, darkroom.PROGRESS))

    def test_the_block_holds_what_the_ticket_wants_recorded(self):
        block = with_block()["darkroom"]
        self.assertEqual(block["recipe"], "turbo_darkroom_portrait@1.0.0")
        self.assertEqual(block["label"], "darkroom_experimental")
        for row in block["phases"]:
            self.assertTrue({"name", "sigma_start", "sigma_end", "weight_first", "weight_last", "cfg", "steps",
                             "expected_sample_forwards", "observed_sample_forwards"} <= set(row), row)
        self.assertEqual(sorted(block["rng"]), ["main_seed", "noise", "reexpose"])
        self.assertEqual(block["settings"]["commit"], {"r_c": 0.06, "r_k": 0.17, "w_compose": 1.0, "w_commit": 1.2})
        self.assertEqual(block["character"], {"mode": "phased_adapter", "stack_index": 1,
                                              "profile_phase_plan_replaced": True, "base_strength": None,
                                              "boost_skipped": None})

    def test_observed_forwards_per_phase_add_up_to_the_expected_ones(self):
        block = with_block()["darkroom"]
        self.assertEqual([p["observed_sample_forwards"] for p in block["phases"]],
                         [p["expected_sample_forwards"] for p in block["phases"]])
        self.assertEqual(sum(s["observed_sample_forwards"] for s in block["segments"]), 17)

    def test_before_the_run_nothing_is_observed(self):
        block = with_block(observed=False)["darkroom"]
        self.assertEqual({p["observed_sample_forwards"] for p in block["phases"]}, {None})
        self.assertIsNone(darkroom.describe(plan_of())["clones_built"])


class RecipeHashTest(unittest.TestCase):
    def test_unchanged_without_a_block_and_sees_the_plan_with_one(self):
        plain = load("report.json")
        self.assertNotIn("darkroom", report_mod._recipe_material(plain))
        a, b = with_block(), with_block()
        self.assertEqual(report_mod.recipe_hash(a), report_mod.recipe_hash(b))
        self.assertNotEqual(report_mod.recipe_hash(a), report_mod.recipe_hash(plain))
        for name, change in (("w_commit", lambda d: d["settings"]["commit"].update(w_commit=1.3)),
                             ("finish cfg", lambda d: d["settings"]["finish"].update(cfg=2.5)),
                             ("recipe", lambda d: d.update(recipe="turbo_darkroom_portrait@1.0.1")),
                             ("seed", lambda d: d["rng"].update(main_seed=7)),
                             ("segment", lambda d: d["segments"][0].update(weight=1.01))):
            with self.subTest(name):
                c = with_block()
                change(c["darkroom"])
                self.assertNotEqual(report_mod.recipe_hash(a), report_mod.recipe_hash(c))

    def test_what_was_observed_is_not_part_of_the_recipe(self):
        a, b = with_block(observed=True), with_block(observed=False)
        self.assertEqual(report_mod.recipe_hash(a), report_mod.recipe_hash(b))
        b["darkroom"]["clones_built"] = 99
        self.assertEqual(report_mod.recipe_hash(a), report_mod.recipe_hash(b))

    def test_the_summary_names_the_run(self):
        text = report_mod.format_summary(with_block())
        self.assertIn("darkroom turbo_darkroom_portrait@1.0.0 (experimental)", text)
        self.assertIn("cfg 1.0 -> 2.0", text)
        self.assertIn("character weight 1.0 -> 1.2", text)
        self.assertNotIn("darkroom", report_mod.format_summary(load("report.json")))


if __name__ == "__main__":
    unittest.main()
