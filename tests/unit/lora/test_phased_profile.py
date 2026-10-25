"""MP-16: a PHOTON_IDENTITY_PROFILE with a phased adapter (phase_plan + weight_profile.kind phased), validated against the
schema and the cross-field rules; the stack rules and the journal for it."""
import copy
import json
import os
import unittest

from jsonschema import Draft202012Validator

from moss_photon.core import errors, lora, phase

from .builders import DMD2, SHA_A, entry

CONTRACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "fixtures", "contracts")
with open(os.path.join(CONTRACTS, "schemas", "identity_profile.schema.json"), "rb") as _f:
    SCHEMA = Draft202012Validator(json.loads(_f.read().decode("utf-8")))


def phased_profile(w0=0.3, w1=0.9, a=0.2, b=0.6, *, constant_first=True, verified=None):
    stack = []
    if constant_first:
        stack.append(entry(0, "distill", "lodestones-8-steps-dmd2-lora.safetensors", DMD2, 1.0, size=1189176800))
    plan = phase.make_plan(len(stack), w0, w1, a, b)
    char = entry(len(stack), "character", "character_a.safetensors", SHA_A, plan.w1, verified=verified)
    char["weight_profile"] = phase.weight_profile_of(plan)
    stack.append(char)
    on_model = sum(e["patch_entries"] for e in stack if e["weight_profile"]["kind"] != "phased")
    att = lora.build_attachment([lora.attachment_entry_of(e) for e in stack], patches_uuid="u-1",
                                patch_entries_total=on_model, baseline_entries=0)
    return lora.build_identity_profile(stack, attachment=att, model_identity=None, phase_plan=phase.plan_to_dict(plan))


def schema_errors(instance):
    return [e.message for e in SCHEMA.iter_errors(instance)]


class PhasedProfileTest(unittest.TestCase):
    def test_a_phased_profile_validates(self):
        for kw in ({}, {"constant_first": False}, {"w0": 0.9, "w1": 0.3}):
            with self.subTest(kw=kw):
                p = phased_profile(**kw)
                self.assertEqual(schema_errors(p), [])
                self.assertEqual(lora.validate_identity_profile(p), [])
                self.assertEqual(p["stack"][-1]["strength_model"], p["phase_plan"]["w1"])
                self.assertEqual(p["summary"]["patch_entries_total"], p["attachment"]["patch_entries_total"])

    def test_the_phased_adapter_adds_no_on_model_entry(self):
        p = phased_profile()
        self.assertEqual(p["attachment"]["patch_entries_total"], p["stack"][0]["patch_entries"])
        wrong = copy.deepcopy(p)
        wrong["attachment"]["patch_entries_total"] += wrong["stack"][1]["patch_entries"]
        wrong["summary"]["patch_entries_total"] = wrong["attachment"]["patch_entries_total"]
        self.assertTrue(any("baseline + entries" in line for line in lora.validate_identity_profile(wrong)))

    def test_inconsistent_plans_are_problems(self):
        base = phased_profile()
        cases = {
            "plan is null but an entry is phased": lambda p: p.update(phase_plan=None),
            "stack_index points elsewhere": lambda p: p["phase_plan"].update(stack_index=0),
            "w0 differs between plan and entry": lambda p: p["phase_plan"].update(w0=0.4),
            "unknown mechanism": lambda p: p["phase_plan"].update(mechanism="hooks"),
            "a not below b": lambda p: p["phase_plan"].update(a=0.7),
            "plan without any phased entry": lambda p: p["stack"][1].update(weight_profile={"kind": "constant", "w": 0.9}),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                p = copy.deepcopy(base)
                mutate(p)
                self.assertTrue(lora.validate_identity_profile(p), name)

    def test_the_schema_is_strict_about_the_plan(self):
        base = phased_profile()
        for name, mutate in {
            "extra key": lambda p: p["phase_plan"].update(extra=1),
            "missing key": lambda p: p["phase_plan"].pop("levels"),
            "levels out of range": lambda p: p["phase_plan"].update(levels=99),
            "a above one": lambda p: p["phase_plan"].update(a=1.5),
            "wrong kind": lambda p: p["phase_plan"].update(kind="linear"),
            "phased entry without w0": lambda p: p["stack"][1]["weight_profile"].pop("w0"),
        }.items():
            with self.subTest(name):
                p = copy.deepcopy(base)
                mutate(p)
                self.assertTrue(schema_errors(p), name)

    def test_check_profile_in_takes_a_phased_profile_that_belongs_to_the_model(self):
        p = phased_profile()
        self.assertIsNone(lora.check_profile_in(p, lora.attachment_json(p["attachment"])))

    def test_the_verified_range_is_checked_at_both_ends(self):
        vs = {"min": 0.5, "max": 1.0, "recommended": 0.8}
        p = phased_profile(w0=0.3, w1=0.9, verified=vs)
        issues = lora.check_stack(p["stack"])
        self.assertEqual([i.code for i in issues], ["lora_strength_outside_verified_range"])
        self.assertEqual(issues[0].detail["strength_model"], 0.3)
        self.assertEqual(lora.check_stack(phased_profile(w0=0.6, w1=0.9, verified=vs)["stack"]), [])
        self.assertEqual(issues[0].detail["index"], 1)

    def test_the_journal_says_phased_and_that_it_goes_last(self):
        line = lora.stack_journal(phased_profile()["stack"])
        self.assertIn("phased 0.3->0.9 over 0.2..0.6", line)
        self.assertIn("applied after the constant ones", line)
        self.assertNotIn("phased", lora.stack_journal([entry(0, "character", "c.safetensors", SHA_A, 0.8)]))

    def test_an_unknown_kind_is_still_refused_in_a_chain(self):
        p = phased_profile()
        p["stack"][0]["weight_profile"] = {"kind": "ramp"}
        with self.assertRaises(errors.UnsupportedFeature):
            lora.check_profile_in(p, lora.attachment_json(p["attachment"]))


if __name__ == "__main__":
    unittest.main()
