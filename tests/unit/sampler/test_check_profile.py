"""solver.check_profile: the four-step profile / MODEL comparison of design 1.5, on dicts and plain facts."""
import copy
import json
import os
import types
import unittest

from moss_photon.core import canon, errors, solver

NODE = "MossPhotonSampler"
CONTRACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "fixtures", "contracts")


def fixture(name="identity_profile.json"):
    with open(os.path.join(CONTRACTS, name), "r", encoding="utf-8") as f:
        return json.load(f)


def facts_for(profile, **kw):
    """Facts of the MODEL a MossPhotonLoRAProfile chain would have produced for ``profile``."""
    att = profile["attachment"]
    base = dict(lora_stack_attachment=canon.canonical_json(att), patches_uuid=att["patches_uuid"],
                patch_entries=att["patch_entries_total"])
    base.update(kw)
    return types.SimpleNamespace(**base)


def clean_facts(**kw):
    base = dict(lora_stack_attachment=None, patches_uuid="u0", patch_entries=0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def codes(check):
    return [i.code for i in check.issues]


class NoProfileTest(unittest.TestCase):
    def test_clean_model(self):
        check = solver.check_profile(None, clean_facts(), False, node=NODE)
        self.assertEqual((codes(check), check.foreign, check.has_lora), ([], False, False))
        self.assertEqual(check.telemetry, {"entries_expected": 0, "entries_observed": 0, "baseline_entries": 0,
                                           "uuid_match": True, "extra_entries": 0, "keys_sample": []})

    def test_any_patch_entry_is_foreign(self):
        check = solver.check_profile(None, clean_facts(patch_entries=252), False, node=NODE)
        self.assertEqual(codes(check), ["foreign_patches"])
        self.assertEqual(check.issues[0].severity, "warning")
        self.assertTrue(check.foreign)
        self.assertEqual((check.telemetry["entries_observed"], check.telemetry["extra_entries"]), (252, 252))
        self.assertFalse(check.telemetry["uuid_match"])

    def test_preset_requiring_a_lora_gets_missing_lora(self):
        check = solver.check_profile(None, clean_facts(), True, node=NODE)
        self.assertEqual(codes(check), ["missing_lora"])
        with self.assertRaises(errors.MissingLora):
            errors.raise_first_error(check.issues)

    def test_an_empty_stack_counts_as_no_profile(self):
        empty = fixture()
        empty["stack"] = []
        self.assertEqual(codes(solver.check_profile(empty, clean_facts(), True, node=NODE)), ["missing_lora"])
        self.assertEqual(codes(solver.check_profile(empty, clean_facts(), False, node=NODE)), [])
        self.assertEqual(codes(solver.check_profile(empty, clean_facts(patch_entries=3), False, node=NODE)),
                         ["foreign_patches"])


    def test_the_empty_profile_of_a_pass_through_node_is_not_incompatible(self):
        """MP-10: lora_name none -> same MODEL, stack [], attachment.entries [], baseline = entries already there, and no
        attachment string on the MODEL."""
        empty = fixture()
        empty["stack"] = []
        empty["attachment"].update({"entries": [], "patch_entries_total": 0, "baseline_entries": 0})
        check = solver.check_profile(empty, clean_facts(), False, node=NODE)
        self.assertEqual((codes(check), check.foreign, check.has_lora), ([], False, False))
        native = solver.check_profile(empty, clean_facts(patch_entries=252), False, node=NODE)
        self.assertEqual(codes(native), ["foreign_patches"])
        # entries == [] alone is enough, whatever the stack says
        only_entries = fixture()
        only_entries["attachment"]["entries"] = []
        self.assertEqual(codes(solver.check_profile(only_entries, clean_facts(), True, node=NODE)), ["missing_lora"])


class WithProfileTest(unittest.TestCase):
    def setUp(self):
        self.profile = fixture()

    def test_clean_chain_is_a_lora_run(self):
        check = solver.check_profile(self.profile, facts_for(self.profile), True, node=NODE)
        self.assertEqual((codes(check), check.foreign, check.has_lora), ([], False, True))
        tel = check.telemetry
        self.assertEqual((tel["entries_expected"], tel["entries_observed"], tel["extra_entries"], tel["uuid_match"]),
                         (252, 252, 0, True))

    def test_attachment_missing_on_the_model(self):
        check = solver.check_profile(self.profile, facts_for(self.profile, lora_stack_attachment=None), True, node=NODE)
        self.assertEqual(codes(check), ["incompatible_lora"])
        with self.assertRaises(errors.IncompatibleLora):
            errors.raise_first_error(check.issues)

    def test_attachment_of_another_chain(self):
        other = copy.deepcopy(self.profile)
        other["attachment"]["entries"][0]["strength_model"] = 0.5
        facts = facts_for(self.profile, lora_stack_attachment=canon.canonical_json(other["attachment"]))
        self.assertEqual(codes(solver.check_profile(self.profile, facts, True, node=NODE)), ["incompatible_lora"])

    def test_the_comparison_is_the_exact_canonical_string(self):
        facts = facts_for(self.profile, lora_stack_attachment=json.dumps(self.profile["attachment"], indent=2))
        self.assertEqual(codes(solver.check_profile(self.profile, facts, True, node=NODE)), ["incompatible_lora"])

    def test_foreign_loader_after_the_chain_by_entry_count(self):
        facts = facts_for(self.profile, patch_entries=252 + 144)
        check = solver.check_profile(self.profile, facts, True, node=NODE)
        self.assertEqual(codes(check), ["foreign_patches"])
        self.assertTrue(check.foreign)
        self.assertEqual((check.telemetry["entries_expected"], check.telemetry["entries_observed"],
                          check.telemetry["extra_entries"]), (252, 396, 144))

    def test_foreign_loader_after_the_chain_by_uuid_alone(self):
        facts = facts_for(self.profile, patches_uuid="someone-else")
        check = solver.check_profile(self.profile, facts, True, node=NODE)
        self.assertEqual(codes(check), ["foreign_patches"])
        self.assertFalse(check.telemetry["uuid_match"])
        self.assertEqual(check.telemetry["extra_entries"], 0)

    def test_baseline_entries_are_foreign_patches_from_before_the_chain(self):
        self.profile["attachment"]["baseline_entries"] = 144
        check = solver.check_profile(self.profile, facts_for(self.profile), True, node=NODE)
        self.assertEqual(codes(check), ["foreign_patches"])
        self.assertEqual(check.telemetry["baseline_entries"], 144)
        self.assertEqual(check.telemetry["extra_entries"], 0)

    def test_telemetry_carries_the_exact_keys_of_the_report(self):
        keys = set(fixture("report.json")["lora"]["foreign_patches"])
        self.assertEqual(set(solver.check_profile(self.profile, facts_for(self.profile), True, node=NODE).telemetry), keys)

    def test_profile_and_facts_are_not_mutated(self):
        before = copy.deepcopy(self.profile)
        solver.check_profile(self.profile, facts_for(self.profile, patch_entries=999), True, node=NODE)
        self.assertEqual(self.profile, before)

    def test_not_a_profile(self):
        for bad in ({}, {"kind": "photon_report"}, {"kind": "photon_identity_profile", "stack": "x", "attachment": {}}):
            with self.subTest(bad=bad):
                check = solver.check_profile(bad, clean_facts(), False, node=NODE)
                self.assertEqual(codes(check), ["incompatible_lora"])

    def test_newer_schema_version(self):
        newer = dict(self.profile, schema_version=99)
        check = solver.check_profile(newer, facts_for(self.profile), False, node=NODE)
        self.assertEqual(codes(check), ["unsupported_feature"])

    def test_the_turbo_fixture_chain_is_clean(self):
        turbo = fixture("identity_profile_turbo.json")
        check = solver.check_profile(turbo, facts_for(turbo), True, node=NODE)
        self.assertEqual((codes(check), check.foreign, check.has_lora), ([], False, True))


if __name__ == "__main__":
    unittest.main()
