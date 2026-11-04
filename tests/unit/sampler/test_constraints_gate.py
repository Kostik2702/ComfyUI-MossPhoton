"""MP-60 (audit F02): the sampler validates the PHOTON_CONSTRAINTS it receives and uses them only for the conditioning
they describe.

* Audit repro C: ``{'blocks_present': {'wardrobe': True}}`` passed the ``wardrobe_safe`` preflight although the validator
  lists 14 missing fields. It is now ``invalid_conditioning`` before the model is touched.
* Constraints whose fingerprint differs from the received positive / negative (another branch, or conditioning changed
  after the prompt node) or carry none: warning ``constraints_conditioning_mismatch``, and every check that reads them
  (block flags, layout extent, trigger words, their warnings, the text-encoder section) is skipped, never passed.
The Scenario of ``test_solver_plan`` passes empty lists as conditioning; ``constraints_fixture`` binds to them."""
import copy
import json
import unittest
from unittest import mock

from moss_photon.core import constraints_contract as cc
from moss_photon.core import errors, report, solver

from .test_solver_plan import (Scenario, Stop, constraints_fixture, good_facts, layout_with_extent, profile_fixture)

MISMATCH = "constraints_conditioning_mismatch"


def elsewhere():
    """Constraints bound to some other conditioning than the Scenario's empty lists."""
    return cc.bind_conditioning(constraints_fixture(bind=False), [["other branch", {}]], [])


class Watched(Scenario):
    """A Scenario that records every report warning of the run."""

    def run(self, **kw):
        self.warned = []
        real = report.ReportBuilder.warn

        def spy(builder, issue, *a, **k):
            self.warned.append(issue if isinstance(issue, dict) else issue.to_dict())
            return real(builder, issue, *a, **k)

        with mock.patch.object(report.ReportBuilder, "warn", spy):
            return super().run(**kw)

    def codes(self):
        return [w["code"] for w in self.warned]

    def mismatch(self):
        found = [w for w in self.warned if w["code"] == MISMATCH]
        self.test.assertEqual(len(found), 1, self.codes())
        return found[0]


class InvalidConstraintsTest(unittest.TestCase):
    def test_audit_repro_c_the_partial_object_is_a_typed_error_before_anything_is_touched(self):
        e = Scenario(self).expect("invalid_conditioning", preset_id="wardrobe_safe",
                                  constraints={"blocks_present": {"wardrobe": True}})
        self.assertIsInstance(e, errors.InvalidConditioning)
        self.assertEqual(e.issue.param, "constraints")
        self.assertEqual(e.issue.detail["problems_total"], 14)  # the 14 missing fields the audit listed
        self.assertIn("missing key", str(e))

    def test_every_planted_defect_of_the_object_stops_the_run(self):
        cases = {
            "not an object": "constraints",
            "wrong kind": dict(constraints_fixture(), kind="other"),
            "blocks_present lies": dict(constraints_fixture(), blocks_present=dict(
                constraints_fixture()["blocks_present"], wardrobe=False)),
            "stale prompt hash": dict(constraints_fixture(), prompt_sha256={"positive": "0" * 64, "negative": "0" * 64}),
            "layout without its space": dict(constraints_fixture(), layout={"items": 1, "extent": [0, 0, 0, 0]}),
            "binding is not hashes": dict(constraints_fixture(), conditioning={"positive_sha256": "x"}),
        }
        for name, value in cases.items():
            with self.subTest(name):
                Scenario(self).expect("invalid_conditioning", constraints=value)

    def test_a_newer_schema_is_unsupported_not_invalid(self):
        Scenario(self).expect("unsupported_feature", constraints=dict(constraints_fixture(), schema_version=2))

    def test_the_message_never_carries_prompt_text(self):
        value = dict(constraints_fixture(), user_text="zkq_secret_prompt", kind="other")
        with self.assertRaises(errors.InvalidConditioning) as cm:
            Scenario(self).run(constraints=value)
        self.assertNotIn("zkq_secret_prompt", str(cm.exception) + json.dumps(cm.exception.to_dict()))

    def test_a_valid_bound_object_passes_and_the_run_reaches_the_clone(self):
        s = Watched(self)
        with self.assertRaises(Stop):
            s.run(constraints=constraints_fixture())
        self.assertNotIn(MISMATCH, s.codes())

    def test_no_constraints_at_all_is_still_fine(self):
        s = Watched(self)
        with self.assertRaises(Stop):
            s.run()
        self.assertNotIn(MISMATCH, s.codes())

    def test_the_input_is_not_mutated(self):
        value = constraints_fixture()
        before = copy.deepcopy(value)
        with self.assertRaises(Stop):
            Scenario(self).run(constraints=value)
        self.assertEqual(value, before)


class MismatchSkipsTheDependentChecksTest(unittest.TestCase):
    def test_a_mismatch_is_one_warning_that_names_the_branch_and_the_skipped_checks(self):
        s = Watched(self)
        with self.assertRaises(Stop):
            s.run(constraints=elsewhere())
        w = s.mismatch()
        self.assertEqual(w["severity"], "warning")
        self.assertEqual(w["detail"]["reason"], "mismatch")
        self.assertEqual(w["detail"]["branches"], ["positive"])
        self.assertEqual(w["detail"]["skipped_checks"], list(cc.SKIPPED_CHECKS))
        self.assertIn("SAME MossPhotonPromptLayout", w["fix"])

    def test_the_empty_wardrobe_block_is_not_judged_on_the_wrong_branch(self):
        c = elsewhere()
        c["blocks_present"]["wardrobe"], c["fields"]["wardrobe"] = False, ""
        s = Watched(self)
        with self.assertRaises(Stop):  # skipped: neither passed nor refused
            s.run(preset_id="wardrobe_safe", constraints=c)
        self.assertIn(MISMATCH, s.codes())

    def test_the_same_empty_block_on_the_right_branch_is_still_refused(self):
        Scenario(self).expect("missing_required_input", preset_id="wardrobe_safe",
                              constraints=constraints_fixture(wardrobe=False))

    def test_a_layout_outside_the_canvas_is_not_judged_on_the_wrong_branch(self):
        c = elsewhere()
        c["layout"] = layout_with_extent([-5000, 0, 0, 0])
        s = Watched(self)
        with self.assertRaises(Stop):
            s.run(constraints=c)
        self.assertIn(MISMATCH, s.codes())
        bound = constraints_fixture()
        bound["layout"] = layout_with_extent([-5000, 0, 0, 0])
        Scenario(self).expect("invalid_layout", constraints=bound)

    def test_the_warnings_of_foreign_constraints_are_not_forwarded(self):
        c = elsewhere()
        c["warnings"] = [{"code": "attribute_conflict", "severity": "warning", "node": "x", "param": "p",
                          "message": "from another prompt", "fix": "f", "detail": None}]
        s = Watched(self)
        with self.assertRaises(Stop):
            s.run(constraints=c)
        self.assertNotIn("attribute_conflict", s.codes())
        bound = constraints_fixture()
        bound["warnings"] = copy.deepcopy(c["warnings"])
        s = Watched(self)
        with self.assertRaises(Stop):
            s.run(constraints=bound)
        self.assertIn("attribute_conflict", s.codes())

    def test_trigger_words_are_not_checked_against_the_wrong_prompt(self):
        profile = profile_fixture()
        profile["stack"][0]["trigger_hints"] = ["zkq"]
        profile["stack"][0]["trigger_present"] = True
        facts = good_facts(lora_stack_attachment=solver.canon.canonical_json(profile["attachment"]),
                           patches_uuid=profile["attachment"]["patches_uuid"],
                           patch_entries=profile["attachment"]["patch_entries_total"])
        bound = constraints_fixture()
        s = Watched(self, facts=facts)
        with self.assertRaises(Stop):
            s.run(constraints=bound, identity_profile=profile)
        self.assertIn("trigger_not_in_prompt", s.codes())  # control: the check runs on the right branch
        c = elsewhere()
        s = Watched(self, facts=facts)
        with self.assertRaises(Stop):
            s.run(constraints=c, identity_profile=profile)
        self.assertNotIn("trigger_not_in_prompt", s.codes())
        self.assertIn(MISMATCH, s.codes())

    def test_unbound_constraints_are_treated_like_a_mismatch(self):
        s = Watched(self)
        c = constraints_fixture(bind=False)
        c["blocks_present"]["scene"], c["fields"]["scene"] = False, ""
        with self.assertRaises(Stop):
            s.run(preset_id="scene_safe", constraints=c)
        w = s.mismatch()
        self.assertEqual((w["detail"]["reason"], w["detail"]["branches"]), ("unbound", ["positive", "negative"]))

    def test_a_negative_branch_mismatch_names_the_negative(self):
        c = cc.bind_conditioning(constraints_fixture(bind=False), [], [["x", {}]])
        s = Watched(self)
        with self.assertRaises(Stop):
            s.run(constraints=c)
        self.assertEqual(s.mismatch()["detail"]["branches"], ["negative"])


class PureHelpersTest(unittest.TestCase):
    def test_check_constraints_input_returns_nothing_for_a_valid_object(self):
        self.assertIsNone(solver.check_constraints_input(constraints_fixture(), node="n"))

    def test_binding_issue_for_unbound_and_mismatch(self):
        a = solver.binding_issue({"reason": "unbound", "branches": ["positive", "negative"]}, node="n")
        b = solver.binding_issue({"reason": "mismatch", "branches": ["negative"]}, node="n")
        self.assertEqual((a.code, a.severity, b.code, b.severity), (MISMATCH, "warning", MISMATCH, "warning"))
        self.assertIn("no fingerprint", a.message)
        self.assertIn("negative conditioning", b.message)
        for issue in (a, b):
            for check in cc.SKIPPED_CHECKS:
                self.assertIn(check, issue.message)


if __name__ == "__main__":
    unittest.main()
