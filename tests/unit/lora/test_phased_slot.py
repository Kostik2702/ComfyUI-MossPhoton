"""MP-73: who keeps the chain's one phased profile (``lora.phased_slot_action``) and the ``phased_lora_demoted`` warning."""
import unittest

from moss_photon.core import errors, lora, phase

from .builders import DMD2, entry


def chain_profile(roles, phased_index=None, shas=None):
    stack = []
    for i, role in enumerate(roles):
        sha = (shas or {}).get(i, ("%x" % (i + 1)) * 64)
        stack.append(entry(i, role, "f%d.safetensors" % i, sha, 0.8))
    plan = None
    if phased_index is not None:
        plan = phase.make_plan(phased_index, 0.3, 0.9, 0.2, 0.6)
        stack[phased_index]["weight_profile"] = phase.weight_profile_of(plan)
    on_model = sum(e["patch_entries"] for e in stack if e["weight_profile"]["kind"] != "phased")
    att = lora.build_attachment([lora.attachment_entry_of(e) for e in stack], patches_uuid="u-1",
                                patch_entries_total=on_model, baseline_entries=0)
    return lora.build_identity_profile(stack, attachment=att, model_identity=None,
                                       phase_plan=phase.plan_to_dict(plan) if plan else None)


class SlotActionTest(unittest.TestCase):
    def test_no_phased_entry_yet_takes_the_slot(self):
        for profile in (None, chain_profile([]), chain_profile(["style", "character"])):
            for is_character in (True, False):
                with self.subTest(profile=profile and len(profile["stack"]), character=is_character):
                    self.assertEqual(lora.phased_slot_action(profile, is_character), "take")

    def test_a_character_takes_the_slot_from_a_non_character(self):
        for role in ("style", "wardrobe", "scene", "other", "distill"):
            with self.subTest(prior=role):
                self.assertEqual(lora.phased_slot_action(chain_profile([role], 0), True), "demote_prior")

    def test_everything_else_demotes_the_new_request(self):
        cases = (("character", True), ("character", False), ("style", False), ("distill", False))
        for prior_role, new_is_character in cases:
            with self.subTest(prior=prior_role, new_character=new_is_character):
                self.assertEqual(lora.phased_slot_action(chain_profile([prior_role], 0), new_is_character), "demote_new")

    def test_the_phased_entry_is_found_by_the_plan_not_by_position(self):
        profile = chain_profile(["distill", "style", "character", "scene"], 1)
        self.assertEqual(lora.phased_slot_action(profile, True), "demote_prior")
        profile = chain_profile(["distill", "style", "character", "scene"], 2)
        self.assertEqual(lora.phased_slot_action(profile, True), "demote_new")

    def test_a_known_distill_file_is_never_the_character_whatever_its_role(self):
        profile = chain_profile(["character"], 0, shas={0: DMD2})
        self.assertEqual(lora.phased_slot_action(profile, True), "demote_prior")


class DemotedIssueTest(unittest.TestCase):
    def test_each_reason_is_a_one_line_warning_in_field_names(self):
        cases = {"another_phased_lora": {}, "apply_text_encoder": {}, "invalid_window": {"a": 0.6, "b": 0.2}}
        for reason, detail in cases.items():
            with self.subTest(reason=reason):
                issue = lora.phased_demoted_issue(reason, 3, 0.45, **detail)
                self.assertEqual((issue.code, issue.severity, issue.param), ("phased_lora_demoted", "warning", "profile"))
                self.assertIn("stack[3]", issue.message)
                self.assertIn("strength_model 0.45", issue.message)
                self.assertEqual(issue.detail["reason"], reason)
                self.assertEqual(issue.detail["index"], 3)
                self.assertNotIn("\n", issue.format())
                self.assertNotIn(".safetensors", issue.format())
                self.assertFalse(issue.fix.startswith(("none", "None")))

    def test_the_code_can_never_be_an_error(self):
        with self.assertRaises(ValueError):
            errors.make_issue("phased_lora_demoted", "n", "p", "m", "f", severity="error")
        with self.assertRaises(ValueError):
            errors.make_issue("lora_setting_ignored", "n", "p", "m", "f", severity="error")


if __name__ == "__main__":
    unittest.main()
