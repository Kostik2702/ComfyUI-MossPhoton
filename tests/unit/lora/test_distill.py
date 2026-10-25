"""Distill role and the known-distill table (docs/03-turbo.md 5.4, Decisions Q1/Q2/Q13; consts.KNOWN_DISTILL_LORAS)."""
import unittest

from moss_photon.core import consts, errors, lora

from .builders import DMD2, SFT_DMD, SHA_A


class DistillTableTest(unittest.TestCase):
    def test_the_two_local_hashes_are_in_the_table(self):
        self.assertEqual(lora.distill_entry(DMD2)["status"], "verified_recipe")
        self.assertEqual(lora.distill_entry(DMD2)["trained_on"], "base")
        self.assertEqual(lora.distill_entry(SFT_DMD)["status"], "incompatible")
        self.assertIsNone(lora.distill_entry(SHA_A))
        self.assertIsNone(lora.distill_entry(None))

    def test_entry_is_a_copy(self):
        row = lora.distill_entry(DMD2)
        row["recipe"]["strength"] = 99
        self.assertEqual(consts.KNOWN_DISTILL_LORAS[DMD2]["recipe"]["strength"], 1.0)


class DistillIssuesTest(unittest.TestCase):
    def test_lodestones_is_verified_recipe_and_raises_nothing(self):
        self.assertEqual(lora.distill_issues("distill", DMD2, "lodestones-8-steps-dmd2-lora.safetensors"), [])

    def test_sft_dmd_file_is_incompatible_lora_in_any_role(self):
        for role in lora.ROLES:
            with self.subTest(role=role):
                issues = lora.distill_issues(role, SFT_DMD, "MOSS_Image_SFT_8_steps_dmd_v0.1.safetensors")
                self.assertEqual([(i.code, i.severity) for i in issues], [("incompatible_lora", "error")])
                with self.assertRaises(errors.IncompatibleLora):
                    errors.raise_first_error(issues)
        text = issues[0].format()
        self.assertIn("MOSS_Image_SFT_8_steps_dmd_v0.1", text)
        self.assertIn("data-space DMD", text)
        self.assertLessEqual(len(text), errors.MAX_MESSAGE_CHARS)

    def test_unknown_file_declared_distill_is_a_warning(self):
        issues = lora.distill_issues("distill", SHA_A, "some_distill.safetensors")
        self.assertEqual([(i.code, i.severity) for i in issues], [("unverified_distill_lora", "warning")])
        errors.raise_first_error(issues)  # a warning never raises

    def test_hash_not_computed_is_unverified_too(self):
        issues = lora.distill_issues("distill", None, "some_distill.safetensors")
        self.assertEqual([i.code for i in issues], ["unverified_distill_lora"])
        self.assertIn("not computed", issues[0].message)

    def test_unknown_file_in_another_role_is_silent(self):
        for role in lora.ROLES:
            if role != "distill":
                self.assertEqual(lora.distill_issues(role, SHA_A, "x.safetensors"), [])

    def test_known_verified_file_in_another_role_is_not_an_issue_here(self):
        self.assertEqual(lora.distill_issues("character", DMD2, "lodestones.safetensors"), [])

    def test_role_distill_exists_in_the_profile_vocabulary(self):
        self.assertIn("distill", lora.ROLES)
        self.assertEqual(lora.ROLES[-1], "distill")  # appended after the five v1 roles, node combo order of 4.4


class LoraBaseMismatchSeverityTest(unittest.TestCase):
    def test_registry_default_stays_warning_and_the_emitter_may_lower_it(self):
        self.assertEqual(errors.REASONS["lora_base_mismatch"].severity, "warning")
        issue = errors.make_issue("lora_base_mismatch", lora.NODE, "lora_name", "m", "f", severity="info")
        self.assertEqual(issue.severity, "info")


if __name__ == "__main__":
    unittest.main()
