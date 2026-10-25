"""Wave 0: turbo contract additions (MP-24, docs/03-turbo.md 5.3/5.4): reason codes, distill table, role distill.

BASELINE_REASONS is a literal snapshot of the REASONS registry at git HEAD 2a09d56 (before MP-24): every row must
still be present, in this order, with this severity and exception-class-ness.
"""
import json
import os
import re
import unittest

from jsonschema import Draft202012Validator

from moss_photon.core import consts
from moss_photon.core.errors import REASON_CODES, REASONS, SEVERITIES, make_issue

BASELINE_REASONS = (
    ("unsupported_model", "error"), ("invalid_model_sampling", "error"), ("conflicting_patch", "error"),
    ("invalid_latent", "error"), ("invalid_dimensions", "error"), ("invalid_conditioning", "error"),
    ("invalid_vae", "error"), ("invalid_layout", "error"), ("missing_required_input", "error"),
    ("unsupported_source_input", "error"), ("invalid_schedule", "error"), ("missing_lora", "error"),
    ("incompatible_lora", "error"), ("invalid_preset", "error"), ("invalid_preset_file", "error"),
    ("preset_version_mismatch", "error"), ("invalid_override", "error"), ("calibration_required", "error"),
    ("budget_exceeded", "error"), ("missing_guard_dependency", "error"), ("non_finite_output", "error"),
    ("unsupported_feature", "error"), ("output_error", "error"),
    ("unscorable_face", "info"), ("guard_not_run", "info"), ("unverified_checkpoint", "warning"),
    ("family_not_verified", "info"), ("quantized_checkpoint_unverified", "warning"),
    ("fingerprint_unavailable", "warning"), ("manifest_missing", "warning"), ("customized_run", "info"),
    ("shift_differs_from_preset", "warning"), ("custom_sigmas", "warning"), ("sigma_step_ratio", "warning"),
    ("multiple_cond_entries", "warning"), ("latent_content_ignored", "info"), ("img2img_not_verified", "warning"),
    ("external_calc_cond_batch", "warning"), ("seed_wrapped", "info"), ("forward_count_mismatch", "warning"),
    ("megapixels_out_of_documented_range", "warning"), ("partial_lora_coverage", "warning"),
    ("lora_low_coverage_forced", "warning"), ("lora_base_mismatch", "warning"), ("lora_layers_missing", "info"),
    ("lora_strength_outside_verified_range", "warning"), ("te_keys_ignored", "warning"),
    ("te_lora_applied", "warning"), ("foreign_patches", "warning"), ("foreign_modifiers", "warning"),
    ("preset_unpinned", "warning"), ("latent_rescaled_by_native_fix", "info"),
    ("multiple_character_loras", "warning"), ("duplicate_lora", "warning"), ("trigger_not_in_prompt", "warning"),
    ("layout_experimental", "info"), ("attribute_conflict", "warning"), ("preview_only_preset", "info"),
    ("non_default_attention_backend", "info"), ("shift_input_ignored", "info"),
    ("zero_strength_lora_skipped", "info"), ("reference_photos_ignored", "warning"),
    ("embedded_unsupported_for_format", "warning"),
)

NEW_CODES = {
    "turbo_setting_outside_verified": "warning",
    "distill_lora_non_turbo": "warning",
    "preset_lora_roles_exceeded": "warning",
    "negative_unused_cfg1": "info",
    "slg_ineffective_cfg1": "info",
    "unverified_distill_lora": "warning",
    # MP-32 (MP-30 review): M6 the recommends_input status, M7 the preset model_families gate, M2 SLG on a turbo preset.
    "reference_photos_recommended": "info",
    "model_family_unverified": "warning",
    "turbo_slg_experimental": "warning",
}

DMD2_SHA = "7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2"
SFT_DMD_SHA = "4abb14e39c1a59018907581e609f43616ec6810126144ec2b1d2e6f8969628fe"

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTRACTS = os.path.join(TESTS_DIR, "fixtures", "contracts")


def _load(*parts):
    with open(os.path.join(CONTRACTS, *parts), "rb") as f:
        return json.loads(f.read().decode("utf-8"))


class OldCodesUnchangedTest(unittest.TestCase):
    def test_baseline_is_a_prefix_of_the_registry(self):
        self.assertEqual(len(BASELINE_REASONS), 63)
        self.assertEqual(tuple(REASON_CODES[:len(BASELINE_REASONS)]), tuple(c for c, _ in BASELINE_REASONS))

    def test_baseline_severities_and_class_presence_unchanged(self):
        for code, sev in BASELINE_REASONS:
            with self.subTest(code=code):
                self.assertEqual(REASONS[code].severity, sev)
                self.assertEqual(REASONS[code].exc is not None, sev == "error")

    def test_registry_is_baseline_plus_new_codes_only(self):
        extra = [c for c in REASON_CODES if c not in dict(BASELINE_REASONS)]
        self.assertEqual(sorted(extra), sorted(NEW_CODES))


class NewCodesTest(unittest.TestCase):
    def test_new_codes_present_with_severity_and_no_class(self):
        for code, sev in NEW_CODES.items():
            with self.subTest(code=code):
                self.assertEqual(REASONS[code].severity, sev)
                self.assertIsNone(REASONS[code].exc)
                self.assertFalse(REASONS[code].in_spec_6_3)

    def test_new_codes_build_issues_and_cannot_be_errors(self):
        for code in NEW_CODES:
            with self.subTest(code=code):
                issue = make_issue(code, "MossPhotonSampler", "cfg", "observed 2.0, verified 1.0-1.1", "set cfg to 1.0")
                self.assertEqual(issue.code, code)
                self.assertIn(issue.severity, SEVERITIES)
                with self.assertRaises(ValueError):
                    make_issue(code, "n", "p", "m", "f", severity="error")

    def test_lora_base_mismatch_downgrades_to_info_on_request(self):
        issue = make_issue("lora_base_mismatch", "MossPhotonLoRAProfile", "lora_name", "trained on base, model sft",
                           "none", severity="info")
        self.assertEqual(issue.severity, "info")
        self.assertEqual(REASONS["lora_base_mismatch"].severity, "warning")


class KnownDistillTableTest(unittest.TestCase):
    def test_keys_are_full_lowercase_sha256(self):
        self.assertEqual(sorted(consts.KNOWN_DISTILL_LORAS), sorted([DMD2_SHA, SFT_DMD_SHA]))
        for key in consts.KNOWN_DISTILL_LORAS:
            self.assertRegex(key, re.compile("^[0-9a-f]{64}$"))

    def test_dmd2_entry(self):
        e = consts.KNOWN_DISTILL_LORAS[DMD2_SHA]
        self.assertEqual(e["name"], "lodestones-8-steps-dmd2-lora")
        self.assertEqual((e["role"], e["status"], e["trained_on"]), ("distill", "verified_recipe", "base"))
        self.assertEqual(e["recipe"], {"steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
                                       "shift": 5.0, "strength": 1.0})

    def test_sft_dmd_entry_is_incompatible_lora(self):
        e = consts.KNOWN_DISTILL_LORAS[SFT_DMD_SHA]
        self.assertEqual(e["name"], "MOSS_Image_SFT_8_steps_dmd_v0.1")
        self.assertEqual((e["role"], e["status"]), ("distill", "incompatible"))
        self.assertEqual(e["reason"], "data-space DMD with own sampler loop")
        self.assertEqual(e["reason_code"], "incompatible_lora")
        self.assertIn(e["reason_code"], REASONS)
        self.assertNotIn("recipe", e)


class DistillRoleSchemaTest(unittest.TestCase):
    def _validator(self, name):
        schema = _load("schemas", name + ".schema.json")
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema)

    def test_turbo_fixture_validates(self):
        doc = _load("identity_profile_turbo.json")
        errors = [e.message for e in self._validator("identity_profile").iter_errors(doc)]
        self.assertEqual(errors, [])
        self.assertEqual([s["role"] for s in doc["stack"]], ["distill", "character"])
        self.assertEqual(doc["summary"]["roles"], ["distill", "character"])
        self.assertEqual(doc["stack"][0]["lora"]["sha256"], DMD2_SHA)
        self.assertEqual(doc["stack"][0]["trained_on"], "base")
        self.assertEqual(doc["summary"]["patch_entries_total"], sum(s["patch_entries"] for s in doc["stack"]))
        self.assertEqual(doc["attachment"]["patch_entries_total"], doc["summary"]["patch_entries_total"])

    def test_distill_role_accepted_in_identity_profile_and_report(self):
        doc = _load("identity_profile.json")
        doc["stack"][0]["role"] = "distill"
        doc["summary"]["roles"] = ["distill"]
        self.assertTrue(self._validator("identity_profile").is_valid(doc))
        rep = _load("report.json")
        rep["lora"]["stack"] = [{
            "role": "distill", "basename": "a.safetensors", "size": 1210000000, "sha256": DMD2_SHA, "strength_model": 1.0,
            "strength_clip": 0.0, "applied_to": {}, "coverage_ratio": 1.0, "layers_missing": [], "keys_patched": 168,
            "patch_entries": 252, "trigger_present": False}]
        self.assertTrue(self._validator("report").is_valid(rep))
        rep["lora"]["stack"][0]["role"] = "villain"
        self.assertFalse(self._validator("report").is_valid(rep))
        rep["lora"]["stack"][0]["role"] = "distill"
        self.assertTrue(self._validator("report").is_valid(rep))

    def test_unknown_role_still_rejected(self):
        doc = _load("identity_profile_turbo.json")
        doc["stack"][0]["role"] = "turbo"
        self.assertFalse(self._validator("identity_profile").is_valid(doc))

    def test_turbo_preset_ids_and_simple_scheduler_accepted(self):
        rep = _load("report.json")
        rep["preset"]["id"] = "turbo_8step"
        self.assertTrue(self._validator("report").is_valid(rep))
        rep["preset"]["id"] = "turbo_other"
        self.assertFalse(self._validator("report").is_valid(rep))
        si = _load("schedule_info.json")
        si["scheduler"] = "simple"
        self.assertTrue(self._validator("schedule_info").is_valid(si))
        si["scheduler"] = "karras"
        self.assertFalse(self._validator("schedule_info").is_valid(si))


if __name__ == "__main__":
    unittest.main()
