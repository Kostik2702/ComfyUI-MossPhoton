"""Wave 0: core/errors.py against design section 3 (3.1 hierarchy, 3.2 message format, 3.3 status codes).

The tables below are an INDEPENDENT copy of the design tables, so a rename in errors.py is caught here.
"""
import dataclasses
import unittest

from moss_photon.core import errors
from moss_photon.core.errors import (
    REASON_CODES,
    REASONS,
    Issue,
    MossPhotonError,
    make_issue,
    raise_first_error,
    split_issues,
)

# code -> (class name, direct parent class name), design 3.1 tree.
ERROR_CODES = {
    "unsupported_model": ("UnsupportedModel", "ModelError"),
    "invalid_model_sampling": ("InvalidModelSampling", "ModelError"),
    "conflicting_patch": ("ConflictingModelPatch", "ModelError"),
    "invalid_latent": ("InvalidLatent", "InputError"),
    "invalid_dimensions": ("InvalidDimensions", "InputError"),
    "invalid_conditioning": ("InvalidConditioning", "InputError"),
    "invalid_vae": ("InvalidVae", "InputError"),
    "invalid_layout": ("InvalidLayout", "InputError"),
    "missing_required_input": ("MissingRequiredInput", "InputError"),
    "unsupported_source_input": ("UnsupportedSourceInput", "InputError"),
    "invalid_schedule": ("InvalidSchedule", "ScheduleError"),
    "missing_lora": ("MissingLora", "LoraError"),
    "incompatible_lora": ("IncompatibleLora", "LoraError"),
    "invalid_preset": ("UnknownPreset", "PresetError"),
    "invalid_preset_file": ("InvalidPresetFile", "PresetError"),
    "preset_version_mismatch": ("PresetVersionMismatch", "PresetError"),
    "invalid_override": ("InvalidOverride", "PresetError"),
    "calibration_required": ("CalibrationRequired", "PresetError"),
    "budget_exceeded": ("BudgetExceeded", "MossPhotonError"),
    "missing_guard_dependency": ("MissingGuardDependency", "DependencyError"),
    "non_finite_output": ("NonFiniteOutput", "SamplingError"),
    "unsupported_feature": ("UnsupportedFeature", "MossPhotonError"),
    "output_error": ("OutputError", "MossPhotonError"),
}

# intermediate classes -> parent (design 3.1)
INTERMEDIATE = {
    "ModelError": "MossPhotonError",
    "InputError": "MossPhotonError",
    "ScheduleError": "MossPhotonError",
    "LoraError": "MossPhotonError",
    "PresetError": "MossPhotonError",
    "DependencyError": "MossPhotonError",
    "SamplingError": "MossPhotonError",
}

# design 3.3: code -> severity; none of them has an exception class.
STATUS_CODES = {
    "unscorable_face": "info", "guard_not_run": "info", "unverified_checkpoint": "warning",
    "family_not_verified": "info", "quantized_checkpoint_unverified": "warning",
    "fingerprint_unavailable": "warning", "manifest_missing": "warning", "customized_run": "info",
    "shift_differs_from_preset": "warning", "custom_sigmas": "warning", "sigma_step_ratio": "warning",
    "multiple_cond_entries": "warning", "latent_content_ignored": "info", "img2img_not_verified": "warning",
    "external_calc_cond_batch": "warning", "seed_wrapped": "info", "forward_count_mismatch": "warning",
    "megapixels_out_of_documented_range": "warning", "partial_lora_coverage": "warning",
    "lora_low_coverage_forced": "warning", "lora_base_mismatch": "warning", "lora_layers_missing": "info",
    "lora_strength_outside_verified_range": "warning", "te_keys_ignored": "warning",
    "te_lora_applied": "warning", "foreign_patches": "warning", "foreign_modifiers": "warning",
    "preset_unpinned": "warning", "latent_rescaled_by_native_fix": "info",
    "multiple_character_loras": "warning", "duplicate_lora": "warning", "trigger_not_in_prompt": "warning",
    "layout_experimental": "info", "attribute_conflict": "warning", "preview_only_preset": "info",
    "non_default_attention_backend": "info", "shift_input_ignored": "info",
    "zero_strength_lora_skipped": "info", "reference_photos_ignored": "warning",
    "embedded_unsupported_for_format": "warning",
}

# SPEC 6.3 typed reasons (seven)
SPEC_6_3 = {
    "unsupported_model", "missing_lora", "incompatible_lora", "missing_guard_dependency",
    "unscorable_face", "budget_exceeded", "invalid_schedule",
}


def _mk(code="invalid_latent", **kw):
    args = dict(node="MossPhotonSampler", param="latent.samples", message="channels = 4, expected 128",
                fix="use MossPhotonEmptyLatent")
    args.update(kw)
    return make_issue(code, args.pop("node"), args.pop("param"), args.pop("message"), args.pop("fix"), **args)


class RegistryTest(unittest.TestCase):
    def test_every_reason_has_class_and_registry_entry(self):
        for code, (cls_name, parent_name) in ERROR_CODES.items():
            with self.subTest(code=code):
                info = REASONS[code]
                self.assertIsNotNone(info.exc)
                self.assertEqual(info.exc.__name__, cls_name)
                self.assertEqual(info.exc.CODE, code)
                self.assertEqual(info.severity, "error")
                self.assertTrue(issubclass(info.exc, MossPhotonError))
                self.assertIn(parent_name, [b.__name__ for b in info.exc.__bases__])
                self.assertIs(getattr(errors, cls_name), info.exc)

    def test_intermediate_classes(self):
        for name, parent in INTERMEDIATE.items():
            with self.subTest(cls=name):
                cls = getattr(errors, name)
                self.assertEqual([b.__name__ for b in cls.__bases__], [parent])
                self.assertIsNone(cls.CODE)

    def test_status_codes_registered_without_class(self):
        for code, sev in STATUS_CODES.items():
            with self.subTest(code=code):
                info = REASONS[code]
                self.assertIsNone(info.exc)
                self.assertEqual(info.severity, sev)

    def test_spec_6_3_codes_present_and_flagged(self):
        flagged = {c for c, info in REASONS.items() if info.in_spec_6_3}
        self.assertEqual(flagged, SPEC_6_3)

    def test_registry_is_append_only_superset_of_design(self):
        self.assertTrue(set(ERROR_CODES) | set(STATUS_CODES) <= set(REASONS))
        self.assertEqual(len(ERROR_CODES), 23)
        self.assertEqual(len(STATUS_CODES), 40)

    def test_reason_codes_are_registry_keys(self):
        self.assertEqual(tuple(REASON_CODES), tuple(REASONS))

    def test_all_errors_catchable_as_base(self):
        for code in ERROR_CODES:
            with self.subTest(code=code):
                with self.assertRaises(MossPhotonError):
                    raise_first_error([_mk(code)])


class IssueTest(unittest.TestCase):
    def test_issue_format_matches_template(self):
        i = make_issue(
            "invalid_latent", "MossPhotonSampler", "latent.samples", "channels = 4, expected 128",
            "use MossPhotonEmptyLatent or EmptyFlux2LatentImage (EmptyLatentImage makes 4-channel latents)")
        self.assertEqual(
            i.format(),
            "[MossPhoton] MossPhotonSampler/latent.samples: channels = 4, expected 128 -> "
            "use MossPhotonEmptyLatent or EmptyFlux2LatentImage (EmptyLatentImage makes 4-channel latents) "
            "[invalid_latent]")

    def test_issue_format_second_and_third_design_examples(self):
        i = make_issue(
            "missing_lora", "MossPhotonSampler", "identity_profile",
            "preset identity_safe requires a character LoRA profile, none connected",
            "add MossPhotonLoRAProfile (role=character) and connect identity_profile")
        self.assertEqual(
            i.format(),
            "[MossPhoton] MossPhotonSampler/identity_profile: preset identity_safe requires a character "
            "LoRA profile, none connected -> add MossPhotonLoRAProfile (role=character) and connect "
            "identity_profile [missing_lora]")

    def test_format_is_one_line_and_at_most_400_chars(self):
        long_msg = "x" * 900 + "\nsecond line\r\nthird"
        i = _mk(message=long_msg, fix="use MossPhotonEmptyLatent")
        text = i.format()
        self.assertLessEqual(len(text), 400)
        self.assertNotIn("\n", text)
        self.assertNotIn("\r", text)
        self.assertTrue(text.endswith("[invalid_latent]"))
        self.assertTrue(text.startswith("[MossPhoton] MossPhotonSampler/latent.samples: "))
        self.assertIn("-> use MossPhotonEmptyLatent", text)

    def test_format_with_huge_fix_still_bounded_and_keeps_code(self):
        text = _mk(message="m", fix="f" * 2000).format()
        self.assertLessEqual(len(text), 400)
        self.assertTrue(text.endswith("[invalid_latent]"))

    def test_format_untouched_when_short(self):
        text = _mk().format()
        self.assertNotIn("...", text)

    def test_to_dict_without_and_with_detail(self):
        d = _mk().to_dict()
        self.assertEqual(sorted(d), ["code", "fix", "message", "node", "param", "severity"])
        d2 = _mk(detail={"observed": 4, "expected": 128}).to_dict()
        self.assertEqual(d2["detail"], {"observed": 4, "expected": 128})
        self.assertEqual(d2["severity"], "error")

    def test_issue_is_frozen(self):
        i = _mk()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            i.code = "other"

    def test_issue_rejects_bad_severity(self):
        with self.assertRaises(ValueError):
            Issue("invalid_latent", "fatal", "n", "p", "m", "f")


class MakeIssueTest(unittest.TestCase):
    def test_default_severity_comes_from_registry(self):
        self.assertEqual(_mk("invalid_latent").severity, "error")
        self.assertEqual(_mk("preset_unpinned").severity, "warning")
        self.assertEqual(_mk("family_not_verified").severity, "info")

    def test_status_codes_cannot_be_errors(self):
        with self.assertRaises(ValueError):
            make_issue("unscorable_face", "n", "p", "m", "f", severity="error")
        with self.assertRaises(ValueError):
            make_issue("foreign_patches", "n", "p", "m", "f", severity="error")

    def test_status_code_severity_can_be_overridden_to_non_error(self):
        self.assertEqual(_mk("custom_sigmas", severity="info").severity, "info")

    def test_error_code_may_be_downgraded(self):
        self.assertEqual(_mk("missing_guard_dependency", severity="warning").severity, "warning")

    def test_unknown_code_and_bad_severity_rejected(self):
        with self.assertRaises(ValueError):
            make_issue("no_such_code", "n", "p", "m", "f")
        with self.assertRaises(ValueError):
            make_issue("invalid_latent", "n", "p", "m", "f", severity="fatal")


class ExceptionTest(unittest.TestCase):
    def test_str_equals_issue_format_and_attributes(self):
        issue = _mk("invalid_schedule", node="MossPhotonSchedule", param="steps")
        exc = errors.InvalidSchedule(issue)
        self.assertEqual(str(exc), issue.format())
        self.assertIs(exc.issue, issue)
        self.assertEqual(exc.code, "invalid_schedule")
        self.assertEqual(exc.to_dict(), issue.to_dict())

    def test_requires_an_issue(self):
        with self.assertRaises(TypeError):
            errors.InvalidLatent("channels = 4")


class RaiseFirstErrorTest(unittest.TestCase):
    def test_raise_first_error_picks_first_error(self):
        w = _mk("preset_unpinned")
        e1 = _mk("invalid_schedule")
        e2 = _mk("invalid_latent")
        with self.assertRaises(errors.InvalidSchedule) as cm:
            raise_first_error([w, e1, e2])
        self.assertIs(cm.exception.issue, e1)

    def test_class_per_code(self):
        for code, (cls_name, _) in ERROR_CODES.items():
            with self.subTest(code=code):
                with self.assertRaises(getattr(errors, cls_name)):
                    raise_first_error([_mk("custom_sigmas"), _mk(code)])

    def test_no_error_returns_none(self):
        self.assertIsNone(raise_first_error([]))
        self.assertIsNone(raise_first_error([_mk("preset_unpinned"), _mk("family_not_verified")]))

    def test_split_issues_keeps_order(self):
        a, b, c, d = _mk("invalid_latent"), _mk("preset_unpinned"), _mk("invalid_schedule"), _mk("seed_wrapped")
        errs, others = split_issues([a, b, c, d])
        self.assertEqual((errs, others), ([a, c], [b, d]))
        self.assertIsInstance(errs, list)
        self.assertIsInstance(others, list)


if __name__ == "__main__":
    unittest.main()
