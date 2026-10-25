"""solver.classify_label: the parity ladder of A21 and the turbo class, as a table (design 6.4 test_parity_class_matrix)."""
import unittest

from moss_photon.core import presets, solver

REF = presets.catalog()["reference"]["sampling"]


def label(preset_id="reference", overrides=None, **kw):
    resolved = presets.resolve(preset_id, overrides=overrides)
    args = dict(preset=resolved.preset, effective=resolved.effective, customized=resolved.customized,
                reference_sampling=REF, steps_used=resolved.effective["sampling"]["steps"],
                cfg=resolved.effective["sampling"]["cfg"], live_shift=5.0, sigmas_bit_equal_native=True,
                noise_policy="native", modifiers=(), foreign_patches=False, family_verified=True, has_lora=False,
                slg_applied=resolved.effective["sampling"]["slg_mode"] == "uncond_skip"
                and resolved.effective["sampling"]["cfg"] != 1.0, calibrated=False)
    args.update(kw)
    return solver.classify_label(**args)


def parity(*a, **kw):
    return label(*a, **kw)["parity_class"]


def reasons(*a, **kw):
    return label(*a, **kw)["not_reference_reasons"]


class LadderTest(unittest.TestCase):
    def test_reference_exact(self):
        self.assertEqual(parity(), "reference_exact")
        self.assertEqual(reasons(), [])

    def test_unverified_family(self):
        self.assertEqual(parity(family_verified=False), "reference_unverified")

    def test_with_lora(self):
        self.assertEqual(parity(has_lora=True), "reference_with_lora")

    def test_unverified_beats_lora(self):
        self.assertEqual(parity(has_lora=True, family_verified=False), "reference_unverified")

    def test_foreign_patches_beat_unverified_and_lora(self):
        self.assertEqual(parity(foreign_patches=True, family_verified=False, has_lora=True),
                         "reference_with_foreign_patches")

    def test_modifiers_beat_foreign_patches(self):
        self.assertEqual(parity(modifiers=("post_cfg_function",), foreign_patches=True),
                         "reference_modified")
        self.assertEqual(label(modifiers=("post_cfg_function",))["modifiers"], ["post_cfg_function"])

    def test_not_reference_beats_everything(self):
        self.assertEqual(parity(sigmas_bit_equal_native=False, modifiers=("x",), foreign_patches=True,
                                family_verified=False), "not_reference")

    def test_candidate_presets_are_numerically_reference(self):
        for pid in ("photo_safe", "identity_safe", "wardrobe_safe", "scene_safe", "night_photo", "advanced"):
            with self.subTest(pid=pid):
                self.assertEqual(parity(pid), "reference_exact")


class NotReferenceReasonsTest(unittest.TestCase):
    def test_each_reason(self):
        self.assertEqual(reasons(overrides={"cfg": 4.0}), ["cfg_differs", "customized"])
        self.assertEqual(reasons(overrides={"steps": 30}, steps_used=30), ["steps_differ", "customized"])
        self.assertEqual(reasons(overrides={"slg_mode": "off"}), ["slg_differs", "customized"])
        self.assertEqual(reasons(overrides={"slg_layers": [21]}), ["slg_differs", "customized"])
        self.assertEqual(reasons(overrides={"slg_end": 0.5}), ["slg_differs", "customized"])
        self.assertEqual(reasons(sigmas_bit_equal_native=False), ["custom_sigmas"])
        self.assertEqual(reasons(live_shift=3.0), ["shift_differs"])
        self.assertEqual(reasons(noise_policy="seed_plus_i"), ["noise_seed_plus_i"])

    def test_a_phased_lora_weight_is_not_reference_with_its_own_reason(self):
        self.assertEqual(reasons(has_lora=True, phased=True), ["lora_phased"])
        self.assertEqual(parity(has_lora=True, phased=True), "not_reference")
        self.assertEqual(parity(has_lora=True), "reference_with_lora")  # control: the same run without the flag
        self.assertEqual(reasons(customized=True, phased=True), ["lora_phased", "customized"])

    def test_a_phased_lora_weight_demotes_a_turbo_run_too(self):
        block = label("turbo_8step", steps_used=8, cfg=1.0, has_lora=True, slg_applied=False, phased=True)
        self.assertEqual((block["parity_class"], block["not_reference_reasons"]), ("not_reference", ["lora_phased"]))

    def test_draft_fast(self):
        self.assertEqual(parity("draft_fast", steps_used=25), "not_reference")
        self.assertEqual(reasons("draft_fast", steps_used=25), ["steps_differ", "draft_preset"])

    def test_steps_used_is_what_ran_not_what_the_preset_says(self):
        self.assertEqual(reasons(steps_used=10), ["steps_differ"])

    def test_cfg_equal_to_the_reference_is_not_a_reason(self):
        self.assertEqual(reasons(overrides={"cfg": 3.0}), [])

    def test_customized_alone(self):
        self.assertEqual(reasons(customized=True), ["customized"])
        self.assertEqual(label(customized=True)["ui_label"], "stable+customized")
        self.assertEqual(label()["ui_label"], "stable")


class BlockTest(unittest.TestCase):
    def test_block_matches_the_report_contract_keys(self):
        self.assertEqual(set(label()), {"preset_status", "ui_label", "parity_class", "not_reference_reasons",
                                        "modifiers", "calibrated", "customized", "recommended_final"})

    def test_values_come_from_the_preset(self):
        block = label("night_photo")
        self.assertEqual((block["preset_status"], block["recommended_final"]), ("experimental", False))
        self.assertEqual(label("reference")["recommended_final"], True)
        self.assertEqual(label(calibrated=True)["calibrated"], True)


class TurboTest(unittest.TestCase):
    def test_is_turbo(self):
        self.assertTrue(solver.is_turbo({"id": "turbo_8step", "lora_policy": "require_distill_profile"}))
        self.assertTrue(solver.is_turbo({"id": "x", "lora_policy": "require_distill_and_character_profile"}))
        self.assertTrue(solver.is_turbo({"id": "turbo_anything", "lora_policy": "use_explicit_user_profile_or_none"}))
        self.assertFalse(solver.is_turbo(presets.catalog()["reference"]))
        self.assertFalse(solver.is_turbo(presets.catalog()["advanced"]))

    def turbo(self, **kw):
        base = dict(steps_used=8, cfg=1.0, has_lora=True, slg_applied=False)
        base.update(kw)
        return label("turbo_8step", **base)

    def test_clean_turbo_run_is_native_turbo(self):
        block = self.turbo()
        self.assertEqual(block["parity_class"], solver.TURBO_PARITY_CLASS)
        self.assertEqual(solver.TURBO_PARITY_CLASS, "native_turbo")
        self.assertEqual(block["not_reference_reasons"], [])
        self.assertFalse(block["recommended_final"])

    def test_a_customized_turbo_run_keeps_native_turbo_and_says_what_differs(self):
        """MP-30 M1 (lead decision): the native graph with the same steps / cfg is still the basis, so the class stays
        ``native_turbo``; the label records customized=true and the reasons steps_differ / cfg_differs /
        turbo_setting_outside_verified instead of overstating comparability."""
        block = self.turbo(steps_used=16, cfg=2.0, customized=True)
        self.assertEqual(block["parity_class"], "native_turbo")
        self.assertTrue(block["customized"])
        self.assertEqual(block["not_reference_reasons"], ["steps_differ", "cfg_differs", "turbo_setting_outside_verified"])
        self.assertEqual(block["ui_label"], "candidate+customized")

    def test_customized_turbo_reasons_one_axis_at_a_time(self):
        self.assertEqual(self.turbo(steps_used=12)["not_reference_reasons"],
                         ["steps_differ", "turbo_setting_outside_verified"])
        self.assertEqual(self.turbo(cfg=1.5)["not_reference_reasons"], ["cfg_differs", "turbo_setting_outside_verified"])
        # cfg 1.05 differs from the preset but is inside the verified range [1.0, 1.1]
        self.assertEqual(self.turbo(cfg=1.05)["not_reference_reasons"], ["cfg_differs"])
        self.assertEqual(self.turbo(cfg=1.05)["parity_class"], "native_turbo")
        self.assertEqual(self.turbo(steps_used=8, cfg=1.0)["not_reference_reasons"], [])

    def test_the_reasons_come_from_the_values_not_from_the_customized_flag(self):
        self.assertEqual(self.turbo(customized=True)["not_reference_reasons"], [])
        self.assertTrue(self.turbo(customized=True)["customized"])

    def test_turbo_identity_uses_its_own_verified_range(self):
        block = label("turbo_identity", steps_used=16, cfg=1.0, has_lora=True, slg_applied=False)
        self.assertEqual(block["not_reference_reasons"], ["steps_differ", "turbo_setting_outside_verified"])

    def test_custom_sigmas_seed_plus_i_shift_and_applied_slg_break_it(self):
        self.assertEqual(self.turbo(sigmas_bit_equal_native=False)["not_reference_reasons"], ["custom_sigmas"])
        self.assertEqual(self.turbo(noise_policy="seed_plus_i")["not_reference_reasons"], ["noise_seed_plus_i"])
        self.assertEqual(self.turbo(live_shift=3.0)["not_reference_reasons"], ["shift_differs"])
        self.assertEqual(self.turbo(slg_applied=True)["not_reference_reasons"], ["slg_differs"])
        self.assertEqual(self.turbo(slg_applied=True)["parity_class"], "not_reference")

    def test_applied_slg_with_a_customized_recipe_is_not_reference_and_lists_both(self):
        """MP-30 M2: the Advanced node may put SLG on a turbo preset; the run is then not the native turbo graph."""
        block = self.turbo(slg_applied=True, cfg=1.5)
        self.assertEqual(block["parity_class"], "not_reference")
        self.assertEqual(block["not_reference_reasons"], ["cfg_differs", "turbo_setting_outside_verified", "slg_differs"])

    def test_a_model_family_outside_the_preset_families_is_not_reference_for_turbo(self):
        """MP-30 M7: turbo is verified on SFT only; a Base checkpoint is not the native turbo basis."""
        block = self.turbo(family_in_preset=False)
        self.assertEqual(block["parity_class"], "not_reference")
        self.assertEqual(block["not_reference_reasons"], ["model_family_unverified"])

    def test_the_family_gate_has_a_default_that_changes_nothing(self):
        self.assertEqual(self.turbo(family_in_preset=True)["parity_class"], "native_turbo")

    def test_ladder_order_holds_for_turbo(self):
        self.assertEqual(self.turbo(modifiers=("x",))["parity_class"], "reference_modified")
        self.assertEqual(self.turbo(foreign_patches=True)["parity_class"], "reference_with_foreign_patches")
        self.assertEqual(self.turbo(family_verified=False)["parity_class"], "reference_unverified")


if __name__ == "__main__":
    unittest.main()
