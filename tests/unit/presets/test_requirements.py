"""MP-7: evaluate_requirements (design 1.3)."""
import unittest

from moss_photon.core import errors, presets


def issues(pid, **ctx):
    return presets.evaluate_requirements(presets.resolve(pid), ctx)


def pairs(found):
    return [(i.code, i.severity) for i in found]


class RequirementsTest(unittest.TestCase):
    def test_identity_safe_without_and_with_character_lora(self):
        got = issues("identity_safe", lora_roles=[])
        self.assertEqual(pairs(got), [("missing_lora", "error"), ("reference_photos_recommended", "info")])
        self.assertEqual(got[0].param, "identity_profile")
        self.assertIn("MossPhotonLoRAProfile", got[0].fix)
        self.assertEqual(pairs(issues("identity_safe", lora_roles=["style"])), [("missing_lora", "error"), ("reference_photos_recommended", "info")])
        self.assertEqual(pairs(issues("identity_safe", lora_roles=["style", "character"])), [("reference_photos_recommended", "info")])
        self.assertEqual(issues("identity_safe", lora_roles=["character"], reference_photos=True), [])

    def test_error_issue_raises_the_registered_exception(self):
        with self.assertRaises(errors.MissingLora):
            errors.raise_first_error(issues("identity_safe", lora_roles=[]))

    def test_wardrobe_safe_empty_and_filled_block(self):
        self.assertEqual(pairs(issues("wardrobe_safe", blocks_present={"wardrobe": False})),
                         [("missing_required_input", "error")])
        self.assertEqual(pairs(issues("wardrobe_safe")), [("missing_required_input", "error")])
        self.assertEqual(pairs(issues("wardrobe_safe", blocks_present={"wardrobe": True, "scene": False})), [])

    def test_scene_safe_block_latent_and_mask(self):
        self.assertEqual(pairs(issues("scene_safe", blocks_present={"scene": True})), [])
        self.assertEqual(pairs(issues("scene_safe", blocks_present={"scene": False})), [("missing_required_input", "error")])
        self.assertEqual(pairs(issues("scene_safe", blocks_present={"scene": True}, latent_is_empty=False)),
                         [("unsupported_source_input", "error")])
        self.assertEqual(pairs(issues("scene_safe", blocks_present={"scene": True}, has_noise_mask=True)),
                         [("unsupported_source_input", "error")])
        self.assertEqual(pairs(issues("wardrobe_safe", blocks_present={"wardrobe": True}, latent_is_empty=False)),
                         [("unsupported_source_input", "error")])

    def test_presets_without_constraints_give_no_issue(self):
        for pid in ("reference", "photo_safe", "night_photo", "draft_fast", "advanced"):
            self.assertEqual(issues(pid, lora_roles=[], blocks_present={}, latent_is_empty=False, has_noise_mask=True), [], pid)

    def test_recommends_input_is_info_and_not_an_error(self):
        got = issues("identity_safe", lora_roles=["character"])
        self.assertEqual(pairs(got), [("reference_photos_recommended", "info")])
        self.assertEqual(got[0].param, "reference_photos")
        self.assertIsNone(errors.raise_first_error(got))
        self.assertTrue(got[0].format().startswith("[MossPhoton] "))

    def test_the_recommendation_uses_a_registered_status_code_and_points_to_the_diagnostics_node(self):
        """MP-30 M6/m7: the literal code "none" was outside the registry and the fix sent the user to a sampler
        socket that does not exist; reference photos are connected to MossPhotonDiagnostics."""
        got = issues("identity_safe", lora_roles=["character"])
        self.assertEqual(got[0].code, "reference_photos_recommended")
        self.assertIn("reference_photos_recommended", errors.REASONS)
        self.assertEqual(errors.REASONS["reference_photos_recommended"].severity, "info")
        self.assertIsNone(errors.REASONS["reference_photos_recommended"].exc)
        self.assertIn("MossPhotonDiagnostics", got[0].fix)
        self.assertIn("reference_images", got[0].fix)
        self.assertNotIn("MossPhotonSampler", got[0].fix)

    def test_every_code_evaluate_requirements_can_emit_is_registered(self):
        """The class of the M6 defect: sweep every preset against contexts that trip each constraint kind."""
        contexts = [{}, {"lora_roles": ["distill"]}, {"lora_roles": ["character"]}, {"lora_roles": ["distill", "character"]},
                    {"lora_roles": ["distill", "style"], "blocks_present": {"wardrobe": True, "scene": True}},
                    {"latent_is_empty": False, "has_noise_mask": True}, {"reference_photos": True}]
        seen = set()
        for pid in presets.ALL_PRESET_IDS:
            resolved = presets.resolve(pid)
            for ctx in contexts:
                for issue in presets.evaluate_requirements(resolved, ctx):
                    seen.add(issue.code)
                    self.assertIn(issue.code, errors.REASONS, "%s %s" % (pid, ctx))
        self.assertIn("reference_photos_recommended", seen)  # control: the sweep reached the old offender

    def test_node_name(self):
        r = presets.resolve("identity_safe")
        self.assertEqual(presets.evaluate_requirements(r, {}, node="MossPhotonSamplerAdvanced")[0].node,
                         "MossPhotonSamplerAdvanced")


if __name__ == "__main__":
    unittest.main()
