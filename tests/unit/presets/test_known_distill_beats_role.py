"""MP-60 (audit F03): a file whose sha256 is a known distill LoRA is a distill for every requirement rule, whatever role
the user set. It never satisfies the character requirement, it stops a non-turbo preset with an instruction that names
the file and its role, and an unknown hash keeps the user's role. No role is ever rewritten.

Audit repro B: the known DMD2 hash with ``role=character`` produced only a load-time warning; ``identity_safe`` then
asked for nothing but reference photos and a 50-step preset did not see the distill."""
import copy
import unittest

from moss_photon.core import errors, presets

DMD2 = "7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2"
SFT_DMD = "4abb14e39c1a59018907581e609f43616ec6810126144ec2b1d2e6f8969628fe"
NAME = "lodestones-8-steps-dmd2-lora.safetensors"
UNKNOWN = "e" * 64


def item(role, sha, basename="x.safetensors"):
    return {"basename": basename, "role": role, "sha256": sha}


def evaluate(pid, stack, **ctx):
    ctx = dict(ctx, lora_stack=stack, lora_roles=[s["role"] for s in stack])
    return presets.evaluate_requirements(presets.resolve(pid), ctx)


def codes(found, *, errors_only=False, only=None):
    return [(i.code, i.severity) for i in found
            if (not errors_only or i.severity == "error") and (only is None or i.code == only)]


class KnownDistillCannotBeACharacterTest(unittest.TestCase):
    def test_audit_repro_b_identity_safe_notes_the_known_distill_given_the_character_role(self):
        got = evaluate("identity_safe", [item("character", DMD2, NAME)])
        # MP-84: the character role is not recorded and the distill that is in the stack does not suit a non-turbo preset:
        # two warnings (errors before), the run goes on
        self.assertEqual(codes(got, errors_only=True), [])
        self.assertEqual(codes(got), [("lora_not_recorded", "warning"), ("reference_photos_recommended", "info"),
                                      ("distill_lora_non_turbo", "warning")])
        text = got[0].message + " " + got[0].fix
        for needle in (NAME, "'character'", "known distill", "role=distill"):
            self.assertIn(needle, text)
        self.assertIsNone(errors.raise_first_error(got))

    def test_an_unknown_hash_with_the_character_role_still_satisfies_it(self):
        self.assertEqual(codes(evaluate("identity_safe", [item("character", UNKNOWN)]), errors_only=True), [])
        self.assertEqual(codes(evaluate("identity_safe", [item("character", None)]), errors_only=True), [])

    def test_a_real_character_next_to_the_known_distill_satisfies_it(self):
        stack = [item("style", DMD2, NAME), item("character", UNKNOWN, "anna.safetensors")]
        self.assertEqual(codes(evaluate("identity_safe", stack), only="lora_not_recorded"), [])

    def test_the_input_is_not_rewritten(self):
        stack = [item("character", DMD2, NAME)]
        before = copy.deepcopy(stack)
        evaluate("identity_safe", stack)
        self.assertEqual(stack, before)


class KnownDistillStopsNonTurboPresetsTest(unittest.TestCase):
    def test_every_role_is_noted_in_a_normal_preset_with_file_and_role_named(self):
        """MP-84: a warning (``incompatible_lora`` error before): the preset decides turbo or not, the run goes on."""
        for role in ("character", "style", "wardrobe", "scene", "other"):
            for pid in ("reference", "photo_safe", "identity_safe"):
                with self.subTest(role=role, preset=pid):
                    got = [i for i in evaluate(pid, [item(role, DMD2, NAME)]) if i.code == "distill_lora_non_turbo"]
                    self.assertEqual(len(got), 1)
                    self.assertEqual(got[0].severity, "warning")
                    for needle in (NAME, "role '%s'" % role, "role=distill", "turbo"):
                        self.assertIn(needle, got[0].message + " " + got[0].fix)
                    self.assertEqual(got[0].detail["known_distill"], [{"basename": NAME, "role": role}])

    def test_the_distill_role_keeps_the_old_text(self):
        got = [i for i in evaluate("reference", [item("distill", DMD2, NAME)]) if i.code == "distill_lora_non_turbo"]
        self.assertEqual(len(got), 1)
        self.assertNotIn("known distill", got[0].message)
        self.assertIsNone(got[0].detail)

    def test_advanced_stays_a_warning_and_names_the_file(self):
        got = evaluate("advanced", [item("character", DMD2, NAME)])
        self.assertEqual(codes(got), [("distill_lora_non_turbo", "warning")])
        self.assertIn(NAME, got[0].message)

    def test_an_unknown_hash_in_the_distill_role_is_still_a_distill(self):
        got = evaluate("reference", [item("distill", UNKNOWN)])
        self.assertEqual(codes(got), [("distill_lora_non_turbo", "warning")])

    def test_an_unknown_hash_with_another_role_is_not_a_distill(self):
        self.assertEqual(evaluate("reference", [item("style", UNKNOWN)]), [])

    def test_the_turbo_presets_accept_the_distill_policy(self):
        got = evaluate("turbo_8step", [item("distill", DMD2, NAME)])
        self.assertEqual([i for i in got if i.code == "distill_lora_non_turbo"], [])


class TurboKeepsTheExplicitRoleTest(unittest.TestCase):
    """MP-50 stays: the file counts as a distill for the rules but does not silently fill the distill requirement."""

    def test_turbo_8step_still_asks_for_the_distill_role_and_names_the_file(self):
        got = evaluate("turbo_8step", [item("character", DMD2, NAME)])
        missing = [i for i in got if i.code == "lora_not_recorded"]
        self.assertEqual(len(missing), 1)
        self.assertIn(NAME, missing[0].message)
        self.assertIn("role 'character'", missing[0].message)
        self.assertEqual(missing[0].severity, "warning", "a distill file under another role is a mistake, not a merged checkpoint")

    def test_turbo_identity_expects_a_real_character_besides_the_distill(self):
        got = evaluate("turbo_identity", [item("distill", DMD2, NAME), item("character", DMD2, "second.safetensors")])
        self.assertTrue([i for i in got if i.code == "lora_not_recorded"])  # the second file is a distill, not a character

    def test_the_known_distill_is_not_an_extra_role_of_turbo_8step(self):
        got = evaluate("turbo_8step", [item("style", DMD2, NAME)])
        self.assertNotIn("preset_lora_roles_exceeded", [i.code for i in got])


class StackRulesTest(unittest.TestCase):
    """The second-distill note and the character count use the same effective role (core/lora.py)."""

    def entry(self, index, role, sha, name):
        return {"index": index, "role": role, "lora": {"basename": name, "size": 1, "sha256": sha}, "strength_model": 1.0,
                "sidecar": {"verified_strength": None}, "weight_profile": {"kind": "constant", "w": 1.0}}

    def test_a_known_distill_under_another_role_is_the_second_distill(self):
        from moss_photon.core import lora
        stack = [self.entry(0, "distill", DMD2, "a.safetensors"), self.entry(1, "character", SFT_DMD, "b.safetensors")]
        got = lora.check_stack(stack)
        self.assertEqual([(i.code, i.severity) for i in got], [("multiple_distill_loras", "warning")])
        self.assertIn("role 'character'", got[0].message)

    def test_a_known_distill_under_the_character_role_is_not_a_character(self):
        from moss_photon.core import lora
        stack = [self.entry(0, "character", DMD2, "a.safetensors"), self.entry(1, "character", UNKNOWN, "b.safetensors")]
        self.assertEqual(lora.check_stack(stack), [])  # not "multiple character LoRAs"


class BlocksUnverifiedTest(unittest.TestCase):
    """MP-60 (audit F02): constraints that do not describe the conditioning leave the block rules unjudged."""

    def test_the_block_requirement_is_skipped_not_passed_or_failed(self):
        failed = presets.evaluate_requirements(presets.resolve("wardrobe_safe"), {"blocks_present": {"wardrobe": False}})
        self.assertEqual([i.code for i in failed], ["missing_required_input"])
        skipped = presets.evaluate_requirements(
            presets.resolve("wardrobe_safe"), {"blocks_present": {}, "blocks_unverified": True})
        self.assertEqual(skipped, [])

    def test_other_rules_still_run_when_blocks_are_unverified(self):
        got = presets.evaluate_requirements(presets.resolve("scene_safe"), {
            "blocks_present": {}, "blocks_unverified": True, "latent_is_empty": False})
        self.assertEqual([i.code for i in got], ["unsupported_source_input"])


if __name__ == "__main__":
    unittest.main()
