"""check_stack (design 1.6, 3.3; docs/03-turbo.md 5.4, R-T5): duplicates, two characters, two distills, strength range."""
import unittest

from moss_photon.core import errors, lora

from .builders import DMD2, SHA_A, SHA_B, entry


def codes(issues):
    return [(i.code, i.severity, i.detail.get("index")) for i in issues]


class StackRulesTest(unittest.TestCase):
    def test_clean_stack(self):
        stack = [entry(0, "distill", "d.safetensors", DMD2, 1.0), entry(1, "character", "c.safetensors", SHA_A, 0.8)]
        self.assertEqual(lora.check_stack(stack), [])

    def test_duplicate_file_by_hash(self):
        stack = [entry(0, "style", "a.safetensors", SHA_A), entry(1, "style", "renamed.safetensors", SHA_A)]
        got = lora.check_stack(stack)
        self.assertEqual(codes(got), [("duplicate_lora", "warning", 1)])
        self.assertEqual(got[0].detail["same_as"], 0)

    def test_duplicate_file_without_hash_uses_basename_and_size(self):
        a = entry(0, "style", "a.safetensors", None, size=10)
        same = entry(1, "style", "a.safetensors", None, size=10)
        other_size = entry(2, "style", "a.safetensors", None, size=11)
        self.assertEqual(codes(lora.check_stack([a, same, other_size])), [("duplicate_lora", "warning", 1)])

    def test_two_character_roles_warn_on_the_second(self):
        stack = [entry(0, "character", "a.safetensors", SHA_A), entry(1, "character", "b.safetensors", SHA_B)]
        self.assertEqual(codes(lora.check_stack(stack)), [("multiple_character_loras", "warning", 1)])

    def test_other_roles_may_repeat(self):
        stack = [entry(0, "style", "a.safetensors", SHA_A), entry(1, "style", "b.safetensors", SHA_B)]
        self.assertEqual(lora.check_stack(stack), [])

    def test_second_distill_is_incompatible_lora_error(self):
        stack = [entry(0, "distill", "a.safetensors", SHA_A, 1.0), entry(1, "distill", "b.safetensors", SHA_B, 1.0)]
        got = lora.check_stack(stack)
        self.assertEqual(codes(got), [("incompatible_lora", "error", 1)])
        with self.assertRaises(errors.IncompatibleLora):
            errors.raise_first_error(got)

    def test_strength_outside_the_sidecar_range_warns_only_for_that_entry(self):
        vs = {"min": 0.6, "max": 0.9, "recommended": 0.8}
        stack = [entry(0, "character", "a.safetensors", SHA_A, 1.2, verified=vs),
                 entry(1, "style", "b.safetensors", SHA_B, 5.0)]  # no verified range: nothing to compare with
        got = lora.check_stack(stack)
        self.assertEqual(codes(got), [("lora_strength_outside_verified_range", "warning", 0)])
        self.assertEqual(got[0].detail["verified"], [0.6, 0.9])

    def test_strength_on_the_boundary_is_inside(self):
        vs = {"min": 0.6, "max": 0.9, "recommended": 0.8}
        for s in (0.6, 0.9, 0.75):
            self.assertEqual(lora.check_stack([entry(0, "character", "a", SHA_A, s, verified=vs)]), [], s)

    def test_distill_with_the_known_table_range(self):
        vs = {"min": 1.0, "max": 1.0, "recommended": 1.0}
        self.assertEqual(lora.check_stack([entry(0, "distill", "d", DMD2, 1.0, verified=vs)]), [])
        self.assertEqual(codes(lora.check_stack([entry(0, "distill", "d", DMD2, 0.5, verified=vs)])),
                         [("lora_strength_outside_verified_range", "warning", 0)])

    def test_issue_text_names_node_param_and_fix(self):
        got = lora.check_stack([entry(0, "character", "a", SHA_A), entry(1, "character", "b", SHA_B)])[0]
        self.assertTrue(got.format().startswith("[MossPhoton] MossPhotonLoRAProfile/role: "))
        self.assertTrue(got.fix)


class StackJournalTest(unittest.TestCase):
    def test_apply_order_is_logged_and_has_no_path_or_trigger(self):
        stack = [entry(0, "distill", "lodestones-8-steps-dmd2-lora.safetensors", DMD2, 1.0),
                 entry(1, "character", "character_a.safetensors", SHA_A, 0.8, hints=["secret_trigger"])]
        line = lora.stack_journal(stack)
        self.assertEqual(line, "LoRA stack, apply order is bit-relevant: 0 distill lodestones-8-steps-dmd2-lora.safetensors (1) "
                               "-> 1 character character_a.safetensors (0.8)")
        self.assertNotIn("secret_trigger", line)

    def test_character_before_distill_is_recorded_as_non_canonical(self):
        stack = [entry(0, "character", "c", SHA_A, 0.8), entry(1, "distill", "d", DMD2, 1.0)]
        self.assertIn("not the canonical turbo order", lora.stack_journal(stack))
        self.assertNotIn("not the canonical", lora.stack_journal(list(reversed(stack))[:1]))


if __name__ == "__main__":
    unittest.main()
