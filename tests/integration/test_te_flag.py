"""Text-encoder keys (design 4.4 step 5, D6): transformer only by default; applied only with the flag, a clip and a log."""
import unittest

from . import lora_synth, tiny_moss


class TeFlagTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.core import consts, errors, lora
        cls.torch, cls.consts, cls.errors, cls.lora = torch, consts, errors, lora

    def setUp(self):
        self.env = lora_synth.install(self)
        self.patcher = self.tiny.patcher
        self.clip = lora_synth.FakeClip()
        sd = lora_synth.make_lora_sd(self.patcher, seed=3, modules=("qkv", "o"))
        sd.update(lora_synth.te_lora_sd())
        self.env.write("with_te.safetensors", sd)
        self.env.write("te_only.safetensors", lora_synth.te_lora_sd())

    def codes(self, entry):
        return [(w["code"], w["severity"]) for w in entry["warnings"]]

    def test_flag_off_ignores_the_te_part_with_a_warning(self):
        for clip in (None, self.clip):
            with self.subTest(clip=clip is not None):
                out, profile, out_clip = lora_synth.run_node(self.patcher, "with_te.safetensors", clip=clip,
                                                             strength_clip=0.7)
                entry = profile["stack"][0]
                self.assertIs(out_clip, clip)  # the very same object, nothing cloned
                self.assertEqual(self.codes(entry), [("te_keys_ignored", "warning")])
                self.assertEqual(entry["applied_to"], {"transformer": True, "text_encoder": False})
                self.assertEqual(entry["strength_clip"], 0.0)  # recorded as 0 unless the flag is on
                self.assertEqual(entry["coverage"]["te_modules_ignored"], 1)
                self.assertEqual((entry["coverage"]["modules_total"], entry["coverage"]["modules_mapped"]), (48, 48))
                self.assertFalse(profile["summary"]["te_applied"])
                self.assertFalse(profile["attachment"]["entries"][0]["te"])
        self.assertEqual(len(self.clip.patcher.patches), 0)

    def test_flag_on_without_clip_is_missing_required_input(self):
        with self.assertRaises(self.errors.MissingRequiredInput) as ctx:
            lora_synth.run_node(self.patcher, "with_te.safetensors", apply_text_encoder=True, strength_clip=0.7)
        self.assertEqual(ctx.exception.code, "missing_required_input")
        self.assertIn("clip", str(ctx.exception))
        self.assertEqual(len(self.patcher.patches), 0)

    def test_flag_on_with_clip_applies_to_a_clone_and_says_so(self):
        with self.assertLogs("MossPhoton", level="WARNING") as logs:
            out, profile, new_clip = lora_synth.run_node(
                self.patcher, "with_te.safetensors", clip=self.clip, apply_text_encoder=True, strength_clip=0.7)
        entry = profile["stack"][0]
        self.assertIsNot(new_clip, self.clip)
        self.assertEqual(len(self.clip.patcher.patches), 0)               # the input clip is untouched
        self.assertEqual(sorted(new_clip.patcher.patches), ["enc.proj.weight"])
        self.assertEqual(self.codes(entry), [("te_lora_applied", "warning")])
        self.assertTrue(any("text-encoder LoRA applied" in line for line in logs.output))
        self.assertEqual(entry["applied_to"], {"transformer": True, "text_encoder": True})
        self.assertEqual(entry["strength_clip"], 0.7)
        self.assertEqual((entry["patch_entries"], entry["coverage"]["modules_total"], entry["coverage"]["modules_mapped"]),
                         (48, 49, 49))  # TE counts in the denominator when applied; patch_entries are the MODEL's
        self.assertEqual(entry["coverage"]["te_modules_ignored"], 0)
        self.assertTrue(profile["summary"]["te_applied"])
        self.assertTrue(profile["attachment"]["entries"][0]["te"])
        self.assertEqual(profile["attachment"]["patch_entries_total"], 48)  # the attachment counts the MODEL's patches only
        self.assertEqual(self.lora.patch_entries_of(out), 48)
        self.assertEqual(self.lora.validate_identity_profile(profile), [])

    def test_te_only_file_is_incompatible_without_the_flag_and_applies_with_it(self):
        with self.assertRaises(self.errors.IncompatibleLora) as ctx:
            lora_synth.run_node(self.patcher, "te_only.safetensors")
        self.assertIn("maps 0/0 modules", str(ctx.exception))
        out, profile, new_clip = lora_synth.run_node(self.patcher, "te_only.safetensors", clip=self.clip,
                                                     apply_text_encoder=True, strength_clip=1.0)
        entry = profile["stack"][0]
        self.assertEqual((entry["patch_entries"], entry["applied_to"]), (0, {"transformer": False, "text_encoder": True}))
        self.assertEqual(self.lora.patch_entries_of(out), 0)
        self.assertEqual(len(new_clip.patcher.patches), 1)

    def test_flag_on_but_zero_strengths_is_a_pass_through(self):
        out, profile, clip = lora_synth.run_node(self.patcher, "with_te.safetensors", clip=self.clip,
                                                 apply_text_encoder=True, strength_clip=0.0, strength_model=0.0)
        self.assertIs(out, self.patcher)
        self.assertIs(clip, self.clip)
        self.assertEqual(profile["stack"], [])

    def test_clip_output_is_the_input_clip_when_the_flag_is_off(self):
        _o, _p, out_clip = lora_synth.run_node(self.patcher, "with_te.safetensors", clip=self.clip)
        self.assertIs(out_clip, self.clip)
        _o, _p, none_clip = lora_synth.run_node(self.patcher, "with_te.safetensors")
        self.assertIsNone(none_clip)


if __name__ == "__main__":
    unittest.main()
