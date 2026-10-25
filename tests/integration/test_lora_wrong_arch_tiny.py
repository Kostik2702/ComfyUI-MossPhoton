"""Wrong-architecture and low-coverage files (design 6.5, 1.9, 3.1): typed errors, the input MODEL untouched."""
import unittest
from unittest import mock

from . import lora_synth, tiny_moss


class WrongArchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.core import errors, lora
        cls.torch, cls.errors, cls.lora = torch, errors, lora
        cls.baseline = tiny_moss.sample_euler(cls.tiny)

    def setUp(self):
        self.env = lora_synth.install(self)
        self.patcher = self.tiny.patcher
        self.uuid0 = str(self.patcher.patches_uuid)

    def assert_untouched(self):
        self.assertEqual((len(self.patcher.patches), str(self.patcher.patches_uuid), self.patcher.attachments),
                         (0, self.uuid0, {}))
        self.assertTrue(self.torch.equal(tiny_moss.sample_euler(self.tiny), self.baseline))

    def unrelated_sd(self):
        import torch
        gen = torch.Generator().manual_seed(1)
        sd = {}
        for n in range(8):
            for m in ("attention.to_q", "attention.to_k", "feed_forward.w1", "adaLN_modulation.1"):
                base = "diffusion_model.layers.%d.%s" % (n, m)
                sd[base + ".lora_A.weight"] = torch.randn(4, 16, generator=gen)
                sd[base + ".lora_B.weight"] = torch.randn(16, 4, generator=gen)
        return sd

    def test_unrelated_keys_are_incompatible_lora_with_counts_and_filename(self):
        self.env.write("cinema_like.safetensors", self.unrelated_sd())
        with self.assertRaises(self.errors.IncompatibleLora) as ctx:
            lora_synth.run_node(self.patcher, "cinema_like.safetensors")
        e = ctx.exception
        self.assertEqual(e.code, "incompatible_lora")
        self.assertIn("maps 0/32 modules to MOSS", str(e))
        self.assertIn("cinema_like.safetensors", str(e))
        self.assertEqual((e.issue.detail["modules_mapped"], e.issue.detail["modules_total"]), (0, 32))
        self.assert_untouched()

    def test_allow_low_coverage_never_rescues_zero_mapped(self):
        self.env.write("cinema_like.safetensors", self.unrelated_sd())
        with self.assertRaises(self.errors.IncompatibleLora):
            lora_synth.run_node(self.patcher, "cinema_like.safetensors", allow_low_coverage=True)
        self.assert_untouched()

    def test_a_file_without_any_weight_module_is_incompatible(self):
        import torch
        self.env.write("empty.safetensors", {"junk.tensor": torch.zeros(2)})
        with self.assertRaises(self.errors.IncompatibleLora) as ctx:
            lora_synth.run_node(self.patcher, "empty.safetensors")
        self.assertIn("maps 0/0 modules", str(ctx.exception))
        self.assert_untouched()

    def test_architecture_named_in_the_metadata_stops_before_mapping(self):
        sd = lora_synth.make_lora_sd(self.patcher, seed=3)  # keys that WOULD map, labelled for another architecture
        self.env.write("krea_labelled.safetensors", sd, {"ss_architecture": "krea2drv", "modelspec.architecture": "Krea-2/lora"})
        with mock.patch.object(self.lora, "map_lora", side_effect=AssertionError("mapped")):
            with self.assertRaises(self.errors.IncompatibleLora) as ctx:
                lora_synth.run_node(self.patcher, "krea_labelled.safetensors")
        self.assertIn("krea2drv", str(ctx.exception))
        self.assert_untouched()

    def test_moss_metadata_passes(self):
        sd = lora_synth.make_lora_sd(self.patcher, seed=3)
        self.env.write("ok.safetensors", sd, {"ss_architecture": "mossimage"})
        out, profile, _ = lora_synth.run_node(self.patcher, "ok.safetensors")
        self.assertEqual(profile["stack"][0]["coverage"]["verdict"], "ok")

    def test_partial_coverage_is_a_warning(self):
        sd = lora_synth.add_bogus_modules(lora_synth.make_lora_sd(self.patcher, seed=3), 20)  # 48 mapped of 68: 0.706
        self.env.write("partial.safetensors", sd)
        out, profile, _ = lora_synth.run_node(self.patcher, "partial.safetensors")
        entry = profile["stack"][0]
        self.assertEqual((entry["coverage"]["modules_mapped"], entry["coverage"]["modules_total"], entry["coverage"]["verdict"]),
                         (48, 68, "warn"))
        self.assertEqual([(w["code"], w["severity"]) for w in entry["warnings"]], [("partial_lora_coverage", "warning")])
        self.assertEqual(len(entry["coverage"]["unmapped_examples"]), 5)
        self.assertEqual(self.lora.patch_entries_of(out), 48)

    def test_low_coverage_is_an_error_unless_forced(self):
        sd = lora_synth.add_bogus_modules(lora_synth.make_lora_sd(self.patcher, seed=3), 60)  # 48 of 108: 0.444
        self.env.write("low.safetensors", sd)
        with self.assertRaises(self.errors.IncompatibleLora) as ctx:
            lora_synth.run_node(self.patcher, "low.safetensors")
        self.assertIn("maps 48/108 modules", str(ctx.exception))
        self.assert_untouched()
        out, profile, _ = lora_synth.run_node(self.patcher, "low.safetensors", allow_low_coverage=True)
        entry = profile["stack"][0]
        self.assertEqual(entry["coverage"]["verdict"], "error-forced")
        self.assertEqual([w["code"] for w in entry["warnings"]], ["lora_low_coverage_forced"])
        self.assertEqual(self.lora.patch_entries_of(out), 48)

    def test_a_full_file_is_ok_without_warnings(self):
        sd = lora_synth.make_lora_sd(self.patcher, seed=3)
        self.env.write("full.safetensors", sd)
        _o, profile, _ = lora_synth.run_node(self.patcher, "full.safetensors")
        self.assertEqual(profile["stack"][0]["coverage"]["verdict"], "ok")
        self.assertEqual(profile["stack"][0]["warnings"], [])

    def test_a_layer_gap_is_info_not_a_coverage_problem(self):
        sd = lora_synth.make_lora_sd(self.patcher, seed=3, skip_layers=(22,))
        self.env.write("no22.safetensors", sd)
        _o, profile, _ = lora_synth.run_node(self.patcher, "no22.safetensors")
        entry = profile["stack"][0]
        self.assertEqual(entry["coverage"]["verdict"], "ok")
        self.assertEqual(entry["coverage"]["layers_missing"], [22])
        self.assertEqual([(w["code"], w["severity"]) for w in entry["warnings"]], [("lora_layers_missing", "info")])

    def test_missing_file_is_comfyuis_own_error(self):
        with self.assertRaises(FileNotFoundError):
            lora_synth.run_node(self.patcher, "gone.safetensors")
        self.assert_untouched()

    def test_non_moss_model_is_refused_before_anything_is_read(self):
        not_moss = lora_synth.FakeClip().patcher  # a real ModelPatcher around a plain torch module
        sd = lora_synth.make_lora_sd(self.patcher, seed=3)
        self.env.write("ok.safetensors", sd)
        with mock.patch.object(self.lora, "load_lora_state", side_effect=AssertionError("read")):
            with self.assertRaises(self.errors.UnsupportedModel) as ctx:
                lora_synth.run_node(not_moss, "ok.safetensors")
        self.assertEqual(ctx.exception.code, "unsupported_model")


if __name__ == "__main__":
    unittest.main()
