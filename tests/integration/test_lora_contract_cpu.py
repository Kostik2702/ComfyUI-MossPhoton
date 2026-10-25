"""MP-9 <-> MP-10 contract on the REAL MossPhotonLoRAProfile node (design 7.2, M4): the sampler accepts exactly what the
node produces. The sampler itself imports nothing of the LoRA module; only this test drives both."""
import unittest

from . import _comfy_env, lora_synth, tiny_moss
from . import sampler_support as ss


class RealLoraNodeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        from moss_photon.core import errors, report
        cls.errors, cls.report = errors, report

    def setUp(self):
        self.env = lora_synth.install(self)
        self.base = self.tiny.patcher
        self.env.write("char.safetensors", lora_synth.make_lora_sd(self.base, seed=31))
        self.env.write("style.safetensors", lora_synth.make_lora_sd(self.base, seed=32, skip_layers=(3,)))
        self.env.write("distill.safetensors", lora_synth.make_lora_sd(self.base, fmt="kohya", seed=33,
                                                                       modules=lora_synth.KOHYA_SPLIT))

    def label(self, model, profile, preset_id="photo_safe", **kw):
        with ss.known_sft(), ss.NoSample():
            return ss.run(self.tiny, model=model, identity_profile=profile, preset_id=preset_id, **kw).report

    def test_a_real_character_chain_is_clean(self):
        model, profile, _ = lora_synth.run_node(self.base, "char.safetensors", strength_model=0.8)
        report = self.label(model, profile, "identity_safe")
        self.assertEqual(report["label"]["parity_class"], "reference_with_lora")
        fp = report["lora"]["foreign_patches"]
        self.assertEqual((fp["entries_expected"], fp["entries_observed"], fp["uuid_match"], fp["extra_entries"]),
                         (ss.entries(model), ss.entries(model), True, 0))
        stack = report["lora"]["stack"]
        self.assertEqual([(s["role"], s["strength_model"], s["basename"]) for s in stack],
                         [("character", 0.8, "char.safetensors")])
        self.assertEqual(report["lora"]["profile_digest"], profile["digest"])
        self.assertEqual(self.report.validate_report(report), [])

    def test_a_real_two_node_chain_lists_both_adapters_in_order(self):
        m1, p1, _ = lora_synth.run_node(self.base, "char.safetensors", strength_model=0.8)
        m2, p2, _ = lora_synth.run_node(m1, "style.safetensors", strength_model=0.5, role="style", profile_in=p1)
        report = self.label(m2, p2, "identity_safe")
        self.assertEqual([s["role"] for s in report["lora"]["stack"]], ["character", "style"])
        self.assertEqual(report["label"]["parity_class"], "reference_with_lora")
        self.assertEqual(report["lora"]["foreign_patches"]["extra_entries"], 0)
        # the first node's profile no longer describes the MODEL after the second node
        with self.assertRaises(self.errors.IncompatibleLora):
            self.label(m2, p1)

    def test_the_pass_through_node_gives_an_empty_profile_that_is_not_incompatible(self):
        model, profile, _ = lora_synth.run_node(self.base, "none")
        self.assertIs(model, self.base)
        self.assertEqual(profile["stack"], [])
        report = self.label(model, profile)
        self.assertEqual(report["label"]["parity_class"], "reference_exact")
        self.assertEqual(report["lora"]["stack"], [])
        self.assertEqual(report["lora"]["foreign_patches"]["entries_observed"], 0)
        with self.assertRaises(self.errors.MissingLora):
            self.label(model, profile, "identity_safe")

    def test_a_zero_strength_node_is_a_pass_through_too(self):
        model, profile, _ = lora_synth.run_node(self.base, "char.safetensors", strength_model=0.0)
        self.assertEqual(self.label(model, profile)["label"]["parity_class"], "reference_exact")

    def test_the_pass_through_node_on_a_model_that_already_has_native_patches(self):
        patched = ss.apply_lora(self.base, ss.make_lora_dict(self.base), 1.0)
        model, profile, _ = lora_synth.run_node(patched, "none")
        report = self.label(model, profile)
        self.assertEqual(report["label"]["parity_class"], "reference_with_foreign_patches")
        self.assertEqual(report["lora"]["foreign_patches"]["extra_entries"], ss.entries(patched))

    def test_a_native_loader_after_the_real_chain_is_seen(self):
        model, profile, _ = lora_synth.run_node(self.base, "char.safetensors")
        after = ss.apply_lora(model, ss.make_lora_dict(self.base, seed=41), 1.0)
        report = self.label(after, profile)
        self.assertEqual(report["label"]["parity_class"], "reference_with_foreign_patches")
        self.assertGreater(report["lora"]["foreign_patches"]["extra_entries"], 0)
        self.assertFalse(report["lora"]["foreign_patches"]["uuid_match"])

    def test_a_native_loader_before_the_real_chain_is_baseline(self):
        pre = ss.apply_lora(self.base, ss.make_lora_dict(self.base, seed=42), 1.0)
        model, profile, _ = lora_synth.run_node(pre, "char.safetensors")
        report = self.label(model, profile)
        self.assertEqual(report["label"]["parity_class"], "reference_with_foreign_patches")
        self.assertEqual(report["lora"]["foreign_patches"]["baseline_entries"], ss.entries(pre))

    def test_the_real_chain_really_samples_with_the_adapter_in_effect(self):
        model, profile, _ = lora_synth.run_node(self.base, "char.safetensors", strength_model=0.8)
        with ss.known_sft():
            with_lora = ss.run(self.tiny, model=model, identity_profile=profile, preset_id="photo_safe",
                               overrides={"steps": 6})
        plain = ss.run(self.tiny, preset_id="photo_safe", overrides={"steps": 6})
        self.assertFalse(with_lora.latent["samples"].equal(plain.latent["samples"]))
        self.assertEqual(with_lora.report["forwards"]["sample_forwards"], 12)
        self.assertTrue(with_lora.report["forwards"]["match"])

    def test_a_real_distill_plus_character_stack_runs_turbo_identity(self):
        m1, p1, _ = lora_synth.run_node(self.base, "distill.safetensors", strength_model=1.0, role="distill")
        m2, p2, _ = lora_synth.run_node(m1, "char.safetensors", strength_model=0.8, profile_in=p1)
        with ss.NoSample():
            report = self.label(m2, p2, "turbo_identity", cfg=1.0)
        self.assertEqual([s["role"] for s in report["lora"]["stack"]], ["distill", "character"])
        self.assertEqual(report["label"]["parity_class"], "native_turbo")
        self.assertEqual(self.report.validate_report(report), [])


if __name__ == "__main__":
    unittest.main()
