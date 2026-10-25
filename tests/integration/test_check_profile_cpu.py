"""M1/M4: the profile / MODEL comparison on real patchers (design 6.4, Appendix A C10).

The profile is built by a TEST HELPER from the contract fixture (no MP-10 import): the attachment string is stored on the
MODEL exactly as a MossPhotonLoRAProfile node would. Sampling itself is skipped where the subject is the label.
"""
import unittest

from . import _comfy_env, tiny_moss
from . import sampler_support as ss


class CheckProfileCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        from moss_photon.core import adapter, errors, solver
        cls.adapter, cls.errors, cls.solver = adapter, errors, solver
        cls.base = cls.tiny.patcher
        cls.chain = ss.apply_lora(cls.base, ss.make_lora_dict(cls.base, seed=11), 0.8)
        cls.profile = ss.make_profile(cls.chain, base_model=cls.base, strength=0.8)

    def label(self, model, profile, preset_id="photo_safe", **kw):
        with ss.known_sft(), ss.NoSample():
            return ss.run(self.tiny, model=model, identity_profile=profile, preset_id=preset_id, **kw).report

    def test_clean_chain_is_reference_with_lora_and_really_samples(self):
        with ss.known_sft():
            result = ss.run(self.tiny, model=self.chain, identity_profile=self.profile, preset_id="photo_safe",
                            overrides={"steps": 6})
        report = result.report
        self.assertEqual(report["lora"]["foreign_patches"], {
            "entries_expected": ss.entries(self.chain), "entries_observed": ss.entries(self.chain),
            "baseline_entries": 0, "uuid_match": True, "extra_entries": 0, "keys_sample": []})
        self.assertEqual([s["basename"] for s in report["lora"]["stack"]], ["synthetic.safetensors"])
        self.assertNotIn("foreign_patches", ss.codes(report))
        plain = ss.run(self.tiny, preset_id="photo_safe", overrides={"steps": 6})
        self.assertFalse(plain.latent["samples"].equal(result.latent["samples"]), "control: the LoRA is in effect")

    def test_label_of_a_clean_chain_without_the_steps_override(self):
        report = self.label(self.chain, self.profile)
        self.assertEqual(report["label"]["parity_class"], "reference_with_lora")
        self.assertEqual(report["label"]["modifiers"], [])

    def test_native_loader_after_the_chain_is_foreign(self):
        after = ss.apply_lora(self.chain, ss.make_lora_dict(self.base, seed=12, skip={3}), 1.0)  # LoraLoaderModelOnly
        gained = ss.entries(after) - ss.entries(self.chain)
        self.assertGreater(gained, 0)
        report = self.label(after, self.profile)
        fp = report["lora"]["foreign_patches"]
        self.assertEqual(report["label"]["parity_class"], "reference_with_foreign_patches")
        self.assertEqual((fp["extra_entries"], fp["uuid_match"], fp["baseline_entries"]), (gained, False, 0))
        self.assertEqual(fp["entries_expected"], ss.entries(self.chain))
        self.assertEqual(fp["entries_observed"], ss.entries(after))
        self.assertTrue(0 < len(fp["keys_sample"]) <= 20)
        self.assertIn("foreign_patches", ss.codes(report))

    def test_native_loader_before_the_chain_is_baseline(self):
        pre = ss.apply_lora(self.base, ss.make_lora_dict(self.base, seed=13), 1.0)
        chain = ss.apply_lora(pre, ss.make_lora_dict(self.base, seed=14, skip={5}), 1.0)
        profile = ss.make_profile(chain, base_model=pre, baseline=ss.entries(pre))
        report = self.label(chain, profile)
        fp = report["lora"]["foreign_patches"]
        self.assertEqual(report["label"]["parity_class"], "reference_with_foreign_patches")
        self.assertEqual((fp["baseline_entries"], fp["uuid_match"], fp["extra_entries"]), (ss.entries(pre), True, 0))
        self.assertIn("foreign_patches", ss.codes(report))

    def test_attachment_of_another_chain_is_incompatible(self):
        other = ss.apply_lora(self.base, ss.make_lora_dict(self.base, seed=15), 0.5)
        other_profile = ss.make_profile(other, base_model=self.base, strength=0.5, name="other.safetensors")
        with self.assertRaises(self.errors.IncompatibleLora):
            self.label(self.chain, other_profile)

    def test_profile_without_the_attachment_on_the_model_is_incompatible(self):
        bare = ss.apply_lora(self.base, ss.make_lora_dict(self.base, seed=16), 1.0)
        profile = ss.make_profile(bare, base_model=self.base, attach=False)
        with self.assertRaises(self.errors.IncompatibleLora):
            self.label(bare, profile)

    def test_native_lora_with_no_profile_at_all_is_foreign(self):
        report = self.label(self.chain, None)
        self.assertEqual(report["label"]["parity_class"], "reference_with_foreign_patches")
        self.assertEqual(report["lora"]["foreign_patches"]["extra_entries"], ss.entries(self.chain))
        self.assertEqual(report["lora"]["stack"], [])

    def test_auraflow_and_slg_clones_after_the_chain_stay_clean(self):
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
        from moss_photon.core import guidance
        aura = ModelSamplingAuraFlow().patch_aura(self.chain, 5.0)[0]
        slg = guidance.apply_slg(self.chain, layers=[22], start=0.0, end=1.0)
        for name, model in (("auraflow", aura), ("slg", slg), ("plain clone", self.chain.clone())):
            with self.subTest(clone=name):
                facts = self.adapter.probe_model(model, reference_dtypes=ss.F32)
                check = self.solver.check_profile(self.profile, facts, False, node="MossPhotonSampler")
                self.assertEqual((list(check.issues), check.foreign, check.has_lora), ([], False, True))

    def test_a_native_loader_after_an_auraflow_clone_is_still_seen(self):
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
        aura = ModelSamplingAuraFlow().patch_aura(self.chain, 5.0)[0]
        after = ss.apply_lora(aura, ss.make_lora_dict(self.base, seed=17), 1.0)
        facts = self.adapter.probe_model(after, reference_dtypes=ss.F32)
        self.assertTrue(self.solver.check_profile(self.profile, facts, False, node="MossPhotonSampler").foreign)

    def test_identity_safe_needs_a_character_in_the_stack(self):
        style = ss.make_profile(self.chain, base_model=self.base, role="style", strength=0.8)
        with self.assertRaises(self.errors.MissingLora):
            self.label(self.chain, style, preset_id="identity_safe")
        ok = ss.make_profile(self.chain, base_model=self.base, role="character", strength=0.8)
        report = self.label(self.chain, ok, preset_id="identity_safe")
        self.assertEqual(report["lora"]["stack"][0]["role"], "character")
        self.assertEqual(report["label"]["parity_class"], "reference_with_lora")

    def test_trigger_warning_needs_the_constraints_and_never_prints_the_trigger(self):
        profile = ss.make_profile(self.chain, base_model=self.base, strength=0.8)
        profile["stack"][0]["trigger_hints"] = ["zkq_secret_token"]
        profile["stack"][0]["trigger_present"] = True
        constraints = ss.read_contract("constraints.json")
        without = self.label(self.chain, profile)
        self.assertNotIn("trigger_not_in_prompt", ss.codes(without))
        report = self.label(self.chain, profile, constraints=constraints)
        self.assertIn("trigger_not_in_prompt", ss.codes(report))
        self.assertNotIn("zkq_secret_token", repr(report))
        constraints["compiled_positive"] += " zkq_secret_token"
        self.assertNotIn("trigger_not_in_prompt", ss.codes(self.label(self.chain, profile, constraints=constraints)))

    def test_the_report_never_carries_prompts(self):
        constraints = ss.read_contract("constraints.json")
        report = self.label(self.chain, self.profile, constraints=constraints)
        text = repr(report)
        for private in ("A woman in a kitchen", "red wool sweater", "small kitchen"):
            self.assertNotIn(private, text)
        self.assertEqual(report["text_encoder"], constraints["text_encoder"])


if __name__ == "__main__":
    unittest.main()
