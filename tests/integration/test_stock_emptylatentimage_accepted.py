"""The stock EmptyLatentImage is accepted: fix first, probe after (design 1.2 M8/LD9, check C8)."""
import unittest

from . import _comfy_env, tiny_moss


class StockEmptyLatentAcceptedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        import comfy.sample
        import comfy.samplers
        import nodes
        from comfy_extras.nodes_flux import EmptyFlux2LatentImage
        from moss_photon.core import adapter, errors
        cls.torch, cls.comfy_sample, cls.comfy_samplers = torch, comfy.sample, comfy.samplers
        cls.nodes, cls.Flux2, cls.adapter, cls.errors = nodes, EmptyFlux2LatentImage, adapter, errors
        cls.patcher = cls.tiny.patcher

    def run_on(self, lat):
        noise = self.comfy_sample.prepare_noise(lat, tiny_moss.SEED)
        guider = self.comfy_samplers.CFGGuider(self.patcher)
        guider.set_conds(self.tiny.make_cond(23), self.tiny.make_cond(9))
        guider.set_cfg(tiny_moss.CFG)
        sigmas = self.adapter.native_sigmas(self.patcher, "sgm_uniform", tiny_moss.STEPS)
        with self.torch.inference_mode():
            return guider.sample(noise, lat.clone(), self.comfy_samplers.sampler_object("euler"), sigmas, seed=tiny_moss.SEED)

    def test_stock_emptylatentimage_is_repaired_to_the_flux2_latent_and_runs_bit_equal(self):
        for width, height in ((128, 128), (128, 192)):
            with self.subTest(size=(width, height)):
                stock = self.nodes.EmptyLatentImage().generate(width, height, 1)[0]
                native = self.Flux2.execute(width, height, 1).args[0]["samples"]
                self.assertEqual(tuple(stock["samples"].shape), (1, 4, height // 8, width // 8))  # 4 channels, ratio 8
                self.assertEqual(stock["downscale_ratio_spacial"], 8)
                fixed = self.adapter.fix_empty_latent(self.patcher, stock)
                self.assertEqual(tuple(fixed.shape), (1, 128, height // 16, width // 16))
                self.assertTrue(self.torch.equal(fixed, native))
                self.assertTrue(self.torch.equal(self.run_on(fixed), self.run_on(native)))

    def test_probe_after_fix_accepts_it_with_an_info(self):
        stock = self.nodes.EmptyLatentImage().generate(128, 192, 1)[0]
        fixed = self.adapter.fix_empty_latent(self.patcher, stock)
        facts = self.adapter.probe_latent(stock, fixed)
        self.assertEqual((facts.channels, facts.height, facts.width), (128, 12, 8))
        self.assertEqual(facts.raw_shape, (1, 4, 24, 16))
        self.assertEqual(facts.raw_downscale_ratio_spacial, 8.0)
        self.assertTrue(facts.rescaled_by_native_fix)
        self.assertTrue(facts.is_empty)
        self.assertEqual((facts.pixel_width, facts.pixel_height), (128, 192))
        issues = self.adapter.validate_latent_facts(facts, channels=128, node="MossPhotonSampler")
        self.assertEqual([(i.code, i.severity) for i in issues], [("latent_rescaled_by_native_fix", "info")])
        self.errors.raise_first_error(issues)

    def test_native_flux2_latent_needs_no_repair(self):
        native = {"samples": self.Flux2.execute(128, 128, 1).args[0]["samples"]}
        fixed = self.adapter.fix_empty_latent(self.patcher, native)
        facts = self.adapter.probe_latent(native, fixed)
        self.assertFalse(facts.rescaled_by_native_fix)
        self.assertEqual(self.adapter.validate_latent_facts(facts, channels=128, node="T"), [])

    def test_non_empty_four_channel_latent_stays_four_channel_and_is_refused(self):
        latent = {"samples": self.torch.ones(1, 4, 8, 8)}
        fixed = self.adapter.fix_empty_latent(self.patcher, latent)
        self.assertEqual(int(fixed.shape[1]), 4)  # ComfyUI never repairs a non-empty latent
        facts = self.adapter.probe_latent(latent, fixed)
        self.assertFalse(facts.is_empty)
        issues = self.adapter.validate_latent_facts(facts, channels=128, node="MossPhotonSampler")
        self.assertEqual([i.code for i in issues], ["invalid_latent"])
        for name in ("MossPhotonEmptyLatent", "EmptyFlux2LatentImage", "EmptyLatentImage"):
            self.assertIn(name, issues[0].fix)
        with self.assertRaises(self.errors.InvalidLatent):
            self.errors.raise_first_error(issues)

    def test_non_empty_latent_with_a_declared_foreign_ratio_is_refused(self):
        latent = {"samples": self.torch.ones(1, 128, 8, 8), "downscale_ratio_spacial": 8}
        facts = self.adapter.probe_latent(latent, self.adapter.fix_empty_latent(self.patcher, latent))
        self.assertEqual([i.code for i in self.adapter.validate_latent_facts(facts, channels=128, node="T")],
                         ["invalid_latent"])

    def test_noise_mask_and_batch_index_are_seen_by_the_probe(self):
        latent = {"samples": self.torch.zeros(1, 128, 8, 8), "noise_mask": self.torch.ones(1, 1, 8, 8), "batch_index": [0]}
        facts = self.adapter.probe_latent(latent, self.adapter.fix_empty_latent(self.patcher, latent))
        self.assertTrue(facts.has_noise_mask and facts.has_batch_index)
        self.assertEqual([i.code for i in self.adapter.validate_latent_facts(facts, channels=128, node="T")],
                         ["unsupported_source_input"])

    def test_inputs_are_not_mutated_by_the_repair(self):
        stock = self.nodes.EmptyLatentImage().generate(128, 128, 1)[0]
        snapshot = (tuple(stock["samples"].shape), dict(stock))
        self.adapter.fix_empty_latent(self.patcher, stock)
        self.assertEqual((tuple(stock["samples"].shape), dict(stock)), snapshot)


if __name__ == "__main__":
    unittest.main()
