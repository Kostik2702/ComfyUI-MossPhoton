"""The REAL ``reference`` preset against the node-level stock path on the tiny 24-layer MOSS (design 6.4, m6).

Stock side: ``nodes.KSampler().sample`` after the native ``SkipLayerGuidanceDiTSimple(single_layers="22")``. The
four spike configurations: prompt lengths different/equal x SLG on/off. Photon side: ``MossPhotonSampler`` through
``solver.execute`` with the preset (SLG on) or with the Advanced override ``slg_mode=off``.
"""
import unittest

from . import _comfy_env, tiny_moss
from . import sampler_support as ss


class ReferenceVsStockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.core import rng, sigma
        cls.torch, cls.rng, cls.sigma = torch, rng, sigma

    def compare(self, pos_tokens, neg_tokens, slg):
        tiny = self.tiny
        pos, neg = tiny.make_cond(pos_tokens), tiny.make_cond(neg_tokens)
        patcher = ss.native_slg(tiny.patcher, 22) if slg else tiny.patcher
        stock_stats = __import__("moss_photon.core.solver", fromlist=["ForwardStats"]).ForwardStats()
        stock = ss.stock_run(tiny, patcher, positive=pos, negative=neg, steps=50, counter=stock_stats)
        overrides = {} if slg else {"slg_mode": "off"}
        result = ss.run(tiny, positive=pos, negative=neg, overrides=overrides)
        return stock, stock_stats, result

    def check(self, pos_tokens, neg_tokens, slg, calls, skip_calls):
        stock, stock_stats, result = self.compare(pos_tokens, neg_tokens, slg)
        self.assertTrue(self.torch.equal(result.latent["samples"], stock), "Photon latent is not torch.equal to stock")
        fw = result.report["forwards"]
        self.assertEqual((fw["model_calls"], fw["slg_skip_calls"]), (calls, skip_calls))
        self.assertEqual((stock_stats.calls, stock_stats.skip_calls), (calls, skip_calls))
        self.assertEqual(fw["sample_forwards"], 100)
        self.assertEqual(fw["expected_sample_forwards"], 100)
        self.assertTrue(fw["match"])
        self.assertEqual(result.report["output"]["latent_sha256"], ss.tensor_sha(stock))
        return result

    def test_different_length_with_slg(self):
        result = self.check(23, 9, True, calls=100, skip_calls=50)
        self.assertEqual(result.report["sampling"]["slg"]["layers"], [22])
        self.assertEqual(result.report["label"]["parity_class"], "reference_unverified")
        self.assertEqual(result.report["label"]["not_reference_reasons"], [])
        self.assertEqual(result.report["sampling"]["sigmas"]["equals_native"], True)

    def test_equal_length_with_slg(self):
        self.check(23, 23, True, calls=100, skip_calls=50)

    def test_different_length_without_slg(self):
        # unequal prompt lengths cannot share a batch: two calls of batch 1 per step
        self.check(23, 9, False, calls=100, skip_calls=0)

    def test_equal_length_without_slg_batches_both_branches(self):
        # equal lengths: one call of batch 2 per step, 50 calls for 100 sample forwards
        stock, stock_stats, result = self.compare(23, 23, False)
        self.assertTrue(self.torch.equal(result.latent["samples"], stock))
        fw = result.report["forwards"]
        self.assertEqual((fw["model_calls"], fw["sample_forwards"], fw["slg_skip_calls"]), (50, 100, 0))
        self.assertEqual(stock_stats.calls, 50)

    def test_noise_and_sigma_hashes_equal_the_stock_side(self):
        tiny = self.tiny
        result = ss.run(tiny)
        stock_sigmas = self.torch.tensor(
            __import__("comfy.samplers", fromlist=["KSampler"]).KSampler(
                tiny.patcher, steps=50, device=self.torch.device("cpu"), sampler="euler",
                scheduler="sgm_uniform").sigmas)
        self.assertEqual(result.report["sampling"]["sigmas"]["sha256"], self.sigma.sigma_hash(stock_sigmas))
        import comfy.sample
        noise = comfy.sample.prepare_noise(tiny.latent, tiny_moss.SEED)
        self.assertEqual(result.report["inputs"]["noise_sha256"], self.rng.tensor_sha256(noise))

    def test_the_run_uses_cfgguider_and_ksampler_euler(self):
        """The route is CFGGuider + KSAMPLER(sample_euler) + guider.sample, nothing else."""
        import comfy.k_diffusion.sampling as kds
        import comfy.samplers
        from moss_photon.core import solver
        sampler = solver.get_sampler("euler")
        self.assertIsInstance(sampler, comfy.samplers.KSAMPLER)
        self.assertIs(sampler.sampler_function, kds.sample_euler)
        guider = solver.build_guider(self.tiny.patcher, self.tiny.make_cond(3), self.tiny.make_cond(3), 3.0)
        self.assertIs(type(guider), comfy.samplers.CFGGuider)
        self.assertEqual(guider.cfg, 3.0)

    def test_report_is_a_valid_photon_report(self):
        from moss_photon.core import report
        result = ss.run(self.tiny)
        self.assertEqual(report.validate_report(result.report), [])
        self.assertEqual(result.report["preset"]["id"], "reference")
        self.assertFalse(result.report["preset"]["pinned"])
        codes = [w["code"] for w in result.report["warnings"]]
        self.assertIn("preset_unpinned", codes)

    def test_pin_accepted_and_recorded(self):
        from moss_photon.core import presets
        pin = presets.pin_of(presets.catalog()["reference"])
        result = ss.run(self.tiny, preset_pin=pin)
        self.assertTrue(result.report["preset"]["pinned"])
        self.assertNotIn("preset_unpinned", [w["code"] for w in result.report["warnings"]])


class NodeLevelTest(unittest.TestCase):
    """The nodes themselves (what ComfyUI calls), against the stock node-level path, 50 steps, SLG 22."""

    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.nodes import sampler
        cls.torch, cls.sampler = torch, sampler

    def test_basic_and_advanced_nodes_equal_the_stock_path(self):
        tiny = self.tiny
        pos, neg = tiny.make_cond(23), tiny.make_cond(9)
        latent = {"samples": tiny.latent.clone()}
        stock = ss.stock_run(tiny, ss.native_slg(tiny.patcher, 22), positive=pos, negative=neg, steps=50)
        basic = self.sampler.MossPhotonSampler().sample(tiny.patcher, pos, neg, latent, tiny_moss.SEED, "reference", 0.0,
                                                        preview=False)  # cfg 0 = the preset value
        advanced = self.sampler.MossPhotonSamplerAdvanced().sample(
            tiny.patcher, pos, neg, latent, tiny_moss.SEED, "reference", 3.0, steps=0, slg_mode="from_preset",
            slg_layers="", slg_start=-1.0, slg_end=-1.0, solver="euler", cfg_profile="constant",
            noise_policy="native", max_forwards=0, fingerprint="off", preview=False)
        for node_out in (basic, advanced):
            out_latent, out_report = node_out["result"]
            self.assertTrue(self.torch.equal(out_latent["samples"], stock))
            self.assertEqual(out_report["forwards"]["sample_forwards"], 100)
            self.assertIn("MossPhoton", node_out["ui"]["text"][0])
        self.assertEqual(basic["result"][1]["node"], "MossPhotonSampler")
        self.assertEqual(advanced["result"][1]["node"], "MossPhotonSamplerAdvanced")
        self.assertEqual(advanced["result"][1]["preset"]["diff"], [])
        self.assertEqual(basic["result"][1]["preset"]["diff"], [])
        self.assertFalse(basic["result"][1]["label"]["customized"])
        self.assertFalse(advanced["result"][1]["label"]["customized"])  # an explicit 3.0 equals the preset: no diff
        self.assertIn("cfg 3 from the preset", basic["ui"]["text"][0])
        self.assertIn("cfg 3 from the user (widget)", advanced["ui"]["text"][0])
        self.assertEqual(basic["result"][1]["recipe_hash"], advanced["result"][1]["recipe_hash"])


class ReferenceExactLabelTest(unittest.TestCase):
    """With the SFT identity (monkeypatched, the fixture has no file) the same run is ``reference_exact``."""

    def test_known_sft_identity_gives_reference_exact(self):
        from unittest import mock
        from moss_photon.core import consts, fingerprint
        tiny = tiny_moss.build_tiny_moss()
        sft = dict(fingerprint.KNOWN_CHECKPOINTS[consts.SFT_BF16_SHA256])
        ident = {"identity": {"sha256": consts.SFT_BF16_SHA256, "size": 10164655808,
                              "basename": "moss_image_sft_bf16.safetensors", "hash_source": "cache",
                              "classified_by": "known_table", "official": True, "variant": "bf16",
                              "verified_matrix": True},
                 "family": sft["family"], "warnings": []}
        with mock.patch.object(fingerprint, "identify_checkpoint", return_value=ident):
            result = ss.run(tiny)
        self.assertEqual(result.report["label"]["parity_class"], "reference_exact")
        self.assertEqual(result.report["model"]["family"], "moss-image-sft")


if __name__ == "__main__":
    unittest.main()
