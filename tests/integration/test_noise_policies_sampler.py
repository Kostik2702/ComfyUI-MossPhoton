"""Noise policies through the sampler (design 6.4, D2, A3).

``native`` batch 2 equals the stock KSampler batch 2 bit for bit. ``seed_plus_i`` item i is ``allclose`` (atol 1e-4,
a batch-2 and a batch-1 forward differ at 1e-6, INV 8.2: no bit claim) to a single run with ``seed + i``, the report
lists the sub-seeds and the run is ``not_reference`` with ``noise_seed_plus_i``.
"""
import unittest

from . import _comfy_env, tiny_moss
from . import sampler_support as ss

STEPS = {"steps": 6}


class NoisePoliciesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        cls.torch = torch

    def batch(self, n):
        return {"samples": self.torch.zeros(n, 128, 8, 8)}

    def test_native_batch2_equals_the_stock_ksampler(self):
        tiny = self.tiny
        pos, neg = tiny.make_cond(23), tiny.make_cond(9)
        stock = ss.stock_run(tiny, ss.native_slg(tiny.patcher, 22), positive=pos, negative=neg, steps=6,
                             latent=self.batch(2))
        ours = ss.run(tiny, positive=pos, negative=neg, latent=self.batch(2), overrides=STEPS)
        self.assertEqual(tuple(ours.latent["samples"].shape), (2, 128, 8, 8))
        self.assertTrue(self.torch.equal(ours.latent["samples"], stock))
        self.assertEqual(ours.report["inputs"]["subseeds"], [tiny_moss.SEED])
        self.assertEqual(ours.report["inputs"]["noise_policy"], "native")
        self.assertEqual(ours.report["inputs"]["batch"], 2)
        fw = ours.report["forwards"]
        self.assertEqual((fw["expected_sample_forwards"], fw["sample_forwards"], fw["match"]), (24, 24, True))

    def test_native_items_are_not_seed_plus_i(self):
        ours = ss.run(self.tiny, latent=self.batch(2), overrides=dict(STEPS, slg_mode="off"))
        single1 = ss.run(self.tiny, seed=tiny_moss.SEED + 1, overrides=dict(STEPS, slg_mode="off"))
        self.assertFalse(self.torch.allclose(ours.latent["samples"][1:2], single1.latent["samples"], atol=1e-4))

    def test_seed_plus_i_item_i_is_the_single_run_with_seed_plus_i(self):
        batch = ss.run(self.tiny, latent=self.batch(2), noise_policy="seed_plus_i", overrides=STEPS)
        for i in (0, 1):
            single = ss.run(self.tiny, seed=tiny_moss.SEED + i, overrides=STEPS)
            self.assertTrue(self.torch.allclose(batch.latent["samples"][i:i + 1], single.latent["samples"], atol=1e-4),
                            "item %d" % i)
        self.assertEqual(batch.report["inputs"]["subseeds"], [tiny_moss.SEED, tiny_moss.SEED + 1])
        self.assertEqual(batch.report["inputs"]["noise_policy"], "seed_plus_i")

    def test_seed_plus_i_marks_the_run_not_reference(self):
        result = ss.run(self.tiny, latent=self.batch(2), noise_policy="seed_plus_i", overrides=STEPS)
        label = result.report["label"]
        self.assertEqual(label["parity_class"], "not_reference")
        self.assertIn("noise_seed_plus_i", label["not_reference_reasons"])

    def test_seed_plus_i_never_touches_the_global_rng(self):
        self.torch.manual_seed(999)
        expected = self.torch.rand(1).item()
        self.torch.manual_seed(999)
        ss.run(self.tiny, latent=self.batch(2), noise_policy="seed_plus_i", overrides=dict(STEPS, slg_mode="off"))
        self.assertEqual(self.torch.rand(1).item(), expected)

    def test_noise_hash_differs_between_the_policies_for_batch_two(self):
        a = ss.run(self.tiny, latent=self.batch(2), overrides=dict(STEPS, slg_mode="off"))
        b = ss.run(self.tiny, latent=self.batch(2), noise_policy="seed_plus_i", overrides=dict(STEPS, slg_mode="off"))
        self.assertNotEqual(a.report["inputs"]["noise_sha256"], b.report["inputs"]["noise_sha256"])


if __name__ == "__main__":
    unittest.main()
