"""Noise policies against comfy.sample.prepare_noise on CPU (design A3, check C4, 6.1)."""
import hashlib
import unittest

from . import _comfy_env


class NoiseCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _comfy_env.setup()
        import torch
        import comfy.sample
        from moss_photon.core import errors, rng
        cls.torch, cls.rng, cls.errors = torch, rng, errors
        cls.prepare_noise = staticmethod(comfy.sample.prepare_noise)  # a plain function must not become a method
        cls.lat3 = torch.zeros(3, 128, 8, 8)

    def test_native_noise_equals_prepare_noise(self):
        for seed in (0, 5, 42, 0xFFFFFFFFFFFFFFFF):
            with self.subTest(seed=seed):
                ours = self.rng.make_noise(self.lat3, seed, "native")
                self.assertTrue(self.torch.equal(ours, self.prepare_noise(self.lat3, seed)))
                self.assertEqual((ours.dtype, ours.device.type), (self.torch.float32, "cpu"))

    def test_native_noise_with_batch_index_equals_prepare_noise(self):
        idx = [0, 2, 2]
        self.assertTrue(self.torch.equal(self.rng.make_noise(self.lat3, 7, "native", batch_index=idx),
                                         self.prepare_noise(self.lat3, 7, idx)))

    def test_own_generator_equals_prepare_noise_batch3(self):
        own = self.rng.own_generator_noise(self.lat3.shape, self.lat3.dtype, 5)
        self.assertTrue(self.torch.equal(own, self.prepare_noise(self.lat3, 5)))

    def test_seed_plus_i_item_equals_single_seed(self):
        out = self.rng.make_noise(self.lat3, 5, "seed_plus_i")
        self.assertEqual(tuple(out.shape), (3, 128, 8, 8))
        for i in range(3):
            single = self.prepare_noise(self.torch.zeros(1, 128, 8, 8), 5 + i)
            with self.subTest(item=i):
                self.assertTrue(self.torch.equal(out[i:i + 1], single))

    def test_native_item1_differs_from_seed_plus_1(self):
        native = self.rng.make_noise(self.lat3, 5, "native")
        spi = self.rng.make_noise(self.lat3, 5, "seed_plus_i")
        self.assertTrue(self.torch.equal(native[0:1], spi[0:1]))  # item 0 is the same stream start
        self.assertFalse(self.torch.equal(native[1:2], spi[1:2]))
        self.assertFalse(self.torch.equal(native[1:2], native[0:1]))

    def test_own_generator_leaves_global_rng(self):
        self.torch.manual_seed(999)
        before = self.torch.rand(1).item()
        self.torch.manual_seed(999)
        self.rng.own_generator_noise(self.lat3.shape, self.lat3.dtype, 5)
        self.rng.make_noise(self.lat3, 5, "seed_plus_i")
        self.assertEqual(self.torch.rand(1).item(), before)

    def test_native_documented_global_reseed(self):
        """native == stock KSampler behaviour: it reseeds the global CPU RNG (documented in rng.py, not hidden)."""
        self.rng.make_noise(self.lat3, 123, "native")
        after = self.torch.rand(1).item()
        self.torch.manual_seed(123)
        self.torch.randn(self.lat3.shape)
        self.assertEqual(self.torch.rand(1).item(), after)

    def test_seed_max_and_wrap(self):
        top = 0xFFFFFFFFFFFFFFFF
        out = self.rng.make_noise(self.torch.zeros(2, 4, 2, 2), top, "seed_plus_i")
        self.assertEqual(tuple(out.shape), (2, 4, 2, 2))
        self.assertTrue(self.torch.equal(out[0:1], self.prepare_noise(self.torch.zeros(1, 4, 2, 2), top)))
        self.assertTrue(self.torch.equal(out[1:2], self.prepare_noise(self.torch.zeros(1, 4, 2, 2), 0)))  # wrapped
        self.assertEqual(self.rng.subseeds(top, 2, "seed_plus_i"), [top, 0])

    def test_dtype_follows_the_latent(self):
        lat = self.torch.zeros(2, 4, 2, 2, dtype=self.torch.bfloat16)
        for policy in self.rng.NOISE_POLICIES:
            with self.subTest(policy=policy):
                out = self.rng.make_noise(lat, 3, policy)
                self.assertEqual(out.dtype, self.torch.bfloat16)
        self.assertTrue(self.torch.equal(self.rng.make_noise(lat, 3, "native"), self.prepare_noise(lat, 3)))

    def test_seed_plus_i_rejects_batch_index(self):
        with self.assertRaises(self.errors.UnsupportedFeature):
            self.rng.make_noise(self.lat3, 1, "seed_plus_i", batch_index=[0, 1, 2])

    def test_tensor_sha256_is_the_float32_hash(self):
        t = self.prepare_noise(self.lat3, 5)
        self.assertEqual(self.rng.tensor_sha256(t), hashlib.sha256(t.numpy().tobytes()).hexdigest())
        self.assertNotEqual(self.rng.tensor_sha256(t), self.rng.tensor_sha256(self.prepare_noise(self.lat3, 6)))
        bf = t.to(self.torch.bfloat16)
        self.assertEqual(self.rng.tensor_sha256(bf), hashlib.sha256(bf.float().numpy().tobytes()).hexdigest())

    def test_native_batch_is_one_stream(self):
        """Item 1 of a native batch continues the stream: it is NOT the single-image noise of seed + 1."""
        native = self.rng.make_noise(self.lat3, 5, "native")
        for i in range(1, 3):
            self.assertFalse(self.torch.equal(native[i:i + 1], self.prepare_noise(self.torch.zeros(1, 128, 8, 8), 5 + i)))


if __name__ == "__main__":
    unittest.main()
