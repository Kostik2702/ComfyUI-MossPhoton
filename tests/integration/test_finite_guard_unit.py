"""guard.check_finite on real tensors (the fake-tensor variant is in tests/unit/sampler/test_guard.py)."""
import unittest

from . import _comfy_env


class FiniteGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _comfy_env.setup()
        import torch
        from moss_photon.core import errors, guard
        cls.torch, cls.errors, cls.guard = torch, errors, guard

    def check(self, t, where="step 2"):
        return self.guard.check_finite(t, node="MossPhotonSampler", where=where)

    def test_finite_tensors_pass(self):
        for dtype in (self.torch.float32, self.torch.bfloat16, self.torch.float16):
            self.assertIsNone(self.check(self.torch.randn(2, 4, 4).to(dtype)))
        self.assertIsNone(self.check(self.torch.zeros(0)))

    def test_nan_inf_and_negative_inf_raise(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            t = self.torch.ones(2, 3, 3)
            t[1, 2, 0] = bad
            with self.subTest(bad=bad):
                with self.assertRaises(self.errors.NonFiniteOutput) as cm:
                    self.check(t)
                self.assertIn("step 2", str(cm.exception))

    def test_a_single_bad_element_in_bfloat16_is_found(self):
        t = self.torch.ones(1, 128, 8, 8, dtype=self.torch.bfloat16)
        t[0, 127, 7, 7] = float("nan")
        with self.assertRaises(self.errors.NonFiniteOutput):
            self.check(t, "final latent")

    def test_the_tensor_is_not_modified(self):
        t = self.torch.randn(3, 3)
        before = t.clone()
        self.check(t)
        self.assertTrue(self.torch.equal(t, before))


if __name__ == "__main__":
    unittest.main()
