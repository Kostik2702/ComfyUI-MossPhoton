"""A forward that returns NaN from step 1 on stops the run with ``non_finite_output`` at step 1 (design 6.4)."""
import unittest

from . import _comfy_env, tiny_moss
from . import sampler_support as ss


class NonFiniteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.core import errors
        cls.torch, cls.errors = torch, errors

    def poisoned_forward(self, from_sigma, calls):
        net = self.tiny.model.diffusion_model
        original = net.forward

        def forward(x, timesteps, *args, **kwargs):
            out = original(x, timesteps, *args, **kwargs)
            calls.append(float(timesteps[0]))
            if float(timesteps[0]) <= from_sigma:
                return out * float("nan")
            return out
        return net, original, forward

    def test_nan_at_step_1_is_reported_at_step_1_and_stops_the_run(self):
        calls = []
        sigmas = self.tiny.sigmas.tolist()
        net, original, forward = self.poisoned_forward(sigmas[1] + 1e-6, calls)
        net.forward = forward
        try:
            with self.assertRaises(self.errors.NonFiniteOutput) as cm:
                ss.run(self.tiny, overrides={"steps": 6})
        finally:
            del net.forward
        self.assertEqual(cm.exception.issue.detail, {"where": "step 1"})
        self.assertIn("step 1", str(cm.exception))
        # step 0 (2 forwards) and step 1 (2 forwards) ran, nothing after the poisoned step
        self.assertEqual(len(calls), 4)

    def test_the_model_is_healthy_again_after_the_failure(self):
        net = self.tiny.model.diffusion_model
        self.assertNotIn("forward", net.__dict__)
        result = ss.run(self.tiny, overrides={"steps": 6})
        self.assertTrue(result.report["output"]["finite"])
        self.assertEqual(result.report["guards"]["finite_check"], "passed")

    def test_nan_only_in_the_last_step_is_still_caught(self):
        calls = []
        sigmas = self.tiny.sigmas.tolist()
        net, original, forward = self.poisoned_forward(sigmas[5] + 1e-6, calls)
        net.forward = forward
        try:
            with self.assertRaises(self.errors.NonFiniteOutput) as cm:
                ss.run(self.tiny, overrides={"steps": 6})
        finally:
            del net.forward
        self.assertEqual(cm.exception.issue.detail, {"where": "step 5"})


if __name__ == "__main__":
    unittest.main()
