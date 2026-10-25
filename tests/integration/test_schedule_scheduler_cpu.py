"""MossPhotonSchedule scheduler widget on the tiny MOSS, CPU (MP-30 M5): the turbo presets use ``simple``."""
import unittest

from . import tiny_moss

# docs/03-turbo.md 3.2: the native turbo graph's 8-step simple schedule at shift 5.
TURBO_8_SIMPLE_SHA256 = "c4fe20a8f589897444d75ce8b7a6bb594b7c5c991c8e8bc7c1d18b3cdc20f693"


class ScheduleSchedulerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        import comfy.samplers
        from moss_photon.core import sigma
        from moss_photon.nodes.schedule import MossPhotonSchedule
        cls.torch, cls.samplers, cls.sigma, cls.node = torch, comfy.samplers, sigma, MossPhotonSchedule
        cls.latent = {"samples": cls.tiny.latent}

    def test_simple_with_8_steps_is_the_native_turbo_schedule(self):
        sigmas, info = self.node().build(self.tiny.patcher, self.latent, 8, 5.0, "reference", "simple")
        self.assertEqual(self.sigma.sigma_hash(sigmas), TURBO_8_SIMPLE_SHA256)
        native = self.samplers.calculate_sigmas(self.tiny.patcher.get_model_object("model_sampling"), "simple", 8)
        self.assertTrue(self.torch.equal(sigmas, native))
        self.assertEqual((info["scheduler"], info["steps"], info["n_sigmas"]), ("simple", 8, 9))
        self.assertTrue(info["is_native_reference"])
        self.assertEqual(info["sigmas_sha256"], TURBO_8_SIMPLE_SHA256)

    def test_sgm_uniform_stays_the_default_and_differs_at_8_steps(self):
        default, info = self.node().build(self.tiny.patcher, self.latent, 8, 5.0, "reference")
        self.assertEqual(info["scheduler"], "sgm_uniform")
        self.assertNotEqual(self.sigma.sigma_hash(default), TURBO_8_SIMPLE_SHA256)  # control: the widget matters


if __name__ == "__main__":
    unittest.main()
