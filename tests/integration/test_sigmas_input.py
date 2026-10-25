"""The optional ``sigmas`` input of the Advanced sampler (design 4.6, 6.4): validated before any forward; a schedule that is
not bit-equal to the native one is labelled ``custom_sigmas`` / ``not_reference``."""
import unittest

from . import _comfy_env, tiny_moss
from . import sampler_support as ss


class SigmasInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.core import errors, sigma, solver
        cls.torch, cls.errors, cls.sigma, cls.solver = torch, errors, sigma, solver

    def custom(self, values):
        return self.torch.tensor(values, dtype=self.torch.float32)

    def test_valid_custom_sigmas_run_and_are_labelled(self):
        values = [1.0, 0.8, 0.55, 0.3, 0.0]
        result = ss.run(self.tiny, sigmas=self.custom(values), overrides={"slg_mode": "off"})
        report = result.report
        self.assertEqual(report["sampling"]["steps"], 4)
        self.assertEqual(report["sampling"]["sigmas"], {
            "source": "input", "n": 5, "sha256": self.sigma.sigma_hash(values), "first": 1.0, "last": 0.0,
            "equals_native": False})
        self.assertIn("custom_sigmas", ss.codes(report))
        self.assertEqual(report["label"]["parity_class"], "not_reference")
        self.assertIn("custom_sigmas", report["label"]["not_reference_reasons"])
        self.assertEqual(report["forwards"]["expected_branch_forwards"], 8)
        self.assertEqual(report["forwards"]["sample_forwards"], 8)

    def test_the_connected_sigmas_are_what_ran(self):
        values = [1.0, 0.8, 0.55, 0.3, 0.0]
        result = ss.run(self.tiny, sigmas=self.custom(values), overrides={"slg_mode": "off"})
        import comfy.sample
        import comfy.samplers
        guider = comfy.samplers.CFGGuider(self.tiny.patcher)
        guider.set_conds(self.tiny.make_cond(23), self.tiny.make_cond(9))
        guider.set_cfg(3.0)
        noise = comfy.sample.prepare_noise(self.tiny.latent, tiny_moss.SEED)
        with self.torch.inference_mode():
            out = guider.sample(noise, self.tiny.latent.clone(), comfy.samplers.sampler_object("euler"),
                                self.custom(values), seed=tiny_moss.SEED)
        self.assertTrue(self.torch.equal(result.latent["samples"], out))

    def test_native_sigmas_passed_in_are_not_custom(self):
        native = self.tiny.sigmas.clone()
        result = ss.run(self.tiny, sigmas=native, overrides={"steps": 6})
        self.assertEqual(result.report["sampling"]["sigmas"]["source"], "input")
        self.assertTrue(result.report["sampling"]["sigmas"]["equals_native"])
        self.assertNotIn("custom_sigmas", ss.codes(result.report))
        plain = ss.run(self.tiny, overrides={"steps": 6})
        self.assertTrue(self.torch.equal(result.latent["samples"], plain.latent["samples"]))

    def test_list_input_is_accepted(self):
        result = ss.run(self.tiny, sigmas=[1.0, 0.6, 0.0], overrides={"slg_mode": "off"})
        self.assertEqual(result.report["sampling"]["sigmas"]["n"], 3)

    def test_invalid_sigmas_raise_before_any_forward(self):
        bad = {
            "nan": [1.0, float("nan"), 0.0],
            "inf": [1.0, float("inf"), 0.0],
            "not_decreasing": [1.0, 0.5, 0.6, 0.0],
            "duplicate": [1.0, 0.5, 0.5, 0.0],
            "not_ending_at_zero": [1.0, 0.7, 0.4, 0.1],
            "above_model_max": [1.5, 0.7, 0.0],
            "negative": [1.0, 0.5, -0.1, 0.0],
            "too_short": [0.0],
            "empty": [],
        }
        for name, values in bad.items():
            with self.subTest(case=name):
                model = self.tiny.patcher.clone()
                stats = self.solver.ForwardStats()
                self.solver.install_counter(model, stats)
                with self.assertRaises(self.errors.InvalidSchedule):
                    ss.run(self.tiny, model=model, sigmas=self.custom(values) if values else values)
                self.assertEqual(stats.calls, 0)

    def test_two_dimensional_sigmas_are_rejected(self):
        with self.assertRaises(self.errors.InvalidSchedule):
            ss.run(self.tiny, sigmas=self.torch.tensor([[1.0, 0.5, 0.0]]))

    def test_a_step_jump_is_only_a_warning_for_connected_sigmas(self):
        values = [1.0, 0.99, 0.98, 0.5, 0.0]  # a x49 jump between consecutive step sizes
        result = ss.run(self.tiny, sigmas=self.custom(values), overrides={"slg_mode": "off"})
        self.assertIn("sigma_step_ratio", ss.codes(result.report))

    def test_non_empty_latent_warnings(self):
        latent = {"samples": self.torch.ones(1, 128, 8, 8)}
        with ss.NoSample():
            a = ss.run(self.tiny, latent=latent, overrides={"steps": 6})
            b = ss.run(self.tiny, latent=latent, sigmas=self.custom([0.8, 0.4, 0.0]))
        self.assertIn("latent_content_ignored", ss.codes(a.report))
        self.assertIn("img2img_not_verified", ss.codes(b.report))
        self.assertFalse(a.report["inputs"]["latent_was_empty"])


if __name__ == "__main__":
    unittest.main()
