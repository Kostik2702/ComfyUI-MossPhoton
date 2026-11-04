"""Typed preflight errors on the tiny model: no forward ever runs (design 6.4, D4)."""
import unittest

from . import _comfy_env, tiny_moss
from . import sampler_support as ss


class InvalidInputsTinyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.core import errors, solver
        cls.torch, cls.errors, cls.solver = torch, errors, solver

    def counted_model(self, base=None):
        model = (base or self.tiny.patcher).clone()
        stats = self.solver.ForwardStats()
        self.solver.install_counter(model, stats)
        return model, stats

    def test_four_channel_non_empty_latent(self):
        model, stats = self.counted_model()
        with self.assertRaises(self.errors.InvalidLatent) as cm:
            ss.run(self.tiny, model=model, latent={"samples": self.torch.ones(1, 4, 8, 8)})
        self.assertEqual(stats.calls, 0)
        self.assertIn("MossPhotonEmptyLatent", str(cm.exception))

    def test_empty_four_channel_latent_is_repaired_by_comfy_like_the_stock_node(self):
        result = ss.run(self.tiny, latent={"samples": self.torch.zeros(1, 4, 16, 16), "downscale_ratio_spacial": 8},
                        overrides={"steps": 6})
        self.assertEqual(tuple(result.latent["samples"].shape), (1, 128, 8, 8))
        self.assertTrue(result.report["inputs"]["latent_rescaled_by_native_fix"])
        self.assertEqual(result.report["inputs"]["latent_raw_shape"], [1, 4, 16, 16])
        self.assertIn("latent_rescaled_by_native_fix", ss.codes(result.report))
        self.assertNotIn("downscale_ratio_spacial", result.latent)

    def test_missing_samples(self):
        with self.assertRaises(self.errors.InvalidLatent):
            ss.run(self.tiny, latent={})

    def test_noise_mask_is_a_second_queue_feature(self):
        latent = {"samples": self.tiny.latent.clone(), "noise_mask": self.torch.ones(1, 1, 8, 8)}
        with self.assertRaises(self.errors.UnsupportedSourceInput):
            ss.run(self.tiny, latent=latent)

    def test_model_sampling_sd3_is_refused_with_the_auraflow_hint(self):
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        sd3 = ModelSamplingSD3().patch(self.tiny.patcher, 3.0)[0]
        model, stats = self.counted_model(sd3)
        with self.assertRaises(self.errors.InvalidModelSampling) as cm:
            ss.run(self.tiny, model=model)
        self.assertIn("ModelSamplingAuraFlow", str(cm.exception))
        self.assertEqual(stats.calls, 0)

    def test_foreign_cond_hook_with_slg_requested_is_conflicting_patch(self):
        hooked = ss.native_slg(self.tiny.patcher, 22)
        model, stats = self.counted_model(hooked)
        with self.assertRaises(self.errors.ConflictingModelPatch):
            ss.run(self.tiny, model=model)
        self.assertEqual(stats.calls, 0)
        # the incoming model keeps its hook untouched
        self.assertIn("sampler_calc_cond_batch_function", hooked.model_options)

    def test_wrong_text_dim(self):
        import torch
        bad = [[torch.randn(1, 5, 32), {}]]
        with self.assertRaises(self.errors.InvalidConditioning):
            ss.run(self.tiny, positive=bad)

    def test_empty_conditioning_list(self):
        with self.assertRaises(self.errors.InvalidConditioning):
            ss.run(self.tiny, negative=[])

    def test_vae_with_the_wrong_latent_space(self):
        import types
        with self.assertRaises(self.errors.InvalidVae):
            ss.run(self.tiny, vae=types.SimpleNamespace(latent_channels=4, downscale_ratio=8))

    def test_a_compatible_vae_is_only_checked(self):
        import types
        result = ss.run(self.tiny, vae=types.SimpleNamespace(latent_channels=128, downscale_ratio=16),
                        overrides={"steps": 6})
        self.assertTrue(result.report["output"]["finite"])

    def test_not_a_moss_model(self):
        import types
        fake = types.SimpleNamespace(model=object(), model_options={}, patches={}, patches_uuid="x",
                                     get_model_object=lambda name: None, model_dtype=lambda: None, load_device="cpu",
                                     is_dynamic=lambda: False)
        with self.assertRaises(self.errors.UnsupportedModel):
            ss.run(self.tiny, model=fake)

    def test_identity_safe_without_a_lora_runs_with_a_note(self):
        """MP-84: a role is never required: the run goes on (it stopped with ``missing_lora`` before) and the report notes it."""
        model, stats = self.counted_model()
        result = ss.run(self.tiny, model=model, preset_id="identity_safe", overrides={"steps": 4})
        self.assertGreater(stats.calls, 0)
        self.assertIn("lora_not_recorded", ss.codes(result.report))

    def test_wardrobe_and_scene_safe_blocks(self):
        constraints = ss.bound_constraints(self.tiny, wardrobe=False)
        with self.assertRaises(self.errors.MissingRequiredInput):
            ss.run(self.tiny, preset_id="wardrobe_safe", constraints=constraints)
        constraints = ss.bound_constraints(self.tiny, wardrobe=True, scene=False)
        with self.assertRaises(self.errors.MissingRequiredInput):
            ss.run(self.tiny, preset_id="scene_safe", constraints=constraints)

    def test_scene_safe_refuses_a_source_latent(self):
        constraints = ss.bound_constraints(self.tiny)
        with ss.NoSample():
            with self.assertRaises(self.errors.UnsupportedSourceInput):
                ss.run(self.tiny, preset_id="scene_safe", constraints=constraints,
                       latent={"samples": self.torch.ones(1, 128, 8, 8)})
            ok = ss.run(self.tiny, preset_id="scene_safe", constraints=constraints)
        self.assertEqual(ok.report["preset"]["id"], "scene_safe")

    def test_pin_and_preset_errors(self):
        with ss.NoSample():
            pinned = ss.run(self.tiny, preset_pin="reference@2.0.0")   # MP-84: a pin is advisory
            slg = ss.run(self.tiny, overrides={"slg_layers": [99]})    # a layer the model does not have is dropped
        self.assertIn("preset_pin_mismatch", ss.codes(pinned.report))
        self.assertFalse(pinned.report["preset"]["pinned"])
        self.assertEqual(slg.report["sampling"]["slg"]["mode"], "off")
        self.assertIn("setting_adjusted", ss.codes(slg.report))
        with self.assertRaises(self.errors.UnknownPreset):
            ss.run(self.tiny, preset_id="turbo_9000")
        with self.assertRaises(self.errors.InvalidOverride):   # a value of the wrong type: only an API caller
            ss.run(self.tiny, overrides={"slg_layers": "22"})

    def test_a_plan_over_max_forwards_is_a_warning_and_the_run_goes_on(self):
        model, stats = self.counted_model()
        result = ss.run(self.tiny, model=model, max_forwards=10, overrides={"steps": 6})
        self.assertIn("forward_budget_exceeded", ss.codes(result.report))
        self.assertEqual(stats.calls, result.report["forwards"]["model_calls"])
        self.assertGreater(stats.calls, 0)

    def test_layout_outside_the_canvas(self):
        constraints = ss.bound_constraints(self.tiny)
        constraints["layout"] = {"items": 1, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
                                 "bboxes": [[-70, -70, 70, 70]], "extent": [-70, -70, 70, 70], "experimental": True}
        with self.assertRaises(self.errors.InvalidLayout):  # the canvas is 128 px: edges allowed up to +-64
            ss.run(self.tiny, constraints=constraints)
        constraints["layout"]["extent"] = [-64, -64, 64, 64]
        with ss.NoSample():
            ss.run(self.tiny, constraints=constraints)

    def test_phase_plan_is_rejected(self):
        tiny = self.tiny
        model = ss.apply_lora(tiny.patcher, ss.make_lora_dict(tiny.patcher))
        profile = ss.make_profile(model, base_model=tiny.patcher)
        profile["phase_plan"] = {"kind": "phased"}
        with self.assertRaises(self.errors.UnsupportedFeature):
            ss.run(tiny, model=model, identity_profile=profile)

    def test_seed_plus_i_needs_no_batch_index(self):
        latent = {"samples": self.tiny.latent.clone(), "batch_index": [0]}
        with self.assertRaises(self.errors.UnsupportedFeature):
            ss.run(self.tiny, latent=latent, noise_policy="seed_plus_i")


if __name__ == "__main__":
    unittest.main()
