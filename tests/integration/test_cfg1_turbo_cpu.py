"""CFG 1 / turbo path (docs/03-turbo.md 3.3, 5.3; MP-9 criterion): one forward per step, SLG not applied, the negative
prompt ignored, info codes ``negative_unused_cfg1`` / ``slg_ineffective_cfg1``, ``parity_class = native_turbo`` for the
turbo presets, the turbo scheduler is ``simple``. The turbo runs use the real ``turbo_8step`` / ``turbo_identity`` files
and synthetic distill / character LoRAs on the tiny model.
"""
import unittest

from . import _comfy_env, tiny_moss
from . import sampler_support as ss


class Cfg1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.core import errors, solver
        cls.torch, cls.errors, cls.solver = torch, errors, solver

    def test_one_forward_per_step_slg_not_applied_and_equal_to_the_stock_cfg1_run(self):
        tiny = self.tiny
        result = ss.run(tiny, cfg=1.0, overrides={"steps": 6})
        fw = result.report["forwards"]
        self.assertEqual((fw["expected_branch_forwards"], fw["model_calls"], fw["sample_forwards"],
                          fw["slg_skip_calls"], fw["match"]), (6, 6, 6, 0, True))
        stock = ss.stock_run(tiny, tiny.patcher, positive=tiny.make_cond(23), negative=tiny.make_cond(9), steps=6,
                             cfg=1.0)
        self.assertTrue(self.torch.equal(result.latent["samples"], stock))
        self.assertNotIn("sampler_calc_cond_batch_function", tiny.patcher.model_options)

    def test_the_negative_prompt_is_ignored(self):
        a = ss.run(self.tiny, cfg=1.0, negative=self.tiny.make_cond(9), overrides={"steps": 6})
        b = ss.run(self.tiny, cfg=1.0, negative=self.tiny.make_cond(31), overrides={"steps": 6})
        self.assertTrue(self.torch.equal(a.latent["samples"], b.latent["samples"]))
        self.assertNotEqual(a.report["conditioning"]["negative"]["sha256"], b.report["conditioning"]["negative"]["sha256"])

    def test_info_codes_and_the_requested_slg_stays_in_the_report(self):
        result = ss.run(self.tiny, cfg=1.0, overrides={"steps": 6})
        by_code = {w["code"]: w for w in result.report["warnings"]}
        self.assertEqual(by_code["negative_unused_cfg1"]["severity"], "info")
        self.assertEqual(by_code["slg_ineffective_cfg1"]["severity"], "info")
        self.assertEqual(result.report["sampling"]["slg"]["layers"], [22])

    def test_with_slg_off_only_the_negative_info_is_emitted(self):
        result = ss.run(self.tiny, cfg=1.0, overrides={"steps": 6, "slg_mode": "off"})
        self.assertIn("negative_unused_cfg1", ss.codes(result.report))
        self.assertNotIn("slg_ineffective_cfg1", ss.codes(result.report))

    def test_above_cfg1_neither_info_is_emitted(self):
        result = ss.run(self.tiny, cfg=1.01, overrides={"steps": 6})
        self.assertEqual(result.report["forwards"]["sample_forwards"], 12)
        self.assertNotIn("negative_unused_cfg1", ss.codes(result.report))
        self.assertNotIn("slg_ineffective_cfg1", ss.codes(result.report))
        self.assertEqual(result.report["forwards"]["slg_skip_calls"], 6)

    def test_a_foreign_cond_hook_does_not_conflict_when_slg_cannot_apply(self):
        hooked = ss.native_slg(self.tiny.patcher, 22)
        with ss.NoSample():
            report = ss.run(self.tiny, model=hooked, cfg=1.0).report
        self.assertEqual(report["label"]["modifiers"], ["cond_batch_function"])


class TurboTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        import comfy.samplers
        from moss_photon.core import errors, presets, report, sigma, solver
        cls.torch, cls.errors, cls.presets, cls.report, cls.sigma, cls.solver = torch, errors, presets, report, sigma, solver
        base = cls.tiny.patcher
        cls.distill = ss.apply_lora(base, ss.make_lora_dict(base, seed=21), 1.0)
        cls.distill_profile = ss.make_profile(cls.distill, base_model=base, role="distill",
                                              name="lodestones-8-steps-dmd2-lora.safetensors")
        cls.stack = ss.apply_lora(cls.distill, ss.make_lora_dict(base, seed=22), 0.8)
        cls.stack_profile = ss.retarget_profile(ss.read_contract("identity_profile_turbo.json"), cls.stack)
        cls.simple8 = comfy.samplers.calculate_sigmas(base.get_model_object("model_sampling"), "simple", 8)

    def turbo(self, preset="turbo_8step", model=None, profile=None, **kw):
        kw.setdefault("cfg", 1.0)
        with ss.known_sft():
            return ss.run(self.tiny, model=model or self.distill, identity_profile=profile or self.distill_profile,
                          preset_id=preset, **kw)

    def test_turbo_8step_is_native_turbo_and_equals_the_native_turbo_graph(self):
        tiny = self.tiny
        result = self.turbo()
        report = result.report
        self.assertEqual(self.report.validate_report(report), [])
        self.assertEqual(report["label"]["parity_class"], "native_turbo")
        self.assertEqual(report["label"]["not_reference_reasons"], [])
        self.assertEqual((report["sampling"]["steps"], report["sampling"]["scheduler"], report["sampling"]["cfg"]),
                         (8, "simple", 1.0))
        self.assertEqual(report["sampling"]["slg"]["mode"], "off")
        self.assertEqual(report["sampling"]["sigmas"]["sha256"], self.sigma.sigma_hash(self.simple8))
        fw = report["forwards"]
        self.assertEqual((fw["expected_branch_forwards"], fw["model_calls"], fw["sample_forwards"],
                          fw["slg_skip_calls"], fw["match"]), (8, 8, 8, 0, True))
        stock = ss.stock_run(tiny, self.distill, positive=tiny.make_cond(23), negative=tiny.make_cond(9), steps=8,
                             cfg=1.0, scheduler="simple")
        self.assertTrue(self.torch.equal(result.latent["samples"], stock))
        self.assertEqual(report["lora"]["stack"][0]["role"], "distill")
        self.assertIn("negative_unused_cfg1", ss.codes(report))
        self.assertFalse(report["label"]["recommended_final"])

    def test_the_turbo_scheduler_is_not_the_reference_scheduler(self):
        reference = self.torch.tensor(self.tiny.sigmas.tolist())
        self.assertNotEqual(self.sigma.sigma_hash(self.simple8), self.sigma.sigma_hash(reference))

    def test_the_default_cfg_widget_of_3_is_refused_by_the_turbo_bounds(self):
        with self.assertRaises(self.errors.InvalidOverride) as cm:
            self.turbo(cfg=3.0)
        self.assertIn("cfg", str(cm.exception))

    def node_sample(self, cfg, preset="turbo_8step"):
        """The simple node, the way ComfyUI calls it, with the distill chain."""
        from moss_photon.nodes import sampler
        tiny = self.tiny
        with ss.known_sft():
            return sampler.MossPhotonSampler().sample(
                self.distill, tiny.make_cond(23), tiny.make_cond(9), {"samples": tiny.latent.clone()}, tiny_moss.SEED,
                preset, cfg, identity_profile=self.distill_profile, preview=False)

    def test_the_simple_node_with_the_default_cfg_runs_turbo_with_the_preset_cfg(self):
        """MP-28: widget default 0 = preset value (1.0 here), so turbo needs no manual cfg."""
        from moss_photon.nodes import sampler
        self.assertEqual(sampler.MossPhotonSampler.INPUT_TYPES()["required"]["cfg"][1]["default"], 0.0)
        out = self.node_sample(0.0)
        report = out["result"][1]
        self.assertEqual((report["sampling"]["cfg"], report["label"]["customized"], report["preset"]["diff"]),
                         (1.0, False, []))
        # the node uses the real bf16 reference set, the float32 fixture reads as a modifier; nothing else differs
        self.assertEqual(report["label"]["modifiers"], ["weight_dtype:torch.float32"])
        self.assertEqual(report["label"]["not_reference_reasons"], [])
        self.assertEqual(report["forwards"]["sample_forwards"], 8)
        self.assertIn("cfg 1 from the preset", out["ui"]["text"][0])
        self.assertNotIn("turbo_setting_outside_verified", ss.codes(report))

    def test_an_explicit_cfg_of_3_on_turbo_is_still_refused(self):
        with self.assertRaises(self.errors.InvalidOverride):
            self.node_sample(3.0)

    def test_an_explicit_cfg_inside_the_bounds_is_an_override_with_a_diff(self):
        report = self.node_sample(1.5)["result"][1]
        self.assertTrue(report["label"]["customized"])
        self.assertIn("sampling.cfg", [d["path"] for d in report["preset"]["diff"]])
        self.assertIn("turbo_setting_outside_verified", ss.codes(report))

    def test_an_explicit_cfg_equal_to_the_preset_is_not_customized(self):
        report = self.node_sample(1.0)["result"][1]
        self.assertEqual((report["label"]["customized"], report["preset"]["diff"]), (False, []))

    def test_turbo_without_a_distill_lora_stops_before_any_forward(self):
        model = self.tiny.patcher.clone()
        stats = self.solver.ForwardStats()
        self.solver.install_counter(model, stats)
        with self.assertRaises(self.errors.MissingLora):
            ss.run(self.tiny, model=model, preset_id="turbo_8step", cfg=1.0)
        self.assertEqual(stats.calls, 0)

    def test_outside_the_verified_range_warns_and_forwards_follow(self):
        tiny = self.tiny
        result = self.turbo(cfg=1.5, overrides={"steps": 12})
        report = result.report
        fw = report["forwards"]
        self.assertEqual((fw["expected_branch_forwards"], fw["sample_forwards"], fw["slg_skip_calls"]), (24, 24, 0))
        outside = [w for w in report["warnings"] if w["code"] == "turbo_setting_outside_verified"]
        self.assertEqual(sorted(w["param"] for w in outside), ["cfg", "steps"])
        self.assertTrue(report["label"]["customized"])
        self.assertEqual(report["label"]["parity_class"], "native_turbo")
        self.assertNotIn("negative_unused_cfg1", ss.codes(report))
        stock = ss.stock_run(tiny, self.distill, positive=tiny.make_cond(23), negative=tiny.make_cond(9), steps=12,
                             cfg=1.5, scheduler="simple")
        self.assertTrue(self.torch.equal(result.latent["samples"], stock))

    def test_the_budget_cap_of_the_preset_holds(self):
        with self.assertRaises(self.errors.BudgetExceeded):
            self.turbo(max_forwards=7)

    def test_custom_sigmas_break_native_turbo(self):
        result = self.turbo(sigmas=[1.0, 0.7, 0.3, 0.0])
        self.assertEqual(result.report["label"]["parity_class"], "not_reference")
        self.assertIn("custom_sigmas", result.report["label"]["not_reference_reasons"])

    def test_turbo_8step_with_a_character_lora_warns(self):
        with ss.NoSample():
            result = self.turbo(model=self.stack, profile=self.stack_profile)
        self.assertIn("preset_lora_roles_exceeded", ss.codes(result.report))

    def test_turbo_identity_runs_the_stack_and_is_native_turbo(self):
        with ss.NoSample():
            result = self.turbo("turbo_identity", model=self.stack, profile=self.stack_profile)
        self.assertEqual(result.report["label"]["parity_class"], "native_turbo")
        self.assertNotIn("preset_lora_roles_exceeded", ss.codes(result.report))
        self.assertEqual([s["role"] for s in result.report["lora"]["stack"]], ["distill", "character"])
        self.assertEqual(result.report["preset"]["id"], "turbo_identity")

    def test_turbo_identity_without_the_character_lora(self):
        with self.assertRaises(self.errors.MissingLora):
            self.turbo("turbo_identity")

    def test_a_distill_lora_in_a_reference_preset_is_incompatible(self):
        with self.assertRaises(self.errors.IncompatibleLora):
            self.turbo("reference", cfg=3.0)

    def test_advanced_with_a_distill_lora_warns_and_ignores_slg_at_cfg1(self):
        with ss.NoSample():
            result = self.turbo("advanced", overrides={"steps": 8})
        codes = ss.codes(result.report)
        self.assertIn("distill_lora_non_turbo", codes)
        self.assertIn("slg_ineffective_cfg1", codes)
        self.assertIn("negative_unused_cfg1", codes)
        self.assertEqual(result.report["label"]["parity_class"], "not_reference")

    def test_the_turbo_report_is_valid_with_the_stack(self):
        with ss.NoSample():
            result = self.turbo("turbo_identity", model=self.stack, profile=self.stack_profile)
        self.assertEqual(self.report.validate_report(result.report), [])


if __name__ == "__main__":
    unittest.main()
