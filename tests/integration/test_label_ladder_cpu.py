"""M2/LD3: each native sampling modifier on the incoming MODEL gives ``reference_modified`` with the right
``label.modifiers`` and the warning ``foreign_modifiers`` (design 6.4, 1.2, Appendix A C9). The sampling steps are skipped
(the label is decided before any forward); the models and their modifiers are real."""
import unittest

from . import _comfy_env, tiny_moss
from . import sampler_support as ss


class LabelLadderCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        import comfy.patcher_extension as pe
        from moss_photon.core import errors
        cls.torch, cls.pe, cls.errors = torch, pe, errors
        cls.base = cls.tiny.patcher

    def report(self, model, **kw):
        with ss.known_sft(), ss.NoSample():
            return ss.run(self.tiny, model=model, **kw).report

    def test_clean_model_is_reference_exact_with_the_sft_identity(self):
        report = self.report(self.base)
        self.assertEqual((report["label"]["parity_class"], report["label"]["modifiers"]), ("reference_exact", []))
        self.assertNotIn("foreign_modifiers", ss.codes(report))

    def expect_modified(self, model, modifiers, **kw):
        report = self.report(model, **kw)
        self.assertEqual(report["label"]["parity_class"], "reference_modified")
        self.assertEqual(report["label"]["modifiers"], modifiers)
        self.assertIn("foreign_modifiers", ss.codes(report))
        warning = next(w for w in report["warnings"] if w["code"] == "foreign_modifiers")
        for code in modifiers:
            self.assertIn(code, warning["message"])
        return report

    def test_cfg_nodes(self):
        from comfy_extras.nodes_cfg import CFGNorm, CFGZeroStar
        from comfy_extras.nodes_model_advanced import RescaleCFG
        self.expect_modified(CFGNorm.execute(self.base, 1.0).args[0], ["post_cfg_function"])
        self.expect_modified(CFGZeroStar.execute(self.base).args[0], ["post_cfg_function"])
        self.expect_modified(RescaleCFG().patch(self.base, 0.7)[0], ["cfg_function"])

    def test_full_slg_node(self):
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiT
        full = SkipLayerGuidanceDiT.execute(self.base, 3.0, 0.01, 1.0, double_layers="", single_layers="22").args[0]
        self.expect_modified(full, ["post_cfg_function"])

    def test_attention_override_function_wrapper_object_patch_and_transformer_wrapper(self):
        clone = self.base.clone()
        clone.set_model_optimized_attention(lambda *a, **k: None)
        report = self.expect_modified(clone, ["attention_override"])
        self.assertIn("non_default_attention_backend", ss.codes(report))
        clone = self.base.clone()
        clone.set_model_unet_function_wrapper(lambda f, a: f(a["input"], a["timestep"], **a["c"]))
        self.expect_modified(clone, ["model_function_wrapper"])
        clone = self.base.clone()
        clone.add_object_patch("diffusion_model.dummy", self.torch.nn.Identity())
        self.expect_modified(clone, ["object_patch:diffusion_model.dummy"])
        clone = self.base.clone()
        self.pe.add_wrapper_with_key(self.pe.WrappersMP.APPLY_MODEL, "x", lambda executor, *a, **k: executor(*a, **k),
                                     clone.model_options, is_model_options=True)
        self.expect_modified(clone, ["transformer_option:wrappers"])

    def test_native_slg_simple_hook_conflicts_when_slg_is_requested_and_labels_when_it_is_off(self):
        hooked = ss.native_slg(self.base, 22)
        with ss.known_sft(), ss.NoSample():
            with self.assertRaises(self.errors.ConflictingModelPatch):
                ss.run(self.tiny, model=hooked)
        report = self.report(hooked, overrides={"slg_mode": "off"})
        self.assertEqual(report["label"]["modifiers"], ["cond_batch_function"])
        self.assertIn("external_calc_cond_batch", ss.codes(report))
        self.assertEqual(report["label"]["parity_class"], "not_reference")  # slg_differs wins over the modifier
        self.assertIn("slg_differs", report["label"]["not_reference_reasons"])

    def test_a_modifier_beats_foreign_patches_in_the_ladder(self):
        from comfy_extras.nodes_cfg import CFGNorm
        patched = ss.apply_lora(self.base, ss.make_lora_dict(self.base), 1.0)  # native loader, no profile
        model = CFGNorm.execute(patched, 1.0).args[0]
        report = self.expect_modified(model, ["post_cfg_function"])
        self.assertGreater(report["lora"]["foreign_patches"]["extra_entries"], 0)
        self.assertIn("foreign_patches", ss.codes(report))

    def test_auraflow_at_the_native_shift_is_not_a_modifier_but_another_shift_is_a_difference(self):
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
        same = self.report(ModelSamplingAuraFlow().patch_aura(self.base, 5.0)[0])
        self.assertEqual((same["label"]["parity_class"], same["label"]["modifiers"]), ("reference_exact", []))
        other = self.report(ModelSamplingAuraFlow().patch_aura(self.base, 3.0)[0])
        self.assertEqual(other["label"]["parity_class"], "not_reference")
        self.assertIn("shift_differs", other["label"]["not_reference_reasons"])
        self.assertIn("shift_differs_from_preset", ss.codes(other))
        self.assertTrue(other["label"]["customized"])
        self.assertEqual(other["sampling"]["shift"], 3.0)
        self.assertIn({"path": "sampling.shift", "base": 5.0, "effective": 3.0}, other["preset"]["diff"])

    def test_dtype_other_than_bf16_is_a_modifier_on_the_real_reference_set(self):
        with ss.known_sft(), ss.NoSample():
            report = ss.run(self.tiny, reference_dtypes=None).report  # the fixture is float32
        self.assertEqual(report["label"]["parity_class"], "reference_modified")
        self.assertEqual(report["label"]["modifiers"], ["weight_dtype:torch.float32"])

    def test_unknown_checkpoint_is_unverified_not_exact(self):
        with ss.NoSample():
            report = ss.run(self.tiny).report
        self.assertEqual(report["label"]["parity_class"], "reference_unverified")
        self.assertIn("fingerprint_unavailable", ss.codes(report))
        self.assertEqual(report["model"]["family"], "unknown-moss")

    def test_the_modifiers_are_part_of_the_recipe_hash(self):
        from comfy_extras.nodes_cfg import CFGNorm
        clean = self.report(self.base)
        modified = self.report(CFGNorm.execute(self.base, 1.0).args[0])
        self.assertNotEqual(clean["recipe_hash"], modified["recipe_hash"])


if __name__ == "__main__":
    unittest.main()
