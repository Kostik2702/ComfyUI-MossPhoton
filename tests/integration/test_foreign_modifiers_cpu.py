"""Foreign sampling modifiers detected on real patchers (design 1.2 LD3/M2, check C9)."""
import unittest

from . import _comfy_env, tiny_moss

F32 = ("torch.float32",)


class ForeignModifiersCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        import comfy.patcher_extension as pe
        from moss_photon.core import adapter
        cls.torch, cls.pe, cls.adapter = torch, pe, adapter
        cls.patcher = cls.tiny.patcher

    def mods(self, patcher, ref=F32):
        return self.adapter.probe_model(patcher, reference_dtypes=ref).foreign_modifiers

    def test_probe_model_on_the_clean_tiny_fixture_has_no_foreign_modifiers(self):
        self.assertEqual(self.mods(self.patcher), ())
        self.assertEqual(self.patcher.model_options, {"transformer_options": {}})  # the premise of the rule table

    def test_clean_but_dtype_must_be_bf16(self):
        self.assertEqual(self.mods(self.patcher, ("torch.bfloat16",)), ("weight_dtype:torch.float32",))

    def test_auraflow_shift_object_patch_is_allowed(self):
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
        self.assertEqual(self.mods(ModelSamplingAuraFlow().patch_aura(self.patcher, 5.0)[0]), ())

    def test_native_cfg_nodes(self):
        from comfy_extras.nodes_cfg import CFGNorm, CFGZeroStar
        from comfy_extras.nodes_model_advanced import RescaleCFG
        self.assertEqual(self.mods(CFGNorm.execute(self.patcher, 1.0).args[0]), ("post_cfg_function",))
        self.assertEqual(self.mods(CFGZeroStar.execute(self.patcher).args[0]), ("post_cfg_function",))
        self.assertEqual(self.mods(RescaleCFG().patch(self.patcher, 0.7)[0]), ("cfg_function",))

    def test_native_slg_nodes(self):
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiT, SkipLayerGuidanceDiTSimple
        full = SkipLayerGuidanceDiT.execute(self.patcher, 3.0, 0.01, 1.0, double_layers="", single_layers="22").args[0]
        simple = SkipLayerGuidanceDiTSimple.execute(self.patcher, 0.0, 1.0, double_layers="", single_layers="22").args[0]
        self.assertEqual(self.mods(full), ("post_cfg_function",))
        self.assertEqual(self.mods(simple), ("cond_batch_function",))

    def test_cond_batch_hook_conflicts_only_when_slg_is_requested(self):
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
        simple = SkipLayerGuidanceDiTSimple.execute(self.patcher, 0.0, 1.0, double_layers="", single_layers="22").args[0]
        facts = self.adapter.probe_model(simple, reference_dtypes=F32)
        from moss_photon.core import errors
        on = self.adapter.validate_model_facts(facts, node="MossPhotonSampler", slg_requested=True)
        with self.assertRaises(errors.ConflictingModelPatch):
            errors.raise_first_error(on)
        off = self.adapter.validate_model_facts(facts, node="MossPhotonSampler", slg_requested=False)
        self.assertEqual([i.code for i in off], ["external_calc_cond_batch", "foreign_modifiers"])
        errors.raise_first_error(off)  # warnings only

    def test_attention_override(self):
        clone = self.patcher.clone()
        clone.set_model_optimized_attention(lambda *a, **k: None)
        self.assertEqual(self.mods(clone), ("attention_override",))

    def test_model_function_wrapper(self):
        clone = self.patcher.clone()
        clone.set_model_unet_function_wrapper(lambda f, a: f(a["input"], a["timestep"], **a["c"]))
        self.assertEqual(self.mods(clone), ("model_function_wrapper",))

    def test_extra_object_patch(self):
        clone = self.patcher.clone()
        clone.add_object_patch("diffusion_model.dummy", self.torch.nn.Identity())
        self.assertEqual(self.mods(clone), ("object_patch:diffusion_model.dummy",))

    def test_transformer_wrapper(self):
        clone = self.patcher.clone()
        self.pe.add_wrapper_with_key(self.pe.WrappersMP.APPLY_MODEL, "x", lambda executor, *a, **k: executor(*a, **k),
                                     clone.model_options, is_model_options=True)
        self.assertEqual(self.mods(clone), ("transformer_option:wrappers",))

    def test_denoise_mask_and_pre_cfg_and_unknown_option(self):
        clone = self.patcher.clone()
        clone.model_options["denoise_mask_function"] = lambda *a: None
        clone.model_options["sampler_pre_cfg_function"] = [lambda a: a["conds_out"]]
        clone.model_options["something_new"] = 1
        self.assertEqual(self.mods(clone), ("denoise_mask_function", "model_option:something_new", "pre_cfg_function"))

    def test_weight_patches_are_not_modifiers(self):
        import comfy.sd
        from tests.integration.test_adapter_cpu import make_lora
        lora = make_lora(self.patcher, self.torch, skip=set(), seed=11)
        with_lora, _ = comfy.sd.load_lora_for_models(self.patcher, None, lora, 1.0, 0)
        self.assertGreater(len(with_lora.patches), 0)
        self.assertEqual(self.mods(with_lora), ())

    def test_every_modifier_is_absent_from_the_original(self):
        """Control for the whole file: the detection tests above ran on clones; the source patcher stayed clean."""
        self.assertEqual(self.mods(self.patcher), ())
        self.assertEqual(self.patcher.model_options, {"transformer_options": {}})
        self.assertEqual(self.patcher.object_patches, {})


if __name__ == "__main__":
    unittest.main()
