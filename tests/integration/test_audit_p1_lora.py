"""MP-60 / audit F01 on the real MossPhotonLoRAProfile node and the tiny 24-layer MOSS: a LoRA whose factors do not fit
the weights is reported by shape and never counted as applied.

Audit repro A: every qkv module with the LoRA input dim 127 instead of 128 mapped 24/24, verdict ok, and the native
``LoRAAdapter.calculate_weight`` logged the error and returned the weight bit-identical. The probe here reads the NATIVE
weight back, so "nothing was silently left unpatched while we said ok" is a measurement, not a retelling."""
import logging
import unittest

from . import lora_synth, tiny_moss

LOG = logging.getLogger("MossPhoton")


def shrink_input_dim(sd, suffixes=("lora_A.weight", "lora_down.weight"), only=None):
    """Audit repro A: the input dimension of the A (down) factor is one smaller than the weight's."""
    out = dict(sd)
    for key in sd:
        if key.endswith(suffixes) and (only is None or only(key)):
            out[key] = sd[key][:, :-1].contiguous()
    return out


class LoraShapeAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.core import errors, lora
        cls.torch, cls.errors, cls.lora = torch, errors, lora

    def setUp(self):
        self.env = lora_synth.install(self)
        self.base = self.tiny.patcher
        self.layers = len(self.base.model.diffusion_model.layers)

    # ---------------------------------------------------------------- helpers
    def native_delta(self, patcher, key):
        """Max abs change the registered patches of ``key`` make to the native weight (calculate_weight, as the
        sampler's weight patching does)."""
        import comfy.lora
        weight = patcher.model.state_dict()[key].detach().clone()
        patched = comfy.lora.calculate_weight(patcher.patches[key], weight.clone(), key)
        return float((patched.float() - weight.float()).abs().max())

    def write(self, name, sd):
        self.env.write(name, sd, {"ss_base_model_version": "moss_image"})

    # ---------------------------------------------------------------- repro A
    def test_repro_a_core_reports_the_mismatch_and_nothing_applied(self):
        bad = shrink_input_dim(lora_synth.make_lora_sd(self.base, modules=("qkv",)))
        mapping, patch_dict = self.lora.map_lora_patches(self.base, bad)
        self.assertEqual((mapping.key_found, mapping.shape_ok, mapping.modules_mapped), (self.layers, 0, 0))
        self.assertEqual(len(mapping.shape_mismatch), self.layers)
        self.assertEqual(patch_dict, {})
        self.assertEqual(self.lora.shape_verdict(mapping)["verdict"], "error")
        self.assertEqual(self.lora.coverage_verdict(mapping.modules_total, mapping.modules_mapped)["verdict"], "error")

    def test_repro_a_the_node_refuses_with_the_module_list(self):
        self.write("bad.safetensors", shrink_input_dim(lora_synth.make_lora_sd(self.base, modules=("qkv",))))
        before = ss_entries(self.base)
        with self.assertRaises(self.errors.IncompatibleLora) as cm:
            lora_synth.run_node(self.base, "bad.safetensors")
        issue = cm.exception.issue
        self.assertEqual(issue.code, "incompatible_lora")
        self.assertEqual((issue.detail["key_found"], issue.detail["shape_ok"], issue.detail["applied"]),
                         (self.layers, 0, 0))
        self.assertEqual(issue.detail["shape_mismatch_total"], self.layers)
        self.assertIn("diffusion_model.layers.0.proj.qkv.weight", issue.detail["shape_mismatch"][0])
        self.assertEqual(ss_entries(self.base), before)  # the input MODEL is untouched

    def test_repro_a_probe_the_native_weight_is_what_the_old_code_claimed_ok_for(self):
        """The old verdict was ok (24/24) while the native weight stayed bit-identical; the new code never says ok."""
        import comfy.sd
        bad = shrink_input_dim(lora_synth.make_lora_sd(self.base, modules=("qkv",)))
        key = "diffusion_model.layers.0.proj.qkv.weight"
        # (1) what the native path does with this file: it registers the patch and then leaves the weight alone
        with self.assertLogs(level="ERROR"):
            native = comfy.sd.load_lora_for_models(self.base, None, bad, 1.0, 0)[0]
            self.assertEqual(self.native_delta(native, key), 0.0)
        self.assertEqual(ss_entries(native) - ss_entries(self.base), self.layers)  # "24 patches registered"
        # (2) the pack now sees it before registering
        mapping, _ = self.lora.map_lora_patches(self.base, bad)
        self.assertEqual(mapping.shape_ok, 0)
        # (3) control: a fitting file of the same shape family changes the same native weight
        good = lora_synth.make_lora_sd(self.base, modules=("qkv",))
        applied = comfy.sd.load_lora_for_models(self.base, None, good, 1.0, 0)[0]
        self.assertGreater(self.native_delta(applied, key), 0.0)
        gm, _ = self.lora.map_lora_patches(self.base, good)
        self.assertEqual((gm.key_found, gm.shape_ok, gm.modules_mapped), (self.layers,) * 3)

    # ---------------------------------------------------------------- partial files
    def test_one_bad_module_among_many_is_a_warning_and_is_not_registered(self):
        sd = lora_synth.make_lora_sd(self.base, modules=("qkv", "o"))
        victim = "diffusion_model.layers.5.proj.o.lora_A.weight"
        self.write("partial.safetensors", shrink_input_dim(sd, only=lambda k: k == victim))
        model, profile, _ = lora_synth.run_node(self.base, "partial.safetensors")
        cov = profile["stack"][0]["coverage"]
        total = 2 * self.layers
        self.assertEqual((cov["key_found"], cov["shape_ok"], cov["applied"], cov["modules_mapped"]),
                         (total, total - 1, total - 1, total - 1))
        self.assertEqual(len(cov["shape_mismatch"]), 1)
        self.assertIn("layers.5.proj.o.weight", cov["shape_mismatch"][0])
        warning = [w for w in profile["stack"][0]["warnings"] if w["code"] == "partial_lora_coverage"]
        self.assertTrue(warning)
        self.assertIn("NOT applied", warning[0]["message"])
        self.assertEqual(profile["stack"][0]["patch_entries"], total - 1)
        # the bad module is not registered, and every registered patch really changes its native weight
        self.assertNotIn("diffusion_model.layers.5.proj.o.weight", model.patches)
        self.assertEqual(ss_entries(model), total - 1)
        for key in model.patches:
            with self.subTest(key=key):
                self.assertGreater(self.native_delta(model, key), 0.0, "registered but the weight does not move")
        self.assertEqual(self.lora.validate_identity_profile(profile), [])

    def test_a_split_qkv_slice_that_does_not_fit_is_named_with_its_slice(self):
        sd = lora_synth.make_lora_sd(self.base, fmt="kohya", modules=lora_synth.KOHYA_SPLIT)
        victim = "diffusion_model.layers.2.proj.k.lora_down.weight"
        self.write("split.safetensors", shrink_input_dim(sd, only=lambda k: k == victim))
        _model, profile, _ = lora_synth.run_node(self.base, "split.safetensors")
        cov = profile["stack"][0]["coverage"]
        self.assertEqual((cov["key_found"] - cov["shape_ok"], len(cov["shape_mismatch"])), (1, 1))
        self.assertIn("layers.2.proj.qkv.weight[dim0 ", cov["shape_mismatch"][0])

    def test_up_factor_taken_from_the_whole_qkv_for_a_third_is_a_slice_mismatch(self):
        sd = lora_synth.make_lora_sd(self.base, fmt="kohya", modules=lora_synth.KOHYA_SPLIT)
        third = sd["diffusion_model.layers.1.proj.q.lora_up.weight"]
        sd["diffusion_model.layers.1.proj.q.lora_up.weight"] = self.torch.cat([third] * 3, dim=0)  # the whole qkv
        mapping, _ = self.lora.map_lora_patches(self.base, sd)
        self.assertEqual(len(mapping.shape_mismatch), 1)
        self.assertIn("layers.1.proj.qkv.weight[dim0 0:", mapping.shape_mismatch[0])

    def test_a_share_of_five_percent_or_more_is_incompatible(self):
        sd = lora_synth.make_lora_sd(self.base, modules=("qkv", "o"))
        victims = {"diffusion_model.layers.%d.proj.o.lora_A.weight" % i for i in range(3)}  # 3 of 48 = 6.25%
        self.write("many.safetensors", shrink_input_dim(sd, only=lambda k: k in victims))
        with self.assertRaises(self.errors.IncompatibleLora) as cm:
            lora_synth.run_node(self.base, "many.safetensors")
        self.assertEqual(cm.exception.issue.detail["shape_mismatch_total"], 3)

    def test_allow_low_coverage_does_not_rescue_a_shape_mismatch(self):
        self.write("bad.safetensors", shrink_input_dim(lora_synth.make_lora_sd(self.base, modules=("qkv",))))
        with self.assertRaises(self.errors.IncompatibleLora):
            lora_synth.run_node(self.base, "bad.safetensors", allow_low_coverage=True)

    # ---------------------------------------------------------------- finite
    def test_a_non_finite_factor_is_refused_and_nothing_is_registered(self):
        sd = lora_synth.make_lora_sd(self.base, modules=("qkv", "o"))
        sd["diffusion_model.layers.7.proj.qkv.lora_B.weight"][2, 1] = float("nan")
        self.write("nan.safetensors", sd)
        before = ss_entries(self.base)
        with self.assertRaises(self.errors.IncompatibleLora) as cm:
            lora_synth.run_node(self.base, "nan.safetensors")
        self.assertIn("NaN or Inf", cm.exception.issue.message)
        self.assertEqual(cm.exception.issue.detail["nonfinite_total"], 1)
        self.assertEqual(ss_entries(self.base), before)

    def test_a_non_finite_alpha_is_refused(self):
        sd = lora_synth.make_lora_sd(self.base, fmt="kohya", modules=lora_synth.KOHYA_SPLIT)
        sd["diffusion_model.layers.0.proj.o.alpha"] = self.torch.tensor(float("inf"))
        self.write("alpha.safetensors", sd)
        with self.assertRaises(self.errors.IncompatibleLora):
            lora_synth.run_node(self.base, "alpha.safetensors")

    # ---------------------------------------------------------------- the good path keeps its numbers
    def test_a_fitting_file_reports_three_equal_counts_and_no_warning(self):
        self.write("good.safetensors", lora_synth.make_lora_sd(self.base, modules=("qkv", "o")))
        model, profile, _ = lora_synth.run_node(self.base, "good.safetensors")
        cov = profile["stack"][0]["coverage"]
        self.assertEqual((cov["key_found"], cov["shape_ok"], cov["applied"], cov["shape_unchecked"], cov["shape_mismatch"]),
                         (2 * self.layers, 2 * self.layers, 2 * self.layers, 0, []))
        self.assertEqual(cov["verdict"], "ok")
        self.assertEqual(profile["stack"][0]["warnings"], [])
        self.assertEqual(ss_entries(model), 2 * self.layers)

    def test_the_kohya_split_distill_style_file_is_fully_verified(self):
        self.write("kohya.safetensors", lora_synth.make_lora_sd(self.base, fmt="kohya", modules=lora_synth.KOHYA_SPLIT))
        _model, profile, _ = lora_synth.run_node(self.base, "kohya.safetensors")
        cov = profile["stack"][0]["coverage"]
        self.assertEqual(cov["key_found"], cov["shape_ok"])
        self.assertEqual(cov["key_found"], cov["applied"])
        self.assertEqual(cov["shape_mismatch"], [])


def ss_entries(patcher):
    return sum(len(v) for v in patcher.patches.values())


if __name__ == "__main__":
    unittest.main()
