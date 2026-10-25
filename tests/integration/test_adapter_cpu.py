"""adapter + Schedule/EmptyLatent nodes on the tiny 24-layer MOSS, CPU (design 6.1, MP-6 acceptance)."""
import hashlib
import json
import os
import unittest

from . import _comfy_env, tiny_moss

F32 = ("torch.float32",)
CONTRACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "contracts")


def read_json(*parts):
    with open(os.path.join(CONTRACTS, *parts), "r", encoding="utf-8") as f:
        return json.load(f)


class AdapterCpuBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        import comfy.samplers
        import comfy.sample
        import comfy.model_management
        from moss_photon.core import adapter, consts, sigma, errors
        cls.torch, cls.comfy_samplers, cls.comfy_sample = torch, comfy.samplers, comfy.sample
        cls.mm = comfy.model_management
        cls.adapter, cls.consts, cls.sigma, cls.errors = adapter, consts, sigma, errors
        cls.patcher = cls.tiny.patcher


class ProbeModelTest(AdapterCpuBase):
    def test_probe_model_tiny(self):
        f = self.adapter.probe_model(self.patcher, reference_dtypes=F32)
        self.assertTrue(f.is_moss)
        self.assertEqual(f.class_name, "MOSSImage")
        self.assertEqual((f.shift, f.multiplier, f.sigma_max), (5.0, 1.0, 1.0))
        self.assertTrue(f.is_flow and f.is_const)
        self.assertEqual(f.model_sampling_class, "ModelSamplingDiscreteFlow+CONST")
        self.assertEqual((f.num_layers, f.dim, f.text_dim), (24, tiny_moss.DIM, 64))
        self.assertEqual((f.latent_channels, f.spacial_downscale_ratio), (128, 16))
        self.assertEqual(f.foreign_modifiers, ())
        self.assertEqual(f.weight_dtype, "torch.float32")
        self.assertIsNone(f.manual_cast_dtype)
        self.assertFalse(f.is_dynamic)
        self.assertEqual(f.patches_uuid, str(self.patcher.patches_uuid))
        self.assertEqual(f.patch_entries, 0)
        self.assertIsNone(f.checkpoint_path)  # built without a loader
        self.assertIsNone(f.lora_stack_attachment)
        self.assertEqual(self.adapter.validate_model_facts(f, node="T", slg_requested=True), [])

    def test_default_reference_dtype_is_bf16_so_float32_is_a_modifier(self):
        f = self.adapter.probe_model(self.patcher)
        self.assertEqual(f.foreign_modifiers, ("weight_dtype:torch.float32",))
        issues = self.adapter.validate_model_facts(f, node="T", slg_requested=False)
        self.assertEqual([(i.code, i.severity) for i in issues], [("foreign_modifiers", "warning")])

    def test_latent_format_constants_equal_the_module_constants(self):
        self.assertEqual(self.adapter.latent_format_constants(), (self.adapter.LATENT_CHANNELS, self.adapter.SPATIAL_RATIO))
        self.assertEqual(self.adapter.latent_format_constants(), (128, 16))

    def test_sd3_style_multiplier_is_a_typed_error_with_the_auraflow_hint(self):
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        sd3 = ModelSamplingSD3().patch(self.patcher, 5.0)[0]  # multiplier 1000
        f = self.adapter.probe_model(sd3, reference_dtypes=F32)
        self.assertEqual(f.multiplier, 1000.0)
        issues = self.adapter.validate_model_facts(f, node="MossPhotonSampler", slg_requested=False)
        with self.assertRaises(self.errors.InvalidModelSampling) as cm:
            self.errors.raise_first_error(issues)
        self.assertIn("ModelSamplingAuraFlow", str(cm.exception))
        # the original is untouched
        self.assertEqual(self.adapter.probe_model(self.patcher, reference_dtypes=F32).multiplier, 1.0)

    def test_auraflow_shift_is_visible_in_the_live_shift_not_as_a_modifier(self):
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
        f = self.adapter.probe_model(ModelSamplingAuraFlow().patch_aura(self.patcher, 3.0)[0], reference_dtypes=F32)
        self.assertEqual((f.shift, f.multiplier, f.foreign_modifiers), (3.0, 1.0, ()))
        self.assertEqual(self.adapter.validate_model_facts(f, node="T", slg_requested=True), [])

    def test_patch_facts_track_a_native_lora_and_the_attachment_survives(self):
        import comfy.sd
        lora = make_lora(self.patcher, self.torch, skip=set(), seed=11)
        with_lora, _ = comfy.sd.load_lora_for_models(self.patcher, None, lora, 1.0, 0)
        f0 = self.adapter.probe_model(self.patcher, reference_dtypes=F32)
        f1 = self.adapter.probe_model(with_lora, reference_dtypes=F32)
        self.assertEqual(f0.patch_entries, 0)
        self.assertEqual(f1.patch_entries, 2 * 24)
        self.assertNotEqual(f1.patches_uuid, f0.patches_uuid)
        self.assertEqual(f1.foreign_modifiers, ())  # weight patches are not modifiers
        with_lora.set_attachments(self.consts.ATTACH_LORA_STACK, '{"v":1}')
        clone = with_lora.clone()
        self.assertEqual(self.adapter.probe_model(clone, reference_dtypes=F32).lora_stack_attachment, '{"v":1}')
        self.assertEqual(self.adapter.probe_model(clone, reference_dtypes=F32).patches_uuid, f1.patches_uuid)
        self.assertIsNone(self.adapter.probe_model(self.patcher, reference_dtypes=F32).lora_stack_attachment)


def make_lora(patcher, torch, skip, seed, rank=4):
    import re
    sd = patcher.model.state_dict()
    g = torch.Generator().manual_seed(seed)
    out = {}
    for key in sd:
        m = re.match(r"^(diffusion_model[.]layers[.](\d+)[.]proj[.](qkv|o))[.]weight$", key)
        if m and int(m.group(2)) not in skip:
            out_f, in_f = sd[key].shape
            out[m.group(1) + ".lora_A.weight"] = torch.randn(rank, in_f, generator=g) * 0.5
            out[m.group(1) + ".lora_B.weight"] = torch.randn(out_f, rank, generator=g) * 0.5
    return out


class SigmaCpuTest(AdapterCpuBase):
    def test_native_sigmas_equals_ksampler_calculate_sigmas(self):
        ks = self.comfy_samplers.KSampler(self.patcher, steps=50, device=self.torch.device("cpu"),
                                          sampler="euler", scheduler="sgm_uniform")
        ours = self.adapter.native_sigmas(self.patcher, "sgm_uniform", 50)
        self.assertTrue(self.torch.equal(ours, ks.sigmas))
        self.assertEqual(ours.dtype, self.torch.float32)
        self.assertEqual(ours.device.type, "cpu")

    def test_native_sigmas_50_equals_literal(self):
        from tests.unit.core.test_sigma import REFERENCE_50
        ours = self.adapter.native_sigmas(self.patcher, "sgm_uniform", 50)
        self.assertEqual(ours.tolist(), REFERENCE_50)
        self.assertEqual(self.sigma.sigma_hash(ours), self.consts.REFERENCE_SIGMAS_SHA256)
        self.assertEqual(self.sigma.sigma_hash(ours.tolist()), self.consts.REFERENCE_SIGMAS_SHA256)
        # control: the hash really is the float32 byte hash (different from the float64 one)
        self.assertNotEqual(hashlib.sha256(self.torch.tensor(REFERENCE_50, dtype=self.torch.float64).numpy().tobytes()).hexdigest(),
                            self.consts.REFERENCE_SIGMAS_SHA256)
        self.assertEqual(self.sigma.validate_sigmas(ours.tolist(), node="T", expect_steps=50), [])

    def test_shifted_sigmas_equals_auraflow_route(self):
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
        aura = ModelSamplingAuraFlow().patch_aura(self.patcher, 3.0)[0]
        via_aura = self.comfy_samplers.calculate_sigmas(aura.get_model_object("model_sampling"), "sgm_uniform", 50)
        mine = self.adapter.shifted_sigmas(self.patcher, "sgm_uniform", 50, 3.0)
        self.assertTrue(self.torch.equal(mine, via_aura))
        self.assertFalse(self.torch.equal(mine, self.adapter.native_sigmas(self.patcher, "sgm_uniform", 50)))
        # the live model is untouched
        ms = self.patcher.get_model_object("model_sampling")
        self.assertEqual((float(ms.shift), float(ms.multiplier)), (5.0, 1.0))
        self.assertEqual(self.sigma.sigma_hash(self.adapter.native_sigmas(self.patcher, "sgm_uniform", 50)),
                         self.consts.REFERENCE_SIGMAS_SHA256)

    def test_shifted_at_the_live_shift_equals_native(self):
        self.assertTrue(self.torch.equal(self.adapter.shifted_sigmas(self.patcher, "sgm_uniform", 50, 5.0),
                                         self.adapter.native_sigmas(self.patcher, "sgm_uniform", 50)))

    def test_schedule_validation_precedes_any_big_allocation(self):
        """validate_sigmas works on the tolist() of the CPU tensor; nothing here touches a device."""
        bad = self.adapter.native_sigmas(self.patcher, "sgm_uniform", 50).clone()
        bad[10] = float("nan")
        issues = self.sigma.validate_sigmas(bad.tolist(), node="T")
        self.assertEqual([i.code for i in issues], ["invalid_schedule"])


class LatentCpuTest(AdapterCpuBase):
    def test_make_empty_latent_equals_native(self):
        from comfy_extras.nodes_flux import EmptyFlux2LatentImage
        ours = self.adapter.make_empty_latent(1024, 1024, 2)["samples"]
        native = EmptyFlux2LatentImage.execute(1024, 1024, 2).args[0]["samples"]
        self.assertEqual((tuple(ours.shape), ours.dtype, ours.device), (tuple(native.shape), native.dtype, native.device))
        self.assertEqual(tuple(ours.shape), (2, 128, 64, 64))
        self.assertTrue(self.torch.equal(ours, native))
        self.assertEqual(int(ours.count_nonzero()), 0)

    def test_make_empty_latent_refuses_instead_of_flooring(self):
        with self.assertRaises(self.errors.InvalidDimensions):
            self.adapter.make_empty_latent(1000, 1024, 1)
        with self.assertRaises(self.errors.InvalidDimensions):
            self.adapter.make_empty_latent(1024, 1017, 1)

    def test_fix_empty_latent_matches_comfy(self):
        latent = {"samples": self.torch.zeros(1, 4, 16, 16), "downscale_ratio_spacial": 8}
        ours = self.adapter.fix_empty_latent(self.patcher, latent)
        direct = self.comfy_sample.fix_empty_latent_channels(self.patcher, latent["samples"], 8, None)
        self.assertTrue(self.torch.equal(ours, direct))
        self.assertEqual(tuple(ours.shape), (1, 128, 8, 8))

    def test_fix_empty_latent_without_samples_is_invalid_latent(self):
        for bad in ({}, {"samples": None}, {"samples": [1, 2]}, None):
            with self.subTest(bad=bad):
                with self.assertRaises(self.errors.InvalidLatent):
                    self.adapter.fix_empty_latent(self.patcher, bad)

    def test_probe_latent_of_the_tiny_fixture(self):
        latent = {"samples": self.tiny.latent}
        fixed = self.adapter.fix_empty_latent(self.patcher, latent)
        f = self.adapter.probe_latent(latent, fixed)
        self.assertEqual((f.batch, f.channels, f.height, f.width, f.ndim), (1, 128, 8, 8, 4))
        self.assertEqual((f.pixel_width, f.pixel_height), (128, 128))
        self.assertTrue(f.is_empty)
        self.assertFalse(f.rescaled_by_native_fix)
        self.assertEqual(f.dtype, "torch.float32")
        self.assertEqual(self.adapter.validate_latent_facts(f, channels=128, node="T"), [])

    def test_to_intermediate(self):
        x = self.torch.ones(1, 2, 2, 2, dtype=self.torch.float64)
        y = self.adapter.to_intermediate(x)
        self.assertEqual(y.device, self.mm.intermediate_device())
        self.assertEqual(y.dtype, self.mm.intermediate_dtype())


class ConditioningCpuTest(AdapterCpuBase):
    def test_probe_conditioning_hash_stable(self):
        a = self.adapter.probe_conditioning(self.tiny.make_cond(23), "positive")
        b = self.adapter.probe_conditioning(self.tiny.make_cond(23), "positive")
        c = self.adapter.probe_conditioning(self.tiny.make_cond(9), "negative")
        self.assertEqual(a, b)
        self.assertNotEqual(a.sha256, c.sha256)
        self.assertEqual((a.entries, a.token_counts, a.dim, a.dtype, a.all_tensors), (1, (23,), 64, "torch.float32", True))
        self.assertFalse(a.has_area_or_hooks)
        expected = hashlib.sha256(self.tiny.make_cond(23)[0][0].float().contiguous().numpy().tobytes()).hexdigest()
        self.assertEqual(a.sha256, expected)
        self.assertEqual(self.adapter.validate_cond_facts(a, name="positive", text_dim=64, node="T"), [])

    def test_conditioning_problems(self):
        torch = self.torch
        empty = self.adapter.probe_conditioning([], "positive")
        self.assertEqual([i.code for i in self.adapter.validate_cond_facts(empty, name="positive", text_dim=64, node="T")],
                         ["invalid_conditioning"])
        wrong_dim = self.adapter.probe_conditioning([[torch.zeros(1, 5, 4096), {}]], "positive")
        self.assertEqual([i.code for i in self.adapter.validate_cond_facts(wrong_dim, name="positive", text_dim=64, node="T")],
                         ["invalid_conditioning"])
        junk = self.adapter.probe_conditioning([["not a tensor", {}]], "positive")
        self.assertFalse(junk.all_tensors)
        with_area = self.adapter.probe_conditioning([[torch.zeros(1, 5, 64), {"area": (1, 1, 0, 0)}]], "positive")
        self.assertTrue(with_area.has_area_or_hooks)
        two = self.adapter.probe_conditioning(self.tiny.make_cond(4) + self.tiny.make_cond(5), "positive")
        self.assertEqual((two.entries, two.token_counts), (2, (4, 5)))
        self.assertEqual([i.code for i in self.adapter.validate_cond_facts(two, name="positive", text_dim=64, node="T")],
                         ["multiple_cond_entries"])


class EnvInfoTest(AdapterCpuBase):
    def test_env_info_keys(self):
        from jsonschema import Draft202012Validator
        env = self.adapter.env_info(self.patcher)
        self.assertEqual(set(env), {"comfyui_version", "comfyui_commit", "python", "torch", "cuda", "device", "device_name",
                                    "attention", "deterministic_flag", "weights_streaming", "inference_dtype"})
        self.assertEqual(set(env["attention"]), {"global", "model_override"})
        self.assertEqual((env["device"], env["device_name"], env["weights_streaming"], env["inference_dtype"]),
                         ("cpu", None, False, "torch.float32"))
        self.assertRegex(env["comfyui_version"], r"^\d+\.\d+\.\d+")
        schema = read_json("schemas", "report.schema.json")["properties"]["env"]
        self.assertEqual(sorted(e.message for e in Draft202012Validator(schema).iter_errors(env)), [])

    def test_env_info_is_memoised_but_returns_copies(self):
        a, b = self.adapter.env_info(), self.adapter.env_info()
        self.assertEqual(a, b)
        a["attention"]["global"] = "mutated"
        self.assertNotEqual(self.adapter.env_info()["attention"]["global"], "mutated")
        self.assertEqual((a["weights_streaming"], a["inference_dtype"]), (False, "unknown"))

    def test_attention_override_name_is_reported(self):
        def sparse_container(*args, **kwargs):
            raise AssertionError("never called")

        def my_attention(*args, **kwargs):
            raise AssertionError("never called")
        my_attention.container_function = sparse_container
        clone = self.patcher.clone()
        clone.set_model_optimized_attention(my_attention)
        self.assertEqual(self.adapter.env_info(clone)["attention"]["model_override"], "sparse_container")
        plain = self.patcher.clone()
        plain.set_model_optimized_attention(lambda *a, **k: None)  # no container: the closure's own name
        self.assertEqual(self.adapter.env_info(plain)["attention"]["model_override"], "optimized_attention_override")
        self.assertIsNone(self.adapter.env_info(self.patcher)["attention"]["model_override"])


class NodesCpuTest(AdapterCpuBase):
    """The two MP-6 nodes end to end on the tiny model."""

    def test_schedule_node_reference(self):
        from moss_photon.nodes.schedule import MossPhotonSchedule
        latent = {"samples": self.tiny.latent}
        sigmas, info = MossPhotonSchedule().build(self.patcher, latent, 50, 5.0, "reference")
        self.assertEqual(self.sigma.sigma_hash(sigmas), self.consts.REFERENCE_SIGMAS_SHA256)
        fixture = read_json("schedule_info.json")
        self.assertEqual(info, fixture)  # the tiny model has the MOSS config, so the info equals the 50-step fixture
        from jsonschema import Draft202012Validator
        self.assertEqual(sorted(e.message for e in Draft202012Validator(
            read_json("schemas", "schedule_info.schema.json")).iter_errors(info)), [])

    def test_schedule_node_shifted_equals_auraflow_and_is_not_reference(self):
        from moss_photon.nodes.schedule import MossPhotonSchedule
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
        latent = {"samples": self.tiny.latent}
        sigmas, info = MossPhotonSchedule().build(self.patcher, latent, 30, 3.0, "shifted")
        aura = ModelSamplingAuraFlow().patch_aura(self.patcher, 3.0)[0]
        expected = self.comfy_samplers.calculate_sigmas(aura.get_model_object("model_sampling"), "sgm_uniform", 30)
        self.assertTrue(self.torch.equal(sigmas, expected))
        self.assertFalse(info["is_native_reference"])
        self.assertEqual((info["mode"], info["shift_model"], info["shift_used"], info["steps"], info["n_sigmas"]),
                         ("shifted", 5.0, 3.0, 30, 31))

    def test_schedule_node_refuses_sd3_multiplier_and_leaves_inputs_alone(self):
        from moss_photon.nodes.schedule import MossPhotonSchedule
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        sd3 = ModelSamplingSD3().patch(self.patcher, 5.0)[0]
        latent = {"samples": self.tiny.latent}
        before = self.tiny.latent.clone()
        with self.assertRaises(self.errors.InvalidModelSampling) as cm:
            MossPhotonSchedule().build(sd3, latent, 50, 5.0, "reference")
        self.assertIn("ModelSamplingAuraFlow", str(cm.exception))
        self.assertTrue(self.torch.equal(before, self.tiny.latent))
        self.assertEqual(set(latent), {"samples"})

    def test_schedule_node_accepts_the_stock_empty_latent(self):
        from moss_photon.nodes.schedule import MossPhotonSchedule
        latent = {"samples": self.torch.zeros(1, 4, 16, 16), "downscale_ratio_spacial": 8}
        _, info = MossPhotonSchedule().build(self.patcher, latent, 50, 5.0, "reference")
        self.assertEqual([w["code"] for w in info["warnings"]], ["latent_rescaled_by_native_fix"])
        self.assertEqual(tuple(latent["samples"].shape), (1, 4, 16, 16))  # not mutated

    def test_schedule_node_refuses_a_non_empty_four_channel_latent(self):
        from moss_photon.nodes.schedule import MossPhotonSchedule
        with self.assertRaises(self.errors.InvalidLatent):
            MossPhotonSchedule().build(self.patcher, {"samples": self.torch.ones(1, 4, 8, 8)}, 50, 5.0, "reference")

    def test_empty_latent_node_equals_native(self):
        from moss_photon.nodes.latent import MossPhotonEmptyLatent
        from comfy_extras.nodes_flux import EmptyFlux2LatentImage
        out = MossPhotonEmptyLatent().generate(1024, 1024, 2)
        native = EmptyFlux2LatentImage.execute(1024, 1024, 2).args[0]["samples"]
        self.assertTrue(self.torch.equal(out["result"][0]["samples"], native))
        self.assertEqual(set(out["result"][0]), {"samples"})
        self.assertIn("[2,128,64,64]", out["ui"]["text"][0])

    def test_empty_latent_node_refuses_non_multiples(self):
        from moss_photon.nodes.latent import MossPhotonEmptyLatent
        for w, h in ((1000, 1024), (1024, 1017)):
            with self.subTest(size=(w, h)):
                with self.assertRaises(self.errors.InvalidDimensions):
                    MossPhotonEmptyLatent().generate(w, h, 1)


class VaeProbeTest(AdapterCpuBase):
    def test_probe_vae_on_a_comfy_vae_like_object(self):
        import types
        vae = types.SimpleNamespace(latent_channels=128, downscale_ratio=16)
        f = self.adapter.probe_vae(vae)
        self.assertEqual(self.adapter.validate_vae_facts(f, channels=128, ratio=16, node="T"), [])


if __name__ == "__main__":
    unittest.main()
