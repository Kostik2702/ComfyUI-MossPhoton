"""Adapter rules that need no ComfyUI: fake facts in, Issues out (design 1.2, 6.1)."""
import dataclasses
import os
import subprocess
import sys
import types
import unittest

from moss_photon.core import adapter, errors

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NODE = "MossPhotonSampler"
REF = ("torch.bfloat16",)


def model_facts(**kw):
    base = dict(
        is_moss=True, class_name="MOSSImage", num_layers=42, dim=3072, text_dim=4096,
        model_sampling_class="ModelSamplingDiscreteFlow+CONST", is_flow=True, is_const=True, shift=5.0,
        multiplier=1.0, sigma_max=1.0, latent_channels=128, spacial_downscale_ratio=16, dtype="torch.bfloat16",
        load_device="cuda:0", is_dynamic=False, weight_dtype="torch.bfloat16", manual_cast_dtype=None,
        foreign_modifiers=(), patches_uuid="u", patch_entries=0, checkpoint_path=None, lora_stack_attachment=None)
    base.update(kw)
    return adapter.ModelFacts(**base)


def latent_facts(**kw):
    base = dict(batch=1, channels=128, height=64, width=64, dtype="torch.float32", device="cpu", is_nested=False,
                is_empty=True, has_noise_mask=False, has_batch_index=False, raw_shape=(1, 128, 64, 64),
                raw_downscale_ratio_spacial=None, rescaled_by_native_fix=False, pixel_width=1024, pixel_height=1024)
    base.update(kw)
    return adapter.LatentFacts(**base)


def codes(issues, severity=None):
    return [i.code for i in issues if severity is None or i.severity == severity]


class ValidateModelFactsTest(unittest.TestCase):
    def run_it(self, slg=False, **kw):
        return adapter.validate_model_facts(model_facts(**kw), node=NODE, slg_requested=slg)

    def test_clean_model_has_no_issues(self):
        self.assertEqual(self.run_it(), [])

    def test_validate_model_facts_not_moss(self):
        issues = self.run_it(is_moss=False, class_name="Flux")
        self.assertEqual(codes(issues), ["unsupported_model"])
        self.assertIn("e087d8f6", issues[0].fix)
        self.assertIn("Flux", issues[0].message)

    def test_validate_model_facts_sd3_style_multiplier(self):
        issues = self.run_it(multiplier=1000.0)
        self.assertEqual(codes(issues, "error"), ["invalid_model_sampling"])
        self.assertIn("ModelSamplingAuraFlow", issues[0].fix)
        self.assertIn("1000", issues[0].message)
        with self.assertRaises(errors.InvalidModelSampling) as cm:
            errors.raise_first_error(issues)
        self.assertIn("ModelSamplingAuraFlow", str(cm.exception))

    def test_multiplier_close_to_one_is_still_wrong(self):
        self.assertEqual(codes(self.run_it(multiplier=1.0000001), "error"), ["invalid_model_sampling"])
        self.assertEqual(codes(self.run_it(multiplier=None), "error"), ["invalid_model_sampling"])

    def test_not_flow_or_not_const(self):
        self.assertEqual(codes(self.run_it(is_flow=False), "error"), ["invalid_model_sampling"])
        self.assertEqual(codes(self.run_it(is_const=False), "error"), ["invalid_model_sampling"])

    def test_validate_model_facts_latent_format(self):
        self.assertEqual(codes(self.run_it(latent_channels=16), "error"), ["unsupported_model"])
        self.assertEqual(codes(self.run_it(spacial_downscale_ratio=8), "error"), ["unsupported_model"])
        self.assertEqual(codes(self.run_it(latent_channels=None), "error"), ["unsupported_model"])

    def test_hook_conflict_is_an_error_when_slg_requested(self):
        issues = self.run_it(slg=True, foreign_modifiers=("cond_batch_function",))
        self.assertEqual(codes(issues, "error"), ["conflicting_patch"])
        with self.assertRaises(errors.ConflictingModelPatch):
            errors.raise_first_error(issues)

    def test_hook_conflict_is_a_warning_when_slg_is_off(self):
        issues = self.run_it(slg=False, foreign_modifiers=("cond_batch_function",))
        self.assertEqual(codes(issues, "error"), [])
        self.assertEqual(codes(issues), ["external_calc_cond_batch", "foreign_modifiers"])

    def test_other_modifiers_warn_but_never_stop(self):
        mods = ("attention_override", "cfg_function", "post_cfg_function", "object_patch:x", "weight_dtype:torch.float32")
        issues = self.run_it(slg=True, foreign_modifiers=mods)
        self.assertEqual(codes(issues, "error"), [])
        fm = [i for i in issues if i.code == "foreign_modifiers"][0]
        self.assertEqual(fm.severity, "warning")
        self.assertEqual(fm.detail["modifiers"], list(mods))
        self.assertIn("non_default_attention_backend", codes(issues))

    def test_every_emitted_code_is_registered(self):
        for issue in self.run_it(slg=True, multiplier=2.0, latent_channels=4,
                                 foreign_modifiers=("cond_batch_function", "attention_override")):
            self.assertIn(issue.code, errors.REASONS)


class ValidateLatentFactsTest(unittest.TestCase):
    def run_it(self, **kw):
        return adapter.validate_latent_facts(latent_facts(**kw), channels=128, node=NODE)

    def test_clean_empty_latent_passes(self):
        self.assertEqual(self.run_it(), [])

    def test_validate_latent_facts_nonempty_four_channels(self):
        issues = self.run_it(channels=4, is_empty=False, raw_shape=(1, 4, 64, 64))
        self.assertEqual(codes(issues), ["invalid_latent"])
        for needle in ("MossPhotonEmptyLatent", "EmptyFlux2LatentImage", "EmptyLatentImage"):
            self.assertIn(needle, issues[0].fix)
        self.assertIn("channels = 4, expected 128", issues[0].message)

    def test_repaired_empty_four_channel_latent_passes_with_info(self):
        # the stock EmptyLatentImage: raw [1,4,16,16] ratio 8 -> repaired to [1,128,8,8] (M8)
        issues = self.run_it(height=8, width=8, raw_shape=(1, 4, 16, 16), raw_downscale_ratio_spacial=8.0,
                             rescaled_by_native_fix=True, pixel_width=128, pixel_height=128)
        self.assertEqual([(i.code, i.severity) for i in issues], [("latent_rescaled_by_native_fix", "info")])

    def test_nonempty_latent_with_a_foreign_ratio(self):
        issues = self.run_it(is_empty=False, raw_downscale_ratio_spacial=8.0)
        self.assertEqual(codes(issues), ["invalid_latent"])
        self.assertEqual(self.run_it(is_empty=False, raw_downscale_ratio_spacial=16.0), [])

    def test_nested_and_wrong_ndim(self):
        self.assertEqual(codes(self.run_it(is_nested=True)), ["invalid_latent"])
        self.assertEqual(codes(self.run_it(ndim=5)), ["invalid_latent"])

    def test_noise_mask_is_unsupported_source_input(self):
        issues = self.run_it(has_noise_mask=True)
        self.assertEqual(codes(issues), ["unsupported_source_input"])
        with self.assertRaises(errors.UnsupportedSourceInput):
            errors.raise_first_error(issues)

    def test_batch_index_alone_is_not_an_error(self):
        self.assertEqual(self.run_it(has_batch_index=True), [])


class ValidateCondVaeTest(unittest.TestCase):
    def cond(self, **kw):
        base = dict(entries=1, token_counts=(23,), dim=4096, dtype="torch.bfloat16", sha256="0" * 64,
                    has_area_or_hooks=False)
        base.update(kw)
        return adapter.CondFacts(**base)

    def test_validate_cond_facts(self):
        run = lambda f, **kw: adapter.validate_cond_facts(f, name="positive", text_dim=kw.get("text_dim", 4096), node=NODE)  # noqa: E731
        self.assertEqual(run(self.cond()), [])
        self.assertEqual(codes(run(self.cond(entries=0, token_counts=(), dim=None))), ["invalid_conditioning"])
        self.assertEqual(codes(run(self.cond(all_tensors=False))), ["invalid_conditioning"])
        mismatch = run(self.cond(dim=2048))
        self.assertEqual(codes(mismatch), ["invalid_conditioning"])
        self.assertIn("positive embedding dim = 2048, expected 4096", mismatch[0].message)
        self.assertEqual(run(self.cond(dim=2048), text_dim=None), [])
        multi = run(self.cond(entries=2, token_counts=(23, 5)))
        self.assertEqual([(i.code, i.severity) for i in multi], [("multiple_cond_entries", "warning")])
        self.assertEqual(multi[0].param, "positive")

    def test_a_conditioning_with_area_mask_hooks_or_control_is_a_foreign_modifier(self):
        """MP-30 m4: ``has_area_or_hooks`` was computed and never read; it is now the modifier ``cond_extras``."""
        run = lambda f, name="positive": adapter.validate_cond_facts(f, name=name, text_dim=4096, node=NODE)  # noqa: E731
        got = run(self.cond(has_area_or_hooks=True), "negative")
        self.assertEqual([(i.code, i.severity, i.param) for i in got], [("foreign_modifiers", "warning", "negative")])
        self.assertEqual(got[0].detail, {"modifiers": [adapter.COND_MODIFIER]})
        self.assertEqual(adapter.COND_MODIFIER, "cond_extras")
        self.assertIn("negative", got[0].message)
        self.assertEqual(run(self.cond(has_area_or_hooks=False)), [])

    def test_validate_vae_facts(self):
        ok = adapter.VaeFacts(128, 16)
        self.assertEqual(adapter.validate_vae_facts(ok, channels=128, ratio=16, node=NODE), [])
        for bad in (adapter.VaeFacts(4, 8), adapter.VaeFacts(128, 8), adapter.VaeFacts(None, None),
                    adapter.VaeFacts(16, 16)):
            self.assertEqual(codes(adapter.validate_vae_facts(bad, channels=128, ratio=16, node=NODE)), ["invalid_vae"])

    def test_probe_vae_reads_plain_attributes(self):
        self.assertEqual(adapter.probe_vae(types.SimpleNamespace(latent_channels=128, downscale_ratio=16)),
                         adapter.VaeFacts(128, 16))
        # other VAE families use tuples / callables for the ratio: not an int, so None
        self.assertEqual(adapter.probe_vae(types.SimpleNamespace(latent_channels=16, downscale_ratio=(lambda a: a, 8, 8))),
                         adapter.VaeFacts(16, None))
        self.assertEqual(adapter.probe_vae(types.SimpleNamespace()), adapter.VaeFacts(None, None))


class ValidateDimensionsTest(unittest.TestCase):
    def run_it(self, w, h, b=1, **kw):
        return adapter.validate_dimensions(w, h, b, node="MossPhotonEmptyLatent", **kw)

    def test_valid(self):
        self.assertEqual(self.run_it(1024, 1024), [])
        self.assertEqual(self.run_it(64, 8192, 64), [])

    def test_non_multiples_name_the_side_and_nearest_values(self):
        (w,) = self.run_it(1000, 1024)
        self.assertEqual((w.code, w.param), ("invalid_dimensions", "width"))
        self.assertIn("nearest: 992 or 1008", w.fix)
        (h,) = self.run_it(1024, 1017)
        self.assertEqual(h.param, "height")
        self.assertIn("nearest: 1008 or 1024", h.fix)
        self.assertEqual(codes(self.run_it(1001, 1017)), ["invalid_dimensions", "invalid_dimensions"])

    def test_bounds_and_batch(self):
        self.assertEqual(codes(self.run_it(16, 1024)), ["invalid_dimensions"])
        self.assertEqual(codes(self.run_it(8208, 1024)), ["invalid_dimensions"])
        self.assertEqual(self.run_it(16, 1024, check_bounds=False), [])
        self.assertEqual(codes(self.run_it(0, 1024, check_bounds=False)), ["invalid_dimensions"])
        self.assertEqual(codes(self.run_it(1024, 1024, 0)), ["invalid_dimensions"])
        self.assertEqual(codes(self.run_it(1024, 1024, 65)), ["invalid_dimensions"])

    def test_non_integers(self):
        self.assertEqual(codes(self.run_it(1024.0, 1024)), ["invalid_dimensions"])
        self.assertEqual(codes(self.run_it(True, 1024)), ["invalid_dimensions"])

    def test_make_empty_latent_never_floors(self):
        with self.assertRaises(errors.InvalidDimensions):
            adapter.make_empty_latent(1000, 1024, 1)  # raises before any comfy import


class DetectModifiersTest(unittest.TestCase):
    """The detection rules on fake patchers (the real patchers are in test_foreign_modifiers_cpu)."""

    def patcher(self, **kw):
        base = dict(model_options={"transformer_options": {}}, object_patches={}, hook_patches={}, injections={},
                    additional_models={}, weight_wrapper_patches={}, wrappers={"a": {"k": []}},
                    callbacks={"c": {}}, forced_hooks=None, current_hooks=None)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def detect(self, p, weight="torch.bfloat16", cast=None, ref=REF):
        return adapter.detect_foreign_modifiers(p, weight_dtype=weight, manual_cast_dtype=cast, reference_dtypes=ref)

    def test_clean_patcher_gives_empty_tuple(self):
        self.assertEqual(self.detect(self.patcher()), ())

    def test_each_rule(self):
        cases = [
            (dict(model_options={"transformer_options": {}, "sampler_post_cfg_function": [print]}), "post_cfg_function"),
            (dict(model_options={"transformer_options": {}, "sampler_cfg_function": print}), "cfg_function"),
            (dict(model_options={"transformer_options": {}, "sampler_pre_cfg_function": [print]}), "pre_cfg_function"),
            (dict(model_options={"transformer_options": {}, "sampler_calc_cond_batch_function": print}), "cond_batch_function"),
            (dict(model_options={"transformer_options": {}, "model_function_wrapper": print}), "model_function_wrapper"),
            (dict(model_options={"transformer_options": {}, "denoise_mask_function": print}), "denoise_mask_function"),
            (dict(model_options={"transformer_options": {}, "disable_cfg1_optimization": True}), "disable_cfg1_optimization"),
            (dict(model_options={"transformer_options": {}, "mystery": 1}), "model_option:mystery"),
            (dict(model_options={"transformer_options": {"optimized_attention_override": print}}), "attention_override"),
            (dict(model_options={"transformer_options": {"wrappers": {"apply_model": {"k": [print]}}}}), "transformer_option:wrappers"),
            (dict(model_options={"transformer_options": {"patches_replace": {"dit": {"x": print}}}}), "transformer_option:patches_replace"),
            (dict(object_patches={"diffusion_model.dummy": 1}), "object_patch:diffusion_model.dummy"),
            (dict(hook_patches={"h": 1}), "hook_patches"),
            (dict(injections={"i": [1]}), "injections"),
            (dict(additional_models={"a": [1]}), "additional_models"),
            (dict(weight_wrapper_patches={"w": [1]}), "weight_wrapper_patches"),
            (dict(wrappers={"a": {"k": [print]}}), "wrappers"),
            (dict(callbacks={"c": {"k": [print]}}), "callbacks"),
            (dict(forced_hooks=object()), "hooks"),
            (dict(current_hooks=object()), "hooks"),
        ]
        for kw, code in cases:
            with self.subTest(code=code):
                self.assertEqual(self.detect(self.patcher(**kw)), (code,))

    def test_allowed_and_empty_things_are_not_modifiers(self):
        shift_patch = self.patcher(object_patches={"model_sampling": object()})
        self.assertEqual(self.detect(shift_patch), ())
        empty = self.patcher(model_options={"transformer_options": {"patches": {}, "wrappers": {"a": {"b": []}}},
                                            "sampler_post_cfg_function": [], "disable_cfg1_optimization": False})
        self.assertEqual(self.detect(empty), ())
        self.assertEqual(self.detect(types.SimpleNamespace()), ())  # a bare object has nothing to report

    def test_dtype_rules(self):
        p = self.patcher()
        self.assertEqual(self.detect(p, weight="torch.float32"), ("weight_dtype:torch.float32",))
        self.assertEqual(self.detect(p, weight="torch.float32", ref=("torch.float32",)), ())
        self.assertEqual(self.detect(p, cast="torch.float16"), ("manual_cast:torch.float16",))

    def test_result_is_sorted_and_deduplicated(self):
        p = self.patcher(forced_hooks=object(), current_hooks=object(),
                         model_options={"transformer_options": {}, "sampler_post_cfg_function": [print],
                                        "sampler_cfg_function": print})
        self.assertEqual(self.detect(p), ("cfg_function", "hooks", "post_cfg_function"))


class PurityTest(unittest.TestCase):
    def test_adapter_and_nodes_import_without_comfy_or_torch(self):
        code = ("import sys\nsys.path.insert(0, sys.argv[1])\n"
                "from moss_photon.core import adapter, fingerprint, sigma, rng, report\n"
                "from moss_photon.nodes import latent, schedule\n"
                "latent.MossPhotonEmptyLatent.INPUT_TYPES()\nschedule.MossPhotonSchedule.INPUT_TYPES()\n"
                "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'comfy', 'numpy', 'PIL', "
                "'folder_paths', 'server'))\nassert not bad, bad[:5]\nprint('OK')\n")
        proc = subprocess.run([sys.executable, "-I", "-c", code, PKG_ROOT], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-600:])

    def test_dataclass_fields_cover_the_design(self):
        names = {f.name for f in dataclasses.fields(adapter.ModelFacts)}
        for needed in ("foreign_modifiers", "patches_uuid", "patch_entries", "checkpoint_path",
                       "lora_stack_attachment", "weight_dtype", "manual_cast_dtype", "is_dynamic"):
            self.assertIn(needed, names)


if __name__ == "__main__":
    unittest.main()
