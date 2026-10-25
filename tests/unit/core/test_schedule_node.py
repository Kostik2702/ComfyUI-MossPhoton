"""MossPhotonSchedule (design 4.2) with a fake adapter: no ComfyUI, no torch."""
import json
import os
import types
import unittest
from unittest import mock

from jsonschema import Draft202012Validator

from moss_photon.core import adapter as real_adapter
from moss_photon.core import consts, errors, sigma
from moss_photon.nodes import schedule
from .test_sigma import REFERENCE_50

CONTRACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "fixtures", "contracts")


class Arr:
    """Stand-in for a float32 tensor: only ``tolist`` is used by the node."""

    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


def facts(**kw):
    base = dict(
        is_moss=True, class_name="MOSSImage", num_layers=24, dim=128, text_dim=64,
        model_sampling_class="ModelSamplingDiscreteFlow+CONST", is_flow=True, is_const=True, shift=5.0,
        multiplier=1.0, sigma_max=1.0, latent_channels=128, spacial_downscale_ratio=16, dtype="torch.float32",
        load_device="cpu", is_dynamic=False, weight_dtype="torch.float32", manual_cast_dtype=None,
        foreign_modifiers=(), patches_uuid="u", patch_entries=0, checkpoint_path=None, lora_stack_attachment=None)
    base.update(kw)
    return real_adapter.ModelFacts(**base)


class FakeAdapter(types.SimpleNamespace):
    def __init__(self, model_facts=None, shifted=None, native=None, lat_facts=None):
        super().__init__(LATENT_CHANNELS=128, calls=[])
        self._facts = model_facts or facts()
        self._shifted = shifted
        self._native = native or REFERENCE_50
        self._lat = lat_facts
        self.validate_model_facts = real_adapter.validate_model_facts
        self.validate_latent_facts = real_adapter.validate_latent_facts

    def probe_model(self, model):
        self.calls.append("probe_model")
        return self._facts

    def fix_empty_latent(self, model, latent, node=None):
        self.calls.append("fix")
        return latent["samples"]

    def probe_latent(self, latent, fixed):
        self.calls.append("probe_latent")
        return self._lat or real_adapter.LatentFacts(
            batch=1, channels=128, height=8, width=8, dtype="torch.float32", device="cpu", is_nested=False,
            is_empty=True, has_noise_mask=False, has_batch_index=False, raw_shape=(1, 128, 8, 8),
            raw_downscale_ratio_spacial=None, rescaled_by_native_fix=False, pixel_width=128, pixel_height=128)

    def native_sigmas(self, model, scheduler, steps):
        self.calls.append(("native", scheduler, steps))
        return Arr(self._native)

    def shifted_sigmas(self, model, scheduler, steps, shift):
        self.calls.append(("shifted", scheduler, steps, shift))
        return Arr(self._shifted)


def run(fake, **kw):
    args = dict(model=object(), latent={"samples": object()}, steps=50, shift=5.0, mode="reference")
    args.update(kw)
    with mock.patch.object(schedule, "adapter", fake):
        return schedule.MossPhotonSchedule().build(**args)


def info_validator():
    with open(os.path.join(CONTRACTS, "schemas", "schedule_info.schema.json"), "r", encoding="utf-8") as f:
        return Draft202012Validator(json.load(f))


class InputTypesTest(unittest.TestCase):
    def test_input_types_equal_the_design(self):
        required = schedule.MossPhotonSchedule.INPUT_TYPES()["required"]
        self.assertEqual(list(required), ["model", "latent", "steps", "shift", "mode", "scheduler"])
        self.assertEqual(required["model"][0], "MODEL")
        self.assertEqual(required["latent"][0], "LATENT")
        kind, o = required["steps"]
        self.assertEqual((kind, o["default"], o["min"], o["max"]), ("INT", 50, 1, 1000))
        kind, o = required["shift"]
        self.assertEqual((kind, o["default"], o["min"], o["max"], o["step"]), ("FLOAT", 5.0, 0.01, 100.0, 0.01))
        options, o = required["mode"]
        self.assertEqual((options, o["default"]), (["reference", "shifted"], "reference"))
        options, o = required["scheduler"]
        self.assertEqual((options, o["default"]), (["sgm_uniform", "simple"], "sgm_uniform"))
        self.assertTrue(o["tooltip"].strip())

    def test_node_surface(self):
        cls = schedule.MossPhotonSchedule
        self.assertEqual(cls.RETURN_TYPES, ("SIGMAS", "PHOTON_SCHEDULE_INFO"))
        self.assertEqual(cls.RETURN_NAMES, ("sigmas", "schedule_info"))
        self.assertEqual((cls.FUNCTION, cls.CATEGORY), ("build", consts.CAT_SAMPLING))
        self.assertEqual(len(cls.OUTPUT_TOOLTIPS), 2)
        self.assertEqual(schedule.NODE_DISPLAY_NAME_MAPPINGS, {"MossPhotonSchedule": "Moss Photon Schedule"})


class BuildTest(unittest.TestCase):
    def test_reference_mode_returns_native_sigmas_and_a_valid_info(self):
        fake = FakeAdapter()
        sigmas, info = run(fake)
        self.assertEqual(sigmas.tolist(), REFERENCE_50)
        self.assertEqual(sorted(e.message for e in info_validator().iter_errors(info)), [])
        self.assertTrue(info["is_native_reference"])
        self.assertEqual(info["sigmas_sha256"], consts.REFERENCE_SIGMAS_SHA256)
        self.assertEqual(info["mode"], "reference")
        self.assertEqual(fake.calls.count(("native", "sgm_uniform", 50)), 2)  # one for the schedule, one fresh to compare
        self.assertNotIn(("shifted", "sgm_uniform", 50, 5.0), fake.calls)

    def test_info_equals_the_contract_fixture(self):
        with open(os.path.join(CONTRACTS, "schedule_info.json"), "r", encoding="utf-8") as f:
            self.assertEqual(run(FakeAdapter())[1], json.load(f))

    def test_shifted_mode_marks_not_reference(self):
        shifted = [1.0 - i / 50.0 for i in range(51)]
        fake = FakeAdapter(shifted=shifted)
        sigmas, info = run(fake, mode="shifted", shift=3.0)
        self.assertEqual(sigmas.tolist(), shifted)
        self.assertFalse(info["is_native_reference"])
        self.assertEqual((info["mode"], info["shift_used"], info["shift_model"]), ("shifted", 3.0, 5.0))
        self.assertFalse(info["native_reference"]["equal"])
        self.assertIn(("shifted", "sgm_uniform", 50, 3.0), fake.calls)
        self.assertEqual(sorted(e.message for e in info_validator().iter_errors(info)), [])

    def test_shift_input_ignored_in_reference_mode(self):
        _, info = run(FakeAdapter(), shift=3.0)
        self.assertEqual([(w["code"], w["severity"]) for w in info["warnings"]], [("shift_input_ignored", "info")])
        self.assertEqual(info["shift_used"], 5.0)
        _, info = run(FakeAdapter(), shift=5.0)
        self.assertEqual(info["warnings"], [])

    def test_sd3_multiplier_is_refused_with_the_auraflow_hint_before_any_sigma_exists(self):
        fake = FakeAdapter(model_facts=facts(multiplier=1000.0))
        with self.assertRaises(errors.InvalidModelSampling) as cm:
            run(fake)
        self.assertIn("ModelSamplingAuraFlow", str(cm.exception))
        self.assertEqual(fake.calls, ["probe_model"])

    def test_not_moss_is_refused(self):
        with self.assertRaises(errors.UnsupportedModel):
            run(FakeAdapter(model_facts=facts(is_moss=False)))

    def test_bad_latent_is_refused(self):
        bad = real_adapter.LatentFacts(
            batch=1, channels=4, height=8, width=8, dtype="torch.float32", device="cpu", is_nested=False,
            is_empty=False, has_noise_mask=False, has_batch_index=False, raw_shape=(1, 4, 8, 8),
            raw_downscale_ratio_spacial=None, rescaled_by_native_fix=False, pixel_width=128, pixel_height=128)
        fake = FakeAdapter(lat_facts=bad)
        with self.assertRaises(errors.InvalidLatent):
            run(fake)
        self.assertFalse([c for c in fake.calls if isinstance(c, tuple)])  # no schedule computed

    def test_stock_latent_repair_is_an_info_not_an_error(self):
        repaired = real_adapter.LatentFacts(
            batch=1, channels=128, height=8, width=8, dtype="torch.float32", device="cpu", is_nested=False,
            is_empty=True, has_noise_mask=False, has_batch_index=False, raw_shape=(1, 4, 16, 16),
            raw_downscale_ratio_spacial=8.0, rescaled_by_native_fix=True, pixel_width=128, pixel_height=128)
        _, info = run(FakeAdapter(lat_facts=repaired))
        self.assertEqual([w["code"] for w in info["warnings"]], ["latent_rescaled_by_native_fix"])

    def test_invalid_schedule_is_raised_and_never_returned(self):
        broken = list(REFERENCE_50)
        broken[5] = float("nan")
        with self.assertRaises(errors.InvalidSchedule) as cm:
            run(FakeAdapter(native=broken))
        self.assertEqual(cm.exception.issue.param, "steps")
        self.assertIn("sigmas[5]", str(cm.exception))

    def test_wrong_length_is_invalid_schedule(self):
        with self.assertRaises(errors.InvalidSchedule):
            run(FakeAdapter(), steps=30)  # the fake still returns 51 values for 30 steps

    def test_step_ratio_is_an_error_in_reference_and_a_warning_in_shifted(self):
        jumpy = [1.0, 0.9, 0.8, 0.7, 0.69, 0.2, 0.0]
        with self.assertRaises(errors.InvalidSchedule):
            run(FakeAdapter(native=jumpy), steps=6)
        _, info = run(FakeAdapter(native=jumpy, shifted=jumpy), steps=6, mode="shifted", shift=2.0)
        self.assertEqual([w["code"] for w in info["warnings"]], ["sigma_step_ratio"])

    def test_unknown_mode_or_scheduler_is_a_typed_error(self):
        """n1 (R3): reachable through the API, so never a bare ValueError."""
        for kw, param in ((dict(mode="other"), "mode"), (dict(scheduler="karras"), "scheduler")):
            with self.subTest(kw=kw):
                with self.assertRaises(errors.UnsupportedFeature) as cm:
                    run(FakeAdapter(), **kw)
                self.assertEqual((cm.exception.code, cm.exception.issue.param), ("unsupported_feature", param))
                self.assertNotIsInstance(cm.exception, ValueError)

    def test_simple_scheduler_is_passed_to_the_native_call_and_recorded(self):
        """M5: the turbo presets use simple; the node must not hard-code sgm_uniform."""
        turbo = [1.0, 0.97222, 0.9375, 0.89286, 0.83333, 0.75, 0.625, 0.41667, 0.0]
        fake = FakeAdapter(native=turbo)
        sigmas, info = run(fake, steps=8, scheduler="simple")
        self.assertEqual(sigmas.tolist(), turbo)
        self.assertEqual(fake.calls.count(("native", "simple", 8)), 2)
        self.assertFalse([c for c in fake.calls if isinstance(c, tuple) and c[1] == "sgm_uniform"])
        self.assertEqual(info["scheduler"], "simple")
        self.assertTrue(info["is_native_reference"])
        self.assertEqual(sorted(e.message for e in info_validator().iter_errors(info)), [])
        fake = FakeAdapter(shifted=turbo)
        run(fake, steps=8, scheduler="simple", mode="shifted", shift=3.0)
        self.assertIn(("shifted", "simple", 8, 3.0), fake.calls)

    def test_default_scheduler_is_sgm_uniform_and_the_old_call_shape_still_works(self):
        fake = FakeAdapter()
        _, info = run(fake)
        self.assertEqual(info["scheduler"], "sgm_uniform")

    def test_unreadable_sigma_max_defaults_like_the_sampler(self):
        """m5: ``sigma_max=None`` crashed validate_sigmas' caller with a TypeError."""
        _, info = run(FakeAdapter(model_facts=facts(sigma_max=None)))
        self.assertTrue(info["is_native_reference"])

    def test_unreadable_shift_is_a_typed_model_error_not_a_type_error(self):
        for mode in ("reference", "shifted"):
            with self.subTest(mode=mode):
                with self.assertRaises(errors.InvalidModelSampling) as cm:
                    run(FakeAdapter(model_facts=facts(shift=None), shifted=REFERENCE_50), mode=mode)
                self.assertIn("shift", str(cm.exception))

    def test_sigma_math_used_is_the_pure_module(self):
        self.assertIs(schedule.sigma, sigma)


if __name__ == "__main__":
    unittest.main()
