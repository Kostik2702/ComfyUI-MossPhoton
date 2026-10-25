"""MossPhotonLoRAProfile surface (design 4.4) and the early exits that need no ComfyUI."""
import sys
import types
import unittest
from unittest import mock

from moss_photon.core import consts, errors, lora
from moss_photon.nodes import lora as node

from .builders import entry, profile_of, SHA_A


class FakeModel:
    """Just enough of a ModelPatcher for the pass-through path (it never calls comfy)."""

    def __init__(self, entries=0, uuid="uuid-0"):
        self.patches = {"w%d" % i: [("p",)] for i in range(entries)}
        self.patches_uuid = uuid


def fake_folder_paths(names):
    mod = types.ModuleType("folder_paths")
    mod.get_filename_list = lambda kind: list(names) if kind == "loras" else []
    return mod


class InputTypesTest(unittest.TestCase):
    def test_input_types_equal_the_design_table(self):
        with mock.patch.dict(sys.modules, {"folder_paths": fake_folder_paths(["a.safetensors", "sub/b.safetensors"])}):
            spec = node.MossPhotonLoRAProfile.INPUT_TYPES()
        required, optional = spec["required"], spec["optional"]
        self.assertEqual(list(required), ["model", "lora_name", "strength_model", "role", "profile"])
        self.assertEqual(list(optional), ["trigger", "profile_in", "clip", "apply_text_encoder", "strength_clip",
                                          "allow_low_coverage", "fingerprint", "w0", "w1", "a", "b"])
        self.assertEqual(required["model"][0], "MODEL")
        choices, opts = required["lora_name"]
        self.assertEqual((choices, opts["default"]), (["none", "a.safetensors", "sub/b.safetensors"], "none"))
        kind, opts = required["strength_model"]
        self.assertEqual((kind, opts["default"], opts["min"], opts["max"], opts["step"]), ("FLOAT", 1.0, -10.0, 10.0, 0.01))
        roles, opts = required["role"]
        self.assertEqual((roles, opts["default"]), (["character", "style", "wardrobe", "scene", "other", "distill"], "character"))
        profiles, opts = required["profile"]
        self.assertEqual((profiles, opts["default"]), (["constant", "phased"], "constant"))
        self.assertEqual(optional["trigger"][0], "STRING")
        self.assertEqual(optional["trigger"][1]["default"], "")
        self.assertEqual(optional["profile_in"][0], "PHOTON_IDENTITY_PROFILE")
        self.assertEqual(optional["clip"][0], "CLIP")
        self.assertEqual((optional["apply_text_encoder"][0], optional["apply_text_encoder"][1]["default"]), ("BOOLEAN", False))
        kind, opts = optional["strength_clip"]
        self.assertEqual((kind, opts["default"], opts["min"], opts["max"], opts["step"]), ("FLOAT", 0.0, -10.0, 10.0, 0.01))
        self.assertEqual((optional["allow_low_coverage"][0], optional["allow_low_coverage"][1]["default"]), ("BOOLEAN", False))
        modes, opts = optional["fingerprint"]
        self.assertEqual((modes, opts["default"]), (["cached_or_compute", "cached_only", "off"], "cached_or_compute"))
        for name, default, low, high in (("w0", 0.3, -10.0, 10.0), ("w1", 0.9, -10.0, 10.0), ("a", 0.2, 0.0, 1.0),
                                         ("b", 0.6, 0.0, 1.0)):
            kind, opts = optional[name]
            self.assertEqual((kind, opts["default"], opts["min"], opts["max"], opts["step"]),
                             ("FLOAT", default, low, high, 0.01), name)

    def test_every_widget_has_a_tooltip(self):
        with mock.patch.dict(sys.modules, {"folder_paths": fake_folder_paths([])}):
            spec = node.MossPhotonLoRAProfile.INPUT_TYPES()
        for group in spec.values():
            for name, (kind, opts) in group.items():
                self.assertTrue(opts.get("tooltip"), name)

    def test_input_types_work_without_comfyui(self):
        with mock.patch.dict(sys.modules, {"folder_paths": None}):  # None in sys.modules makes the import fail
            spec = node.MossPhotonLoRAProfile.INPUT_TYPES()
        self.assertEqual(spec["required"]["lora_name"][0], ["none"])

    def test_node_surface(self):
        cls = node.MossPhotonLoRAProfile
        self.assertEqual(cls.RETURN_TYPES, ("MODEL", "PHOTON_IDENTITY_PROFILE", "CLIP"))
        self.assertEqual(cls.RETURN_NAMES, ("model", "identity_profile", "clip"))
        self.assertEqual(len(cls.OUTPUT_TOOLTIPS), 3)
        self.assertEqual((cls.FUNCTION, cls.CATEGORY), ("apply", consts.CAT_SETUP))
        self.assertTrue(cls.DESCRIPTION)
        self.assertEqual(node.NODE_CLASS_MAPPINGS, {"MossPhotonLoRAProfile": cls})
        self.assertEqual(node.NODE_DISPLAY_NAME_MAPPINGS, {"MossPhotonLoRAProfile": "Moss Photon LoRA Profile"})
        self.assertIn("MossPhotonLoRAProfile", consts.NODE_IDS)


class PassThroughTest(unittest.TestCase):
    def apply(self, model, **kw):
        args = dict(lora_name="none", strength_model=1.0, role="character", profile="constant")
        args.update(kw)
        return node.MossPhotonLoRAProfile().apply(model, **args)

    def test_none_returns_the_same_model_and_clip_objects(self):
        model, clip = FakeModel(), object()
        out_model, profile, out_clip = self.apply(model, clip=clip)
        self.assertIs(out_model, model)
        self.assertIs(out_clip, clip)
        self.assertEqual(lora.validate_identity_profile(profile), [])
        self.assertEqual(profile["stack"], [])

    def test_empty_profile_records_the_existing_entries_as_baseline(self):
        model = FakeModel(entries=7, uuid="uuid-7")
        _m, profile, _c = self.apply(model)
        att = profile["attachment"]
        self.assertEqual((att["patches_uuid"], att["patch_entries_total"], att["baseline_entries"], att["entries"]),
                         ("uuid-7", 7, 7, []))
        self.assertEqual(lora.validate_identity_profile(profile), [])

    def test_none_returns_profile_in_unchanged(self):
        prof = profile_of([entry(0, "character", "c", SHA_A, 0.8)])
        _m, profile, _c = self.apply(FakeModel(), profile_in=prof)
        self.assertIs(profile, prof)

    def test_zero_strength_is_a_pass_through_and_logs_the_skip(self):
        model = FakeModel()
        with self.assertLogs("MossPhoton", level="INFO") as logs:
            out_model, profile, _c = self.apply(model, lora_name="a.safetensors", strength_model=0.0)
        self.assertIs(out_model, model)
        self.assertEqual(profile["stack"], [])
        self.assertTrue(any("zero_strength_lora_skipped" in line for line in logs.output))

    def test_zero_strength_with_a_text_encoder_strength_is_not_a_pass_through(self):
        class Reached(Exception):
            pass

        with mock.patch.object(node.adapter, "probe_model", side_effect=Reached):
            with self.assertRaises(Reached):  # it went past the pass-through into the model probe
                self.apply(FakeModel(), lora_name="a.safetensors", strength_model=0.0, apply_text_encoder=True,
                           strength_clip=0.5, clip=object())

    def test_text_encoder_flag_with_zero_clip_strength_and_zero_model_strength_passes_through(self):
        model = FakeModel()
        out_model, _p, _c = self.apply(model, lora_name="a.safetensors", strength_model=0.0, apply_text_encoder=True,
                                       strength_clip=0.0)
        self.assertIs(out_model, model)


class RefusalsTest(unittest.TestCase):
    def apply(self, **kw):
        args = dict(lora_name="a.safetensors", strength_model=1.0, role="character", profile="constant")
        args.update(kw)
        return node.MossPhotonLoRAProfile().apply(FakeModel(), **args)

    def test_only_constant_and_phased_weight_profiles_exist(self):
        for kind in ("ramp", ""):
            with self.subTest(kind=kind):
                with self.assertRaises(errors.UnsupportedFeature) as ctx:
                    self.apply(profile=kind)
                self.assertEqual(ctx.exception.code, "unsupported_feature")
                self.assertIn("constant", str(ctx.exception))

    def test_even_a_pass_through_refuses_an_unknown_profile(self):
        with self.assertRaises(errors.UnsupportedFeature):
            self.apply(lora_name="none", profile="ramp")

    def test_a_bad_phase_plan_is_refused_before_anything_else(self):
        for kw in ({"w0": float("nan")}, {"w1": 11.0}, {"a": 0.6, "b": 0.2}, {"a": 0.3, "b": 0.3}, {"a": -0.1},
                   {"b": 1.5}):
            with self.subTest(kw=kw):
                with self.assertRaises(errors.UnsupportedFeature) as ctx:
                    self.apply(profile="phased", **kw)
                self.assertEqual(ctx.exception.issue.detail["feature"], "phase_plan")

    def test_a_phased_pass_through_when_both_weights_are_zero_or_there_is_no_file(self):
        model = FakeModel()
        for kw in ({"lora_name": "none"}, {"lora_name": "a.safetensors", "w0": 0.0, "w1": 0.0}):
            with self.subTest(kw=kw):
                out, profile, _c = node.MossPhotonLoRAProfile().apply(
                    model, **dict(dict(lora_name="none", strength_model=1.0, role="character", profile="phased"), **kw))
                self.assertIs(out, model)
                self.assertEqual((profile["stack"], profile["phase_plan"]), ([], None))

    def test_a_phased_adapter_never_takes_the_text_encoder(self):
        with self.assertRaises(errors.UnsupportedFeature) as ctx:
            self.apply(profile="phased", apply_text_encoder=True, strength_clip=0.5, clip=object())
        self.assertEqual(ctx.exception.issue.param, "apply_text_encoder")

    def test_unknown_role_is_a_typed_error(self):
        """n1 (R3): reachable through the API, so never a bare ValueError."""
        with self.assertRaises(errors.UnsupportedFeature) as ctx:
            self.apply(role="boss")
        self.assertEqual((ctx.exception.code, ctx.exception.issue.param), ("unsupported_feature", "role"))
        self.assertNotIsInstance(ctx.exception, ValueError)


class IsChangedTest(unittest.TestCase):
    def test_none_and_missing(self):
        self.assertEqual(node.MossPhotonLoRAProfile.IS_CHANGED(lora_name="none"), "none")
        with mock.patch.object(node, "_resolve_path", side_effect=FileNotFoundError):
            self.assertEqual(node.MossPhotonLoRAProfile.IS_CHANGED(lora_name="gone.safetensors"), "missing")

    def test_size_and_mtime_of_the_file_and_its_sidecar(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.safetensors")
            with open(path, "wb") as f:
                f.write(b"12345")
            with mock.patch.object(node, "_resolve_path", return_value=path):
                first = node.MossPhotonLoRAProfile.IS_CHANGED(lora_name="x.safetensors", strength_model=1.0)
                self.assertTrue(first.startswith("5:") and first.endswith("|-"), first)
                with open(lora.sidecar_path(path), "wb") as f:
                    f.write(b"{}")
                second = node.MossPhotonLoRAProfile.IS_CHANGED(lora_name="x.safetensors")
                self.assertNotEqual(first, second)
                self.assertFalse(second.endswith("|-"))


if __name__ == "__main__":
    unittest.main()
