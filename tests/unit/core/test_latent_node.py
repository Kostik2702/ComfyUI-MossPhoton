"""MossPhotonEmptyLatent (design 4.1) with a fake adapter: no ComfyUI needed."""
import types
import unittest
from unittest import mock

from moss_photon.core import adapter as real_adapter
from moss_photon.core import consts, errors
from moss_photon.nodes import latent


class FakeAdapter(types.SimpleNamespace):
    def __init__(self):
        super().__init__(MIN_SIDE=real_adapter.MIN_SIDE, MAX_SIDE=real_adapter.MAX_SIDE, MAX_BATCH=real_adapter.MAX_BATCH,
                         SPATIAL_RATIO=16, calls=[])
        self.validate_dimensions = real_adapter.validate_dimensions

    def latent_format_constants(self):
        self.calls.append("constants")
        return 128, 16

    def make_empty_latent(self, width, height, batch):
        self.calls.append(("make", width, height, batch))
        return {"samples": ("zeros", batch, 128, height // 16, width // 16)}


class InputTypesTest(unittest.TestCase):
    def test_input_types_equal_the_design(self):
        required = latent.MossPhotonEmptyLatent.INPUT_TYPES()["required"]
        self.assertEqual(list(required), ["width", "height", "batch_size"])
        for side in ("width", "height"):
            kind, opts = required[side]
            self.assertEqual((kind, opts["default"], opts["min"], opts["max"], opts["step"]), ("INT", 1024, 64, 8192, 16))
            self.assertTrue(opts["tooltip"])
        kind, opts = required["batch_size"]
        self.assertEqual((kind, opts["default"], opts["min"], opts["max"]), ("INT", 1, 1, 64))

    def test_node_surface(self):
        cls = latent.MossPhotonEmptyLatent
        self.assertEqual((cls.RETURN_TYPES, cls.RETURN_NAMES, cls.FUNCTION), (("LATENT",), ("latent",), "generate"))
        self.assertEqual(cls.CATEGORY, consts.CAT_SETUP)
        self.assertTrue(cls.DESCRIPTION and len(cls.OUTPUT_TOOLTIPS) == 1)
        self.assertEqual(latent.NODE_CLASS_MAPPINGS, {"MossPhotonEmptyLatent": cls})
        self.assertEqual(latent.NODE_DISPLAY_NAME_MAPPINGS, {"MossPhotonEmptyLatent": "Moss Photon Empty Latent"})
        self.assertIn("MossPhotonEmptyLatent", consts.NODE_IDS)


class GenerateTest(unittest.TestCase):
    def run_node(self, w, h, b=1):
        fake = FakeAdapter()
        with mock.patch.object(latent, "adapter", fake):
            out = latent.MossPhotonEmptyLatent().generate(w, h, b)
        return out, fake

    def test_generate_returns_latent_and_text(self):
        out, fake = self.run_node(1024, 1024)
        self.assertEqual(out["result"][0]["samples"], ("zeros", 1, 128, 64, 64))
        self.assertEqual(out["ui"]["text"], ["1024x1024 -> latent [1,128,64,64], 4096 image tokens, 2.0 MiB latent, 1.05 MP"])
        self.assertEqual(fake.calls, ["constants", ("make", 1024, 1024, 1)])

    def test_batch_scales_the_memory_not_the_tokens(self):
        out, _ = self.run_node(512, 768, 2)
        self.assertEqual(out["ui"]["text"], ["512x768 -> latent [2,128,48,32], 1536 image tokens, 1.5 MiB latent, 0.39 MP"])

    def test_1000_wide_and_1017_high_are_refused_with_nearest_multiples(self):
        for w, h, side, near in ((1000, 1024, "width", "992 or 1008"), (1024, 1017, "height", "1008 or 1024")):
            with self.subTest(side=side):
                with self.assertRaises(errors.InvalidDimensions) as cm:
                    self.run_node(w, h)
                self.assertEqual(cm.exception.issue.param, side)
                self.assertIn(near, str(cm.exception))

    def test_refusal_happens_before_any_latent_exists(self):
        fake = FakeAdapter()
        with mock.patch.object(latent, "adapter", fake):
            with self.assertRaises(errors.InvalidDimensions):
                latent.MossPhotonEmptyLatent().generate(1000, 1000, 1)
        self.assertEqual(fake.calls, [])

    def test_out_of_range_sizes_are_refused(self):
        with self.assertRaises(errors.InvalidDimensions):
            self.run_node(16, 1024)
        with self.assertRaises(errors.InvalidDimensions):
            self.run_node(1024, 1024, 65)

    def test_megapixel_warning_is_text_and_log_not_error(self):
        with self.assertLogs("MossPhoton", level="WARNING") as logs:
            out, _ = self.run_node(64, 64)
        text = out["ui"]["text"][0]
        self.assertIn("megapixels_out_of_documented_range", text)
        self.assertIn("megapixels_out_of_documented_range", logs.output[0])
        with self.assertLogs("MossPhoton", level="WARNING"):
            out, _ = self.run_node(4096, 2048)  # 8.4 MP
        self.assertIn("megapixels_out_of_documented_range", out["ui"]["text"][0])

    def test_no_warning_inside_the_documented_range(self):
        for w, h in ((512, 512), (2048, 1920)):  # 0.26 MP and 3.93 MP
            with self.subTest(size=(w, h)):
                out, _ = self.run_node(w, h)
                self.assertNotIn("megapixels", out["ui"]["text"][0])
        with self.assertLogs("MossPhoton", level="WARNING"):
            out, _ = self.run_node(2048, 2048)  # 4.19 MP
        self.assertIn("megapixels", out["ui"]["text"][0])


if __name__ == "__main__":
    unittest.main()
