"""MossPhotonDiagnostics (design 4.7, 6.6 test_diagnostics_*): measurements only, nothing else changes."""
import copy
import hashlib
import inspect
import unittest

import torch

from moss_photon.core import canon, consts, errors
from moss_photon.core import report as report_mod
from moss_photon.nodes import diagnostics

from ..save.common import load_report


def tensor_hash(t):
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


class Poison:
    """A stand-in for photos: only the attributes in ``allowed`` may be touched (an AssertionError otherwise)."""

    def __init__(self, allowed=(), batch=2):
        object.__setattr__(self, "_allowed", set(allowed))
        object.__setattr__(self, "shape", (batch, 8, 8, 3))
        object.__setattr__(self, "touched", [])

    def __getattribute__(self, name):
        if name in ("_allowed", "touched", "__class__"):
            return object.__getattribute__(self, name)
        object.__getattribute__(self, "touched").append(name)
        if name not in object.__getattribute__(self, "_allowed"):
            raise AssertionError("photo attribute %r was touched" % name)
        return object.__getattribute__(self, name)


class DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.node = diagnostics.MossPhotonDiagnostics()
        self.report = load_report()
        gen = torch.Generator().manual_seed(21)
        self.image = torch.rand((2, 16, 24, 3), generator=gen)
        self.masks = (torch.rand((2, 16, 24), generator=gen) > 0.5).float()

    def inspect(self, **kw):
        args = dict(report=self.report)
        args.update(kw)
        return self.node.inspect(**args)

    def test_inputs_are_bit_identical_after_the_call(self):
        r_before = canon.canonical_json(self.report)
        i_before, m_before = tensor_hash(self.image), tensor_hash(self.masks)
        self.inspect(image=self.image, masks=self.masks)
        self.assertEqual(canon.canonical_json(self.report), r_before)
        self.assertEqual(tensor_hash(self.image), i_before)
        self.assertEqual(tensor_hash(self.masks), m_before)

    def test_there_is_no_image_output(self):
        self.assertEqual(diagnostics.MossPhotonDiagnostics.RETURN_TYPES, ("PHOTON_REPORT", "STRING"))
        self.assertEqual(diagnostics.MossPhotonDiagnostics.RETURN_NAMES, ("report", "summary"))
        out = self.inspect(image=self.image)
        self.assertEqual(len(out["result"]), 2)
        self.assertFalse(any(torch.is_tensor(x) for x in out["result"]))

    def test_returns_a_copy_with_only_a_diagnostics_block_added(self):
        out = self.inspect(image=self.image, masks=self.masks)["result"][0]
        self.assertIsNot(out, self.report)
        block = out["diagnostics"]
        rest = {k: v for k, v in out.items() if k != "diagnostics"}
        self.assertEqual(rest, self.report)  # recipe_hash and every other field unchanged
        self.assertNotIn("diagnostics", self.report)
        self.assertEqual(block["parent_recipe_hash"], self.report["recipe_hash"])
        self.assertEqual(report_mod.validate_report(out), [])

    def test_image_and_mask_metrics_are_attached_per_batch_item(self):
        block = self.inspect(image=self.image, masks=self.masks)["result"][0]["diagnostics"]
        self.assertEqual(len(block["image"]["items"]), 2)
        first = block["image"]["items"][0]
        self.assertEqual((first["width"], first["height"], first["channels"]), (24, 16, 3))
        luma = (0.2126 * self.image[0, :, :, 0] + 0.7152 * self.image[0, :, :, 1] +
                0.0722 * self.image[0, :, :, 2]).double()
        self.assertAlmostEqual(first["mean_luma"], float(luma.mean()), places=6)
        self.assertEqual(len(block["masks"]["items"]), 2)
        self.assertAlmostEqual(block["masks"]["items"][1]["coverage_frac"], float(self.masks[1].mean()), places=6)
        self.assertNotIn("identity", block)

    def test_without_optional_inputs_the_block_only_has_the_parent_hash(self):
        block = self.inspect()["result"][0]["diagnostics"]
        self.assertEqual(block, {"parent_recipe_hash": self.report["recipe_hash"]})

    def test_reference_photos_without_consent_are_ignored_with_a_warning_and_never_read(self):
        photos = Poison()  # any attribute access, even .shape, raises
        out = self.inspect(reference_images=photos, consent_reference_photos=False)
        block = out["result"][0]["diagnostics"]
        self.assertNotIn("identity", block)
        self.assertEqual([w["code"] for w in block["warnings"]], ["reference_photos_ignored"])
        self.assertEqual(block["warnings"][0]["severity"], "warning")
        self.assertEqual(photos.touched, [])
        self.assertIn("reference_photos_ignored", out["result"][1])

    def test_with_consent_only_the_count_is_kept(self):
        photos = Poison(allowed=("shape",), batch=3)
        block = self.inspect(reference_images=photos, consent_reference_photos=True)["result"][0]["diagnostics"]
        self.assertEqual(block["identity"], {"status": "not_run", "reason": "missing_guard_dependency",
                                             "n_reference_images": 3})
        self.assertNotIn("warnings", block)
        self.assertEqual(set(photos.touched), {"shape"})

    def test_reference_photo_pixels_never_reach_the_report(self):
        photos = torch.full((2, 8, 8, 3), 0.123456)
        out = self.inspect(reference_images=photos, consent_reference_photos=True)
        self.assertNotIn("0.123456", canon.canonical_json(out["result"][0]))
        self.assertNotIn("0.123456", out["result"][1])

    def test_summary_has_the_run_facts_and_the_metrics(self):
        out = self.inspect(image=self.image, masks=self.masks)
        self.assertEqual(out["ui"]["text"], [out["result"][1]])
        self.assertTrue(out["result"][1].startswith(report_mod.format_summary(self.report)))
        self.assertIn("image[1]: 24x16", out["result"][1])
        self.assertIn("mask[0]: 24x16", out["result"][1])

    def test_newer_schema_and_invalid_reports_are_typed_errors(self):
        newer = copy.deepcopy(self.report)
        newer["schema_version"] = 2
        with self.assertRaises(errors.UnsupportedFeature):
            self.inspect(report=newer)
        broken = copy.deepcopy(self.report)
        del broken["sampling"]
        with self.assertRaises(errors.OutputError):
            self.inspect(report=broken)
        with self.assertRaises(errors.OutputError):
            self.inspect(report="not a report")

    def test_wrong_tensor_rank_is_a_typed_error(self):
        with self.assertRaises(errors.OutputError):
            self.inspect(image=torch.zeros(8, 8))
        with self.assertRaises(errors.OutputError):
            self.inspect(masks=torch.zeros(1, 1, 8, 8))


class DiagnosticsSchemaTest(unittest.TestCase):
    def test_input_types_equal_design_4_7(self):
        spec = diagnostics.MossPhotonDiagnostics.INPUT_TYPES()
        self.assertEqual({k: v[0] for k, v in spec["required"].items()}, {"report": "PHOTON_REPORT"})
        self.assertEqual({k: v[0] for k, v in spec["optional"].items()},
                         {"image": "IMAGE", "reference_images": "IMAGE", "masks": "MASK",
                          "consent_reference_photos": "BOOLEAN"})
        self.assertIs(spec["optional"]["consent_reference_photos"][1]["default"], False)

    def test_node_attributes_and_registration(self):
        cls = diagnostics.MossPhotonDiagnostics
        self.assertTrue(cls.OUTPUT_NODE)
        self.assertEqual((cls.FUNCTION, cls.CATEGORY), ("inspect", consts.CAT_DIAGNOSTICS))
        self.assertEqual(diagnostics.NODE_CLASS_MAPPINGS, {"MossPhotonDiagnostics": cls})
        self.assertEqual(diagnostics.NODE_DISPLAY_NAME_MAPPINGS, {"MossPhotonDiagnostics": "Moss Photon Diagnostics"})
        declared = {k for g in cls.INPUT_TYPES().values() for k in g}
        self.assertLessEqual(declared, set(inspect.signature(cls.inspect).parameters))


if __name__ == "__main__":
    unittest.main()
