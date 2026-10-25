"""jpeg / webp: no EXIF or other embedded metadata, the sidecar instead (LD10; design 4.8, 6.6)."""
import io
import os
import unittest

import torch
from PIL import Image

from .common import SaveTestCase

EXT = {"jpeg": "jpg", "webp": "webp"}


class JpegWebpTest(SaveTestCase):
    def setUp(self):
        super().setUp()
        self.image = torch.rand((1, 24, 32, 3), generator=torch.Generator().manual_seed(4))

    def test_no_exif_or_embedded_data_and_the_sidecar_is_written(self):
        for fmt in ("jpeg", "webp"):
            for mode in ("both", "embedded", "sidecar"):
                with self.subTest(format=fmt, metadata=mode):
                    prefix = "%s_%s" % (fmt, mode)
                    self.run_save(self.image, filename_prefix=prefix, format=fmt, metadata=mode, quality=80,
                                  prompt={"1": 1}, extra_pnginfo={"workflow": 1}, embed_workflow=True)
                    data = self.read("%s_00001_.%s" % (prefix, EXT[fmt]))
                    img = Image.open(io.BytesIO(data))
                    self.assertEqual(len(img.getexif()), 0)
                    self.assertNotIn("exif", img.info)
                    self.assertNotIn(b"Exif", data)
                    self.assertNotIn(b"EXIF", data)
                    self.assertNotIn(b"moss_photon", data)
                    side = self.read_json("%s_00001_.json" % prefix)
                    self.assertEqual((side["file"]["format"], side["file"]["lossless"]), (fmt, False))
                    self.assertFalse(side["file"]["workflow_embedded"])
                    self.assertEqual(side["report"]["preset"]["id"], "reference")

    def test_warning_recorded_when_embedding_was_asked_for(self):
        for fmt in ("jpeg", "webp"):
            for mode, warned in (("both", True), ("embedded", True), ("sidecar", False)):
                with self.subTest(format=fmt, metadata=mode):
                    prefix = "w_%s_%s" % (fmt, mode)
                    result = self.run_save(self.image, filename_prefix=prefix, format=fmt, metadata=mode)
                    codes = [w["code"] for w in self.read_json(prefix + "_00001_.json")["report"]["warnings"]]
                    self.assertEqual("embedded_unsupported_for_format" in codes, warned)
                    self.assertEqual("embedded_unsupported_for_format" in result["ui"]["text"][0], warned)

    def test_metadata_none_writes_only_the_image(self):
        for fmt in ("jpeg", "webp"):
            with self.subTest(format=fmt):
                self.run_save(self.image, filename_prefix="n_" + fmt, format=fmt, metadata="none")
        self.assertEqual(self.listing(), ["n_jpeg_00001_.jpg", "n_webp_00001_.webp"])

    def test_png_with_embedding_does_not_warn(self):
        result = self.run_save(self.image, filename_prefix="p", metadata="both")
        self.assertNotIn("embedded_unsupported_for_format", result["ui"]["text"][0])
        codes = [w["code"] for w in self.read_json("p_00001_.json")["report"]["warnings"]]
        self.assertNotIn("embedded_unsupported_for_format", codes)

    def test_quality_changes_the_lossy_file_and_four_channel_jpeg_is_accepted(self):
        self.run_save(self.image, filename_prefix="q95", format="jpeg", metadata="none", quality=95)
        self.run_save(self.image, filename_prefix="q10", format="jpeg", metadata="none", quality=10)
        self.assertGreater(len(self.read("q95_00001_.jpg")), len(self.read("q10_00001_.jpg")))
        rgba = torch.rand((1, 8, 8, 4))
        self.run_save(rgba, filename_prefix="rgba", format="jpeg", metadata="none")
        self.assertTrue(os.path.exists(os.path.join(self.out, "rgba_00001_.jpg")))


if __name__ == "__main__":
    unittest.main()
