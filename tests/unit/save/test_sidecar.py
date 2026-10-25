"""The JSON sidecar and what it carries (MP-11 acceptance 2; design 4.8, 6.6 test_sidecar)."""
import hashlib
import io
import json
import unittest

import torch
from PIL import Image

from moss_photon.core import canon, saveio

from .common import SaveTestCase, load_report


class BuildSidecarTest(unittest.TestCase):
    def test_shape_and_inputs_untouched(self):
        rep = load_report()
        before = canon.canonical_json(rep)
        info = {"name": "a.png", "format": "png", "width": 8, "height": 8, "lossless": True}
        side = saveio.build_sidecar(rep, file_info=info, batch_index=3, saved_utc="2026-10-25T12:00:00Z")
        self.assertEqual(sorted(side), ["file", "kind", "report", "saved_utc", "schema_version"])
        self.assertEqual((side["schema_version"], side["kind"]), (1, "photon_sidecar"))
        self.assertEqual(side["file"]["batch_index"], 3)
        self.assertEqual(side["saved_utc"], "2026-10-25T12:00:00Z")
        self.assertEqual(canon.canonical_json(rep), before)
        self.assertNotIn("batch_index", info)


class SidecarFileTest(SaveTestCase):
    def setUp(self):
        super().setUp()
        self.image = torch.rand((2, 16, 24, 3), generator=torch.Generator().manual_seed(9))
        self.run_save(self.image, filename_prefix="sub/img", metadata="sidecar")
        self.side = self.read_json("img_00001_.json", "sub")

    def test_top_level_keys(self):
        self.assertEqual(sorted(self.side), ["file", "kind", "report", "saved_utc", "schema_version"])
        self.assertEqual(self.side["kind"], "photon_sidecar")

    def test_one_sidecar_per_image_with_the_same_number(self):
        self.assertEqual(self.listing("sub"),
                         ["img_00001_.json", "img_00001_.png", "img_00002_.json", "img_00002_.png"])
        self.assertEqual(self.read_json("img_00002_.json", "sub")["file"]["batch_index"], 1)

    def test_file_block_describes_the_written_image(self):
        data = self.read("img_00001_.png", "sub")
        f = self.side["file"]
        self.assertEqual((f["name"], f["format"], f["width"], f["height"], f["lossless"], f["batch_index"]),
                         ("img_00001_.png", "png", 24, 16, True, 0))
        self.assertEqual(f["bytes"], len(data))
        self.assertEqual(f["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(Image.open(io.BytesIO(data)).size, (24, 16))

    def test_report_part_carries_preset_params_and_fingerprints(self):
        rep, src = self.side["report"], self.report
        self.assertEqual((rep["preset"]["id"], rep["preset"]["version"], rep["preset"]["pin"]),
                         ("reference", "1.0.0", "reference@1.0.0#00000000"))
        self.assertEqual(rep["sampling"], src["sampling"])  # resolved parameters, incl. sigma fingerprint
        self.assertEqual(rep["model"]["identity"]["sha256"], src["model"]["identity"]["sha256"])
        self.assertEqual(rep["sampling"]["sigmas"]["sha256"], src["sampling"]["sigmas"]["sha256"])
        self.assertEqual(rep["inputs"]["noise_sha256"], src["inputs"]["noise_sha256"])
        self.assertEqual(rep["conditioning"]["positive"]["sha256"], src["conditioning"]["positive"]["sha256"])
        self.assertEqual(rep["lora"], src["lora"])
        self.assertRegex(rep["recipe_hash"], "^[0-9a-f]{64}$")
        self.assertEqual(rep["recipe_hash"], src["recipe_hash"])

    def test_the_input_report_is_not_modified(self):
        before = canon.canonical_json(self.report)
        self.run_save(self.image, report=self.report, filename_prefix="sub/other", metadata="both")
        self.assertEqual(canon.canonical_json(self.report), before)

    def test_embedded_chunk_and_sidecar_agree(self):
        self.run_save(self.image, filename_prefix="both", metadata="both")
        chunk = json.loads(Image.open(io.BytesIO(self.read("both_00001_.png"))).text["moss_photon"])
        side = self.read_json("both_00001_.json")
        for key in ("bytes", "sha256"):
            side["file"].pop(key)
        side["saved_utc"] = chunk["saved_utc"] = None
        self.assertEqual(chunk, side)


if __name__ == "__main__":
    unittest.main()
