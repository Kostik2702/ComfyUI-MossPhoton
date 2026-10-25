"""MossPhotonSaveImage behaviour: never overwrites, race safe, typed errors, clean failure (design 4.8)."""
import copy
import io
import os
import threading
import unittest
from unittest import mock

import torch
from PIL import Image

from moss_photon.core import errors, saveio
from moss_photon.nodes import save

from .common import SaveTestCase


class SaveNodeTest(SaveTestCase):
    def setUp(self):
        super().setUp()
        self.image = torch.rand((1, 16, 16, 3), generator=torch.Generator().manual_seed(3))

    def test_result_is_ui_images_and_no_outputs(self):
        batch = torch.rand((3, 16, 16, 3))
        out = self.run_save(batch, filename_prefix="sub/img", metadata="both")
        self.assertEqual(out["result"], ())
        self.assertEqual(out["ui"]["images"], [
            {"filename": "img_0000%d_.png" % n, "subfolder": "sub", "type": "output"} for n in (1, 2, 3)])
        self.assertEqual(len(self.listing("sub")), 6)

    def test_existing_files_are_never_overwritten(self):
        for name in ("img_00001_.png", "img_00002_.json"):
            with open(os.path.join(self.out, name), "wb") as f:
                f.write(b"PRECIOUS")
        out = self.run_save(self.image, filename_prefix="img", metadata="both")
        self.assertEqual(self.read("img_00001_.png"), b"PRECIOUS")
        self.assertEqual(self.read("img_00002_.json"), b"PRECIOUS")
        self.assertEqual(out["ui"]["images"][0]["filename"], "img_00003_.png")  # pair skips the stale sidecar number
        self.assertEqual(self.listing(), ["img_00001_.png", "img_00002_.json", "img_00003_.json", "img_00003_.png"])

    def test_concurrent_saves_never_collide(self):
        barrier = threading.Barrier(6)
        failures, names = [], []

        def worker():
            try:
                barrier.wait()
                out = self.run_save(self.image, filename_prefix="race", metadata="both")
                names.append(out["ui"]["images"][0]["filename"])
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(failures, [])
        self.assertEqual(len(set(names)), 6)
        for name in names:
            Image.open(io.BytesIO(self.read(name))).load()  # complete, decodable PNG, not a half-written one
            self.assertEqual(self.read_json(name[:-4] + ".json")["file"]["name"], name)

    def test_unsafe_prefix_is_refused_before_any_file_is_made(self):
        for prefix in ("../escape", "C:/abs", "/abs", ""):
            with self.subTest(prefix=prefix):
                with self.assertRaises(errors.OutputError):
                    self.run_save(self.image, filename_prefix=prefix)
        self.assertEqual(self.listing(), [])

    def test_invalid_report_and_newer_schema_are_typed_errors_and_write_nothing(self):
        broken = copy.deepcopy(self.report)
        del broken["recipe_hash"]
        with self.assertRaises(errors.OutputError):
            self.run_save(self.image, report=broken)
        with self.assertRaises(errors.OutputError):
            self.run_save(self.image, report=[1, 2])
        newer = copy.deepcopy(self.report)
        newer["schema_version"] = 99
        with self.assertRaises(errors.UnsupportedFeature):
            self.run_save(self.image, report=newer)
        self.assertEqual(self.listing(), [])

    def test_bad_image_rank_and_unknown_choices_are_typed_errors(self):
        with self.assertRaises(errors.OutputError):
            self.run_save(torch.rand((16, 16, 3)))
        with self.assertRaises(errors.OutputError):
            self.run_save(self.image, format="bmp")
        with self.assertRaises(errors.OutputError):
            self.run_save(self.image, metadata="exif")
        self.assertEqual(self.listing(), [])

    def test_a_failed_write_leaves_no_reserved_files_behind(self):
        with mock.patch.object(saveio, "encode_image", side_effect=OSError("disk full")):
            with self.assertRaises(errors.OutputError):
                self.run_save(self.image, filename_prefix="img", metadata="both")
        self.assertEqual(self.listing(), [])
        with mock.patch.object(saveio, "encode_image", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):  # interrupt propagates untouched, files still cleaned
                self.run_save(self.image, filename_prefix="img", metadata="both")
        self.assertEqual(self.listing(), [])

    def test_metadata_none_writes_pixels_only(self):
        self.run_save(self.image, filename_prefix="img", metadata="none", prompt={"1": 1}, embed_workflow=True)
        self.assertEqual(self.listing(), ["img_00001_.png"])
        data = self.read("img_00001_.png")
        self.assertNotIn(b"tEXt", data)
        self.assertNotIn(b"iTXt", data)

    def test_png_embedded_only_writes_no_sidecar(self):
        self.run_save(self.image, filename_prefix="img", metadata="embedded")
        self.assertEqual(self.listing(), ["img_00001_.png"])

    def test_a_report_with_a_diagnostics_block_is_saved(self):
        from moss_photon.core import report as report_mod
        rep = report_mod.with_diagnostics(self.report, {"image": {"items": []}})
        self.run_save(self.image, report=rep, filename_prefix="img", metadata="sidecar")
        self.assertEqual(self.read_json("img_00001_.json")["report"]["diagnostics"]["image"], {"items": []})

    def test_batch_num_token_is_expanded_like_the_stock_node(self):
        self.run_save(torch.rand((2, 8, 8, 3)), filename_prefix="b%batch_num%", metadata="none")
        self.assertEqual(self.listing(), ["b0_00001_.png", "b1_00002_.png"])  # the counter runs on, as in the stock node

    def test_module_exports_the_stable_id(self):
        self.assertEqual(save.NODE_CLASS_MAPPINGS, {"MossPhotonSaveImage": save.MossPhotonSaveImage})


if __name__ == "__main__":
    unittest.main()
