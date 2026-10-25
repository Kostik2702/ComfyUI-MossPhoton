"""MossPhotonSaveImage writes the same pixels as the stock SaveImage (design 6.6, MP-11 acceptance 3).

Both nodes get the same tensor and the real ``folder_paths``; the PNGs are decoded and compared, and with no metadata
the files are byte-equal as well.
"""
import io
import json
import os
import shutil
import tempfile
import unittest

from . import _comfy_env


class SavePixelsEqualNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _comfy_env.setup()
        import numpy as np
        import torch
        import folder_paths
        import nodes
        from PIL import Image
        from moss_photon.nodes import save
        cls.np, cls.torch, cls.fp, cls.nodes, cls.Image, cls.save = np, torch, folder_paths, nodes, Image, save
        cls.saved_output_dir = folder_paths.get_output_directory()

    def setUp(self):
        self.out = tempfile.mkdtemp(prefix="mp11_native_")
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)
        self.fp.set_output_directory(self.out)
        self.addCleanup(self.fp.set_output_directory, self.saved_output_dir)
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "fixtures", "contracts", "report.json"), "r", encoding="utf-8") as f:
            self.report = json.load(f)

    def tensor(self, shape, seed):
        gen = self.torch.Generator().manual_seed(seed)
        t = self.torch.rand(shape, generator=gen) * 1.6 - 0.3  # values below 0 and above 1 saturate identically
        t[0, 0, 0, 0], t[0, 0, 1, 0], t[0, 1, 0, 0] = -2.0, 5.0, 1.0
        return t

    @staticmethod
    def slurp(path):
        with open(path, "rb") as f:
            return f.read()

    def decode(self, path):
        data = self.slurp(path)
        img = self.Image.open(io.BytesIO(data))
        img.load()
        return data, self.np.asarray(img)

    def save_both(self, tensor, metadata="none", **kw):
        native_dir, photon_dir = "native", "photon"
        self.nodes.SaveImage().save_images(tensor, filename_prefix=native_dir + "/img")
        self.save.MossPhotonSaveImage().save(tensor, self.report, filename_prefix=photon_dir + "/img",
                                             format="png", metadata=metadata, **kw)
        return os.path.join(self.out, native_dir), os.path.join(self.out, photon_dir)

    def test_decoded_pixels_and_bytes_are_equal_without_metadata(self):
        for shape in ((1, 32, 48, 3), (3, 16, 16, 3), (2, 24, 8, 4)):
            with self.subTest(shape=shape):
                shutil.rmtree(self.out)
                os.makedirs(self.out)
                native, photon = self.save_both(self.tensor(shape, seed=sum(shape)))
                names = sorted(os.listdir(native))
                self.assertEqual(len(names), shape[0])
                self.assertEqual(sorted(os.listdir(photon)), names)  # same names, no sidecar for metadata none
                for name in names:
                    n_bytes, n_pixels = self.decode(os.path.join(native, name))
                    p_bytes, p_pixels = self.decode(os.path.join(photon, name))
                    self.assertTrue(self.np.array_equal(n_pixels, p_pixels))
                    self.assertEqual(n_pixels.dtype, self.np.uint8)
                    self.assertEqual(n_bytes, p_bytes)

    def test_pixels_stay_equal_when_metadata_is_written(self):
        native, photon = self.save_both(self.tensor((1, 32, 32, 3), seed=77), metadata="both")
        _, n_pixels = self.decode(os.path.join(native, "img_00001_.png"))
        p_bytes, p_pixels = self.decode(os.path.join(photon, "img_00001_.png"))
        self.assertTrue(self.np.array_equal(n_pixels, p_pixels))
        self.assertEqual(sorted(os.listdir(photon)), ["img_00001_.json", "img_00001_.png"])
        side = json.loads(self.slurp(os.path.join(photon, "img_00001_.json")).decode("utf-8"))
        self.assertEqual(side["file"]["bytes"], len(p_bytes))
        self.assertEqual(side["report"]["preset"]["id"], "reference")

    def test_real_folder_paths_counter_continues_and_never_overwrites(self):
        t = self.tensor((1, 16, 16, 3), seed=5)
        node = self.save.MossPhotonSaveImage()
        node.save(t, self.report, filename_prefix="run/img", metadata="none")
        before = self.slurp(os.path.join(self.out, "run", "img_00001_.png"))
        node.save(t * 0.5, self.report, filename_prefix="run/img", metadata="none")
        self.assertEqual(sorted(os.listdir(os.path.join(self.out, "run"))), ["img_00001_.png", "img_00002_.png"])
        self.assertEqual(self.slurp(os.path.join(self.out, "run", "img_00001_.png")), before)

    def test_prefix_escaping_the_output_folder_is_refused_by_the_node(self):
        from moss_photon.core import errors
        with self.assertRaises(errors.OutputError):
            self.save.MossPhotonSaveImage().save(self.tensor((1, 16, 16, 3), 1), self.report,
                                                 filename_prefix="../outside", metadata="none")


if __name__ == "__main__":
    unittest.main()
