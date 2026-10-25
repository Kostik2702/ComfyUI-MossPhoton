"""imagestats formulas of design 4.7 (design 6.6 test_image_stats)."""
import math
import unittest

import numpy as np
import torch

from moss_photon.core import errors, imagestats

HALF_BIN = 0.5 / imagestats.HIST_BINS


def gray(value, h=8, w=8, c=3):
    return np.full((h, w, c), value, dtype=np.float32)


def reference(img):
    """Independent pure-python implementation of the 4.7 formulas for small images."""
    h, w, _ = img.shape
    luma = [[0.2126 * float(img[y, x, 0]) + 0.7152 * float(img[y, x, 1]) + 0.0722 * float(img[y, x, 2])
             for x in range(w)] for y in range(h)]
    flat = [v for row in luma for v in row]
    mean = sum(flat) / len(flat)
    std = math.sqrt(sum((v - mean) ** 2 for v in flat) / len(flat))
    lap = [255 * (luma[y - 1][x] + luma[y + 1][x] + luma[y][x - 1] + luma[y][x + 1] - 4 * luma[y][x])
           for y in range(1, h - 1) for x in range(1, w - 1)]
    lmean = sum(lap) / len(lap)
    return {"mean_luma": mean, "std_luma": std,
            "clipped_black_frac": sum(v <= 1 / 255 for v in flat) / len(flat),
            "clipped_white_frac": sum(v >= 254 / 255 for v in flat) / len(flat),
            "laplacian_var": sum((v - lmean) ** 2 for v in lap) / len(lap)}


class ImageStatsTest(unittest.TestCase):
    def test_constant_gray(self):
        s = imagestats.image_stats(torch.full((8, 12, 3), 0.5))
        self.assertEqual((s["width"], s["height"], s["channels"]), (12, 8, 3))
        self.assertAlmostEqual(s["mean_luma"], 0.5, places=6)
        self.assertAlmostEqual(s["std_luma"], 0.0, places=6)
        self.assertAlmostEqual(s["p01_luma"], 0.5, delta=HALF_BIN * 2)
        self.assertAlmostEqual(s["p99_luma"], 0.5, delta=HALF_BIN * 2)
        self.assertEqual((s["clipped_black_frac"], s["clipped_white_frac"]), (0.0, 0.0))
        self.assertAlmostEqual(s["laplacian_var"], 0.0, places=6)
        self.assertEqual(s["nonfinite_count"], 0)

    def test_half_black_half_white(self):
        img = gray(0.0)
        img[:, 4:, :] = 1.0
        s = imagestats.image_stats(img)
        self.assertAlmostEqual(s["mean_luma"], 0.5, places=6)
        self.assertAlmostEqual(s["std_luma"], 0.5, places=6)
        self.assertEqual((s["clipped_black_frac"], s["clipped_white_frac"]), (0.5, 0.5))
        self.assertAlmostEqual(s["p01_luma"], HALF_BIN, places=9)
        self.assertAlmostEqual(s["p99_luma"], 1 - HALF_BIN, places=9)
        self.assertAlmostEqual(s["laplacian_var"], reference(img)["laplacian_var"], places=3)
        self.assertGreater(s["laplacian_var"], 0)

    def test_checkerboard_laplacian_variance_is_exactly_1020_squared(self):
        y, x = np.mgrid[0:8, 0:8]
        board = ((x + y) % 2).astype(np.float32)[:, :, None].repeat(3, axis=2)
        s = imagestats.image_stats(board)
        self.assertAlmostEqual(s["laplacian_var"], 1020.0 ** 2, delta=1.0)  # every interior value is +-4*255
        self.assertAlmostEqual(s["mean_luma"], 0.5, places=6)

    def test_matches_an_independent_implementation_on_random_images(self):
        rng = np.random.default_rng(7)
        for shape in ((9, 11, 3), (16, 16, 4), (5, 5, 3)):
            with self.subTest(shape=shape):
                img = rng.random(shape).astype(np.float32)
                img[0, 0, :3] = 0.0  # a clipped black pixel
                img[1, 1, :3] = 1.0  # a clipped white pixel
                s, r = imagestats.image_stats(img), reference(img)
                for key, want in r.items():
                    self.assertAlmostEqual(s[key], want, delta=1e-6 * max(1.0, abs(want)), msg=key)
                self.assertEqual(s["channels"], shape[2])

    def test_luma_weights_are_rec709_on_stored_values(self):
        for ch, weight in enumerate((0.2126, 0.7152, 0.0722)):
            img = np.zeros((4, 4, 3), np.float32)
            img[:, :, ch] = 1.0
            self.assertAlmostEqual(imagestats.image_stats(img)["mean_luma"], weight, places=6)

    def test_nonfinite_values_are_counted_and_excluded(self):
        img = gray(0.25)
        img[2, 3, 1] = np.nan
        img[5, 5, 0] = np.inf
        s = imagestats.image_stats(img)
        self.assertEqual(s["nonfinite_count"], 2)
        self.assertAlmostEqual(s["mean_luma"], 0.25, places=6)  # the two bad pixels are left out
        self.assertAlmostEqual(s["std_luma"], 0.0, places=6)

    def test_all_nonfinite_gives_none_statistics_but_a_count(self):
        s = imagestats.image_stats(np.full((4, 4, 3), np.nan, np.float32))
        self.assertEqual(s["nonfinite_count"], 48)
        for key in ("mean_luma", "std_luma", "p01_luma", "p99_luma", "clipped_black_frac", "clipped_white_frac",
                    "laplacian_var"):
            self.assertIsNone(s[key])

    def test_values_outside_zero_one_do_not_break_the_histogram(self):
        img = gray(0.0)
        img[:, :4, :] = -5.0
        img[:, 4:, :] = 7.0
        s = imagestats.image_stats(img)
        self.assertEqual((s["clipped_black_frac"], s["clipped_white_frac"]), (0.5, 0.5))
        self.assertAlmostEqual(s["p01_luma"], HALF_BIN, places=9)
        self.assertAlmostEqual(s["p99_luma"], 1 - HALF_BIN, places=9)

    def test_tiny_images_have_no_laplacian_and_one_channel_is_luma(self):
        self.assertIsNone(imagestats.image_stats(gray(0.5, h=2, w=8))["laplacian_var"])
        s = imagestats.image_stats(gray(0.4, c=1))
        self.assertAlmostEqual(s["mean_luma"], 0.4, places=6)

    def test_result_is_json_safe_and_input_untouched(self):
        import json
        img = torch.rand((8, 8, 3))
        before = img.clone()
        json.dumps(imagestats.image_stats(img), allow_nan=False)
        self.assertTrue(torch.equal(img, before))

    def test_batch_helpers(self):
        batch = torch.stack([torch.zeros(8, 8, 3), torch.ones(8, 8, 3)])
        items = imagestats.batch_image_stats(batch)
        self.assertEqual([round(i["mean_luma"], 3) for i in items], [0.0, 1.0])
        self.assertEqual(len(imagestats.batch_image_stats(torch.zeros(8, 8, 3))), 1)
        with self.assertRaises(errors.OutputError):
            imagestats.batch_image_stats(torch.zeros(8, 8))
        with self.assertRaises(errors.OutputError):
            imagestats.image_stats(torch.zeros(2, 8, 8, 3))


class MaskCoverageTest(unittest.TestCase):
    def test_half_covered_mask(self):
        m = np.zeros((8, 10), np.float32)
        m[:, :5] = 1.0
        s = imagestats.mask_coverage(m)
        self.assertEqual((s["width"], s["height"], s["coverage_frac"], s["nonfinite_count"]), (10, 8, 0.5, 0))
        self.assertAlmostEqual(s["mean"], 0.5, places=6)

    def test_threshold_is_strictly_above_half(self):
        self.assertEqual(imagestats.mask_coverage(np.full((4, 4), 0.5, np.float32))["coverage_frac"], 0.0)
        self.assertEqual(imagestats.mask_coverage(np.full((4, 4), 0.51, np.float32))["coverage_frac"], 1.0)

    def test_nonfinite_and_shapes(self):
        m = np.ones((4, 4), np.float32)
        m[0, 0] = np.nan
        self.assertEqual(imagestats.mask_coverage(m)["nonfinite_count"], 1)
        self.assertIsNone(imagestats.mask_coverage(np.full((2, 2), np.nan, np.float32))["coverage_frac"])
        self.assertEqual(len(imagestats.batch_mask_coverage(torch.zeros(3, 4, 4))), 3)
        self.assertEqual(len(imagestats.batch_mask_coverage(torch.zeros(4, 4))), 1)
        with self.assertRaises(errors.OutputError):
            imagestats.batch_mask_coverage(torch.zeros(1, 2, 4, 4))


if __name__ == "__main__":
    unittest.main()
