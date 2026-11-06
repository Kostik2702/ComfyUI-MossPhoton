"""MossPhotonHiresFix, the pure side (finishing/hires.py, numpy only): the release map of the face, the late-entry mask
formula and the judgement of ``face_sigma`` against the schedule.

The formula is the contract of the node: ``v`` is the release map (1 outside the faces, 0 in the face core), the mask the
sampler blends with is 1 where a pixel is free to change and 0 where it is held at the source. A pixel with ``v == 1`` must
come out as exactly 1.0 at every sigma of the run, so that the pixels outside the faces are not touched by the blending at all.
"""
import random
import unittest

import numpy as np

from moss_photon.finishing import hires

S0 = float(np.float32(0.861))
SCHEDULE = tuple(float(np.float32(s)) for s in (0.861, 0.776, 0.668, 0.536, 0.381, 0.202, 0.0))
FACE = (0.4, 0.3, 0.6, 0.7)   # x0, y0, x1, y1 as fractions of the image


class ReleaseMapTest(unittest.TestCase):
    def test_shape_dtype_and_range(self):
        m = hires.release_map(48, 64, [FACE])
        self.assertEqual((m.shape, m.dtype), ((48, 64), np.float32))
        self.assertGreaterEqual(float(m.min()), 0.0)
        self.assertLessEqual(float(m.max()), 1.0)

    def test_no_face_is_all_ones(self):
        m = hires.release_map(16, 16, [])
        self.assertTrue(np.array_equal(m, np.ones((16, 16), np.float32)))

    def test_outside_is_exactly_one_and_the_core_exactly_zero(self):
        m = hires.release_map(100, 100, [FACE])
        self.assertEqual(float(m[0, 0]), 1.0)
        self.assertEqual(float(m[99, 99]), 1.0)
        self.assertEqual(float(m[50, 50]), 0.0)       # the centre of the face
        # the core is the inner (1 - feather) of the grown ellipse: a quarter of the box semi-axis is inside it
        cx, cy = 50, 50
        self.assertEqual(float(m[cy, cx + 3]), 0.0)
        self.assertEqual(float(m[cy - 12, cx]), 0.0)
        self.assertGreater(int((m == 0.0).sum()), 100)
        self.assertGreater(int(((m > 0.0) & (m < 1.0)).sum()), 100)   # and there is a ramp

    def test_the_ramp_is_monotonic_along_a_ray_and_flat_at_both_ends(self):
        m = hires.release_map(400, 400, [FACE])
        row = m[200, 200:]
        self.assertTrue(np.all(np.diff(row) >= 0.0))
        self.assertEqual(float(row[0]), 0.0)
        self.assertEqual(float(row[-1]), 1.0)
        ramp = row[(row > 0.0) & (row < 1.0)]
        self.assertGreater(len(ramp), 10)
        # smoothstep: slow at the ends, fastest in the middle
        steps = np.diff(row)
        self.assertLess(float(steps[(row[:-1] > 0.0)][0]), float(steps.max()) / 3.0)

    def test_the_ramp_is_a_smoothstep_of_the_elliptical_radius(self):
        h = w = 801
        half = 0.1
        m = hires.release_map(h, w, [(0.5 - half, 0.5 - half, 0.5 + half, 0.5 + half)], margin=1.3, feather=0.35)
        # radius 0.825 of the grown ellipse (semi-axis 0.13): t = (0.825 - 0.65) / 0.35 = 0.5 -> smoothstep = 0.5
        x = int(round((0.5 + 0.13 * 0.825) * w - 0.5))
        self.assertAlmostEqual(float(m[400, x]), 0.5, delta=0.02)

    def test_the_ramp_is_elliptical_not_circular(self):
        m = hires.release_map(400, 400, [(0.45, 0.2, 0.55, 0.8)])     # a tall, narrow face
        right = int(np.argmax(m[200, 200:] >= 1.0))
        down = int(np.argmax(m[200:, 200] >= 1.0))
        self.assertGreater(down, 2 * right)

    def test_two_faces_combine_with_min(self):
        a, b = (0.1, 0.1, 0.3, 0.3), (0.6, 0.6, 0.9, 0.9)
        both = hires.release_map(90, 90, [a, b])
        self.assertTrue(np.array_equal(both, np.minimum(hires.release_map(90, 90, [a]), hires.release_map(90, 90, [b]))))
        self.assertEqual(float(both[18, 18]), 0.0)
        self.assertEqual(float(both[67, 67]), 0.0)
        self.assertEqual(float(both[50, 5]), 1.0)

    def test_the_boxes_are_fractions_so_the_resolution_does_not_matter(self):
        small = hires.release_map(64, 64, [FACE])
        large = hires.release_map(256, 256, [FACE])
        pooled = large.reshape(64, 4, 64, 4).mean(axis=(1, 3))
        self.assertLess(float(np.abs(small - pooled).mean()), 0.01)
        self.assertAlmostEqual(float((small == 0.0).mean()), float((large == 0.0).mean()), delta=0.01)

    def test_margin_grows_the_zone_and_feather_widens_the_ramp(self):
        base = hires.release_map(200, 200, [FACE])
        wide = hires.release_map(200, 200, [FACE], margin=1.8)
        self.assertGreater(int((wide < 1.0).sum()), int((base < 1.0).sum()))
        soft = hires.release_map(200, 200, [FACE], feather=0.9)
        self.assertLess(int((soft == 0.0).sum()), int((base == 0.0).sum()))
        self.assertGreater(int(((soft > 0.0) & (soft < 1.0)).sum()), int(((base > 0.0) & (base < 1.0)).sum()))

    def test_a_degenerate_box_is_ignored_and_a_box_outside_the_image_does_not_crash(self):
        self.assertTrue(np.array_equal(hires.release_map(20, 20, [(0.5, 0.5, 0.5, 0.9), (0.7, 0.2, 0.3, 0.4)]),
                                       np.ones((20, 20), np.float32)))
        m = hires.release_map(20, 20, [(-0.5, -0.5, 0.2, 0.2), (0.9, 0.9, 1.6, 1.6)])
        self.assertEqual((m.shape, float(m.min())), ((20, 20), 0.0))


class BoxesToFractionsTest(unittest.TestCase):
    def test_pixels_become_fractions_of_the_image(self):
        got = hires.boxes_to_fractions([(32.0, 16.0, 96.0, 80.0, 0.9)], 128, 64)
        self.assertEqual(got, [(0.25, 0.25, 0.75, 1.0)])

    def test_boxes_are_clipped_and_degenerate_ones_dropped(self):
        got = hires.boxes_to_fractions([(-10, -10, 64, 32), (50, 50, 50, 60), (70, 0, 40, 10), (0, 0, 128, 64)], 128, 64)
        self.assertEqual(got, [(0.0, 0.0, 0.5, 0.5), (0.0, 0.0, 1.0, 1.0)])

    def test_an_empty_image_gives_no_boxes(self):
        self.assertEqual(hires.boxes_to_fractions([(0, 0, 1, 1)], 0, 0), [])


class MaskValueTest(unittest.TestCase):
    FACE_SIGMAS = (0.0, 0.05, 0.2, 0.5, 0.8, 0.86, 0.95)

    def test_band_is_five_hundredths(self):
        self.assertEqual(hires.BAND, 0.05)

    def test_v1_is_exactly_one_at_every_sigma_up_to_s0(self):
        rng = random.Random(91)
        sigmas = list(SCHEDULE) + [S0, 1e-6] + [rng.uniform(1e-6, S0) for _ in range(200)]
        for fs in self.FACE_SIGMAS:
            for sigma in sigmas:
                if sigma == 0.0 or sigma > S0:
                    continue
                self.assertEqual(hires.mask_value(1.0, sigma, S0, fs), 1.0, (fs, sigma))

    def test_v1_is_exactly_one_in_float32_arrays_too(self):
        v = np.ones((3, 2, 2), np.float32)
        for fs in self.FACE_SIGMAS:
            for sigma in SCHEDULE[:-1]:
                out = hires.mask_value(v, np.float32(sigma), S0, fs)
                self.assertEqual(out.dtype, np.float32)
                self.assertTrue(bool((out == 1.0).all()), (fs, sigma))

    def test_v0_is_zero_above_face_sigma_plus_band_and_one_at_or_below_face_sigma(self):
        for fs in (0.1, 0.3, 0.5, 0.7):
            for sigma in (fs + hires.BAND + 1e-6, fs + 0.1, S0 + 0.1, 5.0):
                self.assertEqual(hires.mask_value(0.0, sigma, S0, fs), 0.0, (fs, sigma))
            for sigma in (fs, fs - 1e-6, fs / 2, 1e-6):
                self.assertEqual(hires.mask_value(0.0, sigma, S0, fs), 1.0, (fs, sigma))

    def test_v0_is_linear_inside_the_band(self):
        fs = 0.4
        for k in (0.25, 0.5, 0.75):
            self.assertAlmostEqual(hires.mask_value(0.0, fs + k * hires.BAND, S0, fs), 1.0 - k, places=9)

    def test_the_mask_is_monotonic_in_v(self):
        grid = [i / 20 for i in range(21)]
        for fs in (0.2, 0.5, 0.8):
            for sigma in SCHEDULE[:-1]:
                values = [hires.mask_value(v, sigma, S0, fs) for v in grid]
                self.assertEqual(values, sorted(values), (fs, sigma))

    def test_the_mask_stays_in_zero_one(self):
        rng = random.Random(7)
        for _ in range(500):
            v, sigma, fs = rng.uniform(-0.5, 1.5), rng.uniform(0.0, 2.0), rng.uniform(0.0, 1.0)
            out = hires.mask_value(v, sigma, S0, fs)
            self.assertTrue(0.0 <= out <= 1.0, (v, sigma, fs, out))

    def test_it_equals_the_written_formula(self):
        """r = face_sigma + (s0 - face_sigma) * v ; mask = clamp((r - sigma) / BAND + 1, 0, 1)."""
        rng = random.Random(3)
        for _ in range(500):
            v, sigma, fs = rng.random(), rng.uniform(0.0, S0), rng.uniform(0.0, 1.0)
            r = fs + (S0 - fs) * v
            want = min(max((r - sigma) / hires.BAND + 1.0, 0.0), 1.0)
            self.assertAlmostEqual(hires.mask_value(v, sigma, S0, fs), want, places=9)

    def test_an_array_of_v_is_masked_per_pixel(self):
        v = np.array([[0.0, 0.5, 1.0]], np.float32)
        out = hires.mask_value(v, np.float32(0.6), S0, 0.3)
        self.assertEqual(out.shape, v.shape)
        self.assertEqual(float(out[0, 0]), 0.0)    # held: its release (0.3) is far below the sigma
        self.assertEqual(float(out[0, 2]), 1.0)    # free
        self.assertTrue(0.0 < float(out[0, 1]) < 1.0)   # half released: inside the band


class SigmaJudgementTest(unittest.TestCase):
    def test_the_window_is_the_first_and_the_last_non_zero_sigma(self):
        self.assertEqual(hires.sigma_window(SCHEDULE), (SCHEDULE[0], SCHEDULE[-2]))
        self.assertEqual(hires.sigma_window((0.0,)), (None, None))
        self.assertEqual(hires.sigma_window(()), (None, None))

    def test_a_face_sigma_inside_the_window(self):
        for fs in (0.5, 0.3, 0.8, 0.21):
            self.assertEqual(hires.judge_face_sigma(fs, SCHEDULE), hires.INSIDE, fs)

    def test_a_face_sigma_at_or_above_the_first_sigma_protects_nothing(self):
        for fs in (S0, 0.9, 1.0):
            self.assertEqual(hires.judge_face_sigma(fs, SCHEDULE), hires.ABOVE, fs)

    def test_a_face_sigma_at_or_below_the_last_non_zero_sigma_holds_the_face_to_the_end(self):
        for fs in (SCHEDULE[-2], 0.1, 0.0):
            self.assertEqual(hires.judge_face_sigma(fs, SCHEDULE), hires.BELOW, fs)

    def test_a_schedule_without_a_non_zero_sigma_has_no_window(self):
        self.assertEqual(hires.judge_face_sigma(0.5, (0.0,)), hires.NO_WINDOW)


if __name__ == "__main__":
    unittest.main()
