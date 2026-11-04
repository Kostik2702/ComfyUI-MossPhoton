"""Geometry (crop box, masks, pose proxy, composite) and the LAB colour match of MossPhotonIdentityDetailer (MP-58).

(d) the colour match moves the LAB mean toward the original crop. N02 roll-invariant pose proxy, N05 colour statistics on
the skin only with a narrow gain, R02 a face-shaped paste mask with an edge-aware feather: each with a plant.
"""
import unittest

import numpy as np

from moss_photon.finishing import colormatch, geometry
from moss_photon.finishing import face_detailer as fd

from . import fakes


class SelectAndCropTest(unittest.TestCase):
    def test_select_faces_sorts_by_area_filters_confidence_and_caps(self):
        dets = [(0, 0, 10, 10, 0.9), (0, 0, 50, 50, 0.9), (0, 0, 30, 30, 0.2), (0, 0, 20, 20, 0.9)]
        got = geometry.select_faces(dets, max_faces=2, min_conf=0.45)
        self.assertEqual([d[2] for d in got], [50.0, 20.0])
        self.assertEqual(geometry.select_faces(dets, max_faces=0, min_conf=0.45), [])
        self.assertEqual(geometry.select_faces([], max_faces=4, min_conf=0.45), [])

    def test_equal_areas_keep_detector_order(self):
        dets = [(0, 0, 20, 20, 0.9), (100, 0, 120, 20, 0.8)]
        self.assertEqual(geometry.select_faces(dets, max_faces=2, min_conf=0.1), [tuple(map(float, d)) for d in dets])

    def test_short_side(self):
        self.assertEqual(geometry.short_side((10, 20, 60, 100)), 50.0)
        self.assertEqual(geometry.short_side((10, 20, 10, 100)), 0.0)
        self.assertEqual(geometry.short_side((60, 100, 10, 20)), 0.0)  # inverted box: degenerate, never negative

    def test_crop_is_a_square_of_1_8_times_the_long_side_centred_on_the_face(self):
        bx, by, side = geometry.crop_box((100, 100, 150, 160), 1.8, 400, 400)
        self.assertEqual(side, 108)
        self.assertEqual((bx, by), (71, 76))  # centre (125, 130) minus 54

    def test_crop_is_shifted_not_shrunk_at_the_border(self):
        bx, by, side = geometry.crop_box((0, 0, 50, 60), 1.8, 400, 400)
        self.assertEqual((bx, by, side), (0, 0, 108))
        bx, by, side = geometry.crop_box((350, 340, 400, 400), 1.8, 400, 400)
        self.assertEqual((bx + side, by + side, side), (400, 400, 108))

    def test_crop_is_capped_at_the_short_image_side(self):
        bx, by, side = geometry.crop_box((10, 10, 110, 110), 1.8, 400, 120)
        self.assertEqual(side, 120)
        self.assertEqual(by, 0)
        self.assertTrue(0 <= bx <= 400 - side)

    def test_crop_always_fits_inside_the_image(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            w, h = int(rng.integers(40, 500)), int(rng.integers(40, 500))
            x0, y0 = float(rng.integers(0, w - 10)), float(rng.integers(0, h - 10))
            x1, y1 = x0 + float(rng.integers(5, w)), y0 + float(rng.integers(5, h))
            bx, by, side = geometry.crop_box((x0, y0, x1, y1), 1.8, w, h)
            self.assertGreaterEqual(side, 1)
            self.assertTrue(0 <= bx and bx + side <= w and 0 <= by and by + side <= h, (w, h, bx, by, side))


class FeatherMaskTest(unittest.TestCase):
    def test_shape_dtype_range_and_symmetry(self):
        for side in (63, 64, 108):
            m = geometry.feather_mask(side, 0.08)
            self.assertEqual((m.shape, m.dtype), ((side, side), np.float32))
            self.assertTrue(((m >= 0.0) & (m <= 1.0)).all())
            self.assertTrue(np.array_equal(m, m[::-1]))
            self.assertTrue(np.array_equal(m, m[:, ::-1]))
            self.assertTrue(np.array_equal(m, m.T))

    def test_core_is_one_and_the_four_corners_are_exactly_zero(self):
        m = geometry.feather_mask(108, 0.08)
        self.assertEqual(float(m[54, 54]), 1.0)
        for y, x in ((0, 0), (0, 107), (107, 0), (107, 107)):
            self.assertEqual(float(m[y, x]), 0.0)

    def test_it_is_the_inscribed_ellipse_nothing_leaves_the_circle(self):
        side = 100
        m = geometry.feather_mask(side, 0.08)
        yy, xx = np.mgrid[0:side, 0:side]
        r = np.sqrt((xx - 49.5) ** 2 + (yy - 49.5) ** 2) / 50.0
        self.assertTrue((m[r >= 1.0] == 0.0).all())
        self.assertTrue((m[r < 0.99] > 0.0).all())

    def test_the_ramp_is_8_percent_of_the_crop_and_a_smoothstep(self):
        side = 400
        m = geometry.feather_mask(side, 0.08)
        row = m[199, :200]  # from the left edge to the centre (c = 199.5, a = 200)
        width_px = 0.08 * side  # 32 px inward from the edge of the circle
        # first pixel with the full weight: ~32 px inside the circle edge (x ~ 0 + 32)
        full = int(np.argmax(row >= 1.0))
        self.assertAlmostEqual(full, width_px, delta=2.0)
        # halfway into the ramp the smoothstep is 0.5
        half = int(np.argmin(np.abs(row - 0.5)))
        self.assertAlmostEqual(half, width_px / 2.0, delta=1.5)
        self.assertTrue((np.diff(row) >= -1e-7).all())  # non-decreasing toward the centre

    def test_a_zero_feather_is_a_hard_circle(self):
        m = geometry.feather_mask(64, 0.0)
        self.assertEqual(set(np.unique(m).tolist()) - {0.0, 1.0}, set())
        self.assertEqual(float(m[32, 32]), 1.0)


class CompositeRegionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.src = rng.random((16, 16, 3), dtype=np.float32)
        self.new = rng.random((16, 16, 3), dtype=np.float32)
        self.mask = np.zeros((16, 16), dtype=np.float32)
        self.mask[4:12, 4:12] = 1.0
        self.mask[4, 4:12] = 0.5

    def test_zero_is_copied_one_is_new_half_is_the_average(self):
        out = geometry.composite_region(self.src, self.new, self.mask)
        self.assertTrue(np.array_equal(out[self.mask == 0.0], self.src[self.mask == 0.0]))
        self.assertTrue(np.array_equal(out[self.mask == 1.0], self.new[self.mask == 1.0]))
        np.testing.assert_allclose(out[4, 4:12], 0.5 * self.src[4, 4:12] + 0.5 * self.new[4, 4:12], atol=1e-7)

    def test_inputs_are_not_modified_and_alpha_is_kept(self):
        src4 = np.concatenate([self.src, np.full((16, 16, 1), 0.3, dtype=np.float32)], axis=-1)
        new4 = np.concatenate([self.new, np.full((16, 16, 1), 0.9, dtype=np.float32)], axis=-1)
        before = (src4.copy(), new4.copy(), self.mask.copy())
        out = geometry.composite_region(src4, new4, self.mask)
        self.assertTrue(np.array_equal(src4, before[0]) and np.array_equal(new4, before[1]))
        self.assertTrue(np.array_equal(self.mask, before[2]))
        self.assertTrue((out[..., 3] == 0.3).all())

    def test_a_nan_in_the_new_pixels_outside_the_mask_does_not_leak(self):
        new = self.new.copy()
        new[0, 0] = np.nan
        out = geometry.composite_region(self.src, new, self.mask)
        self.assertTrue(np.isfinite(out).all())


class ColorSpaceTest(unittest.TestCase):
    def test_known_points(self):
        lab = colormatch.rgb_to_lab(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]))
        np.testing.assert_allclose(lab[0], [100.0, 0.0, 0.0], atol=0.05)
        np.testing.assert_allclose(lab[1], [0.0, 0.0, 0.0], atol=1e-9)
        self.assertAlmostEqual(float(lab[2][0]), 53.39, places=1)
        self.assertAlmostEqual(float(lab[2][1]), 0.0, delta=0.05)

    def test_round_trip(self):
        rng = np.random.default_rng(2)
        rgb = rng.random((500, 3))
        np.testing.assert_allclose(colormatch.lab_to_rgb(colormatch.rgb_to_lab(rgb)), rgb, atol=1e-6)

    def test_red_is_positive_a_and_blue_is_negative_b(self):
        lab = colormatch.rgb_to_lab(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
        self.assertGreater(lab[0][1], 50.0)
        self.assertLess(lab[1][2], -50.0)


class LabMatchTest(unittest.TestCase):
    """(d) the LAB mean moves toward the original crop."""

    def setUp(self):
        rng = np.random.default_rng(5)
        base = 0.55 + 0.08 * rng.standard_normal((96, 96, 3))
        self.original = np.clip(base, 0, 1).astype(np.float32)
        warm = base * np.array([1.12, 1.0, 0.82]) + np.array([0.04, 0.0, -0.03])  # the MP-56 warm drift
        self.warm = np.clip(warm, 0, 1).astype(np.float32)
        self.mask = geometry.feather_mask(96, 0.08)

    def lab_mean(self, rgb):
        lab = colormatch.rgb_to_lab(rgb)
        w = self.mask.astype(np.float64)
        return (lab * w[..., None]).sum(axis=(0, 1)) / w.sum()

    def test_the_mean_moves_toward_the_original(self):
        ref = self.lab_mean(self.original)
        before = np.linalg.norm(self.lab_mean(self.warm) - ref)
        matched, stats = colormatch.lab_match(self.warm, self.original, self.mask)
        after = np.linalg.norm(self.lab_mean(matched) - ref)
        self.assertGreater(before, 3.0)  # control: the drift really is visible
        self.assertLess(after, before * 0.05)
        self.assertLess(after, 0.5)
        self.assertAlmostEqual(stats["delta_e_before"], before, places=3)
        self.assertLess(stats["delta_e_after"], stats["delta_e_before"])
        self.assertEqual((matched.dtype, matched.shape), (np.float32, self.warm.shape))

    def test_every_lab_channel_gets_closer_not_only_the_total(self):
        ref = self.lab_mean(self.original)
        before = np.abs(self.lab_mean(self.warm) - ref)
        matched, _ = colormatch.lab_match(self.warm, self.original, self.mask)
        after = np.abs(self.lab_mean(matched) - ref)
        self.assertTrue((after < before).all(), (before, after))

    def test_the_std_moves_toward_the_original_too(self):
        flat = np.clip(0.55 + 0.4 * (self.warm - 0.55), 0, 1).astype(np.float32)  # lower contrast than the original
        matched, stats = colormatch.lab_match(flat, self.original, self.mask)
        self.assertTrue(all(g > 1.0 for g in stats["std_gain"]))
        self.assertGreater(colormatch.rgb_to_lab(matched)[..., 0].std(), colormatch.rgb_to_lab(flat)[..., 0].std())

    def test_matching_an_identical_crop_changes_nothing(self):
        matched, stats = colormatch.lab_match(self.original, self.original, self.mask)
        np.testing.assert_allclose(matched, self.original, atol=1e-5)
        self.assertEqual(stats["delta_e_before"], 0.0)

    def test_the_gain_is_clamped(self):
        flat_ref = np.full((96, 96, 3), 0.5, dtype=np.float32) + 1e-4 * np.random.default_rng(0).random((96, 96, 3), dtype=np.float32)
        _, stats = colormatch.lab_match(self.warm, flat_ref, self.mask)
        self.assertTrue(all(colormatch.GAIN_MIN <= g <= colormatch.GAIN_MAX for g in stats["std_gain"]))
        self.assertEqual(stats["std_gain"][0], colormatch.GAIN_MIN)
        self.assertTrue(stats["gain_clipped"][0])           # and the clipping is reported
        self.assertEqual((colormatch.GAIN_MIN, colormatch.GAIN_MAX), (0.85, 1.15))  # narrow on purpose (N05)

    def test_an_empty_mask_is_refused(self):
        with self.assertRaises(ValueError):
            colormatch.lab_match(self.warm, self.original, np.zeros((96, 96), dtype=np.float32))

    def test_the_inputs_are_not_modified(self):
        a, b = self.warm.copy(), self.original.copy()
        colormatch.lab_match(self.warm, self.original, self.mask)
        self.assertTrue(np.array_equal(a, self.warm) and np.array_equal(b, self.original))


class ColourMatchInThePipelineTest(unittest.TestCase):
    """The same property through ``detail_image``: the redraw drifts warm, the composite stays closer to the original."""

    FACE = ("A", 130, 90, 180, 150)

    def warm_redraw(self, crop):
        return np.clip(crop * np.array([1.15, 1.0, 0.8], dtype=np.float32) + np.array([0.05, 0, -0.04], dtype=np.float32), 0, 1)

    def run_pass(self, color_match):
        img = fakes.make_scene(240, 320, [self.FACE])
        backend = fakes.FakeBackend([(130.0, 90.0, 180.0, 150.0, 0.9)], self.warm_redraw)
        settings = fd.Settings(color_match=color_match, accept_without_references=True)
        out, recs, _ = fd.detail_image(img, settings, backend, [], seed=1)
        return img, out, recs[0]

    def test_the_composite_is_closer_to_the_original_with_the_match(self):
        img, out_on, rec_on = self.run_pass(True)
        _, out_off, rec_off = self.run_pass(False)
        c = rec_on["crop"]
        sl = (slice(c["y"], c["y"] + c["side"]), slice(c["x"], c["x"] + c["side"]))
        mask = fakes.full_mask_of(img.shape[:2], rec_on)[sl] > 0.5

        def lab_gap(out):
            a = colormatch.rgb_to_lab(out[sl])[mask].mean(axis=0)
            b = colormatch.rgb_to_lab(img[sl])[mask].mean(axis=0)
            return float(np.linalg.norm(a - b))

        self.assertLess(lab_gap(out_on), 0.2 * lab_gap(out_off))
        self.assertIsNone(rec_off["color_match"])
        self.assertLess(rec_on["color_match"]["delta_e_after"], rec_on["color_match"]["delta_e_before"])
        self.assertGreater(rec_on["color_match"]["delta_e_before"], 3.0)


def rotate(points, degrees, about=(0.0, 0.0)):
    t = np.deg2rad(degrees)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (np.asarray(points, dtype=np.float64) - about) @ rot.T + about


def legacy_yaw(kps):
    """The pre-fix proxy (audit N02): x coordinates only."""
    ioc = abs(float(kps[1][0]) - float(kps[0][0]))
    if ioc < 1e-6:
        return None
    return abs(float(kps[2][0]) - (float(kps[0][0]) + float(kps[1][0])) / 2.0) / ioc


class YawProxyTest(unittest.TestCase):
    """N02: the lateral offset of the nose is the same wherever the head is rolled to."""
    FRONTAL = fakes.fake_kps((0.0, 0.0, 100.0, 100.0))

    def test_the_legacy_proxy_is_wrong_under_roll(self):
        got = [legacy_yaw(rotate(self.FRONTAL, deg, (50.0, 50.0))) for deg in (0, 45, 60, 90)]
        self.assertAlmostEqual(got[0], 0.0, places=6)
        self.assertGreater(got[1], 0.3)      # a rolled frontal face already looks half-turned
        self.assertGreater(got[2], 0.6)      # past yaw_max = 0.6: a frontal face called a profile
        self.assertTrue(got[3] is None or got[3] > 1e3 or got[3] > 0.6)  # eyes on a vertical line: no value, or nonsense

    def test_the_new_proxy_is_zero_for_a_frontal_face_at_any_roll(self):
        for deg in (0, 15, 45, 60, 90, 135, 180, 270):
            with self.subTest(roll=deg):
                self.assertAlmostEqual(geometry.yaw_proxy(rotate(self.FRONTAL, deg, (50.0, 50.0))), 0.0, places=9)

    def test_a_turned_head_reads_the_same_at_any_roll(self):
        base = fakes.fake_kps((0.0, 0.0, 100.0, 100.0), offset=0.9)
        want = geometry.yaw_proxy(base)
        self.assertAlmostEqual(want, 0.9, places=9)
        for deg in (45, 60, 90, 200):
            self.assertAlmostEqual(geometry.yaw_proxy(rotate(base, deg, (50.0, 50.0))), want, places=9)

    def test_it_grows_with_the_turn_and_is_symmetric(self):
        values = [geometry.yaw_proxy(fakes.fake_kps((0, 0, 100, 100), offset=o)) for o in (0.0, 0.2, 0.5, 0.9)]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(geometry.yaw_proxy(fakes.fake_kps((0, 0, 100, 100), offset=-0.5)), 0.5, places=9)

    def test_it_does_not_depend_on_scale_or_position(self):
        base = fakes.fake_kps((0, 0, 100, 100), offset=0.4)
        self.assertAlmostEqual(geometry.yaw_proxy(base * 3.0 + 40.0), geometry.yaw_proxy(base), places=9)

    def test_missing_or_degenerate_landmarks_give_none(self):
        self.assertIsNone(geometry.yaw_proxy(None))
        self.assertIsNone(geometry.yaw_proxy(np.zeros((2, 2))))
        same_eyes = self.FRONTAL.copy()
        same_eyes[1] = same_eyes[0]
        self.assertIsNone(geometry.yaw_proxy(same_eyes))

    def test_kps_shift_and_iou(self):
        a = self.FRONTAL
        self.assertEqual(geometry.kps_shift(a, a), 0.0)
        self.assertAlmostEqual(geometry.kps_shift(a, a + [20.0, 0.0]), 20.0 / 40.0, places=9)  # ioc = 40
        self.assertEqual(geometry.kps_shift(a, None), float("inf"))
        self.assertEqual(geometry.bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(geometry.bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(geometry.bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1.0 / 3.0)


class FaceMaskTest(unittest.TestCase):
    """R02: the paste mask follows the detector's face box, not the 1.8x crop."""
    SIDE, BOX = 108, (29.0, 24.0, 79.0, 84.0)  # a 50 x 60 face in the middle of a 108 px crop

    def test_it_is_zero_outside_the_dilated_face_ellipse_and_one_in_the_core(self):
        m = geometry.ellipse_mask(self.SIDE, self.BOX, 1.15, 0.08)
        self.assertEqual((m.shape, m.dtype), ((108, 108), np.float32))
        yy, xx = np.mgrid[0:108, 0:108]
        r = np.sqrt(((xx + 0.5 - 54.0) / (25.0 * 1.15)) ** 2 + ((yy + 0.5 - 54.0) / (30.0 * 1.15)) ** 2)
        self.assertTrue((m[r >= 1.0] == 0.0).all())
        self.assertEqual(float(m[54, 54]), 1.0)
        self.assertTrue(((m >= 0) & (m <= 1)).all())

    def test_it_is_much_smaller_than_the_crop_circle_and_taller_than_wide(self):
        m = geometry.ellipse_mask(self.SIDE, self.BOX, 1.15, 0.08)
        crop_circle = geometry.feather_mask(self.SIDE, 0.08)
        self.assertLess(int((m > 0).sum()), 0.6 * int((crop_circle > 0).sum()))
        rows, cols = np.where(m > 0)
        self.assertGreater(rows.max() - rows.min(), cols.max() - cols.min())
        for y, x in ((0, 0), (0, 107), (107, 0), (107, 107), (54, 0), (0, 54), (54, 107), (107, 54)):
            self.assertEqual(float(m[y, x]), 0.0)

    def test_the_ramp_is_a_smoothstep_over_8_percent_of_the_crop(self):
        m = geometry.ellipse_mask(400, (100.0, 100.0, 300.0, 300.0), 1.0, 0.08)  # a circle of radius 100 at (200, 200)
        row = m[200, 100:200]
        width_px = 0.08 * 400
        self.assertAlmostEqual(int(np.argmax(row >= 1.0)), width_px, delta=2.0)
        self.assertTrue((np.diff(row) >= -1e-7).all())

    def test_a_box_at_the_border_is_clipped_not_an_error(self):
        m = geometry.ellipse_mask(64, (-10.0, 5.0, 40.0, 70.0), 1.15, 0.08)
        self.assertEqual(m.shape, (64, 64))
        self.assertGreater(float(m.sum()), 0.0)


class SeamFactorTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.src = np.clip(0.5 + 0.03 * rng.standard_normal((96, 96, 3)), 0, 1).astype(np.float32)
        self.mask = geometry.ellipse_mask(96, (20.0, 16.0, 76.0, 80.0), 1.15, 0.08)

    def test_a_redraw_that_agrees_with_its_surroundings_keeps_the_whole_feather(self):
        f = colormatch.seam_factor(self.src, self.src.copy(), self.mask)
        self.assertTrue(np.array_equal(f, np.ones_like(f)))

    def test_a_redraw_that_disagrees_at_the_ring_lowers_the_weight_only_there(self):
        new = self.src.copy()
        new[..., 0] = np.clip(new[..., 0] + 0.5, 0, 1)  # a large colour jump everywhere
        f = colormatch.seam_factor(self.src, new, self.mask)
        ring = (self.mask > 0) & (self.mask < 1)
        core = self.mask >= 1.0
        self.assertTrue((f[core] == 1.0).all())
        self.assertTrue((f[~(ring | core)] == 1.0).all())
        self.assertLess(float(f[ring].mean()), 0.2)
        self.assertTrue(((f >= 0.0) & (f <= 1.0)).all())

    def test_it_can_only_lower_the_mask(self):
        new = np.clip(self.src + 0.3, 0, 1)
        f = colormatch.seam_factor(self.src, new, self.mask)
        self.assertTrue(((self.mask * f) <= self.mask + 1e-9).all())
        self.assertTrue(((self.mask * f)[self.mask == 0.0] == 0.0).all())

    def test_a_mask_without_a_ring_is_left_alone(self):
        hard = (geometry.ellipse_mask(64, (10, 10, 54, 54), 1.0, 0.0) > 0).astype(np.float32)
        f = colormatch.seam_factor(np.zeros((64, 64, 3), np.float32), np.ones((64, 64, 3), np.float32), hard)
        self.assertTrue((f == 1.0).all())


class SkinStatsTest(unittest.TestCase):
    """N05: colour statistics on skin only, narrow gain, reported."""
    SIDE, BOX = 96, (20.0, 16.0, 76.0, 80.0)

    def scene(self, skin_rgb, hair_rgb, seed):
        """A crop: grey background, a hair ring around the face ellipse, skin inside, dark eyes at the landmarks."""
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:self.SIDE, 0:self.SIDE]
        r = np.sqrt(((xx + 0.5 - 48.0) / 28.0) ** 2 + ((yy + 0.5 - 48.0) / 32.0) ** 2)
        img = np.full((self.SIDE, self.SIDE, 3), 0.4, dtype=np.float64)
        img[r < 1.15] = hair_rgb
        img[r < 0.85] = skin_rgb
        img += 0.01 * rng.standard_normal(img.shape)
        kps = fakes.fake_kps(self.BOX)
        for eye in kps[:2]:
            img[np.hypot(xx + 0.5 - eye[0], yy + 0.5 - eye[1]) < 3.0] = 0.05
        return np.clip(img, 0, 1).astype(np.float32), kps

    def setUp(self):
        self.original, self.kps = self.scene((0.80, 0.62, 0.52), (0.08, 0.08, 0.08), 1)
        self.redraw, _ = self.scene((0.84, 0.62, 0.46), (0.08, 0.08, 0.08), 2)   # a warmer skin, the same hair
        self.skin = geometry.skin_mask(self.SIDE, self.BOX, self.kps)
        self.whole = geometry.ellipse_mask(self.SIDE, self.BOX, 1.15, 0.08)

    def skin_gap(self, rgb):
        a = colormatch.rgb_to_lab(rgb)[self.skin > 0.5].mean(axis=0)
        b = colormatch.rgb_to_lab(self.original)[self.skin > 0.5].mean(axis=0)
        return float(np.linalg.norm(a - b))

    def test_the_skin_mask_excludes_hair_eyes_and_mouth_and_stays_inside_the_face(self):
        s = self.skin
        yy, xx = np.mgrid[0:self.SIDE, 0:self.SIDE]
        r = np.sqrt(((xx + 0.5 - 48.0) / 28.0) ** 2 + ((yy + 0.5 - 48.0) / 32.0) ** 2)
        self.assertTrue((s[r >= 0.85] == 0.0).all())                      # the hair ring and the background are out
        for eye in self.kps[:2]:
            self.assertEqual(float(s[int(eye[1]), int(eye[0])]), 0.0)     # eyes are out
        mouth = (self.kps[3] + self.kps[4]) / 2
        self.assertEqual(float(s[int(mouth[1]), int(mouth[0])]), 0.0)      # lips are out
        self.assertGreater(float(s.sum()), 0.3 * float(self.whole.sum()))  # but it is still most of the face
        plain = geometry.skin_mask(self.SIDE, self.BOX, None)
        self.assertGreater(float(plain.sum()), float(s.sum()))             # without landmarks: the plain ellipse

    def test_the_skin_mask_is_never_empty(self):
        far = np.array([[48.0, 48.0], [148.0, 48.0], [48.0, 48.0], [48.0, 48.0], [48.0, 48.0]])  # huge exclusion radius
        self.assertGreater(float(geometry.skin_mask(self.SIDE, self.BOX, far).sum()), 0.0)

    def test_matching_on_the_skin_corrects_the_skin_where_whole_ellipse_stats_do_not(self):
        """The pre-fix statistics (the whole paste ellipse: hair ring and eyes included) dilute the skin correction."""
        skin_matched, stats = colormatch.lab_match(self.redraw, self.original, self.skin)
        legacy_matched, _ = colormatch.lab_match(self.redraw, self.original, self.whole)
        before = self.skin_gap(self.redraw)
        self.assertGreater(before, 3.0)                                    # control: the drift is real
        self.assertLess(self.skin_gap(skin_matched), 0.1 * before)
        self.assertGreater(self.skin_gap(legacy_matched), 3.0 * self.skin_gap(skin_matched))
        self.assertGreater(stats["skin_px"], 500)

    def test_hair_and_background_colour_cannot_leak_into_the_skin(self):
        """Same skin, a very different hair colour in the REDRAW: the skin statistics, and so the result, do not move."""
        other, _ = self.scene((0.84, 0.62, 0.46), (0.45, 0.30, 0.10), 3)  # blond hair instead of black
        a, sa = colormatch.lab_match(self.redraw, self.original, self.skin)
        b, sb = colormatch.lab_match(other, self.original, self.skin)
        self.assertLess(abs(self.skin_gap(a) - self.skin_gap(b)), 0.5)
        self.assertLess(max(abs(x - y) for x, y in zip(sa["mean_shift_lab"], sb["mean_shift_lab"])), 1.0)

    def test_the_gain_is_narrow_and_the_report_says_what_was_clipped(self):
        harsh = np.clip(0.5 + 3.0 * (self.redraw - 0.5), 0, 1).astype(np.float32)  # far more contrast than the original
        _, stats = colormatch.lab_match(harsh, self.original, self.skin)
        self.assertEqual(stats["std_gain"], [colormatch.GAIN_MIN] * 3)
        self.assertEqual(stats["gain_clipped"], [True, True, True])
        _, ok = colormatch.lab_match(self.redraw, self.original, self.skin)
        self.assertTrue(all(0.85 <= g <= 1.15 for g in ok["std_gain"]))


if __name__ == "__main__":
    unittest.main()
