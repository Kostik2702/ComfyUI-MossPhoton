"""MossPhotonIdentityDetailer pipeline (MP-58): outside-mask equality (a), accept rule (b), size gate (c), and the
audit rules N01-N04 (global binding, after-gates on the SAME face with a margin, pass-through without references),
each with a plant that fails on the pre-fix behaviour."""
import itertools
import unittest
from unittest import mock

import numpy as np

from moss_photon.core import errors
from moss_photon.finishing import face_detailer as fd
from moss_photon.finishing import geometry

from . import fakes

H, W = 240, 320
FACE_A = ("A", 130, 90, 180, 150)    # 50 x 60 px: below the gate, above the 48 px floor
FACE_C = ("B", 30, 150, 80, 205)     # a second small face, another person
FACE_BIG = ("A", 200, 40, 312, 200)  # 112 x 160 px: at/above the 110 px gate
NO_COLOR = fd.Settings(color_match=False)
OPT_IN = fd.Settings(color_match=False, accept_without_references=True)


def det(face, conf=0.9):
    _, x0, y0, x1, y1 = face
    return (float(x0), float(y0), float(x1), float(y1), conf)


def run(faces, backend, characters, settings=NO_COLOR, seed=5, scene_faces=None):
    img = fakes.make_scene(H, W, scene_faces or faces)
    out, recs, issues = fd.detail_image(img, settings, backend, characters, seed=seed)
    return img, out, recs, issues


class OutsideMaskTest(unittest.TestCase):
    """(a) outside the mask np.array_equal holds; a planted 1-pixel leak turns it red."""

    def setUp(self):
        self.backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen)
        self.img = fakes.make_scene(H, W, [FACE_A])

    def honest(self, settings=NO_COLOR):
        out, recs, _ = fd.detail_image(self.img, settings, self.backend, [fakes.reference("A")], seed=1)
        return out, recs

    def test_outside_the_mask_the_result_is_bit_equal(self):
        out, recs = self.honest()
        self.assertEqual(recs[0]["decision"], "accepted")
        full = fakes.full_mask_of((H, W), recs[0])
        outside = full <= 0.0
        self.assertTrue(np.array_equal(out[outside], self.img[outside]))
        self.assertGreater(int(outside.sum()), H * W // 2)  # control: a big part of the frame is outside
        self.assertFalse(np.array_equal(out, self.img))     # control: the face really changed
        inside_changed = (out != self.img).any(axis=-1)
        self.assertTrue(inside_changed.any())
        self.assertFalse((inside_changed & outside).any())

    def test_the_same_with_colour_match_on(self):
        out, recs = self.honest(fd.Settings(color_match=True))
        outside = fakes.full_mask_of((H, W), recs[0]) <= 0.0
        self.assertTrue(np.array_equal(out[outside], self.img[outside]))

    def test_the_corners_of_the_crop_square_are_outside_the_mask(self):
        out, recs = self.honest()
        c = recs[0]["crop"]
        for y, x in ((c["y"], c["x"]), (c["y"], c["x"] + c["side"] - 1), (c["y"] + c["side"] - 1, c["x"])):
            self.assertTrue(np.array_equal(out[y, x], self.img[y, x]))

    def test_the_input_array_is_never_written(self):
        self.img.setflags(write=False)
        before = self.img.copy()
        out, _ = self.honest()
        self.assertTrue(np.array_equal(self.img, before))
        self.assertIsNot(out, self.img)

    def test_a_backend_that_scribbles_on_its_crop_cannot_touch_the_input(self):
        def scribble(crop):
            crop[:] = 0.0
            return crop
        backend = fakes.FakeBackend([det(FACE_A)], scribble)
        img = fakes.make_scene(H, W, [FACE_A])
        before = img.copy()
        fd.detail_image(img, OPT_IN, backend, [], seed=1)
        self.assertTrue(np.array_equal(img, before))

    def test_a_planted_one_pixel_leak_is_caught_by_the_guard(self):
        real = geometry.composite_region

        def leaky(src_region, new_region, mask):
            out = real(src_region, new_region, mask)
            out[0, 0, 0] = np.nextafter(out[0, 0, 0], np.float32(2.0))  # corner pixel: mask is exactly 0 there
            return out

        with mock.patch.object(geometry, "composite_region", leaky):
            with self.assertRaises(errors.OutputError) as ctx:
                fd.detail_image(self.img, NO_COLOR, self.backend, [fakes.reference("A")], seed=1)
        self.assertIn("outside the face mask", str(ctx.exception))

    def test_a_planted_leak_turns_the_equality_check_itself_red(self):
        """With the guard switched off the leak reaches the output, and the test's own np.array_equal fails."""
        real = geometry.composite_region

        def leaky(src_region, new_region, mask):
            out = real(src_region, new_region, mask)
            out[0, 0, 0] = np.nextafter(out[0, 0, 0], np.float32(2.0))
            return out

        with mock.patch.object(geometry, "composite_region", leaky), \
                mock.patch.object(geometry, "assert_outside_mask_equal", lambda *a: None):
            out, recs, _ = fd.detail_image(self.img, NO_COLOR, self.backend, [fakes.reference("A")], seed=1)
        outside = fakes.full_mask_of((H, W), recs[0]) <= 0.0
        self.assertFalse(np.array_equal(out[outside], self.img[outside]))
        self.assertEqual(int((out[outside] != self.img[outside]).any(axis=-1).sum()), 1)

    def test_the_checker_sees_one_ulp(self):
        mask = np.zeros((4, 4), dtype=np.float32)
        mask[1:3, 1:3] = 1.0
        src = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = src.copy()
        geometry.assert_outside_mask_equal(src, out, mask)  # equal: silent
        out[0, 3, 2] = np.nextafter(out[0, 3, 2], np.float32(1.0))
        with self.assertRaises(errors.OutputError):
            geometry.assert_outside_mask_equal(src, out, mask)
        out2 = src.copy()
        out2[1, 1, 0] = 0.9  # inside the mask: allowed
        geometry.assert_outside_mask_equal(src, out2, mask)

    def test_r02_the_paste_mask_is_the_face_not_the_crop_circle(self):
        """The changed area is the face ellipse around the detector box, far smaller than the 1.8x crop's circle."""
        out, recs = self.honest()
        c = recs[0]["crop"]
        changed = int((out != self.img).any(axis=-1).sum())
        crop_circle = int(np.pi * (c["side"] / 2.0) ** 2)
        self.assertGreater(changed, 0.5 * 50 * 60)
        self.assertLess(changed, 0.6 * crop_circle)
        yy, xx = np.mgrid[0:H, 0:W]
        far = np.hypot(xx - 155, yy - 120) > 0.34 * c["side"]  # the outer ring of the crop: hair and background
        self.assertFalse(((out != self.img).any(axis=-1) & far).any())


class PasteQualityTest(unittest.TestCase):
    """N05 and R02 through the pipeline (not only the helpers): skin-only colour statistics and the seam guard."""
    FACE = ("A", 130, 90, 180, 150)

    def skin_scene(self):
        """Grey background, a dark hair ring around the face ellipse, skin inside it; the face box is FACE."""
        rng = np.random.default_rng(9)
        img = np.full((H, W, 3), 0.4, dtype=np.float32) + 0.01 * rng.standard_normal((H, W, 3)).astype(np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx + 0.5 - 155.0) / 28.75) ** 2 + ((yy + 0.5 - 120.0) / 34.5) ** 2)  # 1.15 x the 50 x 60 box
        img[r < 1.0] = 0.08
        r_skin = np.sqrt(((xx + 0.5 - 155.0) / 25.0) ** 2 + ((yy + 0.5 - 120.0) / 30.0) ** 2)
        img[r_skin < 0.85] = (0.78, 0.60, 0.50)
        img += 0.01 * rng.standard_normal(img.shape).astype(np.float32)
        return np.clip(img, 0, 1).astype(np.float32)

    def test_n05_the_pipeline_matches_the_skin_not_the_whole_ellipse(self):
        img = self.skin_scene()

        def warm_skin(crop):  # the skin drifts warm, the dark hair ring stays
            out = crop.copy()
            skin = crop[..., 0] > 0.5
            out[skin] = np.clip(out[skin] * np.array([1.10, 1.0, 0.85], dtype=np.float32), 0, 1)
            return out

        backend = fakes.FakeBackend([det(self.FACE)], warm_skin)
        settings = fd.Settings(color_match=True, accept_without_references=True, feather=0.0)
        out, recs, _ = fd.detail_image(img, settings, backend, [], seed=1)
        from moss_photon.finishing import colormatch
        c = recs[0]["crop"]
        sl = (slice(c["y"], c["y"] + c["side"]), slice(c["x"], c["x"] + c["side"]))
        face_box = (130.0 - c["x"], 90.0 - c["y"], 180.0 - c["x"], 150.0 - c["y"])
        skin = geometry.skin_mask(c["side"], face_box, fakes.fake_kps((c["side"] / 2 - 30.0, c["side"] / 2 - 30.0,
                                                                         c["side"] / 2 + 30.0, c["side"] / 2 + 30.0))) > 0.5
        lab_out = colormatch.rgb_to_lab(out[sl])[skin].mean(axis=0)
        lab_src = colormatch.rgb_to_lab(img[sl])[skin].mean(axis=0)
        drift = colormatch.rgb_to_lab(warm_skin(img[sl][..., :3]))[skin].mean(axis=0)
        self.assertGreater(float(np.linalg.norm(drift - lab_src)), 4.0)       # control: the drift is real
        self.assertLess(float(np.linalg.norm(lab_out - lab_src)), 1.0)         # and the skin is back on the original

    def test_r02_the_seam_guard_keeps_the_original_where_the_redraw_disagrees_at_the_edge(self):
        img = fakes.make_scene(H, W, [self.FACE])
        c_side, face_box = 108, (29.0, 24.0, 79.0, 84.0)
        ring = (lambda m: (m > 0) & (m < 1))(geometry.ellipse_mask(c_side, face_box, 1.15, 0.08))

        def ring_jump(crop):  # the core is untouched, the feather ring gets a large colour jump
            out = crop.copy()
            out[ring, 0] = np.clip(out[ring, 0] + 0.45, 0, 1)
            return out

        backend = fakes.FakeBackend([det(self.FACE)], ring_jump)
        out, recs, _ = fd.detail_image(img, OPT_IN, backend, [], seed=1)
        self.assertEqual(recs[0]["decision"], "accepted_unscorable")
        c = recs[0]["crop"]
        region_out = out[c["y"]:c["y"] + c["side"], c["x"]:c["x"] + c["side"]]
        region_src = img[c["y"]:c["y"] + c["side"], c["x"]:c["x"] + c["side"]]
        pasted = float(np.abs(region_out[..., 0] - region_src[..., 0])[ring].mean())
        unguarded = float((geometry.ellipse_mask(c_side, face_box, 1.15, 0.08)[ring] * 0.45).mean())
        self.assertGreater(unguarded, 0.15)                 # control: without the guard the jump WOULD be pasted
        self.assertLess(pasted, 0.4 * unguarded)            # with it, a fraction of it is


class AcceptRuleTest(unittest.TestCase):
    """(b) when the score drops the original crop comes back; the report says before, after and the decision."""

    def test_a_worse_face_returns_the_original(self):
        backend = fakes.FakeBackend([det(FACE_A)], fakes.flatten)
        img, out, recs, issues = run([FACE_A], backend, [fakes.reference("A")])
        self.assertTrue(np.array_equal(out, img))
        self.assertIs(out, img)  # nothing was accepted: not even a copy
        r = recs[0]
        self.assertEqual(r["decision"], "rejected")
        self.assertEqual(r["rollback"], "score_delta")
        self.assertLess(r["score_after"], r["score_before"])
        self.assertFalse(r["unscorable"])
        self.assertIn("detailer_rejected", [i.code for i in issues])

    def test_a_better_face_is_kept(self):
        backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen)
        img, out, recs, _ = run([FACE_A], backend, [fakes.reference("A")])
        r = recs[0]
        self.assertEqual(r["decision"], "accepted")
        self.assertIsNone(r["rollback"])
        self.assertGreater(r["score_after"], r["score_before"] + 0.01)
        self.assertFalse(np.array_equal(out, img))

    def test_an_equal_score_is_not_an_improvement(self):
        backend = fakes.FakeBackend([det(FACE_A)], lambda crop: crop)
        img, out, recs, _ = run([FACE_A], backend, [fakes.reference("A")])
        self.assertEqual(recs[0]["decision"], "rejected")
        self.assertTrue(np.array_equal(out, img))

    def test_n03_a_gain_below_the_margin_is_rolled_back_and_one_above_is_kept(self):
        """The accept rule needs score_after - score_before >= min_delta (0.01), not just '>'."""
        def result_for(gain):
            backend = fakes.FakeBackend([det(FACE_A)], lambda crop: fakes.sharpen(crop, gain))
            _, _, recs, _ = run([FACE_A], backend, [fakes.reference("A")])
            return recs[0]

        lo, hi, tiny = 0.0, 3.0, None  # bisection for a gain whose delta is real but below the margin
        for _ in range(40):
            mid = (lo + hi) / 2
            r = result_for(mid)
            delta = r["score_after"] - r["score_before"]
            if 0.002 < delta < 0.008:
                tiny = r
                break
            lo, hi = (mid, hi) if delta <= 0.002 else (lo, mid)
        self.assertIsNotNone(tiny, "control: a sub-margin gain exists")
        self.assertGreater(tiny["score_after"], tiny["score_before"])  # the OLD rule ('>') would have accepted it
        self.assertEqual((tiny["decision"], tiny["rollback"]), ("rejected", "score_delta"))
        self.assertEqual(result_for(3.0)["decision"], "accepted")

    def test_n03_the_margin_is_the_setting(self):
        backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen)
        strict = fd.Settings(color_match=False, min_delta=0.5)
        _, _, recs, _ = run([FACE_A], backend, [fakes.reference("A")], settings=strict)
        self.assertEqual(recs[0]["decision"], "rejected")

    def test_no_face_found_after_the_redraw_is_a_rejection(self):
        backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen, after={"missing": True})
        img, out, recs, _ = run([FACE_A], backend, [fakes.reference("A")])
        self.assertEqual((recs[0]["decision"], recs[0]["rollback"]), ("rejected", "no_face_after"))
        self.assertIsNone(recs[0]["score_after"])
        self.assertTrue(np.array_equal(out, img))

    def test_one_redraw_per_face_never_best_of_n(self):
        backend = fakes.FakeBackend([det(FACE_A), det(FACE_C)], fakes.flatten)
        run([FACE_A, FACE_C], backend, [], settings=OPT_IN)
        self.assertEqual(backend.redraw_calls, 2)
        self.assertEqual(len(set(backend.redraw_seeds)), 2)


class AfterGatesTest(unittest.TestCase):
    """N03: the redrawn crop must pass the same gates, on the same face, or the original comes back."""

    def rejected(self, **after):
        backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen, after=after)  # the score WOULD rise: sharpen
        img, out, recs, _ = run([FACE_A], backend, [fakes.reference("A")])
        self.assertIs(out, img)
        self.assertEqual(recs[0]["decision"], "rejected")
        return recs[0]

    def test_a_face_turned_away_after_the_redraw_is_rolled_back_despite_the_higher_score(self):
        rec = self.rejected(yaw=0.9)
        self.assertEqual(rec["rollback"], "profile")
        self.assertGreater(rec["score_after"], rec["score_before"] + 0.01)  # the pre-fix rule alone would have accepted

    def test_an_unclear_face_after_the_redraw_is_rolled_back(self):
        rec = self.rejected(det_score=0.2)
        self.assertEqual(rec["rollback"], "low_detector_confidence")
        self.assertGreater(rec["score_after"], rec["score_before"] + 0.01)

    def test_missing_landmarks_after_the_redraw_are_rolled_back(self):
        self.assertEqual(self.rejected(no_kps=True)["rollback"], "no_landmarks_after")

    def test_a_face_that_moved_is_not_the_same_face(self):
        self.assertEqual(self.rejected(kps_dx=20.0)["rollback"], "face_moved")  # landmarks 20 px sideways, same box

    def test_the_score_belongs_to_the_face_that_was_found_not_the_largest_face(self):
        """After the redraw a LARGER face with a perfect match appears in the crop; the old 'largest face' reading would
        score it and accept. The matched face got flatter, so the redraw is rolled back."""
        backend = fakes.FakeBackend([det(FACE_A)], fakes.flatten, after={"extra_face": True})
        img, out, recs, _ = run([FACE_A], backend, [fakes.reference("A")])
        self.assertEqual((recs[0]["decision"], recs[0]["rollback"]), ("rejected", "score_delta"))
        self.assertTrue(np.array_equal(out, img))
        # the pre-fix behaviour, spelled out: the largest reading of the new crop would have won
        readings = backend.read_faces(fd.to_u8(np.full((108, 108, 3), 0.78, dtype=np.float32)))
        largest = max(readings, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
        self.assertGreater(fd.mean_similarity(largest.embedding, fakes.reference("A")), recs[0]["score_before"] + 0.01)

    def test_the_gates_also_guard_an_opted_in_run_without_references(self):
        backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen, after={"yaw": 0.9})
        img, out, recs, _ = run([FACE_A], backend, [], settings=OPT_IN)
        self.assertEqual((recs[0]["decision"], recs[0]["rollback"]), ("rejected", "profile"))
        self.assertIs(out, img)

    def test_a_clean_redraw_passes_every_gate(self):
        backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen, after={"yaw": 0.2, "kps_dx": 2.0})
        _, _, recs, _ = run([FACE_A], backend, [fakes.reference("A")])
        self.assertEqual(recs[0]["decision"], "accepted")


class PassThroughWithoutReferencesTest(unittest.TestCase):
    """N04: no references -> nothing is redrawn unless the user opted in; the input must be finite."""

    def test_the_default_is_pass_through(self):
        backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen)
        img, out, recs, issues = run([FACE_A], backend, [], settings=NO_COLOR)
        self.assertIs(out, img)
        self.assertEqual(backend.redraw_calls, 0)
        r = recs[0]
        self.assertEqual((r["decision"], r["unscorable_reason"]), ("skipped_unscorable", "no_references"))
        self.assertIn("accept_without_references", [i for i in issues if i.code == "unscorable_face"][0].fix)

    def test_opting_in_keeps_the_redraw_and_marks_it_unscorable(self):
        backend = fakes.FakeBackend([det(FACE_A)], fakes.flatten)  # even a worse face: nothing to compare with
        img, out, recs, issues = run([FACE_A], backend, [], settings=OPT_IN)
        r = recs[0]
        self.assertEqual(r["decision"], "accepted_unscorable")
        self.assertTrue(r["unscorable"])
        self.assertEqual(r["unscorable_reason"], "no_references")
        self.assertIsNone(r["score_before"])
        self.assertIsNone(r["score_after"])
        self.assertFalse(np.array_equal(out, img))
        self.assertIn("unscorable_face", [i.code for i in issues])

    def test_the_opt_in_is_off_by_default(self):
        self.assertFalse(fd.Settings().accept_without_references)

    def test_a_non_finite_input_is_refused_before_anything_runs(self):
        backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen)
        for bad in (np.nan, np.inf, -np.inf):
            img = fakes.make_scene(H, W, [FACE_A])
            img[10, 10, 1] = bad
            with self.subTest(value=bad):
                with self.assertRaises(errors.OutputError) as ctx:
                    fd.detail_image(img, OPT_IN, backend, [], seed=1)
                self.assertIn("NaN or Inf", str(ctx.exception))
        self.assertEqual(backend.redraw_calls, 0)


class SizeGateTest(unittest.TestCase):
    """(c) the size gate skips large faces and leaves them untouched."""

    def test_a_large_face_is_skipped_and_the_image_is_the_input(self):
        backend = fakes.FakeBackend([det(FACE_BIG)], fakes.sharpen)
        img, out, recs, issues = run([FACE_BIG], backend, [fakes.reference("A")])
        self.assertIs(out, img)
        self.assertEqual(backend.redraw_calls, 0)
        r = recs[0]
        self.assertEqual(r["decision"], "skipped_large")
        self.assertFalse(r["size_gate"]["passed"])
        self.assertEqual(r["size_gate"]["threshold_px"], 110.0)
        self.assertEqual(r["short_side_px"], 112.0)
        self.assertIn("face_large_skipped", [i.code for i in issues])
        self.assertEqual([i.severity for i in issues if i.code == "face_large_skipped"], ["info"])

    def test_the_gate_is_a_strict_less_than_on_the_short_side(self):
        for short, expect_run in ((109, True), (110, False), (111, False)):
            with self.subTest(short=short):
                face = ("A", 100, 60, 100 + short, 60 + short + 20)
                backend = fakes.FakeBackend([det(face)], fakes.sharpen)
                img = fakes.make_scene(H, W, [face])
                _, recs, _ = fd.detail_image(img, OPT_IN, backend, [], seed=1)
                self.assertEqual(backend.redraw_calls, 1 if expect_run else 0)
                self.assertEqual(recs[0]["size_gate"]["passed"], expect_run)

    def test_the_gate_is_a_setting(self):
        mid = ("A", 100, 50, 220, 180)  # 120 x 130 px: above the default gate of 110, below 200
        backend = fakes.FakeBackend([det(mid)], fakes.sharpen)
        settings = fd.Settings(size_gate_px=200.0, color_match=False, accept_without_references=True)
        img = fakes.make_scene(H, W, [mid])
        _, recs, _ = fd.detail_image(img, settings, backend, [], seed=1)
        self.assertEqual(backend.redraw_calls, 1)
        self.assertEqual(recs[0]["size_gate"]["threshold_px"], 200.0)

    def test_a_large_face_does_not_stop_a_small_one(self):
        backend = fakes.FakeBackend([det(FACE_BIG), det(FACE_A)], fakes.sharpen)
        img, out, recs, _ = run([FACE_BIG, FACE_A], backend, [fakes.reference("A")])
        self.assertEqual(sorted(r["decision"] for r in recs), ["accepted", "skipped_large"])
        self.assertEqual(backend.redraw_calls, 1)

    def test_a_tiny_face_is_unscorable_and_never_redrawn(self):
        tiny = ("A", 150, 100, 180, 130)  # 30 px
        backend = fakes.FakeBackend([det(tiny)], fakes.sharpen)
        img, out, recs, _ = run([tiny], backend, [fakes.reference("A")])
        self.assertIs(out, img)
        self.assertEqual(backend.redraw_calls, 0)
        self.assertEqual((recs[0]["decision"], recs[0]["unscorable_reason"]), ("skipped_unscorable", "too_small"))


class UnscorableFacesTest(unittest.TestCase):
    """Profile, occluded and not-a-face crops are reported unscorable and are not redrawn."""

    def check(self, backend, reason, characters):
        img, out, recs, issues = run([FACE_A], backend, characters, settings=OPT_IN)
        self.assertIs(out, img)
        self.assertEqual(backend.redraw_calls, 0)
        self.assertEqual((recs[0]["decision"], recs[0]["unscorable"], recs[0]["unscorable_reason"]),
                         ("skipped_unscorable", True, reason))
        self.assertIn("unscorable_face", [i.code for i in issues])

    def test_profile(self):
        for characters in ([fakes.reference("A")], []):
            self.check(fakes.FakeBackend([det(FACE_A)], fakes.sharpen, yaw=0.9), "profile", characters)

    def test_low_detector_confidence_means_occluded(self):
        self.check(fakes.FakeBackend([det(FACE_A)], fakes.sharpen, det_score=0.2), "low_detector_confidence",
                   [fakes.reference("A")])

    def test_no_face_in_the_crop(self):
        self.check(fakes.FakeBackend([det(FACE_A)], fakes.sharpen, no_face_for={"A"}), "no_face_in_crop", [])

    def test_no_landmarks(self):
        self.check(fakes.FakeBackend([det(FACE_A)], fakes.sharpen, before={"no_kps": True}), "no_landmarks", [])

    def test_a_face_that_is_not_where_the_detector_saw_it_is_not_scored(self):
        """A reading whose box does not overlap the detector's box is another face, not this one."""
        class Elsewhere(fakes.FakeBackend):
            def read_faces(self, rgb_u8):
                return [fd.FaceReading(fakes.embed("A", 0.25), 0.99, (0.0, 0.0, 10.0, 10.0), fakes.fake_kps((0, 0, 10, 10)))]
        self.check(Elsewhere([det(FACE_A)], fakes.sharpen), "no_face_in_crop", [fakes.reference("A")])

    def test_a_frontal_face_is_not_unscorable(self):
        backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen, yaw=0.1)
        _, _, recs, _ = run([FACE_A], backend, [fakes.reference("A")])
        self.assertEqual(recs[0]["decision"], "accepted")

    def test_a_rolled_frontal_face_is_not_a_profile(self):
        """N02 through the pipeline: the same frontal landmarks at 60 and 90 degrees of roll."""
        for roll in (0.0, 45.0, 60.0, 90.0):
            with self.subTest(roll=roll):
                backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen, before={"roll": roll}, after={"roll": roll})
                _, _, recs, _ = run([FACE_A], backend, [fakes.reference("A")])
                self.assertEqual(recs[0]["decision"], "accepted")

    def test_without_arcface_an_opted_in_run_is_unscored_and_default_passes_through(self):
        backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen, arcface_available=False)
        _, _, recs, _ = run([FACE_A], backend, [], settings=OPT_IN)
        self.assertEqual((recs[0]["decision"], recs[0]["unscorable_reason"]), ("accepted_unscorable", "arcface_unavailable"))
        backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen, arcface_available=False)
        _, out, recs, _ = run([FACE_A], backend, [], settings=NO_COLOR)
        self.assertEqual(recs[0]["decision"], "skipped_unscorable")
        self.assertEqual(backend.redraw_calls, 0)

    def test_references_without_arcface_are_an_error(self):
        backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen, arcface_available=False)
        with self.assertRaises(errors.MissingGuardDependency):
            run([FACE_A], backend, [fakes.reference("A")])


def greedy_legacy(scores, match_min):
    """The pre-fix binding (audit N01): highest pair first, each side used once."""
    pairs = sorted(((float(s), i, j) for i, row in enumerate(scores) for j, s in enumerate(row)), key=lambda t: -t[0])
    owner, used = [None] * len(scores), set()
    for s, i, j in pairs:
        if s < match_min:
            break
        if owner[i] is None and j not in used:
            owner[i] = j
            used.add(j)
    return owner


class GlobalBindingTest(unittest.TestCase):
    """N01: global assignment with a threshold and a margin; ambiguous faces stay untouched."""
    AUDIT = [[0.90, 0.89], [0.88, 0.10]]

    def test_the_greedy_rule_is_not_the_global_optimum_on_the_audit_matrix(self):
        self.assertEqual(greedy_legacy(self.AUDIT, 0.30), [0, None])          # the defect: sum 0.90, one face unmatched
        self.assertEqual(fd.best_assignment(self.AUDIT, 0.30), [1, 0])        # the optimum: sum 1.77, both matched

    def test_the_audit_matrix_is_ambiguous_so_nobody_is_bound(self):
        owners, status = fd.assign_faces(self.AUDIT, 0.30, 0.05)
        self.assertEqual(owners, [None, None])
        self.assertEqual(status, ["ambiguous", "ambiguous"])

    def test_a_clear_matrix_is_bound(self):
        owners, status = fd.assign_faces([[0.90, 0.30], [0.20, 0.80]], 0.30, 0.05)
        self.assertEqual((owners, status), ([0, 1], ["bound", "bound"]))

    def test_below_the_threshold_nobody_is_bound(self):
        self.assertEqual(fd.best_assignment([[0.2, 0.1]], 0.30), [None])
        self.assertEqual(fd.assign_faces([[0.2, 0.1]], 0.30, 0.05), ([None], ["unmatched"]))

    def test_a_rival_face_within_the_margin_makes_the_binding_ambiguous(self):
        # one character, two faces that look about equally like it
        self.assertEqual(fd.assign_faces([[0.62], [0.60]], 0.30, 0.05), ([None, None], ["ambiguous", "unmatched"]))
        self.assertEqual(fd.assign_faces([[0.80], [0.60]], 0.30, 0.05), ([0, None], ["bound", "unmatched"]))

    def test_empty_inputs(self):
        self.assertEqual(fd.best_assignment([], 0.3), [])
        self.assertEqual(fd.assign_faces([], 0.3, 0.05), ([], []))

    def test_too_many_characters_are_refused(self):
        with self.assertRaises(ValueError):
            fd.best_assignment([[0.5] * (fd.MAX_CHARACTERS + 1)], 0.3)

    def test_the_optimum_is_the_best_sum_found_by_brute_force(self):
        rng = np.random.default_rng(0)
        for _ in range(60):
            n_f, n_c = int(rng.integers(1, 5)), int(rng.integers(1, 4))
            scores = rng.random((n_f, n_c)).round(3).tolist()

            def total(owner):
                return sum(scores[i][c] for i, c in enumerate(owner) if c is not None)

            best = 0.0
            for owner in itertools.product(list(range(n_c)) + [None], repeat=n_f):
                used = [c for c in owner if c is not None]
                if len(used) != len(set(used)) or any(c is not None and scores[i][c] < 0.3 for i, c in enumerate(owner)):
                    continue
                best = max(best, total(owner))
            got = fd.best_assignment(scores, 0.3)
            self.assertAlmostEqual(total(got), best, places=9, msg=str(scores))
            self.assertGreaterEqual(total(got), total(greedy_legacy(scores, 0.3)) - 1e-9)

    def test_the_pipeline_leaves_look_alike_faces_untouched(self):
        """Two faces of look-alike people: both score about equally on the primary -> ambiguous -> no redraw."""
        class LookAlike(fakes.FakeBackend):
            def read_faces(self, rgb_u8):
                reading = super().read_faces(rgb_u8)[0]
                tone = float(fakes.center_patch(rgb_u8.astype(np.float32) / 255.0).mean())
                emb = np.array([1.0, 0.0, 0.75 if tone > 0.5 else 0.7], dtype=np.float32)
                return [fd.FaceReading(emb / np.linalg.norm(emb), reading.det_score, reading.bbox, reading.kps)]

        a, b = ("A", 130, 90, 180, 150), ("B", 30, 150, 80, 205)
        backend = LookAlike([det(a), det(b)], fakes.sharpen)
        ref = np.array([1.0, 0.0, 0.75], dtype=np.float32)
        img, out, recs, _ = run([a, b], backend, [np.stack([ref / np.linalg.norm(ref)] * 3)])
        # one character, two faces about equally like it: the winner is ambiguous, the loser is nobody's
        self.assertEqual(sorted(r["decision"] for r in recs), ["skipped_ambiguous", "skipped_other_character"])
        self.assertEqual(backend.redraw_calls, 0)
        self.assertIs(out, img)


class MultiFaceTest(unittest.TestCase):
    def test_each_face_is_bound_to_its_character_and_only_the_primary_is_redrawn(self):
        backend = fakes.FakeBackend([det(FACE_A), det(FACE_C)], fakes.sharpen)
        img, out, recs, issues = run([FACE_A, FACE_C], backend, [fakes.reference("A"), fakes.reference("B")])
        primary = [r for r in recs if r["decision"] == "accepted"]
        other = [r for r in recs if r["decision"] == "skipped_other_character"]
        self.assertEqual((len(primary), len(other)), (1, 1))
        self.assertEqual((primary[0]["character"], other[0]["character"]), ("primary", "other_1"))
        self.assertEqual(backend.redraw_calls, 1)
        full = fakes.full_mask_of((H, W), primary[0])
        self.assertTrue(np.array_equal(out[full <= 0.0], img[full <= 0.0]))
        self.assertIn("face_not_this_character", [i.code for i in issues])

    def test_a_face_that_matches_nobody_stays_untouched(self):
        backend = fakes.FakeBackend([det(FACE_A), det(FACE_C)], fakes.sharpen)
        img, out, recs, _ = run([FACE_A, FACE_C], backend, [fakes.reference("A")])
        self.assertEqual({r["decision"] for r in recs}, {"accepted", "skipped_other_character"})
        stranger = [r for r in recs if r["decision"] == "skipped_other_character"][0]
        self.assertIsNone(stranger["character"])
        self.assertEqual(backend.redraw_calls, 1)

    def test_with_the_opt_in_every_small_face_is_processed(self):
        backend = fakes.FakeBackend([det(FACE_A), det(FACE_C)], fakes.sharpen)
        _, _, recs, _ = run([FACE_A, FACE_C], backend, [], settings=OPT_IN)
        self.assertEqual([r["decision"] for r in recs], ["accepted_unscorable"] * 2)
        self.assertEqual(backend.redraw_calls, 2)

    def test_max_faces_keeps_the_largest(self):
        backend = fakes.FakeBackend([det(FACE_C), det(FACE_A)], fakes.sharpen)
        settings = fd.Settings(color_match=False, max_faces=1, accept_without_references=True)
        img = fakes.make_scene(H, W, [FACE_A, FACE_C])
        _, recs, _ = fd.detail_image(img, settings, backend, [], seed=1)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["bbox"], [130.0, 90.0, 180.0, 150.0])  # 50x60 beats 50x55

    def test_the_faces_get_different_seeds_and_the_seed_is_reproducible(self):
        def seeds(seed, image_index=0):
            backend = fakes.FakeBackend([det(FACE_A), det(FACE_C)], fakes.sharpen)
            img = fakes.make_scene(H, W, [FACE_A, FACE_C])
            fd.detail_image(img, OPT_IN, backend, [], seed=seed, image_index=image_index)
            return backend.redraw_seeds
        self.assertEqual(seeds(11), seeds(11))
        self.assertNotEqual(seeds(11)[0], seeds(11)[1])
        self.assertNotEqual(seeds(11), seeds(12))
        self.assertNotEqual(seeds(11, 0), seeds(11, 1))

    def test_no_face_is_an_info_and_the_input(self):
        backend = fakes.FakeBackend([])
        img, out, recs, issues = run([], backend, [], scene_faces=[FACE_A])
        self.assertIs(out, img)
        self.assertEqual(recs, [])
        self.assertEqual([i.code for i in issues], ["no_face_found"])

    def test_a_low_confidence_detection_is_ignored(self):
        backend = fakes.FakeBackend([det(FACE_A, conf=0.3)], fakes.sharpen)
        _, out, recs, _ = run([FACE_A], backend, [])
        self.assertEqual(recs, [])


class BadInputsTest(unittest.TestCase):
    def test_a_wrong_dtype_or_shape_is_an_output_error(self):
        backend = fakes.FakeBackend([])
        for bad in (np.zeros((8, 8, 3), dtype=np.float64), np.zeros((8, 8), dtype=np.float32),
                    np.zeros((8, 8, 2), dtype=np.float32), "nope"):
            with self.subTest(type=type(bad).__name__):
                with self.assertRaises(errors.OutputError):
                    fd.detail_image(bad, NO_COLOR, backend, [], seed=1)

    def test_a_redraw_of_the_wrong_size_or_with_nan_is_refused(self):
        for fn in (lambda crop: crop[:-1], lambda crop: crop * np.float32("nan")):
            backend = fakes.FakeBackend([det(FACE_A)], fn)
            with self.assertRaises(errors.OutputError):
                run([FACE_A], backend, [], settings=OPT_IN)

    def test_an_alpha_channel_is_kept(self):
        img = np.concatenate([fakes.make_scene(H, W, [FACE_A]), np.full((H, W, 1), 0.25, dtype=np.float32)], axis=-1)
        backend = fakes.FakeBackend([det(FACE_A)], fakes.sharpen)
        out, recs, _ = fd.detail_image(img, OPT_IN, backend, [], seed=1)
        self.assertEqual(recs[0]["decision"], "accepted_unscorable")
        self.assertTrue(np.array_equal(out[..., 3], img[..., 3]))


if __name__ == "__main__":
    unittest.main()
