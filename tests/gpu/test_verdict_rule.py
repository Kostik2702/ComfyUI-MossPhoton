"""The OD3 verdict rule (design 6.7 step 6) on plain numbers. No GPU, no torch, no comfy: always runs.

The rule is proven by planting each violation and expecting red (CLAUDE.md: an invariant is proven by trying to
break it), and by checking that a skipped or ``None`` check can never read as a pass.
"""
import math
import unittest

from . import _env

verdict = None


def setUpModule():
    global verdict
    _env.require_parity_dir()
    import verdict as module
    verdict = module


def good(**over):
    """All checks green in deterministic mode."""
    kw = dict(deterministic=True, a_sigmas=True, b_forwards=True, c_noise=True, f_png=True, g_label=True,
              latent_equal=True, image_equal=True, latent_pair_max=0.0, image_pair_max_levels=0, F=0.0, F_img=0.0)
    kw.update(over)
    return kw


def tolerant(**over):
    kw = good(deterministic=False, latent_equal=False, image_equal=False, latent_pair_max=0.5, image_pair_max_levels=2,
              F=0.25, F_img=0.5)
    kw.update(over)
    return kw


class FTest(unittest.TestCase):
    def test_f_is_the_largest_pair_difference(self):
        self.assertEqual(verdict.f_of([0.0, 0.25, 0.1]), 0.25)

    def test_no_pairs_is_zero(self):
        self.assertEqual(verdict.f_of([]), 0.0)

    def test_nan_is_not_swallowed(self):
        self.assertTrue(math.isnan(verdict.f_of([0.1, float("nan")])))

    def test_negative_or_non_number_is_rejected(self):
        for bad in ([-1.0], ["x"], [None]):
            with self.assertRaises(ValueError):
                verdict.f_of(bad)


class DeterministicBackendTest(unittest.TestCase):
    PROBES = {"C01": 0.0, "C09": 0.0, "C19": 0.0}

    def test_all_zero_is_deterministic(self):
        self.assertTrue(verdict.backend_deterministic(self.PROBES))

    def test_any_positive_f_is_not_deterministic(self):
        for case in verdict.PROBE_CASES:
            probes = dict(self.PROBES)
            probes[case] = 1e-9
            self.assertFalse(verdict.backend_deterministic(probes), case)

    def test_nan_f_is_not_deterministic(self):
        self.assertFalse(verdict.backend_deterministic(dict(self.PROBES, C09=float("nan"))))

    def test_a_missing_probe_has_no_verdict(self):
        for case in verdict.PROBE_CASES:
            probes = {k: v for k, v in self.PROBES.items() if k != case}
            with self.assertRaises(ValueError):
                verdict.backend_deterministic(probes)

    def test_probe_cases_are_the_designed_three(self):
        self.assertEqual(verdict.PROBE_CASES, ("C01", "C09", "C19"))


class ExactModeTest(unittest.TestCase):
    def test_all_green_passes(self):
        res = verdict.decide_case(**good())
        self.assertTrue(res["pass"])
        self.assertEqual((res["mode"], res["failed"]), ("exact", []))

    def test_each_single_violation_fails_and_is_named(self):
        plants = {
            "a_sigmas": dict(a_sigmas=False), "b_forwards": dict(b_forwards=False), "c_noise": dict(c_noise=False),
            "d_latent": dict(latent_equal=False), "e_image": dict(image_equal=False), "f_png": dict(f_png=False),
            "g_label": dict(g_label=False),
        }
        for name, over in plants.items():
            res = verdict.decide_case(**good(**over))
            self.assertFalse(res["pass"], name)
            self.assertEqual(res["failed"], [name])

    def test_a_tiny_difference_is_not_tolerated_in_exact_mode(self):
        # equal flag False with a difference far below any tolerance: exact mode has no tolerance
        res = verdict.decide_case(**good(latent_equal=False, latent_pair_max=1e-30, F=1.0))
        self.assertEqual(res["failed"], ["d_latent"])

    def test_none_never_reads_as_a_pass(self):
        for key in ("a_sigmas", "b_forwards", "c_noise", "f_png", "g_label", "latent_equal", "image_equal"):
            res = verdict.decide_case(**good(**{key: None}))
            self.assertFalse(res["pass"], key)

    def test_failures_are_listed_in_artefact_order(self):
        res = verdict.decide_case(**good(g_label=False, a_sigmas=False, image_equal=False))
        self.assertEqual(res["failed"], ["a_sigmas", "e_image", "g_label"])


class ToleranceModeTest(unittest.TestCase):
    def test_within_the_bounds_passes(self):
        res = verdict.decide_case(**tolerant(latent_pair_max=0.5, F=0.25, image_pair_max_levels=2, F_img=0.5))
        self.assertTrue(res["pass"], res)
        self.assertEqual(res["mode"], "tolerance")
        self.assertEqual(res["bounds"], {"latent": 0.5, "image_levels": 2.0})

    def test_just_over_the_latent_bound_fails(self):
        res = verdict.decide_case(**tolerant(latent_pair_max=0.5000001, F=0.25))
        self.assertEqual(res["failed"], ["d_latent"])

    def test_just_over_the_image_bound_fails(self):
        res = verdict.decide_case(**tolerant(image_pair_max_levels=3, F_img=0.5))
        self.assertEqual(res["failed"], ["e_image"])

    def test_zero_f_means_exact_latent_but_one_level_of_image(self):
        ok = verdict.decide_case(**tolerant(latent_pair_max=0.0, F=0.0, image_pair_max_levels=1, F_img=0.0))
        self.assertTrue(ok["pass"])
        bad = verdict.decide_case(**tolerant(latent_pair_max=1e-12, F=0.0, image_pair_max_levels=1, F_img=0.0))
        self.assertEqual(bad["failed"], ["d_latent"])
        bad = verdict.decide_case(**tolerant(latent_pair_max=0.0, F=0.0, image_pair_max_levels=2, F_img=0.0))
        self.assertEqual(bad["failed"], ["e_image"])

    def test_sigmas_forwards_noise_png_label_stay_exact_in_tolerance_mode(self):
        for key in ("a_sigmas", "b_forwards", "c_noise", "f_png", "g_label"):
            res = verdict.decide_case(**tolerant(**{key: False}))
            self.assertFalse(res["pass"], key)
            self.assertEqual(len(res["failed"]), 1)

    def test_nan_difference_fails(self):
        res = verdict.decide_case(**tolerant(latent_pair_max=float("nan")))
        self.assertIn("d_latent", res["failed"])
        res = verdict.decide_case(**tolerant(F=float("nan")))
        self.assertIn("d_latent", res["failed"])

    def test_equal_flags_are_ignored_in_tolerance_mode(self):
        res = verdict.decide_case(**tolerant(latent_equal=False, image_equal=False))
        self.assertTrue(res["pass"])


class RepeatsTest(unittest.TestCase):
    def test_deterministic_repeats_must_be_identical(self):
        self.assertTrue(verdict.repeats_ok(deterministic=True, repeat_pair_max=0.0, F=0.0))
        self.assertFalse(verdict.repeats_ok(deterministic=True, repeat_pair_max=1e-12, F=1.0))

    def test_tolerance_repeats_are_within_two_f(self):
        self.assertTrue(verdict.repeats_ok(deterministic=False, repeat_pair_max=0.2, F=0.1))
        self.assertFalse(verdict.repeats_ok(deterministic=False, repeat_pair_max=0.2000001, F=0.1))
        self.assertFalse(verdict.repeats_ok(deterministic=False, repeat_pair_max=float("nan"), F=0.1))


if __name__ == "__main__":
    unittest.main()
