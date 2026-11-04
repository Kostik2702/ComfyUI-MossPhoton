"""MP-57 / MP-71: the darkroom block of a preset, the override names and the validation of settings (guarantee e: every invalid
phase order or bound is refused as a typed error by pure code, before any forward, clone or tensor exists).

The values live in ``presets/turbo_darkroom_*.json`` and nowhere else (``core/darkroom.RECIPES`` is gone). The unit suite runs
without comfy and torch (run_tests fails it when comfy was imported), so the refusal needs neither.
"""
import math
import unittest

from moss_photon.core import darkroom, errors, presets

from .support import DARKROOM_IDS, FULLBODY, OFF, PORTRAIT, preset_of, settings_of


def settings(recipe=PORTRAIT, **overrides):
    return settings_of(recipe, overrides)


class PresetBlockTest(unittest.TestCase):
    def test_the_values_have_one_source_the_preset_file(self):
        for gone in ("RECIPES", "RECIPE_IDS", "recipe_ref", "PRESET"):
            self.assertFalse(hasattr(darkroom, gone), gone)
        self.assertEqual([pid for pid, p in presets.catalog().items() if "darkroom" in p], list(DARKROOM_IDS))

    def test_the_two_presets_are_versioned_and_the_ref_is_id_at_version(self):
        for pid in DARKROOM_IDS:
            with self.subTest(preset=pid):
                preset = preset_of(pid)
                self.assertEqual(preset["preset_version"], "1.0.0")
                self.assertEqual(settings(pid).ref, "%s@%s" % (pid, preset["preset_version"]))

    def test_the_turbo_presets_are_the_values_of_the_mp57_recipes_except_the_calibrated_one(self):
        # portrait: w_commit 1.35 by the MP-72 calibration (research/darkroom_calib/table.md); fullbody is still the MP-57 value
        for pid, w_commit in ((PORTRAIT, 1.35), (FULLBODY, 1.2)):
            with self.subTest(preset=pid):
                st = settings(pid)
                self.assertEqual(preset_of(pid)["sampling"]["steps"], 12)
                self.assertEqual(st.schedule, "native")
                self.assertEqual((st.compose_cfg, st.compose_slg), (1.0, False))
                self.assertEqual((st.r_c, st.r_k), (0.06, 0.17))
                self.assertEqual((st.w_compose, st.w_commit), (1.0, w_commit))
                self.assertFalse(st.reexpose)
                self.assertEqual(st.sigma_r, 0.8)
                self.assertEqual((st.r_f, st.finish_cfg, st.eta, st.switch), (0.17, 2.0, 0.0, False))
                self.assertEqual(st.overrides, ())
                self.assertEqual(st.finish_cfg_source, "preset")

    def test_the_block_of_the_preset_is_the_block_the_settings_report(self):
        for pid in DARKROOM_IDS:
            block = preset_of(pid)["darkroom"]
            self.assertEqual(settings(pid).to_dict()["commit"], block["commit"])
            self.assertEqual(settings(pid).to_dict()["finish"], block["finish"])
            self.assertEqual(settings(pid).to_dict()["reexpose"], block["reexpose"])
            self.assertEqual(block["compose"]["cfg"], preset_of(pid)["sampling"]["cfg"])

    def test_a_neutral_block_changes_nothing(self):
        st = settings(OFF)
        self.assertEqual(st.finish_cfg, st.compose_cfg)
        self.assertEqual(st.w_compose, st.w_commit)
        self.assertEqual((st.compose_slg, st.reexpose, st.eta, st.switch), (False, False, 0.0, False))

    def test_the_defaults_of_the_optional_features_are_off(self):
        for pid in DARKROOM_IDS + (OFF,):
            st = settings(pid)
            self.assertEqual((st.reexpose, st.eta, st.switch), (False, 0.0, False), pid)

    def test_a_preset_without_a_block_is_refused_with_the_node_named(self):
        with self.assertRaises(errors.InvalidOverride) as cm:
            darkroom.resolve_settings(presets.catalog()["turbo_identity"], node="MossPhotonSamplerAdvanced")
        self.assertEqual((cm.exception.issue.param, cm.exception.issue.node), ("preset", "MossPhotonSamplerAdvanced"))

    def test_the_seed_offsets_are_constants_and_wrap_like_the_seed_widget(self):
        self.assertEqual(darkroom.reexpose_seed(42), 42 + darkroom.REEXPOSE_SEED_OFFSET)
        self.assertEqual(darkroom.noise_seed(42), 42 + darkroom.NOISE_SEED_OFFSET)
        self.assertNotEqual(darkroom.reexpose_seed(42), darkroom.noise_seed(42))
        top = 0xFFFFFFFFFFFFFFFF
        self.assertEqual(darkroom.reexpose_seed(top), darkroom.REEXPOSE_SEED_OFFSET - 1)

    def test_the_sa2_schedule_is_the_one_of_mp53(self):
        self.assertEqual(darkroom.sa2_values(), [1.0, 0.96, 0.92, 0.88, 0.83, 0.76, 0.625, 0.4166666567325592, 0.0])


class OverridesTest(unittest.TestCase):
    def test_the_override_names_are_the_phase_knobs_without_steps(self):
        self.assertEqual(darkroom.OVERRIDE_NAMES, ("schedule", "compose_cfg", "compose_slg", "r_c", "r_k", "w_compose",
                                                   "w_commit", "reexpose", "sigma_r", "r_f", "finish_cfg", "eta",
                                                   "switch"))

    def test_none_everywhere_is_the_preset(self):
        none = {name: None for name in darkroom.OVERRIDE_NAMES}
        for pid in DARKROOM_IDS + (OFF,):
            self.assertEqual(darkroom.resolve_settings(preset_of(pid), none), settings(pid))

    def test_a_real_value_is_an_override_and_is_named(self):
        for name, value in (("compose_cfg", 1.5), ("r_c", 0.0), ("eta", 0.0), ("w_commit", 0.0), ("finish_cfg", 3.0),
                            ("schedule", "native"), ("switch", False)):
            with self.subTest(name=name):
                self.assertIn(name, settings(**{name: value}).overrides)

    def test_an_override_equal_to_the_preset_is_still_recorded(self):
        self.assertEqual(settings(finish_cfg=2.0).overrides, ("finish_cfg",))
        self.assertEqual(settings(finish_cfg=2.0).finish_cfg, 2.0)

    def test_none_values_are_dropped_not_zero(self):
        self.assertEqual(darkroom.resolve_settings(preset_of(PORTRAIT), {"eta": None, "r_c": None}), settings())

    def test_the_overrides_never_touch_the_preset_dict(self):
        preset = preset_of(PORTRAIT)
        before = repr(preset)
        darkroom.resolve_settings(preset, {"finish_cfg": 3.0, "r_f": 0.3})
        self.assertEqual(repr(preset), before)


class RefusalTest(unittest.TestCase):
    """A value of the wrong TYPE is refused (invalid_override) by pure code; only an API caller can pass one (a value of the
    right type outside its bounds degrades: ``DegradeTest``)."""

    def refused(self, recipe=PORTRAIT, **overrides):
        with self.assertRaises(errors.InvalidOverride) as cm:
            settings_of(recipe, overrides, node="MossPhotonSampler")
        self.assertEqual(cm.exception.code, "invalid_override")
        self.assertEqual(cm.exception.issue.node, "MossPhotonSampler")
        return cm.exception

    def test_a_number_must_be_a_finite_number(self):
        for name in ("r_c", "r_k", "r_f", "w_commit", "w_compose", "compose_cfg", "finish_cfg", "eta", "sigma_r"):
            for bad in (math.nan, math.inf, -math.inf, True, False, "0.1", []):
                with self.subTest(name=name, bad=bad):
                    self.refused(**{name: bad})

    def test_flags_must_be_booleans(self):
        for name in ("compose_slg", "reexpose", "switch"):
            for bad in (1, 0, "on", 2.0):
                with self.subTest(name=name, bad=bad):
                    self.refused(**{name: bad})

    def test_schedule_and_unknown_names(self):
        self.refused(schedule="linear")
        self.refused(sigma_rr=0.5)
        self.refused(steps_override=12)

    def test_steps_is_not_a_darkroom_override(self):
        """The steps of a run are ``sampling.steps`` of the preset (or its override), never a phase knob."""
        self.refused(**{"steps": 12})


class DegradeTest(unittest.TestCase):
    """MP-84 (owner 2026-11-02): a setting conflict degrades to the nearest usable value, with a note per change."""

    def noted(self, **overrides):
        st = settings(**overrides)
        return st, {n[0] for n in st.notes}

    def test_phase_order_is_sorted(self):
        for name, ov, expected in (("r_c above r_k", dict(r_c=0.5, r_k=0.3, r_f=0.6), (0.3, 0.5, 0.6)),
                                   ("r_k above r_f", dict(r_k=0.3, r_f=0.2), (0.06, 0.2, 0.3)),
                                   ("r_c above r_f", dict(r_c=0.9, r_k=0.9, r_f=0.2), (0.2, 0.9, 0.9)),
                                   ("finish before commit end", dict(r_f=0.1), (0.06, 0.1, 0.17)),
                                   ("compose past the preset's r_k", dict(r_c=0.18), (0.17, 0.17, 0.18))):
            with self.subTest(name):
                st, params = self.noted(**ov)
                self.assertEqual((st.r_c, st.r_k, st.r_f), expected)
                self.assertTrue(st.r_c <= st.r_k <= st.r_f)
                self.assertIn("r_c", params)
                self.assertTrue(any("out of order" in n[1] for n in st.notes))

    def test_a_fraction_outside_zero_to_one_is_moved_to_the_bound(self):
        for name, bad, used in (("r_c", -0.01, 0.0), ("r_c", 1.01, 1.0), ("r_f", 1.01, 1.0), ("r_k", -3.0, 0.0)):
            with self.subTest(name=name, bad=bad):
                st, params = self.noted(**{name: bad})
                self.assertIn(name, params)
                self.assertTrue(0.0 <= st.r_c <= st.r_k <= st.r_f <= 1.0)

    def test_weights_are_moved_into_range_and_an_empty_commit_window_has_no_boost(self):
        st, params = self.noted(w_commit=10.1)
        self.assertEqual((st.w_commit, "w_commit" in params), (10.0, True))
        st, params = self.noted(w_compose=-0.1, w_commit=0.0)
        self.assertEqual((st.w_compose, st.w_commit), (0.0, 0.0))
        st, params = self.noted(r_c=0.1, r_k=0.1, r_f=0.1)
        self.assertEqual(st.w_commit, st.w_compose, "no window, no boost")
        self.assertIn("r_k", params)
        self.assertEqual(settings(r_c=0.1, r_k=0.1, r_f=0.1, w_commit=1.0).notes, ())  # equal weights need no window

    def test_guidance_noise_and_reexpose_values_are_moved_to_their_bounds(self):
        for name, bad, used in (("compose_cfg", 0.5, 1.0), ("compose_cfg", 31.0, 30.0), ("finish_cfg", 0.0, 1.0),
                                ("finish_cfg", 0.99, 1.0), ("eta", -0.1, 0.0), ("eta", 1.1, 1.0),
                                ("sigma_r", 0.0, 0.01), ("sigma_r", 0.009, 0.01), ("sigma_r", 1.1, 1.0)):
            with self.subTest(name=name, bad=bad):
                st, params = self.noted(**{name: bad})
                self.assertEqual(getattr(st, name), used)
                self.assertIn(name, params)

    def test_a_clean_override_has_no_note(self):
        for ov in (dict(), dict(finish_cfg=2.0), dict(r_f=0.3), dict(eta=0.2), dict(compose_cfg=1.5)):
            with self.subTest(ov=ov):
                self.assertEqual(settings(**ov).notes, ())

    def test_the_sa2_schedule_needs_no_step_count(self):
        """The knots are resampled for another count by the solver (``darkroom.sa2_values``); the settings accept it."""
        self.assertEqual(settings(schedule="sa2_8").schedule, "sa2_8")


if __name__ == "__main__":
    unittest.main()
