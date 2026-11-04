"""MP-71: a darkroom request on a preset WITHOUT a darkroom block degrades to warnings, never to an exception (pure part;
the full run is ``tests/integration/test_darkroom_cpu.py``)."""
import unittest

from moss_photon.core import darkroom, errors, presets, solver


class IgnoredIssuesTest(unittest.TestCase):
    PRESET = {"id": "turbo_identity"}

    def issues(self, **kw):
        return solver._darkroom_ignored_issues(darkroom.DarkroomRequest(**kw), self.PRESET, "MossPhotonSamplerAdvanced")

    def test_nothing_set_nothing_said(self):
        self.assertEqual(self.issues(), [])
        self.assertEqual(self.issues(overrides={"r_f": None, "eta": None}), [])

    def test_every_knob_set_is_named_once_and_no_value_is_written(self):
        out = self.issues(overrides={"r_f": 0.3, "finish_cfg": 2.5, "eta": None, "switch": True})
        self.assertEqual([i.code for i in out], ["darkroom_override_ignored"])
        issue = out[0]
        self.assertEqual((issue.severity, issue.node), ("warning", "MossPhotonSamplerAdvanced"))
        self.assertEqual(issue.detail, {"fields": ["finish_cfg", "r_f", "switch"], "preset": "turbo_identity"})
        for value in ("0.3", "2.5", "True"):
            self.assertNotIn(value, issue.message)
        self.assertIn("finish_cfg, r_f, switch", issue.message)
        self.assertNotIn("eta", issue.detail["fields"])

    def test_the_finish_model_gets_its_own_existing_code(self):
        out = self.issues(finish_model=object())
        self.assertEqual([i.code for i in out], ["finish_model_ignored"])
        self.assertEqual(out[0].detail, {"preset": "turbo_identity"})

    def test_a_cfg_widget_without_a_block_is_named_as_cfg(self):
        out = self.issues(cfg_widget=2.0)
        self.assertEqual(out[0].detail["fields"], ["cfg"])

    def test_both_together(self):
        out = self.issues(overrides={"r_c": 0.1}, finish_model=object())
        self.assertEqual([i.code for i in out], ["darkroom_override_ignored", "finish_model_ignored"])

    def test_none_of_it_can_be_an_error(self):
        for i in self.issues(overrides={"r_c": 0.1}, finish_model=object(), cfg_widget=2.0):
            self.assertNotEqual(i.severity, "error")
            with self.assertRaises(ValueError):
                errors.make_issue(i.code, "n", "p", "m", "f", severity="error")

    def test_the_fix_names_the_presets_that_do_have_a_block(self):
        fix = self.issues(overrides={"r_c": 0.1})[0].fix
        for pid, p in presets.catalog().items():
            self.assertEqual(pid in fix, "darkroom" in p, pid)


if __name__ == "__main__":
    unittest.main()
