"""D3: ``guidance.apply_slg`` is bit-equal to the native SkipLayerGuidanceDiTSimple, with the same per-call log
(batch size, context length, skipped blocks), for three sigma windows; the incoming MODEL is never written to."""
import unittest

from . import _comfy_env, tiny_moss
from . import sampler_support as ss

WINDOWS = ((0.0, 1.0), (0.0, 0.5), (0.25, 0.75))


def logged(model, log):
    """A clone of ``model`` that appends (batch, context tokens, sorted skipped blocks) per forward to ``log``."""
    import comfy.patcher_extension as pe
    clone = model.clone()

    def watcher(executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kw):
        replaced = (transformer_options.get("patches_replace") or {}).get("dit") or {}
        log.append((int(x.shape[0]), int(c_crossattn.shape[1]), tuple(sorted(replaced))))
        return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kw)

    pe.add_wrapper_with_key(pe.WrappersMP.APPLY_MODEL, "test_log", watcher, clone.model_options, is_model_options=True)
    return clone


class SlgEqualsNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        from moss_photon.core import guidance
        cls.guidance = guidance

    def test_windows(self):
        for start, end in WINDOWS:
            with self.subTest(window=(start, end)):
                native_log, ours_log = [], []
                native = logged(ss.native_slg(self.tiny.patcher, 22, start, end), native_log)
                ours = logged(self.guidance.apply_slg(self.tiny.patcher, layers=[22], start=start, end=end), ours_log)
                a = tiny_moss.sample_euler(self.tiny, native)
                b = tiny_moss.sample_euler(self.tiny, ours)
                self.assertTrue(a.equal(b), "latent differs from native for window %s" % ((start, end),))
                self.assertEqual(ours_log, native_log)
                self.assertGreater(len(native_log), 0)
                if (start, end) == (0.0, 1.0):
                    self.assertEqual(sum(1 for entry in ours_log if entry[2]), tiny_moss.STEPS)

    def test_a_partial_window_really_limits_the_skip(self):
        log = []
        tiny_moss.sample_euler(self.tiny, logged(
            self.guidance.apply_slg(self.tiny.patcher, layers=[22], start=0.25, end=0.75), log))
        skipped = sum(1 for entry in log if entry[2])
        self.assertTrue(0 < skipped < tiny_moss.STEPS, skipped)

    def test_multiple_layers_equal_native(self):
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
        native = SkipLayerGuidanceDiTSimple.execute(self.tiny.patcher, 0.0, 1.0, double_layers="",
                                                    single_layers="5, 22").args[0]
        ours = self.guidance.apply_slg(self.tiny.patcher, layers="5, 22", start=0.0, end=1.0)
        self.assertTrue(tiny_moss.sample_euler(self.tiny, native).equal(tiny_moss.sample_euler(self.tiny, ours)))

    def test_the_incoming_model_is_untouched(self):
        before = dict(self.tiny.patcher.model_options)
        ours = self.guidance.apply_slg(self.tiny.patcher, layers=[22], start=0.0, end=1.0)
        self.assertIsNot(ours, self.tiny.patcher)
        self.assertNotIn("sampler_calc_cond_batch_function", self.tiny.patcher.model_options)
        self.assertEqual(self.tiny.patcher.model_options, before)
        self.assertIn("sampler_calc_cond_batch_function", ours.model_options)

    def test_no_layers_returns_the_input_object(self):
        self.assertIs(self.guidance.apply_slg(self.tiny.patcher, layers=[], start=0.0, end=1.0), self.tiny.patcher)

    def test_at_cfg_1_the_closure_falls_back_to_the_stock_call(self):
        ours = self.guidance.apply_slg(self.tiny.patcher, layers=[22], start=0.0, end=1.0)
        plain = tiny_moss.sample_euler(self.tiny, self.tiny.patcher, cfg=1.0)
        self.assertTrue(plain.equal(tiny_moss.sample_euler(self.tiny, ours, cfg=1.0)))


if __name__ == "__main__":
    unittest.main()
