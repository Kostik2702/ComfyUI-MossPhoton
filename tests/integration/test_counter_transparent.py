"""A5/C2: the APPLY_MODEL counter wrapper is bit-transparent, counts calls / sample forwards / SLG skip calls, and lives
only on the clone."""
import unittest

from . import _comfy_env, tiny_moss
from . import sampler_support as ss


class CounterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        from moss_photon.core import guidance, solver
        cls.guidance, cls.solver = guidance, solver

    def counted(self, model):
        clone = model.clone()
        stats = self.solver.ForwardStats()
        self.solver.install_counter(clone, stats)
        return clone, stats

    def test_bit_equal_with_and_without_the_wrapper_and_stats_12_12_6(self):
        slg = self.guidance.apply_slg(self.tiny.patcher, layers=[22], start=0.0, end=1.0)
        counted, stats = self.counted(slg)
        plain = tiny_moss.sample_euler(self.tiny, slg)
        with_counter = tiny_moss.sample_euler(self.tiny, counted)
        self.assertTrue(plain.equal(with_counter))
        self.assertEqual((stats.calls, stats.elements, stats.skip_calls), (12, 12, 6))

    def test_without_slg_equal_lengths_are_one_batched_call_per_step(self):
        counted, stats = self.counted(self.tiny.patcher)
        tiny_moss.sample_euler(self.tiny, counted, positive_tokens=23, negative_tokens=23)
        self.assertEqual((stats.calls, stats.elements, stats.skip_calls), (6, 12, 0))

    def test_cfg_one_is_one_forward_per_step(self):
        counted, stats = self.counted(self.tiny.patcher)
        tiny_moss.sample_euler(self.tiny, counted, cfg=1.0)
        self.assertEqual((stats.calls, stats.elements), (6, 6))

    def test_the_wrapper_lives_on_the_clone_only(self):
        before = dict(self.tiny.patcher.model_options["transformer_options"])
        counted, _ = self.counted(self.tiny.patcher)
        self.assertNotIn("wrappers", self.tiny.patcher.model_options["transformer_options"])
        self.assertEqual(self.tiny.patcher.model_options["transformer_options"], before)
        self.assertIn("wrappers", counted.model_options["transformer_options"])

    def test_the_counter_is_a_foreign_modifier_so_it_is_never_installed_before_the_probe(self):
        """The label reads the INCOMING model; a clone carrying our counter would read as transformer_option:wrappers."""
        from moss_photon.core import adapter
        counted, _ = self.counted(self.tiny.patcher)
        self.assertEqual(adapter.probe_model(counted, reference_dtypes=ss.F32).foreign_modifiers,
                         ("transformer_option:wrappers",))
        result = ss.run(self.tiny, overrides={"steps": 6})
        self.assertEqual(result.report["label"]["modifiers"], [])


if __name__ == "__main__":
    unittest.main()
