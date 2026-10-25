"""D10, m7: an interrupt on a LoRA clone releases everything -- the exception propagates unchanged, the original MODEL and
the LoRA clone reproduce their baselines bit for bit afterwards, and the sampler's own clone can be collected.

The interrupt is injected where ComfyUI's own preview callback sits (``latent_preview.prepare_callback`` is replaced by
a stub module that raises ``InterruptProcessingException`` at step 2), which also covers the ``preview=True`` wiring.
"""
import gc
import sys
import types
import unittest
import weakref
from unittest import mock

from . import _comfy_env, tiny_moss
from . import sampler_support as ss

STEPS = {"steps": 6}


class InterruptCleanupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import comfy.model_management as mm
        from moss_photon.core import solver
        cls.mm, cls.solver = mm, solver
        cls.lora = ss.apply_lora(cls.tiny.patcher, ss.make_lora_dict(cls.tiny.patcher), 1.0)

    def stub_preview(self, at_step, seen):
        mm = self.mm

        def prepare_callback(model, steps, x0_output_dict=None):
            seen.append((model, steps))

            def callback(step, x0, x, total):
                if step == at_step:
                    raise mm.InterruptProcessingException()
            return callback
        module = types.ModuleType("latent_preview")
        module.prepare_callback = prepare_callback
        return mock.patch.dict(sys.modules, {"latent_preview": module})

    def run_interrupted(self, model, refs, seen):
        real = self.solver.install_counter

        def spying(clone, stats):
            refs.append(weakref.ref(clone))
            return real(clone, stats)
        with self.stub_preview(2, seen), mock.patch.object(self.solver, "install_counter", spying):
            try:
                ss.run(self.tiny, model=model, preview=True, overrides=STEPS)
            except self.mm.InterruptProcessingException as exc:
                return type(exc)
        self.fail("the interrupt did not propagate")

    def test_interrupt_propagates_unchanged_and_nothing_leaks(self):
        tiny = self.tiny
        base_plain = ss.tensor_sha(tiny_moss.sample_euler(tiny))
        base_lora = ss.tensor_sha(tiny_moss.sample_euler(tiny, self.lora))
        self.assertNotEqual(base_plain, base_lora, "control: the LoRA changes the result")
        refs, seen = [], []
        self.assertIs(self.run_interrupted(self.lora, refs, seen), self.mm.InterruptProcessingException)
        self.assertEqual(len(seen), 1, "the preview callback was built exactly once")
        self.assertEqual(seen[0][1], 6)
        # the original and the LoRA clone are exactly what they were
        self.assertEqual(ss.tensor_sha(tiny_moss.sample_euler(tiny)), base_plain)
        self.assertEqual(ss.tensor_sha(tiny_moss.sample_euler(tiny, self.lora)), base_lora)
        self.assertEqual(len(tiny.patcher.patches), 0)
        # a second, uninterrupted Photon run on the LoRA clone equals a plain run of it
        again = ss.run(tiny, model=self.lora, overrides=STEPS)
        plain_lora = tiny_moss.sample_euler(tiny, ss_slg(self.lora))
        self.assertTrue(again.latent["samples"].equal(plain_lora))

    def test_the_samplers_clone_is_collectable_after_an_interrupt(self):
        refs, seen = [], []
        self.run_interrupted(self.lora, refs, seen)
        self.assertEqual(len(refs), 1)
        del seen[:]
        gc.collect()
        self.assertIsNone(refs[0](), "the sampler's clone is still referenced after the interrupt")

    def test_the_samplers_clone_is_collectable_after_a_normal_run(self):
        refs = []
        real = self.solver.install_counter

        def spying(clone, stats):
            refs.append(weakref.ref(clone))
            return real(clone, stats)
        with mock.patch.object(self.solver, "install_counter", spying):
            ss.run(self.tiny, model=self.lora, overrides=STEPS)
        gc.collect()
        self.assertEqual(len(refs), 1)
        self.assertIsNone(refs[0]())

    def test_a_real_comfy_interrupt_flag_stops_the_run_and_is_cleared(self):
        """The standard ComfyUI path: the flag set from outside is polled by the callback and raised."""
        mm = self.mm
        net = self.tiny.model.diffusion_model
        original = net.forward
        calls = []
        # baselines BEFORE the interrupted run (m18): comparing a run with itself afterwards proves nothing
        base_plain = ss.tensor_sha(tiny_moss.sample_euler(self.tiny))
        base_lora = ss.tensor_sha(tiny_moss.sample_euler(self.tiny, self.lora))
        self.assertNotEqual(base_plain, base_lora, "control: the LoRA changes the result")

        def forward(x, t, *a, **k):
            calls.append(1)
            if len(calls) == 3:
                mm.interrupt_current_processing(True)
            return original(x, t, *a, **k)
        net.forward = forward
        try:
            with self.assertRaises(mm.InterruptProcessingException):
                ss.run(self.tiny, model=self.lora, overrides=STEPS)
        finally:
            del net.forward
            mm.interrupt_current_processing(False)
        self.assertLess(len(calls), 12)
        self.assertEqual(ss.tensor_sha(tiny_moss.sample_euler(self.tiny, self.lora)), base_lora)
        self.assertEqual(ss.tensor_sha(tiny_moss.sample_euler(self.tiny)), base_plain)
        self.assertEqual(len(self.tiny.patcher.patches), 0)


def ss_slg(model):
    from moss_photon.core import guidance
    return guidance.apply_slg(model, layers=[22], start=0.0, end=1.0)


if __name__ == "__main__":
    unittest.main()
