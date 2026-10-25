"""Patch leak (design 6.5, check C5; MP-10 acceptance 2): the LoRA lives on a clone, never on the input MODEL.

baseline == original after a LoRA run == original after an interrupted LoRA run == original after interleaving
[orig, lora, orig, lora, orig, orig]; LoRA runs repeat bit-equal; the original keeps zero patches and its uuid; an
exception raised mid-sampling (or inside the node) leaves the next run without the previous job's LoRA.
"""
import unittest
from unittest import mock

from . import lora_synth, tiny_moss


class LeakTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        import comfy.model_management
        from moss_photon.core import consts, lora
        cls.torch, cls.mm, cls.consts, cls.lora = torch, comfy.model_management, consts, lora
        cls.baseline = tiny_moss.sample_euler(cls.tiny)

    def setUp(self):
        self.env = lora_synth.install(self)
        self.patcher = self.tiny.patcher
        self.env.write("A.safetensors", lora_synth.make_lora_sd(self.patcher, seed=11, modules=("qkv", "o")))
        self.env.write("B.safetensors", lora_synth.make_lora_sd(self.patcher, seed=12, modules=("qkv", "o", "ffn_down")))

    def sample(self, model=None, callback=None):
        return tiny_moss.sample_euler(self.tiny, model, callback=callback)

    def eq(self, a, b):
        return bool(self.torch.equal(a, b))

    def assert_original_pristine(self):
        p = self.patcher
        self.assertEqual(len(p.patches), 0)
        self.assertEqual(str(p.patches_uuid), self.uuid0)
        self.assertEqual(p.attachments, {})
        self.assertEqual(p.model_options, {"transformer_options": {}})
        self.assertEqual(p.object_patches, {})

    def apply(self, name="A.safetensors", **kw):
        return lora_synth.run_node(self.patcher, name, **kw)[0]

    def test_the_original_never_carries_the_lora(self):
        self.uuid0 = str(self.patcher.patches_uuid)
        model = self.apply(strength_model=0.9)
        lora_run = self.sample(model)
        self.assertFalse(self.eq(lora_run, self.baseline))  # the LoRA does something, so equality below means something
        self.assert_original_pristine()
        self.assertTrue(self.eq(self.sample(), self.baseline))
        self.assertTrue(self.eq(self.sample(model), lora_run))  # a LoRA run repeats bit-equal
        self.assert_original_pristine()

    def test_interrupt_in_a_lora_run_leaves_no_trace(self):
        self.uuid0 = str(self.patcher.patches_uuid)
        model = self.apply()
        first = self.sample(model)

        def interrupt(step, x0, x, total):
            if step == 2:
                raise self.mm.InterruptProcessingException()

        with self.assertRaises(self.mm.InterruptProcessingException):
            self.sample(model, callback=interrupt)
        self.assertTrue(self.eq(self.sample(), self.baseline))
        self.assertTrue(self.eq(self.sample(model), first))
        self.assert_original_pristine()

    def test_interleaved_runs_are_clean(self):
        self.uuid0 = str(self.patcher.patches_uuid)
        model = self.apply()
        order = [None, model, None, model, None, None]
        outs = [self.sample(m) for m in order]
        lora_ref = outs[1]
        self.assertFalse(self.eq(lora_ref, self.baseline))
        for out, m in zip(outs, order):
            self.assertTrue(self.eq(out, self.baseline if m is None else lora_ref))
        self.assert_original_pristine()

    def test_exception_in_sampling_does_not_carry_the_lora_into_the_next_job(self):
        self.uuid0 = str(self.patcher.patches_uuid)
        b_alone = self.sample(self.apply("B.safetensors"))                  # reference: B only, on the original
        job1 = self.apply("A.safetensors")                                  # job 1: A applied, then sampling blows up

        def boom(step, x0, x, total):
            if step == 1:
                raise RuntimeError("sampler failed")

        with self.assertRaises(RuntimeError):
            self.sample(job1, callback=boom)
        job2 = self.apply("B.safetensors")                                  # job 2 starts again from the original
        self.assertTrue(self.eq(self.sample(job2), b_alone))                   # no A in it
        self.assertTrue(self.eq(self.sample(), self.baseline))
        self.assert_original_pristine()

    def test_exception_inside_the_node_leaves_the_original_clean(self):
        self.uuid0 = str(self.patcher.patches_uuid)
        with mock.patch.object(self.lora, "provenance_baseline", side_effect=RuntimeError("node failed after apply")):
            with self.assertRaises(RuntimeError):
                self.apply()
        self.assert_original_pristine()
        self.assertTrue(self.eq(self.sample(), self.baseline))
        again = self.apply()
        self.assertFalse(self.eq(self.sample(again), self.baseline))
        self.assert_original_pristine()

    def test_a_chain_is_isolated_from_its_parts(self):
        self.uuid0 = str(self.patcher.patches_uuid)
        m_a, p_a, _ = lora_synth.run_node(self.patcher, "A.safetensors")
        a_run = self.sample(m_a)
        m_ab, _p, _c = lora_synth.run_node(m_a, "B.safetensors", profile_in=p_a)
        self.assertFalse(self.eq(self.sample(m_ab), a_run))
        self.assertTrue(self.eq(self.sample(m_a), a_run))  # the second node did not touch the first node's model
        self.assertEqual(self.lora.patch_entries_of(m_a), 48)
        self.assert_original_pristine()


if __name__ == "__main__":
    unittest.main()
