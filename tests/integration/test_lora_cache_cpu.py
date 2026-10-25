"""MP-30 m17: the LoRA node keeps the last loaded file like the stock LoraLoader and maps it once per execution."""
import os
import unittest
from unittest import mock

from . import lora_synth, tiny_moss


class LoraStateCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        import comfy.lora
        import comfy.utils
        from moss_photon.core import lora
        cls.torch, cls.comfy_lora, cls.comfy_utils, cls.lora = torch, comfy.lora, comfy.utils, lora

    def setUp(self):
        self.env = lora_synth.install(self)
        self.patcher = self.tiny.patcher
        self.env.write("A.safetensors", lora_synth.make_lora_sd(self.patcher, seed=11, modules=("qkv", "o")),
                       {"ss_base_model_version": "moss_image"})
        self.reads = []
        self.maps = []
        real_read, real_map = self.comfy_utils.load_torch_file, self.comfy_lora.load_lora

        def read(path, *a, **k):
            self.reads.append(os.path.basename(path))
            return real_read(path, *a, **k)

        def mapper(*a, **k):
            self.maps.append(1)
            return real_map(*a, **k)
        for target, name, fn in ((self.comfy_utils, "load_torch_file", read), (self.comfy_lora, "load_lora", mapper)):
            patch = mock.patch.object(target, name, fn)
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_same_file_is_read_from_disk_once(self):
        lora_synth.run_node(self.patcher, "A.safetensors")
        lora_synth.run_node(self.patcher, "A.safetensors", strength_model=0.5)
        self.assertEqual(self.reads, ["A.safetensors"])

    def test_the_native_mapper_runs_once_per_node_execution(self):
        lora_synth.run_node(self.patcher, "A.safetensors")
        self.assertEqual(len(self.maps), 1)  # was 2: map_lora and load_lora_for_models
        lora_synth.run_node(self.patcher, "A.safetensors")
        self.assertEqual(len(self.maps), 2)  # the model may differ, so each execution maps for its own model

    def test_a_touched_file_is_read_again(self):
        # Windows keeps a loaded safetensors file mapped, so the edit is simulated by its mtime (part of the key)
        path = self.env.path("A.safetensors")
        self.lora.load_lora_state(path)
        mtime = os.stat(path).st_mtime_ns
        os.utime(path, ns=(mtime + 10 ** 9, mtime + 10 ** 9))
        self.lora.load_lora_state(path)
        self.assertEqual(self.reads, ["A.safetensors", "A.safetensors"])

    def test_the_key_is_path_mtime_and_size(self):
        one, two = self.env.path("one.bin"), self.env.path("two.bin")
        for path, data in ((one, b"abc"), (two, b"abcd")):
            with open(path, "wb") as f:
                f.write(data)
        stamp = os.stat(one).st_mtime_ns
        os.utime(one, ns=(stamp, stamp))
        os.utime(two, ns=(stamp, stamp))
        key = self.lora._state_key
        self.assertNotEqual(key(one), key(two))  # path, and size
        with open(one, "ab") as f:
            f.write(b"d")
        os.utime(one, ns=(stamp, stamp))
        self.assertEqual(os.path.getsize(one), os.path.getsize(two))
        self.assertEqual(key(one)[1:], key(two)[1:])  # control: only the path differs now
        self.assertNotEqual(key(one), key(two))

    def test_two_files_alternating_are_each_read_once(self):
        """MP-36 m17: the cache is an LRU of several slots, so a two-LoRA chain re-run reads no file twice (the
        one-slot cache read A, B, A, B...)."""
        self.env.write("B.safetensors", lora_synth.make_lora_sd(self.patcher, seed=12, modules=("qkv", "o")))
        for _ in range(3):
            lora_synth.run_node(self.patcher, "A.safetensors")
            lora_synth.run_node(self.patcher, "B.safetensors")
        self.assertEqual(self.reads, ["A.safetensors", "B.safetensors"])

    def test_the_chain_gives_the_same_models_on_every_run(self):
        self.env.write("B.safetensors", lora_synth.make_lora_sd(self.patcher, seed=12, modules=("qkv", "o")))
        first = self.chain()
        again = self.chain()
        self.assertEqual(self.reads, ["A.safetensors", "B.safetensors"])
        self.assertEqual(first[1]["digest"], again[1]["digest"])
        self.assertTrue(self.torch.equal(tiny_moss.sample_euler(self.tiny, first[0]),
                                         tiny_moss.sample_euler(self.tiny, again[0])))

    def chain(self):
        model, profile, _ = lora_synth.run_node(self.patcher, "A.safetensors")
        model, profile, _ = lora_synth.run_node(model, "B.safetensors", role="style", profile_in=profile)
        return model, profile

    def test_the_least_recently_used_file_goes_first(self):
        slots = self.lora.STATE_CACHE_SLOTS
        names = ["F%d.safetensors" % i for i in range(slots + 1)]
        for i, name in enumerate(names):
            self.env.write(name, lora_synth.make_lora_sd(self.patcher, seed=20 + i, modules=("qkv", "o")))
        for name in names:
            lora_synth.run_node(self.patcher, name)
        lora_synth.run_node(self.patcher, names[-1])       # still cached
        self.assertEqual(self.reads, names)
        lora_synth.run_node(self.patcher, names[0])        # evicted by the (slots + 1)th file
        self.assertEqual(self.reads, names + [names[0]])

    def test_a_patch_count_mismatch_is_a_typed_error_not_a_runtime_error(self):
        """n1 (R3): an add_patches that behaves differently in another ComfyUI is reported, never a bare RuntimeError."""
        from moss_photon.core import errors
        with mock.patch.object(self.lora, "apply_patches", return_value=(self.patcher, None)):
            with self.assertRaises(errors.UnsupportedFeature) as ctx:
                lora_synth.run_node(self.patcher, "A.safetensors")
        self.assertNotIsInstance(ctx.exception, RuntimeError)
        self.assertIn("add_patches added 0 entries", str(ctx.exception))

    def test_the_returned_metadata_is_a_copy(self):
        path = self.env.path("A.safetensors")
        _, meta = self.lora.load_lora_state(path)
        meta["poison"] = "x"
        _, again = self.lora.load_lora_state(path)
        self.assertNotIn("poison", again)

    def test_the_cached_state_gives_the_same_model_as_a_fresh_read(self):
        m_first, p_first, _ = lora_synth.run_node(self.patcher, "A.safetensors", strength_model=0.8)
        m_cached, p_cached, _ = lora_synth.run_node(self.patcher, "A.safetensors", strength_model=0.8)
        self.assertEqual(self.reads, ["A.safetensors"])
        self.assertEqual(p_first["digest"], p_cached["digest"])
        self.assertTrue(self.torch.equal(tiny_moss.sample_euler(self.tiny, m_first),
                                         tiny_moss.sample_euler(self.tiny, m_cached)))


if __name__ == "__main__":
    unittest.main()
