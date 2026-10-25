"""The node applies a LoRA exactly like the native loader (design 6.5, D6): bit-equal run, clones only.

Synthetic LoRAs on the tiny 24-layer MOSS: a PEFT fused-qkv file (the node vs ``LoraLoaderModelOnly``) and a
kohya-suffix split-q/k/v file shaped like the 8-step distill LoRA (docs/03-turbo.md 4.1: 144 entries, 96 keys).
"""
import json
import os
import unittest

from . import lora_synth, tiny_moss

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "contracts",
                           "schemas", "identity_profile.schema.json")


class ApplyEqualsNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        import nodes
        from jsonschema import Draft202012Validator
        from moss_photon.core import adapter, consts, lora
        cls.torch, cls.nodes, cls.adapter, cls.consts, cls.lora = torch, nodes, adapter, consts, lora
        with open(SCHEMA_PATH, "rb") as f:
            cls.schema = Draft202012Validator(json.loads(f.read().decode("utf-8")))
        cls.baseline = tiny_moss.sample_euler(cls.tiny)

    def setUp(self):
        self.env = lora_synth.install(self)
        self.patcher = self.tiny.patcher

    def run_native(self, name, strength):
        native = self.nodes.LoraLoaderModelOnly()
        return native.load_lora_model_only(self.patcher, name, strength)[0]

    def test_fused_peft_file_is_bit_equal_to_lora_loader_model_only(self):
        sd = lora_synth.make_lora_sd(self.patcher, fmt="peft", modules=("qkv", "o"), seed=11)
        self.env.write("fused.safetensors", sd, {"ss_base_model_version": "moss_image"})
        out, profile, clip = lora_synth.run_node(self.patcher, "fused.safetensors", strength_model=0.7)
        native = self.run_native("fused.safetensors", 0.7)
        mine, theirs = tiny_moss.sample_euler(self.tiny, out), tiny_moss.sample_euler(self.tiny, native)
        self.assertTrue(self.torch.equal(mine, theirs))
        self.assertFalse(self.torch.equal(mine, self.baseline))  # the LoRA has an effect, the equality is not vacuous
        self.assertIsNone(clip)
        self.assertEqual(self.lora.patch_entries_of(out), self.lora.patch_entries_of(native))
        self.assertEqual(sorted(out.patches), sorted(native.patches))

    def test_kohya_split_qkv_file_is_bit_equal_and_counted_per_entry(self):
        sd = lora_synth.make_lora_sd(self.patcher, fmt="kohya", modules=lora_synth.KOHYA_SPLIT, rank=8, seed=21)
        self.env.write("distill_like.safetensors", sd)
        out, profile, _clip = lora_synth.run_node(self.patcher, "distill_like.safetensors", role="distill",
                                                  strength_model=1.0)
        native = self.run_native("distill_like.safetensors", 1.0)
        self.assertTrue(self.torch.equal(tiny_moss.sample_euler(self.tiny, out), tiny_moss.sample_euler(self.tiny, native)))
        e = profile["stack"][0]
        self.assertEqual((e["patch_entries"], e["keys_patched"]), (144, 96))  # 24 layers x 6 modules; q,k,v share qkv
        self.assertEqual(e["key_format"], "kohya")
        self.assertEqual(e["coverage"]["modules_total"], 144)
        self.assertEqual(e["coverage"]["modules_mapped"], 144)
        self.assertEqual(e["coverage"]["tensors_total"], 24 * 6 * 3)

    def test_profile_and_attachment(self):
        sd = lora_synth.make_lora_sd(self.patcher, fmt="peft", modules=("qkv", "o"), seed=11)
        digest = self.env.write("fused.safetensors", sd, {"ss_base_model_version": "moss_image",
                                                           "software": '{"name": "ai-toolkit", "version": "0.13.34"}'})
        out, profile, _ = lora_synth.run_node(self.patcher, "fused.safetensors", strength_model=0.8, trigger=" anna , ")
        self.assertEqual([e.message for e in self.schema.iter_errors(profile)], [])
        self.assertEqual(self.lora.validate_identity_profile(profile), [])
        e = profile["stack"][0]
        self.assertEqual((e["index"], e["role"], e["strength_model"], e["strength_clip"]), (0, "character", 0.8, 0.0))
        self.assertEqual(e["lora"], {"basename": "fused.safetensors", "size": os.path.getsize(self.env.path("fused.safetensors")),
                                     "sha256": digest, "metadata": {"ss_base_model_version": "moss_image",
                                                                    "software": "ai-toolkit 0.13.34"}})
        self.assertEqual((e["patch_entries"], e["keys_patched"]), (48, 48))
        self.assertEqual(e["key_format"], "peft_fused_qkv")
        cov = e["coverage"]
        self.assertEqual((cov["modules_total"], cov["modules_mapped"], cov["ratio"], cov["verdict"]), (48, 48, 1.0, "ok"))
        self.assertEqual((cov["layers_total"], cov["layers_covered"], cov["layers_missing"]), (24, 24, []))
        self.assertEqual((e["trigger_present"], e["trigger_hints"]), (True, ["anna"]))
        self.assertEqual(e["applied_to"], {"transformer": True, "text_encoder": False})
        self.assertEqual(e["warnings"], [])  # nothing to warn about: tiny model has no identity, sha is unknown to the tables
        att = profile["attachment"]
        self.assertEqual((att["patch_entries_total"], att["baseline_entries"], att["patches_uuid"]),
                         (48, 0, str(out.patches_uuid)))
        self.assertEqual(out.get_attachment(self.consts.ATTACH_LORA_STACK), self.lora.attachment_json(att))
        self.assertIsInstance(out.get_attachment(self.consts.ATTACH_LORA_STACK), str)
        facts = self.adapter.probe_model(out, reference_dtypes=("torch.float32",))
        self.assertEqual((facts.lora_stack_attachment, facts.patches_uuid, facts.patch_entries),
                         (self.lora.attachment_json(att), att["patches_uuid"], 48))
        self.assertEqual(profile["model_identity"], None)  # the tiny model was not loaded from a file
        self.assertEqual(profile["summary"]["patch_entries_total"], 48)

    def test_native_lora_metadata_attachment_is_kept(self):
        sd = lora_synth.make_lora_sd(self.patcher, fmt="peft", seed=3)
        self.env.write("m.safetensors", sd, {"ss_base_model_version": "moss_image", "ss_network_dim": "4"})
        out, _p, _c = lora_synth.run_node(self.patcher, "m.safetensors")
        native = self.run_native("m.safetensors", 1.0)
        self.assertEqual(out.get_attachment("lora_metadata"), native.get_attachment("lora_metadata"))
        self.assertEqual(out.get_attachment("lora_metadata")["ss_network_dim"], "4")

    def test_input_model_is_untouched(self):
        before = (len(self.patcher.patches), str(self.patcher.patches_uuid), dict(self.patcher.attachments),
                  repr(sorted(self.patcher.model_options)), len(self.patcher.object_patches))
        sd = lora_synth.make_lora_sd(self.patcher, fmt="peft", seed=3)
        self.env.write("m.safetensors", sd)
        out, _p, _c = lora_synth.run_node(self.patcher, "m.safetensors")
        self.assertIsNot(out, self.patcher)
        after = (len(self.patcher.patches), str(self.patcher.patches_uuid), dict(self.patcher.attachments),
                 repr(sorted(self.patcher.model_options)), len(self.patcher.object_patches))
        self.assertEqual(before, after)
        self.assertEqual(len(self.patcher.patches), 0)
        self.assertIsNone(self.patcher.get_attachment(self.consts.ATTACH_LORA_STACK))
        self.assertTrue(self.torch.equal(tiny_moss.sample_euler(self.tiny), self.baseline))

    def test_none_and_zero_strength_return_the_same_objects(self):
        sd = lora_synth.make_lora_sd(self.patcher, fmt="peft", seed=3)
        self.env.write("m.safetensors", sd)
        m1, p1, _ = lora_synth.run_node(self.patcher, "none")
        self.assertIs(m1, self.patcher)
        m2, p2, _ = lora_synth.run_node(self.patcher, "m.safetensors", strength_model=0.0)
        self.assertIs(m2, self.patcher)
        self.assertEqual(p2["stack"], [])
        self.assertEqual(self.lora.validate_identity_profile(p2), [])


if __name__ == "__main__":
    unittest.main()
