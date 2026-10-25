"""M1 / check C10 (design 6.5): a native loader before, between or after the Photon chain is detectable from the
attachment (string + patches_uuid + entry count + baseline); AuraFlow and SLG clones after the chain stay clean."""
import unittest

from . import lora_synth, tiny_moss


class ForeignPatchesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        import comfy.sd
        from moss_photon.core import adapter, consts, errors, lora
        cls.torch, cls.sd, cls.adapter, cls.consts, cls.errors, cls.lora = torch, comfy.sd, adapter, consts, errors, lora

    def setUp(self):
        self.env = lora_synth.install(self)
        self.patcher = self.tiny.patcher
        self.env.write("A.safetensors", lora_synth.make_lora_sd(self.patcher, seed=11, modules=("qkv", "o")))
        self.env.write("B.safetensors", lora_synth.make_lora_sd(self.patcher, seed=12, modules=("qkv", "o"), skip_layers=(3,)))
        # the "foreign" loader: a different file, split q/k/v + ffn (144 entries), applied with the NATIVE function
        self.foreign_sd = lora_synth.make_lora_sd(self.patcher, fmt="kohya", modules=lora_synth.KOHYA_SPLIT, rank=8, seed=77)
        self.foreign_entries = 144

    def native(self, model):
        out, _ = self.sd.load_lora_for_models(model, None, self.foreign_sd, 1.0, 0)
        return out

    def facts(self, model):
        return self.adapter.probe_model(model, reference_dtypes=("torch.float32",))

    def test_native_loader_before_the_first_node_is_baseline(self):
        dirty = self.native(self.patcher)
        _m, profile, _ = lora_synth.run_node(dirty, "A.safetensors")
        att = profile["attachment"]
        self.assertEqual(att["baseline_entries"], self.foreign_entries)
        self.assertEqual(att["patch_entries_total"], self.foreign_entries + 48)
        self.assertEqual([e["patch_entries"] for e in att["entries"]], [48])
        self.assertEqual(self.lora.validate_identity_profile(profile), [])

    def test_clean_model_has_zero_baseline(self):
        _m, profile, _ = lora_synth.run_node(self.patcher, "A.safetensors")
        self.assertEqual(profile["attachment"]["baseline_entries"], 0)

    def test_native_loader_between_two_nodes_grows_the_baseline_by_its_entries(self):
        m_a, p_a, _ = lora_synth.run_node(self.patcher, "A.safetensors")
        between = self.native(m_a)
        # the attachment string survives the native loader (clone copies attachments), so the chain continues
        self.assertEqual(between.get_attachment(self.consts.ATTACH_LORA_STACK), self.lora.attachment_json(p_a["attachment"]))
        m_ab, p_ab, _ = lora_synth.run_node(between, "B.safetensors", profile_in=p_a)
        att = p_ab["attachment"]
        self.assertEqual(att["baseline_entries"], self.foreign_entries)
        self.assertEqual(att["patch_entries_total"], 48 + self.foreign_entries + 46)
        self.assertEqual(att["patch_entries_total"], self.lora.patch_entries_of(m_ab))
        self.assertEqual(att["baseline_entries"] + sum(e["patch_entries"] for e in att["entries"]), att["patch_entries_total"])
        self.assertEqual(self.lora.validate_identity_profile(p_ab), [])
        self.assertEqual(p_a["attachment"]["baseline_entries"], 0)  # the earlier profile is not rewritten

    def test_two_foreign_loaders_between_nodes_add_up(self):
        m_a, p_a, _ = lora_synth.run_node(self.patcher, "A.safetensors")
        twice = self.native(self.native(m_a))
        _m, p_ab, _ = lora_synth.run_node(twice, "B.safetensors", profile_in=p_a)
        self.assertEqual(p_ab["attachment"]["baseline_entries"], 2 * self.foreign_entries)

    def test_native_loader_before_and_between_both_count(self):
        dirty = self.native(self.patcher)
        m_a, p_a, _ = lora_synth.run_node(dirty, "A.safetensors")
        _m, p_ab, _ = lora_synth.run_node(self.native(m_a), "B.safetensors", profile_in=p_a)
        self.assertEqual(p_ab["attachment"]["baseline_entries"], 2 * self.foreign_entries)

    def test_native_loader_after_the_chain_is_detected_by_uuid_and_count(self):
        m_a, p_a, _ = lora_synth.run_node(self.patcher, "A.safetensors")
        att = p_a["attachment"]
        clean = self.facts(m_a)
        self.assertEqual((clean.lora_stack_attachment, clean.patches_uuid, clean.patch_entries),
                         (self.lora.attachment_json(att), att["patches_uuid"], att["patch_entries_total"]))
        after = self.facts(self.native(m_a))
        self.assertEqual(after.lora_stack_attachment, self.lora.attachment_json(att))  # string alone cannot tell
        self.assertNotEqual(after.patches_uuid, att["patches_uuid"])
        self.assertEqual(after.patch_entries - att["patch_entries_total"], self.foreign_entries)

    def test_auraflow_and_slg_clones_after_the_chain_stay_clean(self):
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
        m_a, p_a, _ = lora_synth.run_node(self.patcher, "A.safetensors")
        att = p_a["attachment"]
        for name, clone in (("auraflow", ModelSamplingAuraFlow().patch_aura(m_a, 5.0)[0]),
                            ("slg", SkipLayerGuidanceDiTSimple.execute(m_a, 0.0, 1.0, double_layers="", single_layers="22").args[0])):
            with self.subTest(clone=name):
                f = self.facts(clone)
                self.assertEqual((f.lora_stack_attachment, f.patches_uuid, f.patch_entries),
                                 (self.lora.attachment_json(att), att["patches_uuid"], att["patch_entries_total"]))

    def test_profile_of_another_chain_is_refused(self):
        m_a, p_a, _ = lora_synth.run_node(self.patcher, "A.safetensors")
        m_b, p_b, _ = lora_synth.run_node(self.patcher, "B.safetensors")
        with self.assertRaises(self.errors.IncompatibleLora) as ctx:
            lora_synth.run_node(m_a, "B.safetensors", profile_in=p_b)  # the MODEL of chain A, the profile of chain B
        self.assertIn("does not belong to this MODEL", str(ctx.exception))
        with self.assertRaises(self.errors.IncompatibleLora):
            lora_synth.run_node(self.patcher, "B.safetensors", profile_in=p_a)  # a MODEL without any chain

    def test_an_empty_profile_starts_a_chain_on_a_model_that_already_has_native_patches(self):
        dirty = self.native(self.patcher)
        _m, empty, _ = lora_synth.run_node(dirty, "none")  # an empty profile starts a chain on any model
        self.assertEqual(empty["attachment"]["baseline_entries"], self.foreign_entries)
        _m2, profile, _ = lora_synth.run_node(dirty, "A.safetensors", profile_in=empty)
        self.assertEqual(profile["attachment"]["baseline_entries"], self.foreign_entries)
        self.assertEqual(len(profile["stack"]), 1)

    def test_malformed_and_unsupported_profile_in(self):
        m_a, p_a, _ = lora_synth.run_node(self.patcher, "A.safetensors")
        with self.assertRaises(self.errors.IncompatibleLora):
            lora_synth.run_node(m_a, "B.safetensors", profile_in={"kind": "something"})
        phased = dict(p_a, phase_plan={"phases": []})
        with self.assertRaises(self.errors.UnsupportedFeature):
            lora_synth.run_node(m_a, "B.safetensors", profile_in=phased)

    def test_pass_through_keeps_the_chain_intact(self):
        m_a, p_a, _ = lora_synth.run_node(self.patcher, "A.safetensors")
        m_same, p_same, _ = lora_synth.run_node(m_a, "none", profile_in=p_a)
        self.assertIs(m_same, m_a)
        self.assertIs(p_same, p_a)
        m_ab, p_ab, _ = lora_synth.run_node(m_same, "B.safetensors", profile_in=p_same)
        self.assertEqual([e["basename"] for e in p_ab["attachment"]["entries"]], ["A.safetensors", "B.safetensors"])


if __name__ == "__main__":
    unittest.main()
