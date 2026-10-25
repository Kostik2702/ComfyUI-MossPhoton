"""Module grouping, key format and mapping summary (design 1.6, 1.9 / C6): golden key lists, no ComfyUI."""
import unittest

from moss_photon.core import lora

from . import golden_keys as gk


class GroupingTest(unittest.TestCase):
    def test_golden_tensor_and_module_counts(self):
        for name, (gen, tensors, modules, _fmt) in gk.GOLDEN.items():
            with self.subTest(name=name):
                keys = gen()
                self.assertEqual(len(keys), tensors)
                self.assertEqual(len(lora.weight_modules(lora.group_lora_modules(keys))), modules)

    def test_alpha_only_modules_are_dropped(self):
        grouped = lora.group_lora_modules(gk.lokr())
        self.assertEqual(len(grouped), 164 + 123)
        weighted = lora.weight_modules(grouped)
        self.assertEqual(len(weighted), 164)
        dropped = set(grouped) - set(weighted)
        self.assertEqual(len(dropped), 123)
        self.assertTrue(all(grouped[m] == frozenset({".alpha"}) for m in dropped))

    def test_without_the_refinement_a_working_lokr_would_read_as_partial(self):
        # 164 mapped of 287 grouped modules is 0.57: the failure the refinement removes (design 1.9)
        self.assertAlmostEqual(164 / 287, 0.571, places=3)
        self.assertEqual(lora.coverage_verdict(164, 164)["verdict"], "ok")

    def test_longest_suffix_wins(self):
        grouped = lora.group_lora_modules(["a.b.lora_A.weight", "a.b.lora_B", "c.diff_b", "c.diff", "d.lora.up.weight"])
        self.assertEqual(grouped["a.b"], frozenset({".lora_A.weight", ".lora_B"}))
        self.assertEqual(grouped["c"], frozenset({".diff_b", ".diff"}))
        self.assertEqual(grouped["d"], frozenset({".lora.up.weight"}))

    def test_unknown_suffix_is_its_own_non_weight_module(self):
        grouped = lora.group_lora_modules(["weird.tensor"])
        self.assertEqual(grouped, {"weird.tensor": frozenset({""})})
        self.assertEqual(lora.weight_modules(grouped), {})

    def test_dora_scale_is_a_scale_not_a_weight(self):
        grouped = lora.group_lora_modules(["m.lora_A.weight", "m.lora_B.weight", "m.dora_scale", "n.dora_scale"])
        self.assertEqual(sorted(lora.weight_modules(grouped)), ["m"])

    def test_every_suffix_of_the_design_is_known(self):
        for suffix in lora.WEIGHT_SUFFIXES:
            self.assertEqual(lora.group_lora_modules(["x" + suffix]), {"x": frozenset({suffix})}, suffix)
        for suffix in lora.SCALE_SUFFIXES:
            self.assertEqual(lora.group_lora_modules(["x" + suffix]), {"x": frozenset({suffix})}, suffix)


class KeyFormatTest(unittest.TestCase):
    def test_formats_of_the_real_files(self):
        for name, (gen, _t, _m, fmt) in gk.GOLDEN.items():
            with self.subTest(name=name):
                self.assertEqual(lora.classify_key_format(gen()), fmt)

    def test_distill_files_are_kohya_by_their_suffix_family(self):
        # docs/03-turbo.md 5.4 asks MP-10 to decide: lora_down/lora_up is the kohya family, the prefix is not what counts
        keys = gk.diffusion_kohya()
        self.assertTrue(all(k.startswith("diffusion_model.") for k in keys))
        self.assertEqual(lora.classify_key_format(keys), "kohya")

    def test_unknown_and_empty(self):
        self.assertEqual(lora.classify_key_format([]), "unknown")
        self.assertEqual(lora.classify_key_format(["a.hada_w1_a", "a.hada_w1_b"]), "unknown")


class TePrefixTest(unittest.TestCase):
    def test_te_prefixes(self):
        mods = ["text_encoders.qwen.transformer.layers.0.q_proj", "text_encoder.model.layers.0.q", "lora_te_x_y",
                "diffusion_model.layers.0.proj.q", "lora_unet_layers_0_proj_q"]
        self.assertEqual(lora.te_prefixed_modules(mods),
                         ["lora_te_x_y", "text_encoder.model.layers.0.q", "text_encoders.qwen.transformer.layers.0.q_proj"])


class SummarizeMappingTest(unittest.TestCase):
    @staticmethod
    def weight_key(layer, tail="proj.qkv.weight"):
        return "diffusion_model.layers.%d.%s" % (layer, tail)

    def test_full_peft_split_file(self):
        keys = gk.peft_split()
        accepted = []
        for n in gk.ALL42:
            for off in range(3):
                accepted.append((self.weight_key(n), (0, off * 8, 8)))
            for tail in ("proj.o.weight", "proj.ffn_up_gate.weight", "proj.ffn_down.weight"):
                accepted.append(self.weight_key(n, tail))
        names = {m for m in lora.weight_modules(lora.group_lora_modules(keys))}
        r = lora.summarize_mapping(keys, accepted_model=accepted, accepted_te=[], mapper_names=names,
                                   num_layers=42, te_active=False)
        self.assertEqual((r.tensors_total, r.modules_total, r.modules_mapped), (504, 252, 252))
        self.assertEqual((r.model_entries, r.keys_patched), (252, 168))  # q,k,v land on the fused qkv key: character_a 168
        self.assertEqual((r.layers_total, r.layers_covered, r.layers_missing), (42, 42, ()))
        self.assertEqual((r.te_entries, r.te_modules_ignored, r.unmapped_examples), (0, 0, ()))
        self.assertEqual(r.key_format, "peft_split_qkv")

    def test_layer_22_missing_is_reported_not_penalised(self):
        keys = gk.peft_fused()
        accepted = [self.weight_key(n, t) for n in gk.NO22 for t in
                    ("proj.qkv.weight", "proj.o.weight", "proj.ffn_up_gate.weight", "proj.ffn_down.weight")]
        r = lora.summarize_mapping(keys, accepted_model=accepted, accepted_te=[], mapper_names=set(),
                                   num_layers=42, te_active=False)
        self.assertEqual((r.layers_covered, r.layers_missing), (41, (22,)))
        self.assertEqual(lora.coverage_verdict(r.modules_total, r.modules_mapped)["verdict"], "ok")

    def test_unmapped_examples_are_capped_and_sorted(self):
        keys = gk.other_architecture()
        r = lora.summarize_mapping(keys, accepted_model=[], accepted_te=[], mapper_names=set(), num_layers=42,
                                   te_active=False)
        self.assertEqual(r.modules_mapped, 0)
        self.assertEqual(len(r.unmapped_examples), 5)
        self.assertEqual(list(r.unmapped_examples), sorted(r.unmapped_examples))

    def test_te_modules_are_ignored_without_the_flag_and_counted_with_it(self):
        keys = (gk.peft_fused() + ["text_encoders.qwen.transformer.layers.%d.q_proj.lora_%s.weight" % (n, ab)
                                   for n in range(3) for ab in "AB"])
        accepted = [self.weight_key(n, "proj.qkv.weight") for n in gk.NO22]
        off = lora.summarize_mapping(keys, accepted_model=accepted, accepted_te=["t.weight"], mapper_names=set(),
                                     num_layers=42, te_active=False)
        self.assertEqual((off.modules_total, off.te_modules_ignored, off.te_entries, off.modules_mapped),
                         (164, 3, 0, 41))
        on = lora.summarize_mapping(keys, accepted_model=accepted, accepted_te=["t.weight"], mapper_names=set(),
                                    num_layers=42, te_active=True)
        self.assertEqual((on.modules_total, on.te_modules_ignored, on.te_entries, on.modules_mapped),
                         (167, 0, 1, 42))

    def test_unknown_layer_count(self):
        r = lora.summarize_mapping(["a.lora_A.weight", "a.lora_B.weight"], accepted_model=[], accepted_te=[],
                                   mapper_names=set(), num_layers=None, te_active=False)
        self.assertEqual((r.layers_total, r.layers_covered, r.layers_missing), (0, 0, ()))


if __name__ == "__main__":
    unittest.main()
