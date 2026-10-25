"""B1 (REQUIRED, design 6.5, check C7): coverage is per FILE, so B after A reads exactly like B alone.

Also the distill + character stack (owner 2026-10-25): order recorded and logged, roles, known-hash rules.
"""
import unittest
from unittest import mock

from . import lora_synth, tiny_moss


class ChainABTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.core import consts, lora
        cls.torch, cls.consts, cls.lora = torch, consts, lora

    def setUp(self):
        self.env = lora_synth.install(self)
        self.patcher = self.tiny.patcher
        # A touches qkv and o of all 24 layers; B touches the SAME weight keys but skips layer 3
        self.env.write("A.safetensors", lora_synth.make_lora_sd(self.patcher, seed=11, modules=("qkv", "o")))
        self.env.write("B.safetensors", lora_synth.make_lora_sd(self.patcher, seed=12, modules=("qkv", "o"), skip_layers=(3,)))

    def test_b_after_a_maps_exactly_like_b_alone(self):
        sd_b, _ = self.lora.load_lora_state(self.env.path("B.safetensors"))
        m_a, p_a, _ = lora_synth.run_node(self.patcher, "A.safetensors")
        alone = self.lora.map_lora(self.patcher, sd_b)
        on_top = self.lora.map_lora(m_a, sd_b)
        self.assertEqual(alone, on_top)  # the whole MappingResult: counts, layers, key format
        self.assertEqual((alone.modules_total, alone.modules_mapped, alone.model_entries), (46, 46, 46))
        self.assertEqual(alone.layers_missing, (3,))
        self.assertEqual(self.lora.coverage_verdict(alone.modules_total, alone.modules_mapped)["verdict"], "ok")

    def test_node_chain_a_then_b(self):
        m_a, p_a, _ = lora_synth.run_node(self.patcher, "A.safetensors", strength_model=0.8)
        m_ab, p_ab, _ = lora_synth.run_node(m_a, "B.safetensors", strength_model=0.6, profile_in=p_a, role="style")
        m_b, p_b, _ = lora_synth.run_node(self.patcher, "B.safetensors", strength_model=0.6, role="style")
        e = self.lora.patch_entries_of
        # B is accepted although it adds no new distinct key (the v1 "gained a new distinct key" rule would reject it)
        self.assertEqual(len(m_ab.patches) - len(m_a.patches), 0)
        self.assertEqual(e(m_ab) - e(m_a), 46)
        self.assertEqual(e(m_ab) - e(m_a), p_b["stack"][0]["patch_entries"])
        self.assertNotEqual(str(m_ab.patches_uuid), str(m_a.patches_uuid))
        # per-file numbers of B do not depend on A
        for key in ("patch_entries", "keys_patched", "coverage", "key_format", "applied_to"):
            self.assertEqual(p_ab["stack"][1][key], p_b["stack"][0][key], key)
        self.assertEqual(p_ab["stack"][1]["coverage"]["layers_missing"], [3])
        self.assertEqual([x["index"] for x in p_ab["stack"]], [0, 1])
        self.assertEqual([x["role"] for x in p_ab["stack"]], ["character", "style"])
        # the result changes with B on top of A
        def run(m):
            return tiny_moss.sample_euler(self.tiny, m)

        self.assertFalse(self.torch.equal(run(m_ab), run(m_a)))
        self.assertFalse(self.torch.equal(run(m_ab), run(m_b)))
        # attachment of the second node
        att = p_ab["attachment"]
        self.assertEqual((att["patch_entries_total"], att["baseline_entries"], att["patches_uuid"]),
                         (e(m_ab), 0, str(m_ab.patches_uuid)))
        self.assertEqual([x["basename"] for x in att["entries"]], ["A.safetensors", "B.safetensors"])
        self.assertEqual([x["patch_entries"] for x in att["entries"]], [48, 46])
        self.assertEqual(m_ab.get_attachment(self.consts.ATTACH_LORA_STACK), self.lora.attachment_json(att))
        self.assertEqual(self.lora.validate_identity_profile(p_ab), [])
        # chain inputs are not mutated by the second node
        self.assertEqual(e(m_a), 48)
        self.assertEqual(p_a["attachment"]["patch_entries_total"], 48)
        self.assertEqual(len(p_a["stack"]), 1)

    def test_swapping_the_order_changes_recipe_and_attachment_not_per_file_numbers(self):
        m_a, p_a, _ = lora_synth.run_node(self.patcher, "A.safetensors")
        _m_ab, p_ab, _ = lora_synth.run_node(m_a, "B.safetensors", profile_in=p_a)
        m_b, p_b, _ = lora_synth.run_node(self.patcher, "B.safetensors")
        _m_ba, p_ba, _ = lora_synth.run_node(m_b, "A.safetensors", profile_in=p_b)
        self.assertNotEqual(self.lora.attachment_json(p_ab["attachment"]), self.lora.attachment_json(p_ba["attachment"]))
        self.assertNotEqual(p_ab["digest"], p_ba["digest"])
        self.assertEqual(p_ab["attachment"]["patch_entries_total"], p_ba["attachment"]["patch_entries_total"])

    def test_old_rule_would_have_rejected_b(self):
        # the trap of v1 (MP-5 B1): b adds entries and a new uuid, never a new key
        import comfy.sd
        sd_a, _ = self.lora.load_lora_state(self.env.path("A.safetensors"))
        sd_b, _ = self.lora.load_lora_state(self.env.path("B.safetensors"))
        m_a, _ = comfy.sd.load_lora_for_models(self.patcher, None, sd_a, 1.0, 0)
        m_ab, _ = comfy.sd.load_lora_for_models(m_a, None, sd_b, 1.0, 0)
        self.assertEqual(len(m_ab.patches) - len(m_a.patches), 0)
        self.assertGreater(self.lora.patch_entries_of(m_ab), self.lora.patch_entries_of(m_a))

    def test_mapping_does_not_read_the_models_patches(self):
        # the property B1 stands on, tested directly: a model whose `patches` explode on access still maps
        class Boom:
            def _no(self, *args, **kwargs):
                raise AssertionError("coverage read model.patches")

            __contains__ = __iter__ = __len__ = __getitem__ = keys = values = items = get = __getattr__ = _no

        sd_b, _ = self.lora.load_lora_state(self.env.path("B.safetensors"))
        spy = lora_synth.namespace(model=self.patcher.model, patches=Boom(), patches_uuid=Boom())
        self.assertEqual(self.lora.map_lora(spy, sd_b).modules_mapped, 46)


class DistillCharacterStackTest(unittest.TestCase):
    """Distill + character (owner 2026-10-25): validated, order fixed by the nodes, recorded and logged."""

    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.core import consts, errors, lora
        from moss_photon.nodes import lora as node
        cls.torch, cls.consts, cls.errors, cls.lora, cls.node = torch, consts, errors, lora, node

    def setUp(self):
        self.env = lora_synth.install(self)
        self.patcher = self.tiny.patcher
        distill_sd = lora_synth.make_lora_sd(self.patcher, fmt="kohya", modules=lora_synth.KOHYA_SPLIT, rank=8, seed=21)
        self.distill_sha = self.env.write("distill.safetensors", distill_sd, {"rank": "8", "alpha": "8.0"})
        self.char_sha = self.env.write("char.safetensors",
                                       lora_synth.make_lora_sd(self.patcher, seed=31, modules=("qkv", "o"), rank=4),
                                       {"ss_base_model_version": "moss_image"})

    def known(self, status="verified_recipe", trained_on="base"):
        row = {"name": "synthetic-distill", "role": "distill", "status": status, "trained_on": trained_on,
               "recipe": {"steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "shift": 5.0, "strength": 1.0}}
        if status == "incompatible":
            row.update(reason="data-space DMD with own sampler loop", reason_code="incompatible_lora")
        return mock.patch.dict(self.consts.KNOWN_DISTILL_LORAS, {self.distill_sha: row})

    def sft_identity(self, sha=None):
        ident = {"identity": {"sha256": sha or self.consts.SFT_BF16_SHA256}, "family": "moss-image-sft", "warnings": []}
        return mock.patch.object(self.node.fp, "identify_checkpoint", return_value=ident)

    def chain(self, distill_role="distill", distill_strength=1.0):
        m1, p1, _ = lora_synth.run_node(self.patcher, "distill.safetensors", role=distill_role, strength_model=distill_strength)
        m2, p2, _ = lora_synth.run_node(m1, "char.safetensors", role="character", strength_model=0.8, profile_in=p1)
        return m2, p2

    def codes(self, entry):
        return [(w["code"], w["severity"]) for w in entry["warnings"]]

    def test_known_distill_then_character_on_sft_is_valid_and_ordered(self):
        with self.known(), self.sft_identity(), self.assertLogs("MossPhoton", level="INFO") as logs:
            _m, profile = self.chain()
        self.assertEqual(self.lora.validate_identity_profile(profile), [])
        d, c = profile["stack"]
        self.assertEqual((d["index"], d["role"], c["index"], c["role"]), (0, "distill", 1, "character"))
        self.assertEqual(profile["summary"]["roles"], ["distill", "character"])
        self.assertEqual((profile["summary"]["has_distill"], profile["summary"]["has_character"]), (True, True))
        # Base-trained distill LoRA on an SFT model: info, not warning (owner Q13); verified, so no unverified warning
        self.assertEqual(self.codes(d), [("lora_base_mismatch", "info")])
        self.assertEqual((d["trained_on"], d["base_match"]), ("base", False))
        self.assertEqual(d["sidecar"], {"found": False, "curated": True, "dataset": None, "recommended_prompt": None,
                                        "verified_strength": {"min": 1.0, "max": 1.0, "recommended": 1.0}})
        self.assertEqual(self.codes(c), [])
        self.assertEqual(profile["model_identity"], {"family": "moss-image-sft", "sha256": self.consts.SFT_BF16_SHA256})
        # the order is journalled
        journal = [line for line in logs.output if "apply order is bit-relevant" in line]
        self.assertEqual(len(journal), 2)  # one line per node
        self.assertIn("0 distill distill.safetensors (1) -> 1 character char.safetensors (0.8)", journal[-1])
        self.assertNotIn("not the canonical", journal[-1])

    def test_character_before_distill_is_recorded_not_refused(self):
        with self.known(), self.sft_identity(), self.assertLogs("MossPhoton", level="INFO") as logs:
            m1, p1, _ = lora_synth.run_node(self.patcher, "char.safetensors", role="character", strength_model=0.8)
            _m2, p2, _ = lora_synth.run_node(m1, "distill.safetensors", role="distill", strength_model=1.0, profile_in=p1)
        self.assertEqual(p2["summary"]["roles"], ["character", "distill"])
        self.assertEqual(self.lora.validate_identity_profile(p2), [])
        self.assertTrue(any("not the canonical turbo order" in line for line in logs.output))

    def test_the_two_orders_are_different_recipes(self):
        with self.known(), self.sft_identity():
            _m, dc = self.chain()
            m1, p1, _ = lora_synth.run_node(self.patcher, "char.safetensors", role="character", strength_model=0.8)
            _m2, cd, _ = lora_synth.run_node(m1, "distill.safetensors", role="distill", strength_model=1.0, profile_in=p1)
        self.assertNotEqual(dc["digest"], cd["digest"])
        self.assertNotEqual(self.lora.attachment_json(dc["attachment"]), self.lora.attachment_json(cd["attachment"]))

    def test_unknown_file_declared_distill_warns_unverified(self):
        _m, profile, _ = lora_synth.run_node(self.patcher, "distill.safetensors", role="distill")
        self.assertEqual(self.codes(profile["stack"][0]), [("unverified_distill_lora", "warning")])
        self.assertEqual(profile["stack"][0]["sidecar"]["curated"], False)

    def test_fingerprint_off_cannot_verify_a_distill_file(self):
        with self.known():
            _m, profile, _ = lora_synth.run_node(self.patcher, "distill.safetensors", role="distill", fingerprint="off")
        self.assertIsNone(profile["stack"][0]["lora"]["sha256"])
        self.assertEqual(self.codes(profile["stack"][0]), [("unverified_distill_lora", "warning")])

    def test_distill_strength_outside_the_verified_value_warns(self):
        with self.known():
            _m, profile, _ = lora_synth.run_node(self.patcher, "distill.safetensors", role="distill", strength_model=0.5)
        self.assertEqual(self.codes(profile["stack"][0]), [("lora_strength_outside_verified_range", "warning")])

    def test_incompatible_hash_stops_before_the_file_is_read(self):
        with self.known(status="incompatible", trained_on="sft"):
            with mock.patch.object(self.lora, "load_lora_state", side_effect=AssertionError("file was read")):
                for role in ("distill", "character"):
                    with self.assertRaises(self.errors.IncompatibleLora) as ctx:
                        lora_synth.run_node(self.patcher, "distill.safetensors", role=role)
                    self.assertEqual(ctx.exception.code, "incompatible_lora")
                    self.assertIn("data-space DMD", str(ctx.exception))

    def test_second_distill_in_the_chain_is_incompatible_lora(self):
        with self.known(), self.sft_identity():
            m1, p1, _ = lora_synth.run_node(self.patcher, "distill.safetensors", role="distill")
            self.env.write("distill2.safetensors", lora_synth.make_lora_sd(
                self.patcher, fmt="kohya", modules=lora_synth.KOHYA_SPLIT, rank=8, seed=22))
            with self.assertRaises(self.errors.IncompatibleLora) as ctx:
                lora_synth.run_node(m1, "distill2.safetensors", role="distill", profile_in=p1)
            self.assertIn("distill LoRAs are used one at a time", str(ctx.exception))
            self.assertEqual(len(p1["stack"]), 1)

    def test_two_characters_warn(self):
        _m, p, _ = lora_synth.run_node(self.patcher, "char.safetensors")
        self.env.write("char2.safetensors", lora_synth.make_lora_sd(self.patcher, seed=32, modules=("qkv", "o")))
        m1, p1, _ = lora_synth.run_node(self.patcher, "char.safetensors")
        _m2, p2, _ = lora_synth.run_node(m1, "char2.safetensors", profile_in=p1)
        self.assertEqual(self.codes(p2["stack"][1]), [("multiple_character_loras", "warning")])
        self.assertEqual(self.codes(p2["stack"][0]), [])

    def test_same_file_twice_warns_duplicate(self):
        m1, p1, _ = lora_synth.run_node(self.patcher, "char.safetensors")
        _m2, p2, _ = lora_synth.run_node(m1, "char.safetensors", profile_in=p1)
        self.assertIn(("duplicate_lora", "warning"), self.codes(p2["stack"][1]))

    def test_distill_hash_in_another_role_is_logged(self):
        with self.known(), self.assertLogs("MossPhoton", level="WARNING") as logs:
            _m, profile, _ = lora_synth.run_node(self.patcher, "distill.safetensors", role="style")
        self.assertTrue(any("set role to distill" in line for line in logs.output))
        self.assertEqual(profile["stack"][0]["role"], "style")

    def test_base_trained_file_on_an_sft_model_is_a_warning_when_it_is_not_a_known_distill(self):
        meta = {"ss_base_model_version": "moss_image", "ss_sd_model_name": "x/moss_image_base_int8_convrot.safetensors"}
        self.env.write("basey.safetensors", lora_synth.make_lora_sd(self.patcher, seed=41), meta)
        with self.sft_identity():
            _m, profile, _ = lora_synth.run_node(self.patcher, "basey.safetensors")
        self.assertEqual(self.codes(profile["stack"][0]), [("lora_base_mismatch", "warning")])
        self.assertEqual((profile["stack"][0]["trained_on"], profile["stack"][0]["base_match"]), ("base", False))


if __name__ == "__main__":
    unittest.main()
