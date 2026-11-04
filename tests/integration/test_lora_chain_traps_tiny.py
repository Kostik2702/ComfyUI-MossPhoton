"""MP-73: setting conflicts in the MossPhotonLoRAProfile chain degrade to a warning instead of an exception (tiny MOSS, CPU).

Owner principle (2026-10-25): a conflict of settings -> automatic degradation + a warning in the report; raise only when
truly impossible (missing file, wrong model, corrupt LoRA). The live failure behind it: ``profile=phased`` on all six LoRA
nodes of a chain was ``unsupported_feature`` ("one phased adapter per chain"). The rules under test:

* one phased profile per chain and it stays on the character LoRA (the first one); every other phased request runs
  constant at its own ``strength_model`` with ``phased_lora_demoted`` (field names and stack indices only, no file names);
* the character node may come AFTER a node that already took the slot: that node's LoRA is re-applied constant on the
  character node's MODEL, its stack entry becomes constant, and the incoming MODEL / profile are never touched;
* phased with ``apply_text_encoder`` and a window with ``a >= b`` run constant; ``apply_text_encoder`` without a clip is
  ``lora_setting_ignored``.
"""
import copy

from . import lora_synth, sampler_support as ss
from .test_lora_phase_cpu import ADV, STEPS_O, PhaseCpuBase

ROLES6 = ("style", "wardrobe", "character", "scene", "other", "style")   # the owner's chain: character at position 3
CODE = "phased_lora_demoted"
DMD2 = "7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2"


def strength_of(i):
    return round(0.3 + 0.1 * i, 2)


class TrapsBase(PhaseCpuBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from moss_photon.core import lora
        cls.lora = lora

    def setUp(self):
        super().setUp()
        for i in range(6):
            self.env.write("L%d.safetensors" % i, lora_synth.make_lora_sd(self.patcher, seed=100 + i, modules=("qkv", "o")))

    def chain(self, roles=ROLES6, kinds=None, model=None):
        """The chain node by node, every node asking for ``profile=phased`` unless ``kinds`` says otherwise."""
        model, profile = self.patcher if model is None else model, None
        outs = []
        for i, role in enumerate(roles):
            kind = "phased" if kinds is None else kinds[i]
            model, profile, _ = self.node(
                model, "L%d.safetensors" % i, role=role, profile=kind, strength_model=strength_of(i),
                w0=0.3, w1=0.9, a=0.2, b=0.6, profile_in=profile)
            outs.append((model, profile))
        return model, profile, outs

    @staticmethod
    def demoted(profile):
        return {e["index"]: [w for w in e["warnings"] if w["code"] == CODE] for e in profile["stack"]}

    def kinds(self, profile):
        return [e["weight_profile"]["kind"] for e in profile["stack"]]


class OwnerGraphTest(TrapsBase):
    def test_six_phased_nodes_run_the_character_keeps_the_profile_the_others_are_constant(self):
        model, profile, _outs = self.chain()
        self.assertEqual(self.kinds(profile), ["constant", "constant", "phased", "constant", "constant", "constant"])
        self.assertEqual(profile["phase_plan"]["stack_index"], 2)
        self.assertEqual([e["strength_model"] for i, e in enumerate(profile["stack"]) if i != 2],
                         [strength_of(i) for i in (0, 1, 3, 4, 5)], "each demoted node runs at its own strength_model")
        self.assertEqual(self.lora.validate_identity_profile(profile), [])
        carried = self.lora_phase.adapter_of(model)
        self.assertEqual((carried.stack_index, carried.strength_model), (2, strength_of(2)))
        self.assertEqual(self.lora.patch_entries_of(model), profile["attachment"]["patch_entries_total"])
        self.assertGreater(self.lora.patch_entries_of(model), 0)

    def test_the_warning_sits_on_every_demoted_entry_and_names_no_file(self):
        _model, profile, _outs = self.chain()
        found = self.demoted(profile)
        self.assertEqual({i: len(w) for i, w in found.items()}, {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 1})
        for i, (warn,) in ((i, w) for i, w in found.items() if w):
            self.assertEqual(warn["param"], "profile")
            self.assertEqual(warn["detail"]["reason"], "another_phased_lora")
            self.assertEqual(warn["detail"]["index"], i)
            self.assertNotIn("L%d" % i, warn["message"] + warn["fix"], "file names are private")
            self.assertNotIn(".safetensors", warn["message"] + warn["fix"])

    def test_the_sampler_runs_it_and_the_report_has_the_warnings(self):
        from moss_photon.core import report as report_mod
        model, profile, _outs = self.chain()
        res = self.run_adv(model, profile)
        self.assertEqual(report_mod.validate_report(res.report), [])
        self.assertEqual(sorted(w["detail"]["stack_index"] for w in res.report["warnings"] if w["code"] == CODE),
                         [0, 1, 3, 4, 5])
        self.assertEqual(res.report["lora"]["phase"]["stack_index"], 2)
        self.assertEqual([s["role"] for s in res.report["lora"]["stack"]], list(ROLES6))

    def test_it_equals_the_chain_built_with_constant_nodes_in_the_same_patch_order(self):
        """The demoted nodes are ordinary constant LoRAs: node 1 patches first (node 0 is re-applied on the character
        node's MODEL), then node 0, 3, 4, 5. The reference chain builds exactly that order by hand."""
        model, profile, _outs = self.chain()
        got = self.run_adv(model, profile).latent["samples"]
        ref_model, ref_profile = self.patcher, None
        for i, kind in ((1, "constant"), (0, "constant"), (2, "phased"), (3, "constant"), (4, "constant"), (5, "constant")):
            ref_model, ref_profile, _ = self.node(
                ref_model, "L%d.safetensors" % i, role=ROLES6[i], profile=kind, strength_model=strength_of(i),
                w0=0.3, w1=0.9, a=0.2, b=0.6, profile_in=ref_profile)
        want = self.run_adv(ref_model, ref_profile).latent["samples"]
        self.assertTrue(self.equal(got, want))
        self.assertFalse(self.equal(got, self.run_adv(*self.chain(kinds=["constant"] * 6)[:2]).latent["samples"]),
                         "the character LoRA really is phased (ramps), not constant")

    def test_the_demoting_node_never_touches_its_inputs(self):
        _m, _p, outs = self.chain()
        model_in, profile_in = outs[1]            # what node 2 (the character) receives
        profile_before = copy.deepcopy(profile_in)
        patches_before = self.lora.patch_entries_of(model_in)
        adapter_before = self.lora_phase.adapter_of(model_in)
        self.node(model_in, "L2.safetensors", role="character", profile="phased", strength_model=0.5, profile_in=profile_in)
        self.assertEqual(profile_in, profile_before, "the incoming profile still says node 0 is phased")
        self.assertEqual(self.lora.patch_entries_of(model_in), patches_before)
        self.assertIs(self.lora_phase.adapter_of(model_in), adapter_before)
        self.assertEqual(profile_in["stack"][0]["weight_profile"]["kind"], "phased")


class SlotRulesTest(TrapsBase):
    def test_the_first_of_several_phased_characters_keeps_the_profile(self):
        _m, profile, _o = self.chain(roles=("character", "character", "style", "character"))
        self.assertEqual(self.kinds(profile), ["phased", "constant", "constant", "constant"])
        self.assertEqual(profile["phase_plan"]["stack_index"], 0)
        self.assertEqual([i for i, w in self.demoted(profile).items() if w], [1, 2, 3])
        self.assertIn("multiple_character_loras", [w["code"] for w in profile["stack"][1]["warnings"]])
        self.assertEqual(self.lora.validate_identity_profile(profile), [])

    def test_a_character_later_takes_the_profile_from_an_earlier_non_character(self):
        model, profile, _o = self.chain(roles=("style", "character"))
        self.assertEqual(self.kinds(profile), ["constant", "phased"])
        self.assertEqual(profile["phase_plan"]["stack_index"], 1)
        self.assertEqual(self.lora_phase.adapter_of(model).stack_index, 1)
        self.assertEqual(self.lora.validate_identity_profile(profile), [])
        self.assertEqual(self.lora.patch_entries_of(model), profile["stack"][0]["patch_entries"])

    def test_without_any_character_the_first_phased_request_keeps_it(self):
        _m, profile, _o = self.chain(roles=("style", "scene", "other"))
        self.assertEqual(self.kinds(profile), ["phased", "constant", "constant"])
        self.assertEqual(profile["phase_plan"]["stack_index"], 0)

    def test_a_known_distill_file_is_not_a_character_lora(self):
        from unittest import mock
        name = "lodestones-8-steps-dmd2-lora.safetensors"
        self.env.write(name, lora_synth.make_lora_sd(self.patcher, seed=7, modules=("qkv", "o")))
        real = self.lora.describe_file

        def known_hash(path, **kw):   # the file is synthetic; its sha256 is the real distill hash (as in the MP-60 tests)
            info = real(path, **kw)
            return dict(info, sha256=DMD2) if info["basename"] == name else info
        with mock.patch.object(self.lora, "describe_file", known_hash):
            m0, p0, _ = self.node(self.patcher, name, role="character", profile="phased")
            self.assertEqual(self.kinds(p0), ["phased"])
            m1, p1, _ = self.node(m0, "L1.safetensors", role="character", profile="phased", profile_in=p0)
        self.assertEqual(self.kinds(p1), ["constant", "phased"], "a real character LoRA outranks the distill file")
        self.assertEqual(p1["phase_plan"]["stack_index"], 1)
        self.assertEqual(self.lora.validate_identity_profile(p1), [])

    def test_constant_nodes_around_the_phased_one_are_untouched(self):
        _m, profile, _o = self.chain(roles=("distill", "character", "style"), kinds=("constant", "phased", "constant"))
        self.assertEqual(self.kinds(profile), ["constant", "phased", "constant"])
        self.assertEqual([w for e in profile["stack"] for w in e["warnings"] if w["code"] == CODE], [])


class SettingConflictsTest(TrapsBase):
    def test_phased_with_the_text_encoder_runs_constant_with_the_encoder_applied(self):
        sd = lora_synth.make_lora_sd(self.patcher, seed=3, modules=("qkv", "o"))
        sd.update(lora_synth.te_lora_sd())
        self.env.write("WithTe.safetensors", sd)
        clip = lora_synth.FakeClip()
        out, profile, new_clip = self.node(
            self.patcher, "WithTe.safetensors", profile="phased", strength_model=0.8, apply_text_encoder=True,
            clip=clip, strength_clip=0.7)
        entry = profile["stack"][0]
        self.assertEqual(entry["weight_profile"], {"kind": "constant", "w": 0.8})
        self.assertIsNone(profile["phase_plan"])
        self.assertEqual(entry["applied_to"], {"transformer": True, "text_encoder": True})
        self.assertEqual(sorted(new_clip.patcher.patches), ["enc.proj.weight"])
        self.assertIsNone(self.lora_phase.adapter_of(out))
        (warn,) = [w for w in entry["warnings"] if w["code"] == CODE]
        self.assertEqual(warn["detail"]["reason"], "apply_text_encoder")
        self.assertEqual(self.lora.validate_identity_profile(profile), [])

    def test_a_window_with_a_not_below_b_runs_constant(self):
        for a, b in ((0.6, 0.2), (0.4, 0.4)):
            with self.subTest(a=a, b=b):
                out, profile, _ = self.node(self.patcher, "L0.safetensors", profile="phased", strength_model=0.7, a=a, b=b)
                entry = profile["stack"][0]
                self.assertEqual(entry["weight_profile"], {"kind": "constant", "w": 0.7})
                self.assertIsNone(profile["phase_plan"])
                (warn,) = [w for w in entry["warnings"] if w["code"] == CODE]
                self.assertEqual((warn["detail"]["reason"], warn["detail"]["a"], warn["detail"]["b"]),
                                 ("invalid_window", a, b))
                self.assertGreater(self.lora.patch_entries_of(out), 0)
                self.assertEqual(self.lora.validate_identity_profile(profile), [])

    def test_a_demoted_request_equals_the_constant_node(self):
        const_model, const_profile, _ = self.node(self.patcher, "L0.safetensors", strength_model=0.7)
        demoted_model, demoted_profile, _ = self.node(self.patcher, "L0.safetensors", profile="phased", strength_model=0.7,
                                                      a=0.6, b=0.2)
        want = self.run_const_model(const_model, const_profile)
        got = self.run_const_model(demoted_model, demoted_profile)
        self.assertTrue(self.equal(got, want))

    def run_const_model(self, model, profile):
        return ss.run(self.tiny, model=model, identity_profile=profile, overrides=STEPS_O).latent["samples"]

    def test_the_text_encoder_flag_without_a_clip_is_ignored_with_a_warning(self):
        sd = lora_synth.make_lora_sd(self.patcher, seed=3, modules=("qkv", "o"))
        sd.update(lora_synth.te_lora_sd())
        self.env.write("WithTe.safetensors", sd)
        out, profile, out_clip = self.node(self.patcher, "WithTe.safetensors", profile="phased", apply_text_encoder=True,
                                           strength_clip=0.7)
        entry = profile["stack"][0]
        self.assertIsNone(out_clip)
        self.assertIn("lora_setting_ignored", [w["code"] for w in entry["warnings"]])
        # no clip means no text encoder, so the phased request itself is fine
        self.assertEqual(entry["weight_profile"]["kind"], "phased")
        self.assertEqual(self.lora.validate_identity_profile(profile), [])

    def test_what_is_truly_impossible_still_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.node(self.patcher, "missing.safetensors", profile="phased")
        m, p, _ = self.phased(role="style")
        stripped = m.clone()
        stripped.remove_attachments(self.lora_phase.ATTACH_PHASE_LORA)
        with self.assertRaises(self.errors.IncompatibleLora):
            # the profile says node 0 is phased but its adapter is gone from the MODEL: nothing to demote
            self.node(stripped, "L1.safetensors", role="character", profile="phased", profile_in=p)
        with self.assertRaises(self.errors.IncompatibleLora):
            self.node(self.patcher, "L1.safetensors", profile_in=p)   # the profile of another MODEL
