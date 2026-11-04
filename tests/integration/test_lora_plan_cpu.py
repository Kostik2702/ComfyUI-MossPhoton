"""MP-84: MossPhotonLoRAPhase on the tiny 24-layer MOSS (CPU, synthetic LoRA files, no weights).

The guarantees of the ticket, each with the test that goes red when it is violated:

(a) the node patches nothing, validates the file (hard errors for a missing, unmappable or broken one) and chains by MODEL ->
    ``NodeTest``; a MODEL that carries a plan leaks nothing into the MODEL it came from, in a run or after it -> ``NoLeakTest``;
(b) a preset without Darkroom phases applies every entry for the whole run: ``all`` equals the ordinary loader bit for bit, any
    other phase equals it too and says so (``phase_plan_ignored``); a run with no plan is the run of a model without the pack's
    node -> ``ConstantRunTest``;
(c) a Darkroom preset applies each entry in the phases the owner's table names, the identity and composition+identity entries take the commit boost (``all`` is an ordinary loader and takes none), and
    the result equals a chain of stock CFGGuider + euler segments on stock LoRA-loader models -> ``DarkroomPhasesTest``;
(d) the segment clones are DIRECT clones of the incoming MODEL and two neighbouring segments with the same LoRA set share one ->
    ``DarkroomPhasesTest.test_neighbouring_segments_with_the_same_set_share_one_level_clone``;
(e) legacy phased profiles, second distill LoRAs and missing roles degrade to warnings -> ``DegradationTest``.

The references are built from stock comfy parts (``comfy.sd.load_lora_for_models`` = what LoraLoaderModelOnly does, ``CFGGuider``
+ euler and a hand-written resume), never from ``core/darkroom.py`` or ``core/lora_plan.py``.
"""
import copy
import gc
import unittest
from unittest import mock

from . import graph_cache, lora_synth, test_darkroom_cpu as dk_tests, tiny_moss
from . import sampler_support as ss

PORTRAIT = dk_tests.PORTRAIT
SEED = tiny_moss.SEED
TURBO = "turbo_8step"


class PlanBase(dk_tests.DarkroomBase):
    chain = dk_tests.HandoverTest.chain

    def setUp(self):
        super().setUp()
        from moss_photon.core import consts, lora_plan
        from moss_photon.nodes.lora_plan import MossPhotonLoRAPhase
        self.consts, self.lora_plan, self.node = consts, lora_plan, MossPhotonLoRAPhase()

    # ---- nodes
    def phase(self, model, name="A.safetensors", strength=1.0, phase="identity"):
        return self.node.apply(model, name, strength, phase)[0]

    def native(self, model, sd, strength=1.0):
        """What a stock LoraLoaderModelOnly does with the file."""
        return ss.apply_lora(model, sd, strength)

    # ---- runs
    def run_plain(self, model, preset=TURBO, **kw):
        """The solver the way the nodes call it, with the plan loader injected (``cfg`` = the preset's own unless given)."""
        cfg = kw.pop("cfg", None)
        if cfg is None:
            cfg = self.presets.catalog()[preset]["sampling"]["cfg"]
        with ss.known_sft():
            return ss.run(self.tiny, model=model, preset_id=preset, cfg=cfg, lora_loader=self.lora_core.plan_patches, **kw)

    def run_phased(self, model, preset=PORTRAIT, finish_model=None, split_at=(), profile=None, **knobs):
        """The darkroom the way ``run_dk`` drives it, with the plan loader injected."""
        knobs.setdefault("w_commit", dk_tests.PINNED_W_COMMIT)
        compose = knobs.pop("compose_cfg", 1.0)
        with ss.known_sft():
            return ss.run(
                self.tiny, model=model, preset_id=preset, cfg=compose, node_id=dk_tests.NODE, allow_phased_lora=True,
                preview=False, positive=self.tiny.make_cond(23), negative=self.tiny.make_cond(9),
                lora_loader=self.lora_core.plan_patches, identity_profile=profile,
                darkroom=self.darkroom.DarkroomRequest(overrides=knobs, finish_model=finish_model,
                                                       split_at=tuple(split_at), boost_loader=self.lora_core.boost_patches))

    def codes(self, out):
        return [w["code"] for w in self.report(out)["warnings"]]


class NodeTest(PlanBase):
    def test_the_node_patches_nothing_and_records_the_file(self):
        before = ss.entries(self.patcher), str(self.patcher.patches_uuid)
        out = self.phase(self.patcher, strength=0.8, phase="identity")
        self.assertIsNot(out, self.patcher)
        self.assertEqual((ss.entries(out), ss.entries(self.patcher), str(self.patcher.patches_uuid)),
                         (0, before[0], before[1]), "no patch on the new MODEL, the input is untouched")
        self.assertEqual(self.lora_plan.entries_of(self.patcher), ())
        [entry] = self.lora_plan.entries_of(out)
        self.assertEqual((entry["lora_name"], entry["strength"], entry["phase"], entry["basename"]),
                         ("A.safetensors", 0.8, "identity", "A.safetensors"))
        self.assertEqual(entry["sha256"], self.env.written["A.safetensors"])
        self.assertEqual(entry["size"], len(open(self.env.path("A.safetensors"), "rb").read()))
        self.assertEqual(entry["warnings"], [])

    def test_nodes_chain_by_model_and_keep_their_order(self):
        one = self.phase(self.patcher, "A.safetensors", 0.5, "composition")
        two = self.phase(one, "D.safetensors", -1.5, "texture")
        self.assertEqual([(e["lora_name"], e["strength"], e["phase"]) for e in self.lora_plan.entries_of(two)],
                         [("A.safetensors", 0.5, "composition"), ("D.safetensors", -1.5, "texture")])
        self.assertEqual(len(self.lora_plan.entries_of(one)), 1)

    def test_none_and_zero_strength_pass_the_same_model_through(self):
        self.assertIs(self.phase(self.patcher, "none"), self.patcher)
        self.assertIs(self.phase(self.patcher, strength=0.0), self.patcher)

    def test_a_missing_file_is_the_one_comfy_error(self):
        with self.assertRaises(FileNotFoundError):
            self.phase(self.patcher, "nope.safetensors")

    def test_a_file_that_maps_nothing_is_incompatible(self):
        sd = {k: v for k, v in lora_synth.add_bogus_modules({}, 6).items()}
        self.env.write("bogus.safetensors", sd)
        with self.assertRaises(self.errors.IncompatibleLora):
            self.phase(self.patcher, "bogus.safetensors")

    def test_non_finite_factors_are_incompatible(self):
        sd = lora_synth.make_lora_sd(self.patcher, seed=3)
        key = next(k for k in sd if k.endswith("lora_A.weight"))
        sd[key] = sd[key].clone()
        sd[key][0, 0] = float("nan")
        self.env.write("nan.safetensors", sd)
        with self.assertRaises(self.errors.IncompatibleLora):
            self.phase(self.patcher, "nan.safetensors")

    def test_a_trainer_flag_of_one_is_incompatible_and_a_flag_of_zero_is_dropped(self):
        import torch
        sd = lora_synth.make_lora_sd(self.patcher, seed=4)
        self.env.write("rs1.safetensors", dict(sd, **{"diffusion_model.layers.0.proj.qkv.use_rslora_flag": torch.tensor(1.0)}))
        with self.assertRaises(self.errors.IncompatibleLora):
            self.phase(self.patcher, "rs1.safetensors")
        self.env.write("rs0.safetensors", dict(sd, **{"diffusion_model.layers.0.proj.qkv.use_rslora_flag": torch.tensor(0.0)}))
        out = self.phase(self.patcher, "rs0.safetensors")
        self.assertEqual(self.lora_plan.entries_of(out)[0]["warnings"], [], "the flag is not an unmapped key")

    def test_a_partly_mapped_file_is_a_warning_on_the_entry_and_in_the_report(self):
        sd = lora_synth.add_bogus_modules(lora_synth.make_lora_sd(self.patcher, seed=5, modules=("qkv", "o")), 30)
        self.env.write("partial.safetensors", sd)
        out = self.phase(self.patcher, "partial.safetensors")
        [entry] = self.lora_plan.entries_of(out)
        self.assertEqual([w["code"] for w in entry["warnings"]], ["partial_lora_coverage"])
        noted = [w for w in self.report(self.run_plain(out))["warnings"] if w["code"] == "partial_lora_coverage"]
        self.assertEqual([w["detail"]["plan_index"] for w in noted], [0], "the sampler's report carries it, with its index")

    def test_the_unsupported_data_space_dmd_file_is_refused_in_any_phase(self):
        sd = lora_synth.make_lora_sd(self.patcher, seed=6)
        sha = self.env.write("sftdmd.safetensors", sd)
        row = dict(self.consts.KNOWN_DISTILL_LORAS["4abb14e39c1a59018907581e609f43616ec6810126144ec2b1d2e6f8969628fe"])
        with mock.patch.dict(self.consts.KNOWN_DISTILL_LORAS, {sha: row}):
            with self.assertRaises(self.errors.IncompatibleLora):
                self.phase(self.patcher, "sftdmd.safetensors", phase="texture")

    def test_the_plan_does_not_survive_into_the_stock_sampler(self):
        """Another sampler sees the MODEL without the LoRA: the plan is not a patch."""
        out = self.phase(self.patcher)
        tiny_moss.sample_euler(self.tiny)
        plain = ss.tensor_sha(tiny_moss.sample_euler(self.tiny))
        self.assertEqual(ss.tensor_sha(tiny_moss.sample_euler(self.tiny, out)), plain)


class NoLeakTest(PlanBase):
    def snapshot(self, model):
        return {"patches": sorted(model.patches), "uuid": str(model.patches_uuid),
                "options": copy.deepcopy(model.model_options), "attachments": sorted(model.attachments),
                "hooks": (dict(model.hook_patches), model.current_hooks)}

    def test_a_run_changes_neither_the_plan_model_nor_the_model_it_came_from(self):
        one = self.phase(self.patcher, "A.safetensors", 1.0, "identity")
        two = self.phase(one, "D.safetensors", 0.7, "texture")
        before = {n: self.snapshot(m) for n, m in (("base", self.patcher), ("one", one), ("two", two))}
        self.run_phased(two)
        self.run_plain(two)
        gc.collect()
        after = {n: self.snapshot(m) for n, m in (("base", self.patcher), ("one", one), ("two", two))}
        self.assertEqual(after, before)
        self.assertEqual(len(self.patcher.patches), 0)

    def test_the_plan_never_reaches_the_input_model_options(self):
        two = self.phase(self.phase(self.patcher), "D.safetensors", 1.0, "all")
        self.assertNotIn(self.lora_plan.PLAN_KEY, self.patcher.model_options)
        self.assertEqual(len(self.lora_plan.entries_of(two)), 2)

    def test_a_run_repeats_bit_equal_and_the_base_model_is_clean_afterwards(self):
        two = self.phase(self.phase(self.patcher, phase="composition"), "D.safetensors", 1.0, "texture")
        first = ss.tensor_sha(self.samples(self.run_phased(two)))
        base = ss.tensor_sha(tiny_moss.sample_euler(self.tiny))
        again = ss.tensor_sha(self.samples(self.run_phased(two)))
        self.assertEqual(first, again)
        self.assertEqual(ss.tensor_sha(tiny_moss.sample_euler(self.tiny)), base, "no LoRA stayed on the shared module")


class SamplingPatcherTest(PlanBase):
    """MP-84 review MAJOR-1, MP-88: the patcher a run samples from is a DIRECT clone of the incoming MODEL (a clone of a throwaway
    clone leaves a dead ``LoadedModel`` behind: the MP-82 / KreaPhoton leak), and no ``LoadedModel`` is dead after the run, also
    after an interrupt. ``graph_cache.install()`` makes a dead entry possible at all: without a reference cycle round the
    sampling patcher it dies by refcount and its throwaway parent unwinds behind it, so the count alone would never see it."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        graph_cache.install()

    def dead(self):
        gc.collect()
        return sum(1 for m in self.mm.current_loaded_models if m.is_dead())

    def interrupting(self, at_step):
        """``solver.make_callback`` whose callback raises the interrupt ComfyUI raises, at step ``at_step``."""
        real = self.solver.make_callback

        def make(*a, **kw):
            inner = real(*a, **kw)

            def callback(step, x0, x, total):
                if step == at_step:
                    raise self.mm.InterruptProcessingException()
                return inner(step, x0, x, total)
            callback.preview_active = getattr(inner, "preview_active", False)
            return callback
        return make

    def sampled_from(self, incoming, run, interrupt_at=None):
        """For the patchers ``build_guider`` received while ``run()`` ran: whether each was a direct clone of ``incoming``, whether
        it carried the SLG hook, and the dead ``LoadedModel`` count before and after. Neither the patchers nor their parents are
        kept: a held throwaway parent would hide the dead ``LoadedModel`` it causes. ``interrupt_at``: the run is interrupted
        at that step (the interrupt must propagate)."""
        real, direct, hooked = self.solver.build_guider, [], []

        def spy(patcher, *a, **kw):
            direct.append(patcher.parent is incoming)
            hooked.append(bool(self.solver.guidance.cond_hook_slot(patcher)))
            return real(patcher, *a, **kw)

        gc.collect()                                # dead entries earlier tests of this process left behind are not this run's
        self.mm.current_loaded_models[:] = [m for m in self.mm.current_loaded_models if not m.is_dead()]
        before = self.dead()
        self.assertEqual(before, 0, "precondition: nothing is dead before the run")
        with mock.patch.object(self.solver, "build_guider", spy):
            for _ in range(2):
                if interrupt_at is None:
                    run()
                    continue
                with mock.patch.object(self.solver, "make_callback", self.interrupting(interrupt_at)):
                    with self.assertRaises(self.mm.InterruptProcessingException):
                        run()
        return direct, hooked, before, self.dead()

    def assert_direct_and_clean(self, incoming, run, interrupt_at=None):
        direct, hooked, before, after = self.sampled_from(incoming, run, interrupt_at)
        self.assertTrue(direct)
        problems = []
        if not all(direct):
            problems.append("the patcher that samples is a clone of a clone of the incoming MODEL")
        if after != before:
            problems.append("%d dead LoadedModel left behind" % (after - before))
        self.assertEqual(problems, [])
        return hooked

    def test_a_plan_on_a_preset_without_darkroom_phases_samples_from_a_direct_clone(self):
        model = self.phase(self.patcher, phase="all")
        self.assert_direct_and_clean(model, lambda: self.run_plain(model))

    def test_a_plan_with_slg_samples_from_a_direct_clone(self):
        model = self.phase(self.patcher, phase="all")
        self.assert_direct_and_clean(model, lambda: self.run_plain(model, preset="identity_safe"))

    def test_a_phased_profile_demoted_on_the_basic_sampler_samples_from_a_direct_clone(self):
        m, p, _ = self.lora(self.patcher, "A.safetensors", role="character", profile="phased", w0=0.1, w1=0.9,
                            strength_model=0.7)
        self.assert_direct_and_clean(m, lambda: self.run_plain(m, identity_profile=p, node_id="MossPhotonSampler"))

    def test_a_plan_with_a_phased_profile_on_the_advanced_sampler_equals_a_stock_loader_below_the_phased_one(self):
        m, p, _ = self.lora(self.patcher, "A.safetensors", role="character", profile="phased", w0=0.3, w1=0.9, a=0.2, b=0.6)
        model = self.phase(m, "D.safetensors", 0.9, "all")
        run = dict(identity_profile=p, node_id="MossPhotonSamplerAdvanced", allow_phased_lora=True)
        out = self.run_plain(model, **run)
        d_sd = lora_synth.make_lora_sd(self.patcher, seed=21, modules=("qkv", "o", "ffn_down"))
        m_ref, p_ref, _ = self.lora(self.native(self.patcher, d_sd, 0.9), "A.safetensors", role="character",
                                    profile="phased", w0=0.3, w1=0.9, a=0.2, b=0.6)
        ref = self.run_plain(m_ref, **dict(run, identity_profile=p_ref))
        self.assertIn("phase", self.report(out)["lora"], "control: the run was phased")
        self.assertTrue(self.equal(self.samples(out), self.samples(ref)))
        self.assert_direct_and_clean(model, lambda: self.run_plain(model, **run))

    def test_a_phased_profile_with_slg_on_the_advanced_sampler_samples_from_a_direct_clone(self):
        m, p, _ = self.lora(self.patcher, "A.safetensors", role="character", profile="phased", w0=0.1, w1=0.9, strength_model=0.7)
        run = dict(identity_profile=p, node_id="MossPhotonSamplerAdvanced", allow_phased_lora=True, preset="identity_safe")
        hooked = self.assert_direct_and_clean(m, lambda: self.run_plain(m, **run))
        self.assertTrue(all(hooked), "control: every segment sampled with the SLG hook")

    def test_a_darkroom_segment_with_slg_samples_from_a_direct_clone(self):
        model = self.phase(self.patcher, phase="all")
        hooked = self.assert_direct_and_clean(model, lambda: self.run_phased(model, compose_slg=True, compose_cfg=1.5))
        self.assertIn(True, hooked, "control: a segment sampled with the SLG hook")
        self.assertIn(False, hooked, "control: a segment without it")

    def test_a_darkroom_run_with_slg_interrupted_in_the_compose_segment_leaves_nothing_dead(self):
        model = self.phase(self.patcher, phase="all")
        hooked = self.assert_direct_and_clean(
            model, lambda: self.run_phased(model, compose_slg=True, compose_cfg=1.5), interrupt_at=1)
        self.assertEqual(hooked, [True, True], "control: the interrupted segment was the SLG one, in both runs")

    def test_a_darkroom_run_without_a_plan_and_with_slg_samples_from_a_direct_clone_after_an_interrupt(self):
        self.assert_direct_and_clean(
            self.patcher, lambda: self.run_phased(self.patcher, compose_slg=True, compose_cfg=1.5), interrupt_at=1)

    def test_the_slg_hook_does_not_stay_on_a_level_clone_the_next_segment_shares(self):
        """The darkroom hands two neighbouring segments with the same LoRA set the same level clone (one weight patch). The
        SLG hook is put on that clone itself, so it must be taken off again when its segment ends."""
        model = self.phase(self.patcher, phase="all")
        real, slots, built = self.lora_plan.apply_to, [], []

        def apply_spy(target, *a, **kw):
            built.append(1)
            return real(target, *a, **kw)

        def guider_spy(patcher, *a, **kw):
            slots.append(bool(self.solver.guidance.cond_hook_slot(patcher)))
            return real_guider(patcher, *a, **kw)

        real_guider = self.solver.build_guider
        with mock.patch.object(self.lora_plan, "apply_to", apply_spy), mock.patch.object(self.solver, "build_guider", guider_spy):
            out = self.run_phased(model, compose_slg=True, compose_cfg=1.5, w_commit=1.0)
        segments = self.report(out)["darkroom"]["segments"]
        self.assertEqual(slots, [s["slg"] for s in segments], "each segment sampled with the hook exactly when it is an SLG one")
        self.assertIn(True, slots)
        self.assertIn(False, slots)
        self.assertLess(len(built), len(segments), "control: two segments shared a level clone")


class ConstantRunTest(PlanBase):
    """A preset without Darkroom phases (``turbo_8step``, 8 steps at cfg 1)."""

    def reference(self, model):
        return self.samples(self.run_plain(model))

    def test_all_equals_the_ordinary_loader_bit_for_bit(self):
        expected = self.reference(self.native(self.patcher, self.a_sd))
        out = self.run_plain(self.phase(self.patcher, phase="all"))
        self.assertTrue(self.equal(self.samples(out), expected))
        plain = self.reference(self.patcher)
        self.assertFalse(self.equal(plain, expected), "control: the LoRA changes the latent")
        self.assertNotIn("phase_plan_ignored", self.codes(out))

    def test_every_other_phase_runs_constant_and_says_so(self):
        expected = self.reference(self.native(self.patcher, self.a_sd))
        for phase in ("composition", "texture", "composition+identity"):
            with self.subTest(phase=phase):
                out = self.run_plain(self.phase(self.patcher, phase=phase))
                self.assertTrue(self.equal(self.samples(out), expected))
                warn = next(w for w in self.report(out)["warnings"] if w["code"] == "phase_plan_ignored")
                self.assertEqual(warn["detail"]["phases"], [phase])
                self.assertEqual(warn["severity"], "warning")

    def test_identity_is_the_whole_run_on_a_preset_without_phases_and_says_nothing(self):
        """MP-86: identity acts in every darkroom segment, so there is nothing to ignore."""
        out = self.run_plain(self.phase(self.patcher, phase="identity"))
        self.assertTrue(self.equal(self.samples(out), self.reference(self.native(self.patcher, self.a_sd))))
        self.assertNotIn("phase_plan_ignored", self.codes(out))

    def test_a_chain_applies_in_node_order_like_two_loaders(self):
        d_sd = lora_synth.make_lora_sd(self.patcher, seed=21, modules=("qkv", "o", "ffn_down"))
        expected = self.reference(self.native(self.native(self.patcher, self.a_sd, 0.6), d_sd, 0.9))
        out = self.run_plain(self.phase(self.phase(self.patcher, "A.safetensors", 0.6, "all"), "D.safetensors", 0.9, "all"))
        self.assertTrue(self.equal(self.samples(out), expected))
        swapped = self.reference(self.native(self.native(self.patcher, d_sd, 0.9), self.a_sd, 0.6))
        self.assertFalse(self.equal(expected, swapped), "control: the order is bit-relevant")

    def test_a_stock_loader_and_a_plan_entry_add_up(self):
        ordinary = self.native(self.patcher, self.a_sd, 0.5)
        d_sd = lora_synth.make_lora_sd(self.patcher, seed=21, modules=("qkv", "o", "ffn_down"))
        expected = self.reference(self.native(ordinary, d_sd, 0.9))
        out = self.run_plain(self.phase(ordinary, "D.safetensors", 0.9, "all"))
        self.assertTrue(self.equal(self.samples(out), expected))
        self.assertNotIn("lora_not_recorded", self.codes(out))
        self.assertNotIn("lora_plan_after_patches", self.codes(out), "the loader is before the node: graph order")

    def test_a_loader_after_a_plan_entry_runs_before_it_and_the_report_says_so(self):
        """MP-84 review MAJOR-2: plan entries are applied on top of every patch of the MODEL, whatever the node order. A graph
        ``LoRA Phase(A) -> LoraLoader(D) -> sampler`` runs D, then A; the info says so (and documents it, docs/USAGE.md 3.3)."""
        d_sd = lora_synth.make_lora_sd(self.patcher, seed=21, modules=("qkv", "o", "ffn_down"))
        graph_order = self.reference(self.native(self.native(self.patcher, self.a_sd, 0.6), d_sd, 0.9))
        applied_order = self.reference(self.native(self.native(self.patcher, d_sd, 0.9), self.a_sd, 0.6))
        self.assertFalse(self.equal(graph_order, applied_order), "control: the two orders differ bit for bit")
        out = self.run_plain(self.native(self.phase(self.patcher, "A.safetensors", 0.6, "all"), d_sd, 0.9))
        self.assertTrue(self.equal(self.samples(out), applied_order))
        [note] = [w for w in self.report(out)["warnings"] if w["code"] == "lora_plan_after_patches"]
        self.assertEqual((note["severity"], note["detail"]["plan_indices"], note["param"]), ("info", [0], "model"))

    def test_plan_nodes_one_after_another_need_no_order_note(self):
        out = self.run_plain(self.phase(self.phase(self.patcher, "A.safetensors", 0.6, "all"), "D.safetensors", 0.9, "all"))
        self.assertNotIn("lora_plan_after_patches", self.codes(out))
        late = self.phase(self.native(self.patcher, self.a_sd, 0.5), "D.safetensors", 0.9, "all")
        self.assertNotIn("lora_plan_after_patches", self.codes(self.run_plain(late)))

    def test_a_negative_strength_is_the_ordinary_negative_strength(self):
        expected = self.reference(self.native(self.patcher, self.a_sd, -2.0))
        self.assertTrue(self.equal(self.reference(self.phase(self.patcher, strength=-2.0, phase="all")), expected))

    def test_the_report_records_the_plan_and_the_run_is_labelled(self):
        out = self.run_plain(self.phase(self.phase(self.patcher, phase="texture"), "D.safetensors", 0.4, "all"))
        rep = self.report(out)
        self.assertEqual([(e["basename"], e["strength"], e["phase"], e["applied_in"]) for e in rep["lora"]["plan"]],
                         [("A.safetensors", 1.0, "texture", ["run"]), ("D.safetensors", 0.4, "all", ["run"])])
        self.assertEqual(rep["lora"]["plan"][0]["sha256"], self.env.written["A.safetensors"])
        self.assertEqual(rep["lora"]["stack"], [])
        self.assertEqual(rep["label"]["parity_class"], "native_turbo")
        plain = self.report(self.run_plain(self.patcher))
        self.assertNotIn("plan", plain["lora"], "a run without a plan has no plan key")
        self.assertNotEqual(rep["recipe_hash"], plain["recipe_hash"])

    def test_a_run_with_no_plan_equals_the_stock_turbo_run(self):
        out = self.run_plain(self.patcher)
        stock = ss.stock_run(self.tiny, self.patcher, positive=self.tiny.make_cond(23), negative=self.tiny.make_cond(9),
                             steps=8, cfg=1.0, scheduler="simple")
        self.assertTrue(self.equal(self.samples(out), stock.to(self.samples(out).dtype)))

    def test_zero_strength_entries_are_not_applied(self):
        """An entry that reached the plan with strength 0 (an API caller) loads nothing and changes nothing."""
        plan = self.lora_plan.add_to_plan(self.patcher, self.lora_plan.make_entry(
            "A.safetensors", 0.0, "all", "A.safetensors", 1, None))
        self.assertTrue(self.equal(self.reference(plan), self.reference(self.patcher)))


class DarkroomPhasesTest(PlanBase):
    def levels(self):
        return dk_tests.textbook_applied(self.sigmas12.tolist(), 1.0, dk_tests.PINNED_W_COMMIT, 0.06, 0.17)

    def expected_chain(self, strengths):
        """strengths: the LoRA strength of every one of the 12 steps (``None`` = the bare model), cfg 1 for steps 0..6 and
        2 from step 7 (the portrait recipe); consecutive equal steps are one stock segment."""
        cfgs = [1.0] * 7 + [2.0] * 5
        runs, start = [], 0
        for k in range(1, 13):
            if k == 12 or (strengths[k], cfgs[k]) != (strengths[start], cfgs[start]):
                runs.append((strengths[start], cfgs[start], start, k))
                start = k
        segments = [(self.patcher if s is None else self.native(self.patcher, self.a_sd, s), cfg, first, stop)
                    for s, cfg, first, stop in runs]
        return self.chain(segments), runs

    def test_each_phase_runs_exactly_where_the_owners_table_puts_it(self):
        # the phases that take no commit boost (composition, texture) run at a constant strength; the boosted one
        # (identity) is pinned in ``test_identity_*`` below, composition+identity in ``test_composition_plus_identity_*``
        table = {"composition": [0, 1, 2], "texture": [7, 8, 9, 10, 11]}
        for phase, steps in table.items():
            with self.subTest(phase=phase):
                out = self.run_phased(self.phase(self.patcher, phase=phase))
                expected, _ = self.expected_chain([1.0 if k in steps else None for k in range(12)])
                self.assertTrue(self.equal(self.samples(out), expected.to(self.samples(out).dtype)))
                rep = self.report(out)
                self.assertEqual(rep["lora"]["plan"][0]["applied_in"],
                                 [p for p in ("compose", "commit", "hold", "finish")
                                  if p in self.lora_plan.PHASE_SEGMENTS[phase] and p != "hold"])

    def test_all_equals_the_ordinary_loader_through_the_whole_darkroom_and_takes_no_boost(self):
        """MP-86: ``all`` is exactly an ordinary loader: no commit boost, even though the preset asks for one."""
        out = self.run_phased(self.phase(self.patcher, phase="all"))
        ordinary = self.run_phased(self.native(self.patcher, self.a_sd))
        self.assertTrue(self.equal(self.samples(out), self.samples(ordinary)))
        expected, _ = self.expected_chain([1.0] * 12)
        self.assertTrue(self.equal(self.samples(out), expected.to(self.samples(out).dtype)))
        rep = self.report(out)["darkroom"]["character"]
        self.assertEqual(rep["boost_skipped"], "no_identity_lora")
        self.assertNotIn("boosted_entries", rep)

    def test_a_slider_at_minus_two_under_all_keeps_minus_two_in_every_segment(self):
        """MP-86 review: the additive boost would have made it -1.65 (weaker, the wrong way); next to a boosted character."""
        slider = self.phase(self.patcher, "A.safetensors", -2.0, "all")
        out = self.run_phased(slider)
        ordinary = self.run_phased(self.native(self.patcher, self.a_sd, -2.0))
        self.assertTrue(self.equal(self.samples(out), self.samples(ordinary)))
        expected, _ = self.expected_chain([-2.0] * 12)
        self.assertTrue(self.equal(self.samples(out), expected.to(self.samples(out).dtype)))
        both = self.run_phased(self.phase(slider, "D.safetensors", 1.0, "identity"))
        reference = self.run_phased(self.phase(self.native(self.patcher, self.a_sd, -2.0), "D.safetensors", 1.0, "identity"))
        self.assertTrue(self.equal(self.samples(both), self.samples(reference)), "the slider is the same as a stock loader")
        self.assertEqual(self.report(both)["darkroom"]["character"]["boosted_entries"], ["D.safetensors"])

    def test_identity_acts_from_the_first_step_and_takes_the_commit_boost(self):
        """MP-86 (MP-85 evidence): the compose segment sets hair and silhouette, so the character phase starts there."""
        out = self.run_phased(self.phase(self.patcher, strength=1.0, phase="identity"))
        weights = self.levels()
        # the entry's own strength plus (weight - w_compose), added in the commit, hold and finish steps
        strengths = [1.0] * 3 + [1.0 + (weights[k] - 1.0) for k in range(3, 12)]
        expected, runs = self.expected_chain(strengths)
        self.assertGreater(len(runs), 3)
        self.assertTrue(self.equal(self.samples(out), expected.to(self.samples(out).dtype)))
        plain, _ = self.expected_chain([1.0] * 12)
        self.assertFalse(self.equal(self.samples(out), plain.to(self.samples(out).dtype)), "control: the boost changes the latent")
        rep = self.report(out)["darkroom"]
        self.assertEqual((rep["character"]["mode"], rep["character"]["boost_skipped"], rep["character"]["boosted_entries"]),
                         ("lora_plan_extra_patch", None, ["A.safetensors"]))
        self.assertEqual(self.report(out)["lora"]["plan"][0]["applied_in"], ["compose", "commit", "finish"])
        self.assertNotIn("darkroom_boost_skipped", self.codes(out))
        from moss_photon.core import report as report_core
        self.assertEqual(report_core.validate_report(self.report(out)), [], "boosted_entries is in the report contract")

    def test_composition_plus_identity_keeps_its_strength_in_every_segment_it_acts_in(self):
        """MP-86 A/B (research/v2_normalize_ab): no commit boost; a slider at -2 stays -2, next to a boosted character."""
        out = self.run_phased(self.phase(self.patcher, strength=1.0, phase="composition+identity"))
        expected, _ = self.expected_chain([1.0] * 7 + [None] * 5)
        self.assertTrue(self.equal(self.samples(out), expected.to(self.samples(out).dtype)))
        rep = self.report(out)["darkroom"]
        self.assertEqual((rep["character"]["boost_skipped"], rep["segments"][0]["weight"]), ("no_identity_lora", None))
        self.assertNotIn("boosted_entries", rep["character"])
        slider = self.run_phased(self.phase(self.patcher, "A.safetensors", -2.0, "composition+identity"))
        expected, _ = self.expected_chain([-2.0] * 7 + [None] * 5)
        self.assertTrue(self.equal(self.samples(slider), expected.to(self.samples(slider).dtype)))
        both = self.run_phased(self.phase(self.phase(self.patcher, "A.safetensors", -2.0, "composition+identity"),
                                          "D.safetensors", 1.0, "identity"))
        self.assertEqual(self.report(both)["darkroom"]["character"]["boosted_entries"], ["D.safetensors"],
                         "the slider is not named, the character is")

    def test_a_negative_identity_strength_stays_in_every_segment_and_the_run_says_so(self):
        """MP-86 review: the node default is ``identity``; -2 would have become -1.65 and -0.2 would have flipped to +0.15."""
        for strength in (-2.0, -0.2):
            with self.subTest(strength=strength):
                out = self.run_phased(self.phase(self.patcher, "A.safetensors", strength, "identity"))
                expected, runs = self.expected_chain([strength] * 12)
                self.assertTrue(self.equal(self.samples(out), expected.to(self.samples(out).dtype)))
                [note] = [w for w in self.report(out)["warnings"] if w["code"] == "darkroom_boost_skipped"]
                self.assertEqual((note["severity"], note["detail"]["reason"], note["detail"]["basename"],
                                  note["detail"]["strength"]), ("warning", "negative_strength", "A.safetensors", strength))
                self.assertIn("A.safetensors", note["message"])
                rep = self.report(out)["darkroom"]
                self.assertEqual(rep["character"]["boost_skipped"], "no_identity_lora")
                self.assertNotIn("boosted_entries", rep["character"])
        both = self.run_phased(self.phase(self.phase(self.patcher, "A.safetensors", -2.0, "identity"),
                                          "D.safetensors", 1.0, "identity"))
        rep = self.report(both)
        self.assertEqual(rep["darkroom"]["character"]["boosted_entries"], ["D.safetensors"], "a boosted character stays boosted")
        self.assertEqual([w["detail"]["basename"] for w in rep["warnings"] if w["code"] == "darkroom_boost_skipped"],
                         ["A.safetensors"], "one warning, for the slider")
        quiet = self.run_phased(self.phase(self.patcher, "A.safetensors", -2.0, "identity"), w_commit=1.0)
        self.assertNotIn("darkroom_boost_skipped", self.codes(quiet), "no boost asked for, nothing refused")

    def test_the_boost_set_leaves_composition_and_texture_unboosted(self):
        for phase, steps in (("composition", range(0, 3)), ("texture", range(7, 12)), ("composition+identity", range(0, 7))):
            with self.subTest(phase=phase):
                out = self.run_phased(self.phase(self.patcher, strength=1.0, phase=phase))
                expected, _ = self.expected_chain([1.0 if k in steps else None for k in range(12)])
                self.assertTrue(self.equal(self.samples(out), expected.to(self.samples(out).dtype)))
                rep = self.report(out)["darkroom"]
                self.assertEqual(rep["character"]["boost_skipped"], "no_identity_lora")
                self.assertNotIn("boosted_entries", rep["character"])

    def test_a_known_distill_in_phase_identity_is_not_boosted_and_a_character_next_to_it_is(self):
        """MP-86: a distill (known file hash) put through a LoRA Phase node keeps its strength; the document says to use a
        stock loader for it, this is the safety net."""
        sha = self.env.written["D.safetensors"]
        row = {"name": "synthetic-distill", "role": "distill", "status": "verified_recipe", "trained_on": "base",
               "recipe": {"steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "shift": 5.0, "strength": 1.0}}
        with mock.patch.dict(self.consts.KNOWN_DISTILL_LORAS, {sha: row}):
            alone = self.run_phased(self.phase(self.patcher, "D.safetensors", 1.0, "identity"))
            d_sd =lora_synth.make_lora_sd(self.patcher, seed=21, modules=("qkv", "o", "ffn_down"))
            ordinary = self.run_phased(self.native(self.patcher, d_sd))   # a stock loader: never in the plan, never boosted
            self.assertTrue(self.equal(self.samples(alone), self.samples(ordinary)))
            rep = self.report(alone)["darkroom"]
            self.assertEqual(rep["character"]["boost_skipped"], "no_identity_lora")
            self.assertIn("darkroom_boost_skipped", self.codes(alone))
            both = self.run_phased(self.phase(self.phase(self.patcher, "D.safetensors", 1.0, "identity"),
                                              "A.safetensors", 1.0, "identity"))
            self.assertEqual(self.report(both)["darkroom"]["character"]["boosted_entries"], ["A.safetensors"])
        unknown = self.run_phased(self.phase(self.patcher, "D.safetensors", 1.0, "identity"))
        self.assertFalse(self.equal(self.samples(unknown), self.samples(ordinary)),
                         "control: the same file not known as a distill IS boosted")
        self.assertEqual(self.report(unknown)["darkroom"]["character"]["boosted_entries"], ["D.safetensors"])

    def test_without_an_identity_entry_the_boost_is_skipped_with_a_warning(self):
        for model in (self.patcher, self.phase(self.patcher, phase="texture"), self.native(self.patcher, self.a_sd)):
            with self.subTest(plan=len(self.lora_plan.entries_of(model))):
                out = self.run_phased(model)
                rep = self.report(out)["darkroom"]
                self.assertEqual(rep["character"]["boost_skipped"], "no_identity_lora")
                self.assertIn("darkroom_boost_skipped", self.codes(out))
                self.assertTrue(self.torch.isfinite(self.samples(out)).all())
                self.assertEqual({s["weight"] for s in rep["segments"]}, {None})

    def test_a_constant_character_profile_next_to_an_identity_entry_loses_the_boost_with_a_warning(self):
        """MP-84 review: the boost goes to the identity entries; the profile's constant character LoRA keeps its nominal
        strength, and the run says so (it was silent)."""
        m, p = self.constant_chain(strength=0.8)
        out = self.run_phased(self.phase(m, "D.safetensors", 0.5, "identity"), profile=p)
        [note] = [w for w in self.report(out)["warnings"] if w["code"] == "darkroom_boost_skipped"]
        self.assertEqual((note["severity"], note["detail"]["reason"], note["detail"]["basename"], note["param"]),
                         ("warning", "identity_plan_takes_boost", "A.safetensors", "identity_profile"))
        self.assertIn("A.safetensors", note["message"])
        self.assertEqual(self.report(out)["darkroom"]["character"]["mode"], "lora_plan_extra_patch")
        quiet = self.run_phased(self.phase(m, "D.safetensors", 0.5, "identity"), profile=p, w_commit=1.0)
        self.assertNotIn("darkroom_boost_skipped", self.codes(quiet), "no boost asked for, nothing lost")

    def test_a_phased_adapter_and_an_identity_entry_are_both_boosted_and_the_report_names_both(self):
        from moss_photon.core import report as report_core
        m, p, _d_model, _d_profile = self.phased_chain()
        out = self.run_phased(self.phase(m, "D.safetensors", 0.5, "identity"), profile=p)
        rep = self.report(out)
        self.assertEqual(rep["darkroom"]["character"]["mode"], "phased_adapter+lora_plan_extra_patch")
        self.assertEqual(report_core.validate_report(rep), [])
        self.assertNotIn("darkroom_boost_skipped", self.codes(out))
        alone = self.report(self.run_phased(m, profile=p))
        self.assertEqual(alone["darkroom"]["character"]["mode"], "phased_adapter")

    def test_no_boost_is_asked_for_when_the_weights_are_equal(self):
        out = self.run_phased(self.phase(self.patcher, phase="identity"), w_commit=1.0)
        self.assertNotIn("darkroom_boost_skipped", self.codes(out))
        self.assertEqual(self.report(out)["darkroom"]["character"]["mode"], "constant_profile")
        expected, _ = self.expected_chain([1.0] * 12)
        self.assertTrue(self.equal(self.samples(out), expected.to(self.samples(out).dtype)), "the LoRA runs at its strength")

    def test_composition_and_texture_split_the_run_like_two_stock_segments(self):
        both = self.phase(self.phase(self.patcher, "A.safetensors", 1.0, "composition"), "A.safetensors", 0.5, "texture")
        out = self.run_phased(both)
        strengths = [1.0] * 3 + [None] * 4 + [0.5] * 5
        expected, _ = self.expected_chain(strengths)
        self.assertTrue(self.equal(self.samples(out), expected.to(self.samples(out).dtype)))

    def test_the_finish_model_has_its_own_plan(self):
        finish = self.phase(self.patcher, "A.safetensors", 1.0, "all")
        main = self.phase(self.patcher, "A.safetensors", 0.5, "composition")
        out = self.run_phased(main, finish_model=finish, switch=True)
        rep = self.report(out)["darkroom"]
        self.assertEqual([s["model"] for s in rep["segments"]], ["main", "main", "finish"])
        self.assertNotIn("finish_model_ignored", self.codes(out))
        expected = self.chain([(self.native(self.patcher, self.a_sd, 0.5), 1.0, 0, 3), (self.patcher, 1.0, 3, 7),
                               (self.native(self.patcher, self.a_sd, 1.0), 2.0, 7, 12)])
        self.assertTrue(self.equal(self.samples(out), expected.to(self.samples(out).dtype)))

    def test_an_entry_that_acts_in_no_phase_of_its_model_is_reported_as_inactive_with_a_warning(self):
        """MP-84 re-review MAJOR-A: the finish segments run on the finish MODEL, which never sees the main MODEL's entries.
        A main ``texture`` entry next to a switched finish model does nothing, so ``applied_in`` is empty and the report
        warns (the run goes on); the same holds for a finish entry whose phase is not ``finish``."""
        finish = self.phase(self.patcher, "A.safetensors", 0.0, "all")
        main = self.phase(self.patcher, "A.safetensors", 0.5, "texture")
        out = self.run_phased(main, finish_model=finish, switch=True)
        rep = self.report(out)
        [entry] = [e for e in rep["lora"]["plan"] if e["model"] == "main"]
        self.assertEqual(entry["applied_in"], [])
        [noted] = [w for w in rep["warnings"] if w["code"] == "lora_plan_entry_inactive"]
        self.assertEqual((noted["severity"], noted["detail"]["model"], noted["detail"]["plan_index"],
                          noted["detail"]["phase"]), ("warning", "main", 0, "texture"))
        without = self.run_phased(self.patcher, finish_model=finish, switch=True)
        self.assertTrue(self.equal(self.samples(out), self.samples(without)), "the entry really changed nothing")
        late = self.phase(self.patcher, "A.safetensors", 1.0, "composition")
        rep = self.report(self.run_phased(self.patcher, finish_model=late, switch=True))
        [noted] = [w for w in rep["warnings"] if w["code"] == "lora_plan_entry_inactive"]
        self.assertEqual((noted["detail"]["model"], noted["detail"]["plan_index"]), ("finish", 0))
        for fine in (self.run_phased(self.phase(self.patcher, "A.safetensors", 0.5, "composition"),
                                     finish_model=self.phase(self.patcher, "A.safetensors", 1.0, "all"), switch=True),
                     self.run_phased(self.phase(self.patcher, "A.safetensors", 0.5, "texture")),
                     self.run_phased(self.phase(self.patcher, "A.safetensors", 0.0, "texture"))):
            self.assertNotIn("lora_plan_entry_inactive", self.codes(fine))

    def test_the_finish_models_plan_is_in_the_report_the_recipe_hash_and_the_warnings(self):
        """MP-84 review MAJOR-3: a finish entry changes the picture, so it is in ``lora.plan`` (marked ``finish``), in the
        recipe hash, and its warnings reach the report."""
        from moss_photon.core import report as report_core
        partial = lora_synth.add_bogus_modules(lora_synth.make_lora_sd(self.patcher, seed=5, modules=("qkv", "o")), 30)
        self.env.write("partial.safetensors", partial)
        main = self.phase(self.patcher, "A.safetensors", 0.5, "composition")
        reports = []
        for strength in (1.0, 0.3):
            finish = self.phase(self.patcher, "partial.safetensors", strength, "all")
            reports.append(self.report(self.run_phased(main, finish_model=finish, switch=True)))
        for rep in reports:
            self.assertEqual([(e["basename"], e["model"], e["applied_in"]) for e in rep["lora"]["plan"]],
                             [("A.safetensors", "main", ["compose"]), ("partial.safetensors", "finish", ["finish"])])
            self.assertEqual(report_core.validate_report(rep), [])
            [noted] = [w for w in rep["warnings"] if w["code"] == "partial_lora_coverage"]
            self.assertEqual((noted["detail"]["model"], noted["detail"]["plan_index"]), ("finish", 0))
        self.assertEqual([e["strength"] for e in (r["lora"]["plan"][1] for r in reports)], [1.0, 0.3])
        self.assertNotEqual(reports[0]["recipe_hash"], reports[1]["recipe_hash"])

    def test_neighbouring_segments_with_the_same_set_share_one_level_clone(self):
        real = self.lora_plan.apply_to
        calls = []

        def spy(model, entries, darkroom_phase=None, boost=0.0):
            calls.append((darkroom_phase, boost))
            return real(model, entries, darkroom_phase, boost)

        with mock.patch.object(self.lora_plan, "apply_to", spy):
            out = self.run_phased(self.phase(self.patcher, phase="texture"), split_at=(8, 10))
        self.assertEqual(self.report(out)["darkroom"]["clones_built"], 4, "compose+commit, and three finish segments")
        finish_calls = [c for c in calls if c[0] == "finish"]
        self.assertEqual(len(finish_calls), 1, "three finish segments, one patched level clone")
        whole = self.run_phased(self.phase(self.patcher, phase="texture"))
        self.assertTrue(self.equal(self.samples(out), self.samples(whole)), "the cuts change nothing (euler is stateless)")
        for run in (out, whole):   # a shared level clone must not stack a second forward counter on itself
            fw = self.report(run)["forwards"]
            self.assertEqual((fw["match"], fw["sample_forwards"], fw["model_calls"]), (True, 17, 17))
        slg = self.run_phased(self.phase(self.patcher, phase="all"), compose_slg=True, compose_cfg=1.5, split_at=(2, 5))
        self.assertTrue(self.report(slg)["forwards"]["match"], "a counter inherited by an SLG clone counts once")

    def test_a_darkroom_run_with_no_plan_is_the_run_of_the_old_engine(self):
        """No plan: ``level_for`` takes the stock path; the result equals a chain of stock segments on the bare model."""
        out = self.run_phased(self.patcher, w_commit=1.0)
        expected, _ = self.expected_chain([None] * 12)
        self.assertTrue(self.equal(self.samples(out), expected.to(self.samples(out).dtype)))
        self.assertNotIn("plan", self.report(out)["lora"])

    def test_the_models_the_plan_is_applied_to_are_direct_clones_of_the_incoming_model(self):
        """``ModelPatcher.clone`` sets ``parent``; a clone of a clone would have a clone for a parent (the LoadedModel weakref
        leak of KreaPhoton's docstring)."""
        real = self.lora_plan.apply_to
        parents = []

        def spy(model, entries, darkroom_phase=None, boost=0.0):
            parents.append(model.parent)
            return real(model, entries, darkroom_phase, boost)

        two = self.phase(self.phase(self.patcher, phase="identity"), "D.safetensors", 0.7, "texture")
        with mock.patch.object(self.lora_plan, "apply_to", spy):
            self.run_phased(two)
            self.run_plain(two)
        self.assertGreaterEqual(len(parents), 3, "a darkroom run and a constant run both applied the plan")
        self.assertTrue(all(parent is two for parent in parents), "an apply_to target is not a direct clone of the incoming model")

    def test_the_boost_with_two_identity_entries_is_added_to_each(self):
        d_sd = lora_synth.make_lora_sd(self.patcher, seed=21, modules=("qkv", "o", "ffn_down"))
        out = self.run_phased(self.phase(self.phase(self.patcher, "A.safetensors", 1.0, "identity"),
                                         "D.safetensors", 0.5, "identity"))
        weights = self.levels()
        cfgs = [1.0] * 7 + [2.0] * 5
        segs, start = [], 0
        stack = [(1.0, 0.5)] * 3 + [(1.0 + (weights[k] - 1.0), 0.5 + (weights[k] - 1.0)) for k in range(3, 12)]
        for k in range(1, 13):
            if k == 12 or (stack[k], cfgs[k]) != (stack[start], cfgs[start]):
                s = stack[start]
                model = self.patcher if s is None else self.native(self.native(self.patcher, self.a_sd, s[0]), d_sd, s[1])
                segs.append((model, cfgs[start], start, k))
                start = k
        expected = self.chain(segs)
        self.assertTrue(self.equal(self.samples(out), expected.to(self.samples(out).dtype)))


class DegradationTest(PlanBase):
    def test_a_distill_and_a_second_distill_is_a_warning(self):
        m, p, _ = self.lora(self.patcher, "D.safetensors", role="distill", strength_model=1.0)
        self.env.write("D2.safetensors", lora_synth.make_lora_sd(self.patcher, seed=22, modules=("qkv", "o")))
        m2, p2, _ = self.lora(m, "D2.safetensors", role="distill", strength_model=1.0, profile_in=p)
        self.assertIn("multiple_distill_loras", [w["code"] for w in p2["stack"][1]["warnings"]])
        out = self.run_plain(m2, identity_profile=p2)
        self.assertIn("multiple_distill_loras", self.codes(out))
        self.assertTrue(self.torch.isfinite(self.samples(out)).all())

    def test_a_distill_profile_on_a_preset_without_a_turbo_policy_is_a_warning(self):
        m, p, _ = self.lora(self.patcher, "D.safetensors", role="distill", strength_model=1.0)
        out = self.run_plain(m, preset="reference", identity_profile=p, overrides={"steps": 4})
        self.assertIn("distill_lora_non_turbo", self.codes(out))
        self.assertTrue(self.torch.isfinite(self.samples(out)).all())

    def test_a_turbo_preset_without_any_lora_runs_with_a_note(self):
        for preset in (TURBO, "turbo_identity"):
            with self.subTest(preset=preset):
                out = self.run_plain(self.patcher, preset=preset)
                self.assertIn("lora_not_recorded", self.codes(out))
                self.assertTrue(self.torch.isfinite(self.samples(out)).all())

    def test_a_model_with_a_stock_loader_needs_no_profile_and_no_note(self):
        out = self.run_plain(self.native(self.patcher, self.a_sd), preset="turbo_identity")
        codes = self.codes(out)
        self.assertNotIn("lora_not_recorded", codes)
        self.assertIn("foreign_patches", codes)
        info = next(w for w in self.report(out)["warnings"] if w["code"] == "foreign_patches")
        self.assertEqual(info["severity"], "info")

    def test_a_phased_profile_on_the_basic_sampler_runs_constant_at_its_nominal_strength(self):
        m, p, _ = self.lora(self.patcher, "A.safetensors", role="character", profile="phased", w0=0.1, w1=0.9,
                            strength_model=0.7)
        out = self.run_plain(m, identity_profile=p, node_id="MossPhotonSampler")
        self.assertIn("phased_lora_demoted", self.codes(out))
        expected = self.run_plain(self.native(self.patcher, self.a_sd, 0.7))
        self.assertTrue(self.equal(self.samples(out), self.samples(expected)))
        self.assertNotIn("phase", self.report(out)["lora"])

    def test_a_phased_profile_that_is_not_a_character_runs_constant_in_the_darkroom(self):
        m, p, _ = self.lora(self.patcher, "A.safetensors", role="style", profile="phased", w0=0.1, w1=0.9,
                            strength_model=0.7)
        with ss.known_sft():
            out = ss.run(self.tiny, model=m, identity_profile=p, preset_id=PORTRAIT, cfg=1.0, node_id=dk_tests.NODE,
                         allow_phased_lora=True, lora_loader=self.lora_core.plan_patches,
                         darkroom=self.darkroom.DarkroomRequest(overrides={"w_commit": 1.0}))
        self.assertIn("phased_lora_demoted", self.codes(out))
        self.assertTrue(self.torch.isfinite(self.samples(out)).all())
        ordinary = self.run_phased(self.native(self.patcher, self.a_sd, 0.7), w_commit=1.0)
        self.assertTrue(self.equal(self.samples(out), self.samples(ordinary)))

    def test_the_node_runs_end_to_end_with_the_phase_node_and_a_stock_loader(self):
        ordinary = self.native(self.patcher, self.a_sd, 0.3)
        model = self.phase(ordinary, "D.safetensors", 0.8, "identity")
        out = self.run_adv(model, None, PORTRAIT, steps=12)
        self.assertTrue(self.torch.isfinite(self.samples(out)).all())
        self.assertEqual(self.report(out)["lora"]["plan"][0]["applied_in"], ["compose", "commit", "finish"])
        out = self.run_simple(model, None, TURBO)
        self.assertNotIn("phase_plan_ignored", self.codes(out), "identity is the whole run on a preset without phases")
        late = self.run_simple(self.phase(ordinary, "D.safetensors", 0.8, "texture"), None, TURBO)
        self.assertIn("phase_plan_ignored", self.codes(late))


if __name__ == "__main__":
    unittest.main()
