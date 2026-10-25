"""MP-16: phased LoRA weight by sigma progress on the tiny 24-layer MOSS (CPU, synthetic LoRAs, no weights).

Mechanism under test: clone-per-level (core/lora_phase.py). Invariants (each has a planted violation in the MP-16 report):
(a) w0 == w1 is bit-equal to the constant-weight run; (b) w0 != w1 differs and the weights follow smoothstep of the walked
sigma distance; (c) the incoming MODEL is never mutated; (d) no residue after success, an exception or an interrupt, and a
following constant run equals its baseline; (e) distill + character with only the character phased; (f) a turbo preset with a
phased character LoRA runs only in the Advanced node; (g) forwards per step are unchanged.
"""
import copy
import gc
import hashlib
import struct
import sys
import types
import unittest
import weakref
from unittest import mock

from . import lora_synth, tiny_moss
from . import sampler_support as ss

STEPS = 10
STEPS_O = {"steps": STEPS}
ADV = "MossPhotonSamplerAdvanced"


def textbook_applied(sigmas, w0, w1, a, b, levels=6):
    """The per-step planned and applied weights, from the SPEC formula, with no code of the pack."""
    total = sum(abs(sigmas[i + 1] - sigmas[i]) for i in range(len(sigmas) - 1))
    planned, applied = [], []
    for k in range(len(sigmas) - 1):
        r = sum(abs(sigmas[j + 1] - sigmas[j]) for j in range(k)) / total
        t = min(1.0, max(0.0, (r - a) / (b - a)))
        s = 3 * t ** 2 - 2 * t ** 3
        planned.append(w0 + (w1 - w0) * s)
        j = int(round(s * (levels - 1)))
        applied.append(w0 if j == 0 else w1 if j == levels - 1 else w0 + (w1 - w0) * (j / (levels - 1)))
    return planned, applied


def pack(values):
    return hashlib.sha256(struct.pack("<%dd" % len(values), *values)).hexdigest()


class PhaseCpuBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss(steps=STEPS)
        import torch
        import comfy.model_management as mm
        from moss_photon.core import errors, lora_phase, phase, solver
        cls.torch, cls.mm, cls.errors, cls.lora_phase, cls.phase, cls.solver = torch, mm, errors, lora_phase, phase, solver

    def setUp(self):
        self.env = lora_synth.install(self)
        self.patcher = self.tiny.patcher
        self.env.write("A.safetensors", lora_synth.make_lora_sd(self.patcher, seed=11, modules=("qkv", "o")))
        self.env.write("D.safetensors", lora_synth.make_lora_sd(self.patcher, seed=21, modules=("qkv", "o", "ffn_down")))

    # ---- builders
    def node(self, model, name="A.safetensors", **kw):
        return lora_synth.run_node(model, name, **kw)

    def phased(self, model=None, name="A.safetensors", w0=0.3, w1=0.9, a=0.2, b=0.6, **kw):
        return self.node(self.patcher if model is None else model, name, profile="phased", w0=w0, w1=w1, a=a, b=b, **kw)

    def run_adv(self, model, profile, **kw):
        kw.setdefault("overrides", STEPS_O)
        return ss.run(self.tiny, model=model, identity_profile=profile, node_id=ADV, allow_phased_lora=True, **kw)

    def run_const(self, strength=0.8, name="A.safetensors", **kw):
        m, p, _ = self.node(self.patcher, name, strength_model=strength)
        kw.setdefault("overrides", STEPS_O)
        return ss.run(self.tiny, model=m, identity_profile=p, **kw)

    def equal(self, a, b):
        return bool(self.torch.equal(a, b))

    def baseline_sha(self):
        return ss.tensor_sha(tiny_moss.sample_euler(self.tiny))


class InvariantATest(PhaseCpuBase):
    def test_equal_weights_are_bit_equal_to_the_constant_run(self):
        const = self.run_const(0.8)
        m, p, _ = self.phased(w0=0.8, w1=0.8)
        phased = self.run_adv(m, p)
        self.assertFalse(self.equal(const.latent["samples"], tiny_moss.sample_euler(self.tiny)),
                         "control: the LoRA changes the result")
        self.assertTrue(self.equal(phased.latent["samples"], const.latent["samples"]))
        phase = phased.report["lora"]["phase"]
        self.assertEqual([(s["first_step"], s["last_step"], s["weight"]) for s in phase["segments"]],
                         [(0, STEPS - 1, 0.8)])
        self.assertEqual(phase["clones_built"], 1)

    def test_equal_weights_hold_for_other_values(self):
        for w in (0.35, 1.0, -0.5):
            with self.subTest(w=w):
                const = self.run_const(w)
                m, p, _ = self.phased(w0=w, w1=w, a=0.1, b=0.9)
                self.assertTrue(self.equal(self.run_adv(m, p).latent["samples"], const.latent["samples"]))


class InvariantBTest(PhaseCpuBase):
    def spy_segments(self):
        """Patch the two seams to record (weight, steps) of every segment."""
        seen = []
        real_level, real_sample = self.lora_phase.make_level_model, None
        import comfy.samplers
        real_sample = comfy.samplers.CFGGuider.sample
        weights = []

        def level(model, adapter, weight):
            weights.append(weight)
            return real_level(model, adapter, weight)

        def sample(guider, noise, latent, sampler, sigmas, *a, **k):
            seen.append((weights[-1], int(sigmas.shape[0]) - 1))
            return real_sample(guider, noise, latent, sampler, sigmas, *a, **k)

        stack = (mock.patch.object(self.lora_phase, "make_level_model", level),
                 mock.patch.object(comfy.samplers.CFGGuider, "sample", sample))
        return seen, stack

    def test_a_different_pair_of_weights_changes_the_result_and_the_steps_follow_smoothstep(self):
        sigmas = self.tiny.sigmas.tolist()
        m, p, _ = self.phased(w0=0.3, w1=0.9, a=0.2, b=0.6)
        seen, patches = self.spy_segments()
        with patches[0], patches[1]:
            phased = self.run_adv(m, p)
        planned, applied = textbook_applied(sigmas, 0.3, 0.9, 0.2, 0.6)
        per_step = [w for w, n in seen for _ in range(n)]
        self.assertEqual(per_step, applied)
        self.assertEqual(len(seen), len(set(applied)))
        self.assertGreater(len(seen), 2)
        # the report says the same thing in run-length form and with the hashes of the exact and applied weights
        phase = phased.report["lora"]["phase"]
        expanded = [s["weight"] for s in phase["segments"] for _ in range(s["last_step"] - s["first_step"] + 1)]
        self.assertEqual(expanded, applied)
        self.assertEqual(phase["weights_sha256"], pack(applied))
        self.assertEqual(phase["planned_sha256"], pack(planned))
        for exact, used in zip(planned, applied):
            self.assertLessEqual(abs(exact - used), (0.9 - 0.3) / (2 * 5) + 1e-12)
        # it is neither constant end
        out = phased.latent["samples"]
        self.assertFalse(self.equal(out, self.run_const(0.3).latent["samples"]))
        self.assertFalse(self.equal(out, self.run_const(0.9).latent["samples"]))

    def test_the_direction_matters(self):
        up = self.run_adv(*self.phased(w0=0.3, w1=0.9)[:2])
        down = self.run_adv(*self.phased(w0=0.9, w1=0.3)[:2])
        self.assertFalse(self.equal(up.latent["samples"], down.latent["samples"]))

    def test_progress_is_the_walked_sigma_distance_not_the_step_number(self):
        """A custom schedule that spends most of its distance early moves the weight early, whatever the step count."""
        import torch
        sigmas = torch.tensor([1.0, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.04, 0.02, 0.0])
        m, p, _ = self.phased(w0=0.3, w1=0.9, a=0.2, b=0.6)
        seen, patches = self.spy_segments()
        with patches[0], patches[1]:
            self.run_adv(m, p, sigmas=sigmas)
        _planned, applied = textbook_applied(sigmas.tolist(), 0.3, 0.9, 0.2, 0.6)
        self.assertEqual([w for w, n in seen for _ in range(n)], applied)
        by_index = [0.3 if k / 10.0 <= 0.2 else None for k in range(10)]
        self.assertNotEqual(applied[2], by_index[2], "step 2 of 10 is already past a by distance")
        self.assertEqual(applied[2], 0.9)

    def test_the_handover_between_segments_is_exact(self):
        """The phased result equals a hand-chained reference: the stock constant-weight model per segment, euler resumed
        from the previous segment's x (no code of lora_phase in the reference)."""
        import comfy.k_diffusion.sampling as kd
        import comfy.sample
        import comfy.samplers
        from moss_photon.core import guidance
        torch = self.torch
        sigmas = self.tiny.sigmas
        m, p, _ = self.phased(w0=0.3, w1=0.9, a=0.2, b=0.6)
        phased = self.run_adv(m, p)
        sched = self.phase.build_schedule(sigmas.tolist(), self.phase.make_plan(0, 0.3, 0.9, 0.2, 0.6))
        lora_sd = lora_synth.make_lora_sd(self.patcher, seed=11, modules=("qkv", "o"))
        carry = {"x": None}

        def resume(model, x, sig, extra_args=None, callback=None, disable=None, **opts):
            start = x if carry["x"] is None else carry["x"]
            carry["x"] = kd.sample_euler(model, start, sig, extra_args=extra_args, callback=callback, disable=disable)
            return carry["x"]

        noise = comfy.sample.prepare_noise(self.tiny.latent, tiny_moss.SEED)
        out = None
        for seg in sched.segments:
            level = ss.apply_lora(self.patcher, lora_sd, seg.weight)
            level = guidance.apply_slg(level, layers=[22], start=0.0, end=1.0)
            guider = comfy.samplers.CFGGuider(level)
            guider.set_conds(self.tiny.make_cond(23), self.tiny.make_cond(9))
            guider.set_cfg(3.0)
            with torch.inference_mode():
                out = guider.sample(noise, self.tiny.latent.clone(), comfy.samplers.KSAMPLER(resume),
                                    sigmas[seg.first:seg.stop + 1], seed=tiny_moss.SEED)
        self.assertGreater(len(sched.segments), 2)
        self.assertTrue(self.equal(phased.latent["samples"], out.to(phased.latent["samples"].dtype)))


class InvariantCTest(PhaseCpuBase):
    def snapshot(self, p, weights=True):
        weights_hash = hashlib.sha256()
        for name, param in sorted(p.model.diffusion_model.named_parameters()):
            weights_hash.update(name.encode())
            weights_hash.update(param.detach().contiguous().numpy().tobytes())
        return {
            "patch_keys": sorted(p.patches), "patch_entries": sum(len(v) for v in p.patches.values()),
            "uuid": str(p.patches_uuid), "attachment_keys": sorted(p.attachments),
            "attachment_ids": {k: id(v) for k, v in p.attachments.items()},
            "model_options": copy.deepcopy(p.model_options), "object_patches": dict(p.object_patches),
            "hook_patches": dict(p.hook_patches), "hook_backup": dict(p.hook_backup), "current_hooks": p.current_hooks,
            "weights": weights_hash.hexdigest() if weights else None,
        }

    def test_the_incoming_model_and_the_base_model_are_never_mutated(self):
        m, p, _ = self.phased()
        base_before, in_before = self.snapshot(self.patcher), self.snapshot(m)
        in_light = self.snapshot(m, weights=False)
        self.assertIsNotNone(self.lora_phase.adapter_of(m))
        self.assertEqual(in_before["patch_entries"], 0, "the phased node puts no patch on the MODEL")
        self.assertEqual(in_before["uuid"], base_before["uuid"])
        result = self.run_adv(m, p)
        self.assertIsNotNone(result.report["lora"]["phase"])
        self.assertEqual(self.snapshot(m, weights=False), in_light)
        # the shared module may still hold the last level's weights until ComfyUI loads another clone; load one
        tiny_moss.sample_euler(self.tiny)
        self.assertEqual(self.snapshot(self.patcher), base_before)
        self.assertEqual(self.snapshot(m), in_before)

    def test_the_node_does_not_touch_its_input(self):
        before = self.snapshot(self.patcher)
        self.phased()
        self.assertEqual(self.snapshot(self.patcher), before)


class InvariantDTest(PhaseCpuBase):
    def setUp(self):
        super().setUp()
        self.base_plain = self.baseline_sha()
        self.base_const = ss.tensor_sha(self.run_const(0.8).latent["samples"])
        self.phased_model, self.phased_profile, _ = self.phased()
        self.phased_sha = ss.tensor_sha(self.run_adv(self.phased_model, self.phased_profile).latent["samples"])
        self.assertNotEqual(self.base_plain, self.base_const)
        self.assertNotEqual(self.phased_sha, self.base_const)

    def assert_clean(self):
        gc.collect()
        self.assertEqual(self.baseline_sha(), self.base_plain)
        self.assertEqual(ss.tensor_sha(self.run_const(0.8).latent["samples"]), self.base_const)
        self.assertEqual(ss.tensor_sha(self.run_adv(self.phased_model, self.phased_profile).latent["samples"]),
                         self.phased_sha, "a phased run repeats bit-equal")
        for p in (self.patcher, self.phased_model):
            self.assertEqual(p.hook_patches, {})
            self.assertEqual(p.hook_backup, {})
            self.assertIsNone(p.current_hooks)
        self.assertEqual(len(self.patcher.patches), 0)
        self.assertEqual(len(self.phased_model.patches), 0)

    def track_levels(self):
        refs = []
        real = self.lora_phase.make_level_model

        def level(model, adapter, weight):
            made = real(model, adapter, weight)
            refs.append(weakref.ref(made))
            return made
        return refs, mock.patch.object(self.lora_phase, "make_level_model", level)

    def stub_preview(self, at_step, exc_factory):
        def prepare_callback(model, steps, x0_output_dict=None):
            def callback(step, x0, x, total):
                if step == at_step:
                    raise exc_factory()
            return callback
        module = types.ModuleType("latent_preview")
        module.prepare_callback = prepare_callback
        return mock.patch.dict(sys.modules, {"latent_preview": module})

    def test_success_leaves_nothing_and_the_level_clones_are_collectable(self):
        refs, patch = self.track_levels()
        with patch:
            self.run_adv(self.phased_model, self.phased_profile)
        gc.collect()
        self.assertGreater(len(refs), 2)
        self.assertEqual([r() for r in refs], [None] * len(refs))
        self.assert_clean()

    def test_an_interrupt_in_the_middle_of_the_schedule_leaves_nothing(self):
        refs, patch = self.track_levels()
        with patch, self.stub_preview(8, self.mm.InterruptProcessingException):
            with self.assertRaises(self.mm.InterruptProcessingException):
                self.run_adv(self.phased_model, self.phased_profile, preview=True)
        gc.collect()
        self.assertGreaterEqual(len(refs), 2, "the interrupt came after the first segment")
        self.assertEqual([r() for r in refs], [None] * len(refs))
        self.assert_clean()

    def test_a_real_comfy_interrupt_flag_between_forwards_leaves_nothing(self):
        net = self.tiny.model.diffusion_model
        original, calls = net.forward, []
        mm = self.mm

        def forward(x, t, *a, **k):
            calls.append(1)
            if len(calls) == 13:
                mm.interrupt_current_processing(True)
            return original(x, t, *a, **k)
        net.forward = forward
        try:
            with self.assertRaises(mm.InterruptProcessingException):
                self.run_adv(self.phased_model, self.phased_profile)
        finally:
            del net.forward
            mm.interrupt_current_processing(False)
        self.assertLess(len(calls), 2 * STEPS)
        self.assert_clean()

    def test_an_error_while_sampling_a_later_segment_leaves_nothing(self):
        with self.stub_preview(8, lambda: RuntimeError("sampler failed")):
            with self.assertRaises(RuntimeError):
                self.run_adv(self.phased_model, self.phased_profile, preview=True)
        self.assert_clean()

    def test_an_error_while_building_the_second_level_leaves_nothing(self):
        real = self.lora_phase.make_level_model
        calls = []

        def level(model, adapter, weight):
            calls.append(weight)
            if len(calls) == 2:
                raise RuntimeError("clone failed")
            return real(model, adapter, weight)
        with mock.patch.object(self.lora_phase, "make_level_model", level):
            with self.assertRaises(RuntimeError):
                self.run_adv(self.phased_model, self.phased_profile)
        self.assert_clean()


class InvariantETest(PhaseCpuBase):
    """distill + character where only the character is phased."""

    def chain(self, **phased_kw):
        d_model, d_profile, _ = self.node(self.patcher, "D.safetensors", role="distill", strength_model=1.0)
        return d_model, d_profile, self.phased(d_model, role="character", profile_in=d_profile, **phased_kw)

    def turbo(self, model, profile, **kw):
        with ss.known_sft():
            return ss.run(self.tiny, model=model, identity_profile=profile, preset_id="turbo_identity", cfg=1.0,
                          node_id=ADV, allow_phased_lora=True, **kw)

    def test_the_stack_has_both_adapters_and_only_the_character_is_phased(self):
        d_model, d_profile, (c_model, c_profile, _clip) = self.chain()
        self.assertEqual([e["role"] for e in c_profile["stack"]], ["distill", "character"])
        self.assertEqual([e["weight_profile"]["kind"] for e in c_profile["stack"]], ["constant", "phased"])
        self.assertEqual(c_profile["phase_plan"]["stack_index"], 1)
        self.assertEqual(self.lora.validate_identity_profile(c_profile), [])
        self.assertEqual(len(c_model.patches), len(d_model.patches), "the phased adapter adds no patch to the MODEL")
        self.assertEqual(c_profile["attachment"]["patch_entries_total"], self.lora.patch_entries_of(c_model))

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from moss_photon.core import lora
        cls.lora = lora

    def test_it_runs_equals_the_constant_stack_when_the_weights_are_equal_and_differs_otherwise(self):
        _d, _dp, (c_model, c_profile, _c) = self.chain(w0=0.8, w1=0.8)
        d_model, d_profile, _ = self.node(self.patcher, "D.safetensors", role="distill", strength_model=1.0)
        k_model, k_profile, _ = self.node(d_model, "A.safetensors", role="character", strength_model=0.8,
                                          profile_in=d_profile)
        with ss.known_sft():
            const = ss.run(self.tiny, model=k_model, identity_profile=k_profile, preset_id="turbo_identity", cfg=1.0)
        phased = self.turbo(c_model, c_profile)
        self.assertTrue(self.equal(phased.latent["samples"], const.latent["samples"]))
        ramp_model, ramp_profile, _ = self.chain(w0=0.3, w1=0.9)[2]
        ramp = self.turbo(ramp_model, ramp_profile)
        self.assertFalse(self.equal(ramp.latent["samples"], const.latent["samples"]))
        self.assertEqual(self.report_ok(ramp.report), [])

    def report_ok(self, report):
        from moss_photon.core import report as report_mod
        return report_mod.validate_report(report)

    def test_the_label_and_the_warning_of_a_phased_turbo_run(self):
        _d, _dp, (c_model, c_profile, _c) = self.chain()
        report = self.turbo(c_model, c_profile).report
        self.assertEqual(report["label"]["parity_class"], "not_reference")
        self.assertIn("lora_phased", report["label"]["not_reference_reasons"])
        self.assertIn("phase_lora_experimental", ss.codes(report))
        self.assertEqual([s["role"] for s in report["lora"]["stack"]], ["distill", "character"])
        self.assertEqual(report["lora"]["phase"]["stack_index"], 1)
        # forwards are the turbo ones: one per step
        self.assertEqual((report["forwards"]["sample_forwards"], report["forwards"]["match"]), (8, True))


class InvariantFGTest(PhaseCpuBase):
    def test_only_the_advanced_node_accepts_a_phased_profile(self):
        from moss_photon.nodes import sampler
        tiny = self.tiny
        m, p, _ = self.phased()
        made = []
        real = self.lora_phase.make_level_model
        with mock.patch.object(self.lora_phase, "make_level_model", lambda *a: made.append(a) or real(*a)):
            with self.assertRaises(self.errors.UnsupportedFeature) as ctx:
                sampler.MossPhotonSampler().sample(
                    m, tiny.make_cond(23), tiny.make_cond(9), {"samples": tiny.latent.clone()}, tiny_moss.SEED,
                    "reference", 0.0, identity_profile=p, preview=False)
            self.assertEqual(made, [], "refused before any clone was built")
            self.assertIn("MossPhotonSamplerAdvanced", str(ctx.exception))
            self.assertEqual(ctx.exception.issue.node, "MossPhotonSampler")
            out = sampler.MossPhotonSamplerAdvanced().sample(
                m, tiny.make_cond(23), tiny.make_cond(9), {"samples": tiny.latent.clone()}, tiny_moss.SEED,
                "reference", 0.0, STEPS, "from_preset", "", -1.0, -1.0, "euler", "constant", "native", 0,
                "cached_or_compute", identity_profile=p, preview=False)
        report = out["result"][1]
        self.assertEqual(report["label"]["parity_class"], "not_reference")
        self.assertTrue(made)

    def test_a_turbo_preset_with_a_phased_character_lora_is_allowed_only_in_advanced(self):
        from moss_photon.nodes import sampler
        tiny = self.tiny
        d_model, d_profile, _ = self.node(self.patcher, "D.safetensors", role="distill", strength_model=1.0)
        m, p, _ = self.phased(d_model, role="character", profile_in=d_profile)
        with ss.known_sft():
            with self.assertRaises(self.errors.UnsupportedFeature) as ctx:
                sampler.MossPhotonSampler().sample(
                    m, tiny.make_cond(23), tiny.make_cond(9), {"samples": tiny.latent.clone()}, tiny_moss.SEED,
                    "turbo_identity", 0.0, identity_profile=p, preview=False)
            self.assertIn("MossPhotonSamplerAdvanced", str(ctx.exception))
            report = ss.run(tiny, model=m, identity_profile=p, preset_id="turbo_identity", cfg=1.0, node_id=ADV,
                            allow_phased_lora=True).report
        self.assertIn("lora_phased", report["label"]["not_reference_reasons"])

    def test_forwards_per_step_are_those_of_the_constant_run(self):
        const = self.run_const_report()
        m, p, _ = self.phased(w0=0.3, w1=0.9)
        phased = ss.run(self.tiny, model=m, identity_profile=p, node_id=ADV, allow_phased_lora=True,
                        overrides=STEPS_O).report
        self.assertEqual(phased["forwards"], const["forwards"])
        self.assertEqual((phased["forwards"]["match"], phased["forwards"]["sample_forwards"],
                          phased["forwards"]["slg_skip_calls"]), (True, 2 * STEPS, STEPS))
        self.assertGreater(phased["lora"]["phase"]["clones_built"], 2)

    def run_const_report(self):
        m, p, _ = self.node(self.patcher, "A.safetensors", strength_model=0.8)
        return ss.run(self.tiny, model=m, identity_profile=p, overrides=STEPS_O).report

    def test_a_constant_run_has_no_phase_block_and_keeps_its_label(self):
        report = self.run_const_report()
        self.assertNotIn("phase", report["lora"])
        self.assertNotIn("lora_phased", report["label"]["not_reference_reasons"])
        self.assertNotIn("phase_lora_experimental", ss.codes(report))


class ContractTest(PhaseCpuBase):
    def test_the_report_validates_and_the_recipe_hash_sees_the_plan(self):
        from moss_photon.core import report as report_mod
        reports = []
        for w0, w1 in ((0.3, 0.9), (0.3, 0.9), (0.4, 0.9)):
            m, p, _ = self.phased(w0=w0, w1=w1)
            reports.append(self.run_adv(m, p).report)
        for r in reports:
            self.assertEqual(report_mod.validate_report(r), [])
            self.assertIn("lora.phase", report_mod._recipe_material(r))
        self.assertEqual(reports[0]["recipe_hash"], reports[1]["recipe_hash"])
        self.assertNotEqual(reports[0]["recipe_hash"], reports[2]["recipe_hash"])
        self.assertIn("phased lora", report_mod.format_summary(reports[0]))

    def test_label_and_warning(self):
        m, p, _ = self.phased()
        report = self.run_adv(m, p).report
        self.assertEqual(report["label"]["parity_class"], "not_reference")
        self.assertIn("lora_phased", report["label"]["not_reference_reasons"])
        warn = [w for w in report["warnings"] if w["code"] == "phase_lora_experimental"]
        self.assertEqual((len(warn), warn[0]["severity"]), (1, "warning"))

    def test_the_phased_entry_in_the_report_stack(self):
        m, p, _ = self.phased()
        entry = p["stack"][0]
        self.assertEqual(entry["weight_profile"], {"kind": "phased", "w0": 0.3, "w1": 0.9, "a": 0.2, "b": 0.6})
        self.assertEqual(entry["strength_model"], 0.9)
        self.assertEqual(p["phase_plan"]["mechanism"], "clone_per_level")
        self.assertGreater(entry["patch_entries"], 0, "describes the file")
        self.assertEqual(p["attachment"]["patch_entries_total"], 0)
        report = self.run_adv(m, p).report
        self.assertEqual(report["lora"]["stack"][0]["strength_model"], 0.9)

    def test_a_model_without_the_adapter_is_refused_before_sampling(self):
        m, p, _ = self.phased()
        stripped = m.clone()
        stripped.remove_attachments(self.lora_phase.ATTACH_PHASE_LORA)
        with self.assertRaises(self.errors.IncompatibleLora):
            self.run_adv(stripped, p)

    def test_a_second_phased_adapter_in_one_chain_is_refused_and_a_constant_one_after_it_works(self):
        m, p, _ = self.phased()
        with self.assertRaises(self.errors.UnsupportedFeature):
            self.phased(m, name="D.safetensors", profile_in=p)
        m2, p2, _ = self.node(m, "D.safetensors", strength_model=0.5, profile_in=p, role="style")
        self.assertEqual(p2["phase_plan"], p["phase_plan"])
        self.assertEqual(self.lora_phase.adapter_of(m2).stack_index, 0)
        self.assertEqual(len(m2.patches) > 0, True)
        report = self.run_adv(m2, p2).report
        self.assertEqual(report["lora"]["phase"]["stack_index"], 0)

    def test_no_preset_enables_it(self):
        import json
        import os
        from moss_photon.core import presets
        root = os.path.join(os.path.dirname(presets.__file__), "..", "..", "presets")
        names = [f for f in os.listdir(root) if f.endswith(".json") and f != "schema.json"]
        self.assertGreaterEqual(len(names), 10)

        def keys(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield k
                    yield from keys(v)
            elif isinstance(obj, list):
                for v in obj:
                    yield from keys(v)
        for name in names:
            with open(os.path.join(root, name), "r", encoding="utf-8") as f:
                found = {k for k in keys(json.load(f)) if "phase" in k or k in ("weight_profile", "w0", "w1")}
            self.assertEqual(found, set(), name)
        report = self.run_const_report()
        self.assertNotIn("phase", report["lora"])

    def run_const_report(self):
        m, p, _ = self.node(self.patcher, "A.safetensors", strength_model=0.8)
        return ss.run(self.tiny, model=m, identity_profile=p, overrides=STEPS_O).report


if __name__ == "__main__":
    unittest.main()
