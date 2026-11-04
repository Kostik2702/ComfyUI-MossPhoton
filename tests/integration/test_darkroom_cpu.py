"""MP-57 / MP-71: the darkroom phase engine inside MossPhotonSampler / MossPhotonSamplerAdvanced (presets ``turbo_darkroom_*``)
on the tiny 24-layer MOSS (CPU, synthetic LoRAs, no weights).

The guarantees of the ticket, each with the test that goes red when it is violated (the plants are listed in the reports):

(a) an all-neutral phase plan (``NEUTRAL``: no guidance change, no weight change) is bit-equal to the stock ``turbo_identity``
    run with the same steps -> ``OffEqualsStockTest``; a preset without a darkroom block never reaches the engine and ignores
    the Advanced knobs and ``finish_model`` with warnings -> ``StockPresetIgnoresTheKnobsTest``;
(b) each phase changes only its own segment, the hand-over is exact, re-expose and noise use independent RNGs and never touch
    the main noise stream -> ``PhaseIsolationTest``, ``HandoverTest``, ``IndependentRngTest``;
(c) the incoming MODEL is never mutated, nothing leaks across phases, on exceptions or on an interrupt, and at most one extra
    clone is alive -> ``NoLeakTest``;
(d) forwards are counted per phase (cfg 1 = 1 per step, cfg above 1 = 2 per step), the report records phases, sigma per phase,
    weights, RNG seeds, recipe id@version and the label -> ``ForwardsAndReportTest``;
(e) invalid phase orders or bounds are refused before any forward -> ``RefusedBeforeAnyForwardTest``.

The references are built from stock comfy parts and the textbook formulas (``CFGGuider`` + euler + a hand-written resume),
never from ``core/darkroom.py``.
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

NODE = "MossPhotonSamplerAdvanced"
PORTRAIT = "turbo_darkroom_portrait"
FULLBODY = "turbo_darkroom_fullbody"
OFF = "darkroom_off"   # run_dk: the portrait preset with every phase neutral (NEUTRAL), i.e. the stock turbo run
NEUTRAL = dict(finish_cfg=1.0, w_commit=1.0)
PINNED_W_COMMIT = 1.2  # the MP-57 value; run_dk pins it so the hand-built references do not move with a calibration
SEED = tiny_moss.SEED


def textbook_applied(sigmas, w0, w1, a, b, levels=6):
    """The applied (six-level) character weight of every step from the SPEC formula, with no code of the pack."""
    total = sum(abs(sigmas[i + 1] - sigmas[i]) for i in range(len(sigmas) - 1))
    out = []
    for k in range(len(sigmas) - 1):
        r = sum(abs(sigmas[j + 1] - sigmas[j]) for j in range(k)) / total
        t = min(1.0, max(0.0, (r - a) / (b - a)))
        s = 3 * t ** 2 - 2 * t ** 3
        j = int(round(s * (levels - 1)))
        out.append(w0 if j == 0 else w1 if j == levels - 1 else w0 + (w1 - w0) * (j / (levels - 1)))
    return out


class DarkroomBase(unittest.TestCase):
    STEPS = 12

    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss(steps=cls.STEPS)
        import torch
        import comfy.k_diffusion.sampling as kd
        import comfy.model_management as mm
        import comfy.sample
        import comfy.samplers
        from moss_photon.core import darkroom, errors, lora, lora_phase, presets, solver
        from moss_photon.nodes import sampler as sampler_node
        cls.torch, cls.kd, cls.mm, cls.sample, cls.samplers = torch, kd, mm, comfy.sample, comfy.samplers
        cls.darkroom, cls.errors, cls.lora_phase, cls.solver, cls.lora_core = darkroom, errors, lora_phase, solver, lora
        cls.presets, cls.sampler_node = presets, sampler_node
        cls.sigmas12 = comfy.samplers.calculate_sigmas(cls.tiny.patcher.get_model_object("model_sampling"), "simple", 12)

    def setUp(self):
        self.env = lora_synth.install(self)
        self.patcher = self.tiny.patcher
        self.a_sd = lora_synth.make_lora_sd(self.patcher, seed=11, modules=("qkv", "o"))
        self.env.write("A.safetensors", self.a_sd)
        self.env.write("D.safetensors", lora_synth.make_lora_sd(self.patcher, seed=21, modules=("qkv", "o", "ffn_down")))

    # ---- chains
    def lora(self, model, name, **kw):
        return lora_synth.run_node(model, name, **kw)

    def distill(self, model=None):
        m, p, _ = self.lora(self.patcher if model is None else model, "D.safetensors", role="distill", strength_model=1.0)
        return m, p

    def phased_chain(self, w0=1.0, w1=1.2):
        d_model, d_profile = self.distill()
        m, p, _ = self.lora(d_model, "A.safetensors", role="character", profile="phased", w0=w0, w1=w1, a=0.06, b=0.17,
                            profile_in=d_profile)
        return m, p, d_model, d_profile

    def constant_chain(self, strength=1.0):
        d_model, d_profile = self.distill()
        m, p, _ = self.lora(d_model, "A.safetensors", role="character", strength_model=strength, profile_in=d_profile)
        return m, p

    # ---- runs
    def block(self, preset=PORTRAIT):
        """The ``darkroom`` block of a shipped preset: the one source of the values."""
        return self.presets.catalog()[preset]["darkroom"]

    def settings(self, preset=PORTRAIT, **ov):
        return self.darkroom.resolve_settings(self.presets.catalog()[preset], ov)

    def run_dk(self, model, profile, preset=PORTRAIT, finish_model=None, split_at=(), preview=False, positive=None,
               negative=None, sigmas=None, steps=None, cfg_widget=None, pinned=True, **knobs):
        """The solver, the way MossPhotonSamplerAdvanced calls it: ``knobs`` are the phase knobs by engine name
        (``compose_cfg`` is the request cfg). ``OFF`` = the portrait preset with ``NEUTRAL``; otherwise ``pinned`` fixes
        ``w_commit`` at the MP-57 value unless given, so a calibration of the shipped presets moves none of the references."""
        if preset == OFF:
            preset, knobs = PORTRAIT, dict(NEUTRAL, **knobs)
        elif pinned:
            knobs.setdefault("w_commit", PINNED_W_COMMIT)
        compose = knobs.pop("compose_cfg", 1.0)
        with ss.known_sft():
            return ss.run(
                self.tiny, model=model, identity_profile=profile, preset_id=preset, cfg=compose, node_id=NODE,
                allow_phased_lora=True, preview=preview, overrides={"steps": steps} if steps else {}, sigmas=sigmas,
                positive=positive if positive is not None else self.tiny.make_cond(23),
                negative=negative if negative is not None else self.tiny.make_cond(9),
                darkroom=self.darkroom.DarkroomRequest(
                    overrides=knobs, finish_model=finish_model, split_at=tuple(split_at), cfg_widget=cfg_widget,
                    boost_loader=self.lora_core.boost_patches))

    def run_simple(self, model, profile, preset, cfg=0.0, **kw):
        """MossPhotonSampler (no darkroom widget at all)."""
        tiny = self.tiny
        with ss.known_sft():
            return self.sampler_node.MossPhotonSampler().sample(
                model, tiny.make_cond(23), tiny.make_cond(9), {"samples": tiny.latent.clone()}, SEED, preset, cfg,
                identity_profile=profile, preview=False, **kw)

    def run_adv(self, model, profile, preset, cfg=0.0, steps=0, finish_model=None, **widgets):
        """MossPhotonSamplerAdvanced with the widgets at their sentinels except ``widgets`` (knob widget names)."""
        tiny = self.tiny
        base = dict(slg_mode="from_preset", slg_layers="", slg_start=-1.0, slg_end=-1.0, solver="euler",
                    cfg_profile="constant", noise_policy="native", max_forwards=0, fingerprint="cached_or_compute")
        with ss.known_sft():
            return self.sampler_node.MossPhotonSamplerAdvanced().sample(
                model, tiny.make_cond(23), tiny.make_cond(9), {"samples": tiny.latent.clone()}, SEED, preset, cfg, steps,
                identity_profile=profile, finish_model=finish_model, preview=False, **dict(base, **widgets))

    def run_stock(self, model, profile, steps=None):
        """The stock ``turbo_identity`` run (MossPhotonSampler); ``steps`` through the solver's override."""
        tiny = self.tiny
        with ss.known_sft():
            return self.run_simple(model, profile, "turbo_identity") if steps is None else ss.run(
                tiny, model=model, identity_profile=profile, preset_id="turbo_identity", cfg=1.0,
                overrides={"steps": steps})

    def samples(self, out):
        if isinstance(out, dict):
            return out["result"][0]["samples"]
        return out.latent["samples"]

    def report(self, out):
        return out["result"][1] if isinstance(out, dict) else out.report

    def equal(self, a, b):
        return bool(self.torch.equal(a, b))

    def resume(self, carry, real_fn=None):
        """A KSAMPLER that runs ``real_fn`` (euler by default) from the carried x after the first segment (the stock hand-over)."""
        real_fn = real_fn or self.kd.sample_euler

        def resume(model, x, sig, extra_args=None, callback=None, disable=None, **opts):
            start = x if carry["x"] is None else carry["x"]
            carry["x"] = real_fn(model, start, sig, extra_args=extra_args, callback=callback, disable=disable)
            return carry["x"]
        return self.samplers.KSAMPLER(resume)

    # ---- instruments
    def recorder(self):
        """Patch ``latent_preview.prepare_callback`` to record the input ``x`` of every step (a copy)."""
        seen = []

        def prepare_callback(model, steps, x0_output_dict=None):
            def callback(step, x0, x, total):
                seen.append((step, x.detach().clone(), x0.detach().clone()))
            return callback
        module = types.ModuleType("latent_preview")
        module.prepare_callback = prepare_callback
        return seen, mock.patch.dict(sys.modules, {"latent_preview": module})

    def forward_counter(self):
        """Count the real elements that enter the network, per step (a step ends at its callback)."""
        net = self.tiny.model.diffusion_model
        original = net.forward
        state = {"elements": 0, "per_step": [], "calls": 0}

        def forward(x, t, *a, **k):
            state["elements"] += int(x.shape[0])
            state["calls"] += 1
            return original(x, t, *a, **k)

        def prepare_callback(model, steps, x0_output_dict=None):
            def callback(step, x0, x, total):
                state["per_step"].append(state["elements"] - sum(state["per_step"]))
            return callback
        module = types.ModuleType("latent_preview")
        module.prepare_callback = prepare_callback
        net.forward = forward
        self.addCleanup(lambda: net.__dict__.pop("forward", None))
        return state, mock.patch.dict(sys.modules, {"latent_preview": module})


class OffEqualsStockTest(DarkroomBase):
    """Guarantee (a)."""

    def test_an_all_neutral_plan_is_bit_equal_to_the_stock_turbo_identity_run(self):
        m, p = self.constant_chain()
        stock = self.run_stock(m, p, steps=12)
        off = self.run_dk(m, p, OFF)
        self.assertTrue(self.equal(self.samples(off), self.samples(stock)))
        self.assertEqual(self.report(off)["output"]["latent_sha256"], self.report(stock)["output"]["latent_sha256"])
        self.assertEqual(self.report(off)["inputs"]["noise_sha256"], self.report(stock)["inputs"]["noise_sha256"])
        self.assertEqual(self.report(off)["sampling"]["steps"], 12)
        self.assertEqual(self.report(off)["sampling"]["sigmas"], self.report(stock)["sampling"]["sigmas"])
        control = self.run_dk(m, p, OFF, finish_cfg=1.5, r_f=0.3)
        self.assertFalse(self.equal(self.samples(control), self.samples(stock)), "control: a real phase changes the latent")

    def test_the_same_with_other_step_counts(self):
        m, p = self.constant_chain()
        for steps in (8, 12, 16):
            with self.subTest(steps=steps):
                stock = self.run_stock(m, p, steps=steps)
                off = self.run_dk(m, p, OFF, steps=steps)
                self.assertTrue(self.equal(self.samples(off), self.samples(stock)))
                self.assertEqual(self.report(off)["darkroom"]["steps"], steps)

    def test_neutral_through_the_advanced_node_is_the_stock_run_too(self):
        m, p = self.constant_chain()
        stock = self.run_stock(m, p, steps=12)
        off = self.run_adv(m, p, PORTRAIT, finish_cfg=1.0, w_commit=1.0)
        self.assertTrue(self.equal(self.samples(off), self.samples(stock)))
        self.assertTrue(self.report(off)["darkroom"]["neutral"])

    def test_off_is_one_segment_on_one_clone_with_the_stock_sampler(self):
        m, p = self.constant_chain()
        block = self.report(self.run_dk(m, p, OFF))["darkroom"]
        self.assertEqual((len(block["segments"]), block["clones_built"], block["neutral"]), (1, 1, True))
        self.assertEqual(block["character"]["mode"], "constant_profile")

    def test_off_equals_stock_at_batch_two(self):
        m, p = self.constant_chain()
        lat = {"samples": self.torch.zeros(2, *self.tiny.latent.shape[1:])}
        stock = ss.run(self.tiny, model=m, identity_profile=p, preset_id="turbo_identity", cfg=1.0, latent=lat,
                       overrides={"steps": 12})
        with ss.known_sft():
            off = ss.run(self.tiny, model=m, identity_profile=p, preset_id=PORTRAIT, cfg=1.0, latent=lat, node_id=NODE,
                         darkroom=self.darkroom.DarkroomRequest(overrides=dict(NEUTRAL)))
        self.assertTrue(self.equal(off.latent["samples"], stock.latent["samples"]))

    def test_a_forced_split_of_the_neutral_run_is_still_the_stock_run(self):
        m, p = self.constant_chain()
        stock = self.run_stock(m, p, steps=12)
        cut = self.run_dk(m, p, OFF, split_at=(3, 8), steps=12)
        self.assertEqual(len(self.report(cut)["darkroom"]["segments"]), 3)
        self.assertTrue(self.equal(self.samples(cut), self.samples(stock)))

    def test_off_through_a_phased_profile_equals_the_constant_profile(self):
        m, p, _, _ = self.phased_chain()
        k_model, k_profile = self.constant_chain(1.0)
        const = self.run_dk(k_model, k_profile, "darkroom_off", steps=12)
        phased = self.run_dk(m, p, "darkroom_off", steps=12)
        self.assertTrue(self.equal(self.samples(phased), self.samples(const)))
        self.assertEqual(self.report(phased)["darkroom"]["character"]["mode"], "phased_adapter")



class HandoverTest(DarkroomBase):
    """Guarantee (b), the hand-over: the darkroom result equals a chain of stock CFGGuider + euler segments, each on the
    stock constant-weight model, resumed from the previous segment's x. No code of ``core/darkroom.py`` in the reference."""

    def chain(self, segments, noise=None):
        """segments: [(model, cfg, first, stop)] over ``self.sigmas12``; returns the final latent of the stock chain."""
        torch = self.torch
        noise = self.sample.prepare_noise(self.tiny.latent, SEED) if noise is None else noise
        carry, out = {"x": None}, None
        for model, cfg, first, stop in segments:
            guider = self.samplers.CFGGuider(model)
            guider.set_conds(self.tiny.make_cond(23), self.tiny.make_cond(9))
            guider.set_cfg(cfg)
            with torch.inference_mode():
                out = guider.sample(noise, self.tiny.latent.clone(), self.resume(carry), self.sigmas12[first:stop + 1],
                                    seed=SEED)
        return out

    def test_the_portrait_run_equals_the_hand_chained_stock_segments(self):
        m, p, d_model, d_profile = self.phased_chain()
        result = self.run_dk(m, p, PORTRAIT)
        weights = textbook_applied(self.sigmas12.tolist(), 1.0, 1.2, 0.06, 0.17)
        cfgs = [1.0] * 7 + [2.0] * 5
        runs, start = [], 0
        for k in range(1, 13):
            if k == 12 or (weights[k], cfgs[k]) != (weights[start], cfgs[start]):
                runs.append((weights[start], cfgs[start], start, k))
                start = k
        self.assertGreater(len(runs), 3)
        segments = [(self.lora(d_model, "A.safetensors", role="character", strength_model=w, profile_in=d_profile)[0],
                     cfg, first, stop) for w, cfg, first, stop in runs]
        expected = self.chain(segments)
        self.assertTrue(self.equal(self.samples(result), expected.to(self.samples(result).dtype)))
        self.assertEqual([(s["first_step"], s["last_step"], s["cfg"], s["weight"])
                          for s in self.report(result)["darkroom"]["segments"]],
                         [(first, stop - 1, cfg, w) for w, cfg, first, stop in runs])

    def test_a_guidance_switch_alone_equals_two_stock_segments(self):
        m, p = self.constant_chain()
        result = self.run_dk(m, p, "darkroom_off", steps=12, finish_cfg=2.0, r_f=0.17)
        expected = self.chain([(m, 1.0, 0, 7), (m, 2.0, 7, 12)])
        self.assertTrue(self.equal(self.samples(result), expected.to(self.samples(result).dtype)))

    def test_the_handover_is_exact_at_every_cut_point(self):
        """One more cut inside a deterministic segment changes nothing (euler is stateless between steps)."""
        m, p = self.constant_chain()
        whole = self.run_dk(m, p, "darkroom_off", steps=12, finish_cfg=2.0, r_f=0.17)
        for cut in (1, 4, 7, 9, 11):
            with self.subTest(cut=cut):
                split = self.run_dk(m, p, "darkroom_off", steps=12, finish_cfg=2.0, r_f=0.17, split_at=(cut,))
                self.assertTrue(self.equal(self.samples(split), self.samples(whole)))

    def test_the_model_switch_equals_the_stock_segments_on_the_second_model(self):
        m, p, d_model, d_profile = self.phased_chain()
        finish_model = self.lora(self.patcher, "A.safetensors", role="character", strength_model=1.2)[0]
        switched = self.run_dk(m, p, PORTRAIT, finish_model=finish_model, switch=True)
        plain = self.run_dk(m, p, PORTRAIT)
        self.assertFalse(self.equal(self.samples(switched), self.samples(plain)), "control: the second model is used")
        weights = textbook_applied(self.sigmas12.tolist(), 1.0, 1.2, 0.06, 0.17)
        segments, start = [], 0
        for k in range(1, 8):
            if k == 7 or weights[k] != weights[start]:
                segments.append((self.lora(d_model, "A.safetensors", role="character", strength_model=weights[start],
                                           profile_in=d_profile)[0], 1.0, start, k))
                start = k
        segments.append((finish_model, 2.0, 7, 12))
        expected = self.chain(segments)
        self.assertTrue(self.equal(self.samples(switched), expected.to(self.samples(switched).dtype)))
        block = self.report(switched)["darkroom"]
        self.assertEqual([s["model"] for s in block["segments"]], ["main"] * (len(block["segments"]) - 1) + ["finish"])

    def test_sa2_schedule_equals_the_stock_run_with_the_same_sigmas(self):
        m, p = self.constant_chain()
        sa2 = self.darkroom.sa2_values()
        off = self.run_dk(m, p, OFF, schedule="sa2_8", steps=8)
        with ss.known_sft():
            stock = ss.run(self.tiny, model=m, identity_profile=p, preset_id="turbo_identity", cfg=1.0, sigmas=sa2,
                           node_id="MossPhotonSamplerAdvanced")
        self.assertTrue(self.equal(self.samples(off), self.samples(stock)))
        report = self.report(off)
        self.assertEqual(report["sampling"]["sigmas"]["source"], "input")
        self.assertIn("custom_sigmas", ss.codes(report))
        self.assertIn("custom_sigmas", report["label"]["not_reference_reasons"])
        self.assertEqual(report["darkroom"]["schedule"], "sa2_8")
        # the phases of the portrait preset on it: r = .00 .04 | .08 .12 | .17 .24 .375 .583
        full = self.run_dk(*self.phased_chain()[:2], PORTRAIT, schedule="sa2_8", steps=8)
        rows = {p["name"]: p["steps"] for p in self.report(full)["darkroom"]["phases"]}
        self.assertEqual(rows, {"compose": 2, "commit": 2, "finish": 4})
        self.assertEqual(self.report(full)["forwards"]["sample_forwards"], 4 * 1 + 4 * 2)


class PhaseIsolationTest(DarkroomBase):
    """Guarantee (b), isolation: a setting of one phase changes the latent only from that phase on."""

    def inputs_of(self, **ov):
        m, p, _, _ = self.phased_chain()
        seen, patch = self.recorder()
        with patch:
            out = self.run_dk(m, p, PORTRAIT, preview=True, **ov)
        self.assertEqual([s[0] for s in seen], list(range(len(seen))))
        return [s[1] for s in seen], [s[2] for s in seen], out

    def first_difference(self, a, b):
        diff = [i for i in range(len(a)) if not self.equal(a[i], b[i])]
        return diff[0] if diff else None

    def test_finish_settings_leave_every_earlier_step_alone(self):
        base_x, base_x0, _ = self.inputs_of()
        # guidance acts inside step 7 (its x0 differs); noise is added after the step (only the next input differs)
        for name, ov, first_x0 in (("finish cfg", dict(finish_cfg=3.0), 7), ("noise", dict(eta=0.3), 8)):
            with self.subTest(name):
                x, x0, _ = self.inputs_of(**ov)
                # step 7 is the first finish step: its input and x0 readout still come from the unchanged commit phase
                self.assertEqual(self.first_difference(x, base_x), 8, "the input of step 8 is the first one that differs")
                self.assertEqual(self.first_difference(x0, base_x0), first_x0)

    def test_the_negative_prompt_matters_only_in_the_finish_phase(self):
        base_x, _, _ = self.inputs_of()
        m, p, _, _ = self.phased_chain()
        seen, patch = self.recorder()
        with patch:
            self.run_dk(m, p, PORTRAIT, preview=True, negative=self.tiny.make_cond(31))
        x = [s[1] for s in seen]
        self.assertEqual(self.first_difference(x, base_x), 8)

    def test_the_commit_weight_leaves_compose_and_the_first_commit_step_alone(self):
        base_x, base_x0, _ = self.inputs_of()
        x, x0, _ = self.inputs_of(w_commit=1.5)
        # steps 0-3 run at level 0 (weight 1.0) in both runs; step 4 is the first one at another weight
        self.assertEqual(self.first_difference(x, base_x), 5)
        self.assertEqual(self.first_difference(x0, base_x0), 4)

    def test_the_compose_guidance_changes_everything_after_step_zero_only(self):
        base_x, _, _ = self.inputs_of()
        x, _, _ = self.inputs_of(compose_cfg=1.5, r_f=0.17)
        self.assertEqual(self.first_difference(x, base_x), 1)

    def test_the_end_of_commit_moves_only_the_steps_after_the_old_boundary(self):
        base_x, _, _ = self.inputs_of()
        # r_k 0.17 -> 0.4 stretches the ramp: the first weight change is at step 4 in both, the levels differ from step 4
        x, _, _ = self.inputs_of(r_k=0.4, r_f=0.4)
        first = self.first_difference(x, base_x)
        self.assertIsNotNone(first)
        self.assertGreaterEqual(first, 4)

    def test_the_reexpose_leaves_the_steps_before_r_k_alone(self):
        base_x, _, _ = self.inputs_of(r_k=0.3, r_f=0.3)
        x, _, _ = self.inputs_of(r_k=0.3, r_f=0.3, reexpose=True, sigma_r=0.6)
        # r_k = 0.3 is reached at step 9 of the tiny schedule: inputs 0..9 are identical, the jump is at the input of step 9
        k = next(i for i, r in enumerate(self.progress()) if r >= 0.3)
        self.assertEqual(self.first_difference(x[:k], base_x[:k]), None)
        self.assertEqual(self.first_difference(x[k:k + 1], base_x[k:k + 1]), 0, "step k starts from the re-noised x")

    def progress(self):
        s = self.sigmas12.tolist()
        return [(s[0] - v) / (s[0] - s[-1]) for v in s[:-1]]


class IndependentRngTest(DarkroomBase):
    """Guarantee (b), randomness: re-expose and finish noise use their own seeded generators and never touch the main
    noise stream or torch's global generator."""

    def reexpose_args(self):
        return dict(steps=12, reexpose=True, r_k=0.3, r_f=0.3, sigma_r=0.6)

    def test_reexpose_equals_the_hand_built_renoise_with_its_own_seed(self):
        torch = self.torch
        m, p = self.constant_chain()
        result = self.run_dk(m, p, "darkroom_off", **self.reexpose_args())
        sig = self.sigmas12.tolist()
        r = [(sig[0] - v) / (sig[0] - sig[-1]) for v in sig[:-1]]
        k = next(i for i, v in enumerate(r) if v >= 0.3)
        tail = [0.6] + [v for v in sig[1:] if v < 0.6 - 1e-4]
        carry, last = {"x": None}, {}

        def euler_keep(model, x, sigs, extra_args=None, callback=None, disable=None, **opts):
            def cb(info):
                last["x0"] = info["denoised"]
                if callback is not None:
                    callback(info)
            return self.kd.sample_euler(model, x if carry["x"] is None else carry["x"], sigs, extra_args=extra_args,
                                        callback=cb, disable=disable)

        def run(sigmas):
            guider = self.samplers.CFGGuider(m)
            guider.set_conds(self.tiny.make_cond(23), self.tiny.make_cond(9))
            guider.set_cfg(1.0)
            noise = self.sample.prepare_noise(self.tiny.latent, SEED)
            sampler = self.samplers.KSAMPLER(lambda *a, **kw: carry.update(x=euler_keep(*a, **kw)) or carry["x"])
            with torch.inference_mode():
                return guider.sample(noise, self.tiny.latent.clone(), sampler, torch.tensor(sigmas, dtype=torch.float32),
                                     seed=SEED)

        run(sig[:k + 1])
        gen = torch.Generator().manual_seed(SEED + 1000003)
        fresh = torch.randn(tuple(last["x0"].shape), dtype=torch.float32, generator=gen)
        carry["x"] = 0.6 * fresh.to(last["x0"].dtype) + (1.0 - 0.6) * last["x0"]
        expected = run(tail)
        self.assertTrue(self.equal(self.samples(result), expected.to(self.samples(result).dtype)))
        block = self.report(result)["darkroom"]
        self.assertEqual(block["rng"]["reexpose"], {"enabled": True, "offset": 1000003, "seed": SEED + 1000003})
        self.assertEqual([p_["steps"] for p_ in block["paths"]], [k, len(tail) - 1])

    def test_finish_noise_equals_the_hand_built_ancestral_steps_with_its_own_seed(self):
        torch = self.torch
        m, p = self.constant_chain()
        result = self.run_dk(m, p, "darkroom_off", steps=12, eta=0.3, r_f=0.17)
        gen = torch.Generator().manual_seed(SEED + 2000003)

        def ancestral(model, x, sigs, extra_args=None, callback=None, disable=None, **opts):
            def noise_sampler(a, b):
                return torch.randn(tuple(x.shape), dtype=torch.float32, generator=gen).to(x.dtype)
            return self.kd.sample_euler_ancestral_RF(model, x, sigs, extra_args=extra_args, callback=callback,
                                                     disable=disable, eta=0.3, s_noise=1.0, noise_sampler=noise_sampler)

        carry = {"x": None}
        noise = self.sample.prepare_noise(self.tiny.latent, SEED)
        out = None
        for first, stop, fn in ((0, 7, self.kd.sample_euler), (7, 12, ancestral)):
            guider = self.samplers.CFGGuider(m)
            guider.set_conds(self.tiny.make_cond(23), self.tiny.make_cond(9))
            guider.set_cfg(1.0)
            with torch.inference_mode():
                out = guider.sample(noise, self.tiny.latent.clone(), self.resume(carry, fn),
                                    self.sigmas12[first:stop + 1], seed=SEED)
        self.assertTrue(self.equal(self.samples(result), out.to(self.samples(result).dtype)))
        self.assertEqual(self.report(result)["darkroom"]["rng"]["noise"],
                         {"enabled": True, "offset": 2000003, "seed": SEED + 2000003})
        self.assertTrue(self.report(result)["sampling"]["ancestral_noise"])

    def test_the_main_noise_stream_is_untouched(self):
        m, p = self.constant_chain()
        stock = self.run_stock(m, p, steps=12)
        sha = self.report(stock)["inputs"]["noise_sha256"]
        for name, ov in (("reexpose", self.reexpose_args()), ("noise", dict(steps=12, eta=0.3, r_f=0.17)),
                         ("both", dict(self.reexpose_args(), eta=0.3))):
            with self.subTest(name):
                self.assertEqual(self.report(self.run_dk(m, p, "darkroom_off", **ov))["inputs"]["noise_sha256"], sha)
        self.assertEqual(sha, self.solver.rng.tensor_sha256(self.sample.prepare_noise(self.tiny.latent, SEED)))

    def test_every_random_draw_after_the_main_noise_uses_a_private_generator(self):
        torch = self.torch
        m, p = self.constant_chain()
        calls = []
        real = {name: getattr(torch, name) for name in ("randn", "randn_like", "rand", "rand_like", "normal", "randint")}

        def spy(name):
            def wrapper(*a, **k):
                calls.append((name, k.get("generator")))
                return real[name](*a, **k)
            return wrapper

        patches = [mock.patch.object(torch, name, spy(name)) for name in real]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.run_dk(m, p, "darkroom_off", **dict(self.reexpose_args(), eta=0.3))
        for patch in patches:
            patch.stop()
        self.assertEqual([c for c in calls if c[0] != "randn"], [], "only torch.randn with an explicit generator is used")
        default = [c for c in calls if c[1] is torch.default_generator]
        private = [c for c in calls if c[1] is not None and c[1] is not torch.default_generator]
        self.assertEqual(len(default), 1, "the main noise only")
        self.assertGreaterEqual(len(private), 1 + 5, "one re-noise and the finish-noise steps")
        self.assertFalse([c for c in calls if c[1] is None])

    def test_the_global_generator_is_where_the_main_noise_left_it(self):
        torch = self.torch
        m, p = self.constant_chain()
        self.sample.prepare_noise(self.tiny.latent, SEED)
        after_main_noise = torch.get_rng_state().clone()
        self.run_dk(m, p, "darkroom_off", **dict(self.reexpose_args(), eta=0.3))
        self.assertTrue(self.equal(torch.get_rng_state(), after_main_noise))

    def test_the_streams_are_independent_of_each_other(self):
        m, p = self.constant_chain()
        a = self.run_dk(m, p, "darkroom_off", **dict(self.reexpose_args()))
        b = self.run_dk(m, p, "darkroom_off", **dict(self.reexpose_args(), eta=0.3))
        c = self.run_dk(m, p, "darkroom_off", **dict(self.reexpose_args(), eta=0.3))
        self.assertFalse(self.equal(self.samples(a), self.samples(b)))
        self.assertTrue(self.equal(self.samples(b), self.samples(c)), "a seeded run repeats bit-equal")
        self.assertNotEqual(self.darkroom.reexpose_seed(SEED), self.darkroom.noise_seed(SEED))
        self.assertNotEqual(self.darkroom.reexpose_seed(SEED), SEED)


class NoLeakTest(DarkroomBase):
    """Guarantee (c)."""

    def snapshot(self, p, weights=True):
        digest = hashlib.sha256()
        for name, param in sorted(p.model.diffusion_model.named_parameters()):
            digest.update(name.encode())
            digest.update(param.detach().contiguous().numpy().tobytes())
        return {
            "patch_keys": sorted(p.patches), "patch_entries": sum(len(v) for v in p.patches.values()),
            "uuid": str(p.patches_uuid), "attachment_keys": sorted(p.attachments),
            "attachment_ids": {k: id(v) for k, v in p.attachments.items()},
            "model_options": copy.deepcopy(p.model_options), "object_patches": dict(p.object_patches),
            "hook_patches": dict(p.hook_patches), "hook_backup": dict(p.hook_backup), "current_hooks": p.current_hooks,
            "weights": digest.hexdigest() if weights else None,
        }

    def setUp(self):
        super().setUp()
        self.m, self.p, self.d_model, _ = self.phased_chain()
        self.finish_model = self.lora(self.patcher, "A.safetensors", role="character", strength_model=1.2)[0]
        self.base_plain = ss.tensor_sha(tiny_moss.sample_euler(self.tiny))
        self.base_run = ss.tensor_sha(self.samples(self.run_dk(self.m, self.p, PORTRAIT)))

    def assert_clean(self):
        gc.collect()
        self.assertEqual(ss.tensor_sha(tiny_moss.sample_euler(self.tiny)), self.base_plain)
        self.assertEqual(ss.tensor_sha(self.samples(self.run_dk(self.m, self.p, PORTRAIT))), self.base_run,
                         "a darkroom run repeats bit-equal")
        for model in (self.patcher, self.m, self.d_model, self.finish_model):
            self.assertEqual(model.hook_patches, {})
            self.assertEqual(model.hook_backup, {})
            self.assertIsNone(model.current_hooks)
        self.assertEqual(len(self.patcher.patches), 0)
        self.assertEqual(len(self.m.patches), len(self.d_model.patches), "the phased adapter adds no patch to the MODEL")

    def stub_preview(self, at_step, exc_factory):
        def prepare_callback(model, steps, x0_output_dict=None):
            def callback(step, x0, x, total):
                if step == at_step:
                    raise exc_factory()
            return callback
        module = types.ModuleType("latent_preview")
        module.prepare_callback = prepare_callback
        return mock.patch.dict(sys.modules, {"latent_preview": module})

    def full_run(self, **kw):
        return self.run_dk(self.m, self.p, PORTRAIT, finish_model=self.finish_model, switch=True, eta=0.2, **kw)

    def test_no_incoming_model_is_mutated_by_a_full_run(self):
        light = {n: self.snapshot(model, weights=False) for n, model in (("main", self.m), ("finish", self.finish_model))}
        tiny_moss.sample_euler(self.tiny)  # the shared module holds the last run's weights from setUp: reload the base
        before = {n: self.snapshot(model) for n, model in (("base", self.patcher), ("main", self.m),
                                                           ("distill", self.d_model), ("finish", self.finish_model))}
        result = self.full_run()
        self.assertEqual(self.report(result)["forwards"]["match"], True)
        self.assertEqual({n: self.snapshot(model, weights=False) for n, model in (("main", self.m),
                                                                                   ("finish", self.finish_model))}, light)
        tiny_moss.sample_euler(self.tiny)  # load another clone so the shared module drops the last level's patches
        after = {n: self.snapshot(model) for n, model in (("base", self.patcher), ("main", self.m),
                                                          ("distill", self.d_model), ("finish", self.finish_model))}
        self.assertEqual(after, before)

    def test_at_most_one_segment_clone_is_alive_at_a_time(self):
        real_level, real_counter = self.lora_phase.make_level_model, self.solver.install_counter
        levels, runs = [], []

        def level(model, adapter, weight):
            gc.collect()
            self.assertEqual([r() for r in levels], [None] * len(levels), "the previous level clone is still alive")
            made = real_level(model, adapter, weight)
            levels.append(weakref.ref(made))
            return made

        def counter(model_clone, stats):
            gc.collect()
            alive = [r for r in runs if r() is not None]
            self.assertEqual(alive, [], "the previous segment's model is still alive")
            runs.append(weakref.ref(model_clone))
            return real_counter(model_clone, stats)

        with mock.patch.object(self.lora_phase, "make_level_model", level), \
                mock.patch.object(self.solver, "install_counter", counter):
            self.run_dk(self.m, self.p, PORTRAIT)
        self.assertGreaterEqual(len(levels), 5)
        gc.collect()
        self.assertEqual([r() for r in levels + runs], [None] * (len(levels) + len(runs)))

    def test_success_leaves_nothing(self):
        self.full_run()
        self.assert_clean()

    def test_an_interrupt_in_the_finish_phase_leaves_nothing(self):
        with self.stub_preview(9, self.mm.InterruptProcessingException):
            with self.assertRaises(self.mm.InterruptProcessingException):
                self.full_run(preview=True)
        self.assert_clean()

    def test_an_interrupt_in_the_commit_phase_leaves_nothing(self):
        with self.stub_preview(4, self.mm.InterruptProcessingException):
            with self.assertRaises(self.mm.InterruptProcessingException):
                self.full_run(preview=True)
        self.assert_clean()

    def test_a_real_interrupt_flag_between_forwards_leaves_nothing(self):
        net, calls, mm = self.tiny.model.diffusion_model, [], self.mm
        original = net.forward

        def forward(x, t, *a, **k):
            calls.append(1)
            if len(calls) == 11:
                mm.interrupt_current_processing(True)
            return original(x, t, *a, **k)
        net.forward = forward
        try:
            with self.assertRaises(mm.InterruptProcessingException):
                self.full_run()
        finally:
            del net.forward
            mm.interrupt_current_processing(False)
        self.assertLess(len(calls), 17)
        self.assert_clean()

    def test_an_error_while_sampling_a_later_segment_leaves_nothing(self):
        with self.stub_preview(9, lambda: RuntimeError("sampler failed")):
            with self.assertRaises(RuntimeError):
                self.full_run(preview=True)
        self.assert_clean()

    def test_an_error_while_building_a_later_clone_leaves_nothing(self):
        real, calls = self.lora_phase.make_level_model, []

        def level(model, adapter, weight):
            calls.append(weight)
            if len(calls) == 3:
                raise RuntimeError("clone failed")
            return real(model, adapter, weight)
        with mock.patch.object(self.lora_phase, "make_level_model", level):
            with self.assertRaises(RuntimeError):
                self.full_run()
        self.assert_clean()

    def test_an_error_in_the_finish_model_clone_leaves_nothing(self):
        def broken(*a, **k):
            raise RuntimeError("finish clone failed")
        with mock.patch.object(self.finish_model, "clone", broken):
            with self.assertRaises(RuntimeError):
                self.full_run()
        self.assert_clean()


class ForwardsAndReportTest(DarkroomBase):
    """Guarantee (d)."""

    def test_forwards_per_step_and_per_phase_are_counted_from_the_network(self):
        m, p, _, _ = self.phased_chain()
        state, patch = self.forward_counter()
        with patch:
            result = self.run_dk(m, p, PORTRAIT, preview=True)
        self.assertEqual(state["per_step"], [1] * 7 + [2] * 5, "cfg 1 = 1 per step, cfg above 1 = 2 per step")
        report = self.report(result)
        self.assertEqual(report["forwards"]["sample_forwards"], sum(state["per_step"]))
        self.assertEqual((report["forwards"]["expected_sample_forwards"], report["forwards"]["match"]), (17, True))
        rows = {r["name"]: r for r in report["darkroom"]["phases"]}
        self.assertEqual([rows[n]["observed_sample_forwards"] for n in ("compose", "commit", "finish")],
                         [sum(state["per_step"][0:3]), sum(state["per_step"][3:7]), sum(state["per_step"][7:12])])
        for row in rows.values():
            self.assertEqual(row["observed_sample_forwards"], row["expected_sample_forwards"])

    def test_the_report_records_phases_sigmas_weights_seeds_recipe_and_label(self):
        from moss_photon.core import report as report_mod
        m, p, _, _ = self.phased_chain()
        report = self.report(self.run_dk(m, p, PORTRAIT))
        self.assertEqual(report_mod.validate_report(report), [])
        block = report["darkroom"]
        version = self.presets.catalog()[PORTRAIT]["preset_version"]
        self.assertEqual((block["recipe"], block["label"]), ("%s@%s" % (PORTRAIT, version), "darkroom_experimental"))
        sigmas = self.sigmas12.tolist()
        rows = {r["name"]: r for r in block["phases"]}
        self.assertEqual([rows[n]["steps"] for n in ("compose", "commit", "finish")], [3, 4, 5])
        self.assertAlmostEqual(rows["compose"]["sigma_end"], 1.0 - 0.06, places=12)
        self.assertEqual(rows["commit"]["weight_first"], 1.0)
        self.assertEqual(rows["commit"]["weight_last"], 1.2)
        self.assertEqual((rows["finish"]["cfg"], rows["compose"]["cfg"]), (2.0, 1.0))
        self.assertEqual(block["paths"][0]["sigmas_sha256"], report["sampling"]["sigmas"]["sha256"])
        self.assertEqual(block["rng"]["main_seed"], SEED)
        self.assertEqual(block["settings"]["commit"], {"r_c": 0.06, "r_k": 0.17, "w_compose": 1.0, "w_commit": 1.2})
        self.assertEqual(report["label"]["parity_class"], "not_reference")
        self.assertIn("darkroom_experimental", report["label"]["not_reference_reasons"])
        self.assertEqual(report["sampling"]["steps"], 12)
        self.assertEqual(len(sigmas), 13)
        self.assertEqual(block["clones_built"], len(block["segments"]))
        self.assertNotIn("phase", report["lora"], "the darkroom replaces the profile's own phase plan")
        self.assertIn("darkroom %s@%s" % (PORTRAIT, version), report_mod.format_summary(report))
        self.assertEqual(report["preset"]["id"], PORTRAIT)
        self.assertEqual(report["preset"]["version"], version)

    def test_warnings(self):
        m, p, _, _ = self.phased_chain()
        report = self.report(self.run_dk(m, p, PORTRAIT))
        codes = ss.codes(report)
        self.assertEqual(codes.count("darkroom_experimental"), 1)
        self.assertNotIn("phase_lora_experimental", codes)
        self.assertNotIn("negative_unused_cfg1", codes, "the finish phase uses the negative prompt")
        warn = [w for w in report["warnings"] if w["code"] == "darkroom_experimental"][0]
        self.assertEqual(warn["severity"], "warning")
        self.assertIn("%s@%s" % (PORTRAIT, self.presets.catalog()[PORTRAIT]["preset_version"]), warn["message"])
        self.assertEqual(warn["detail"]["finish_cfg_source"], "preset")
        k_model, k_profile = self.constant_chain()
        off = self.report(self.run_dk(k_model, k_profile, "darkroom_off"))
        self.assertIn("negative_unused_cfg1", ss.codes(off), "no phase uses the negative prompt")

    def test_off_is_labelled_experimental_too(self):
        k_model, k_profile = self.constant_chain()
        label = self.report(self.run_dk(k_model, k_profile, "darkroom_off"))["label"]
        self.assertEqual(label["parity_class"], "not_reference")
        self.assertIn("darkroom_experimental", label["not_reference_reasons"])

    def test_the_recipe_hash_sees_the_recipe(self):
        m, p, _, _ = self.phased_chain()
        a = self.report(self.run_dk(m, p, PORTRAIT))
        b = self.report(self.run_dk(m, p, PORTRAIT))
        c = self.report(self.run_dk(m, p, PORTRAIT, finish_cfg=3.0))
        self.assertEqual(a["recipe_hash"], b["recipe_hash"])
        self.assertNotEqual(a["recipe_hash"], c["recipe_hash"])

    def test_slg_in_compose_skips_only_while_the_guidance_is_above_one_and_never_in_the_finish(self):
        k_model, k_profile = self.constant_chain()
        report = self.report(self.run_dk(k_model, k_profile, "darkroom_off", steps=12, compose_cfg=1.5, compose_slg=True,
                                         r_f=0.4, finish_cfg=1.5))
        # r < 0.4: 10 steps with SLG; the 2 finish steps run plain CFG 1.5
        self.assertEqual(report["forwards"]["slg_skip_calls"], 10)
        self.assertEqual((report["forwards"]["sample_forwards"], report["forwards"]["match"]), (24, True))
        self.assertTrue(report["darkroom"]["phases"][0]["slg"])
        self.assertFalse(report["darkroom"]["phases"][-1]["slg"])

    def test_batch_two_counts_sample_forwards(self):
        m, p, _, _ = self.phased_chain()
        lat = {"samples": self.torch.zeros(2, *self.tiny.latent.shape[1:])}
        with ss.known_sft():
            result = ss.run(self.tiny, model=m, identity_profile=p, preset_id=PORTRAIT, cfg=1.0, latent=lat,
                            node_id=NODE, allow_phased_lora=True, darkroom=self.darkroom.DarkroomRequest())
        fw = self.report(result)["forwards"]
        self.assertEqual((fw["expected_branch_forwards"], fw["expected_sample_forwards"], fw["sample_forwards"]),
                         (17, 34, 34))

    def test_the_simple_node_returns_the_text_and_the_report(self):
        m, p, _, _ = self.phased_chain()
        out = self.run_simple(m, p, PORTRAIT)
        self.assertEqual(sorted(out), ["result", "ui"])
        self.assertIn("darkroom %s@" % PORTRAIT, out["ui"]["text"][0])
        self.assertEqual(out["result"][1]["node"], "MossPhotonSampler")
        self.assertEqual(out["result"][1]["darkroom"]["character"]["mode"], "phased_adapter")


class RefusedBeforeAnyForwardTest(DarkroomBase):
    """Guarantee (e), and the typed errors of the requirements."""

    def instruments(self):
        state, patch = self.forward_counter()
        spies = {"level": mock.patch.object(self.lora_phase, "make_level_model", side_effect=AssertionError("clone built")),
                 "sample": mock.patch.object(self.samplers.CFGGuider, "sample", side_effect=AssertionError("sampled")),
                 "noise": mock.patch.object(self.solver.rng, "make_noise", side_effect=AssertionError("noise drawn"))}
        for spy in spies.values():
            spy.start()
            self.addCleanup(spy.stop)
        patch.start()
        self.addCleanup(patch.stop)
        return state

    def refused(self, exc, call):
        state = self.instruments()
        with self.assertRaises(exc) as cm:
            call()
        self.assertEqual(state["calls"], 0, "no forward ran")
        return cm.exception

    def test_a_finish_model_of_another_architecture(self):
        other = tiny_moss.build_tiny_moss(layers=4).patcher
        m, p, _, _ = self.phased_chain()
        err = self.refused(self.errors.UnsupportedModel, lambda: self.run_dk(m, p, PORTRAIT, finish_model=other,
                                                                             switch=True))
        self.assertIn("num_layers", str(err))

    def test_the_phased_adapter_must_be_on_the_model(self):
        m, p, _, _ = self.phased_chain()
        stripped = m.clone()
        stripped.remove_attachments(self.lora_phase.ATTACH_PHASE_LORA)
        self.refused(self.errors.IncompatibleLora, lambda: self.run_dk(stripped, p, PORTRAIT))


class DegradedThroughTheNodesTest(DarkroomBase):
    """MP-84 (owner 2026-11-02): what used to be refused before any forward is a warning and the run goes on with the nearest
    usable value (the former ``RefusedBeforeAnyForwardTest`` cases). Each case runs, counts its forwards right and names the
    adjustment in the report."""

    def ran(self, out, code, param=None):
        rep = self.report(out)
        self.assertTrue(self.torch.isfinite(self.samples(out)).all())
        self.assertTrue(rep["forwards"]["match"])
        hits = [w for w in rep["warnings"] if w["code"] == code and (param is None or w["param"] == param)]
        self.assertTrue(hits, "no %s%s in %s" % (code, "/" + param if param else "", [(w["code"], w["param"]) for w in rep["warnings"]]))
        return rep

    def test_invalid_orders_and_bounds_through_the_advanced_node(self):
        m, p, _, _ = self.phased_chain()
        for name, widgets, check in (
                ("r_c above r_k", dict(r_c=0.5, r_k=0.3), lambda s: (s["commit"]["r_c"], s["commit"]["r_k"], s["finish"]["r_f"]) == (0.17, 0.3, 0.5)),
                ("r_k above r_f", dict(r_k=0.4, r_f=0.2), lambda s: (s["commit"]["r_c"], s["commit"]["r_k"], s["finish"]["r_f"]) == (0.06, 0.2, 0.4)),
                ("finish before the commit end", dict(r_f=0.1), lambda s: (s["commit"]["r_k"], s["finish"]["r_f"]) == (0.1, 0.17)),
                ("weight window", dict(r_c=0.1, r_k=0.1, r_f=0.1), lambda s: s["commit"]["w_commit"] == s["commit"]["w_compose"]),
                ("cfg below one", dict(finish_cfg=0.5), lambda s: s["finish"]["cfg"] == 1.0),
                ("eta", dict(eta=1.5), lambda s: s["finish"]["eta"] == 1.0)):
            with self.subTest(name):
                rep = self.ran(self.run_adv(m, p, PORTRAIT, **widgets), "setting_adjusted")
                self.assertTrue(check(rep["darkroom"]["settings"]), rep["darkroom"]["settings"])

    def test_sa2_with_another_step_count_resamples_its_knots(self):
        m, p, _, _ = self.phased_chain()
        rep = self.ran(self.run_adv(m, p, PORTRAIT, schedule="sa2_8"), "setting_adjusted", "schedule")
        self.assertEqual((rep["darkroom"]["schedule"], rep["sampling"]["steps"]), ("sa2_8", 12))
        exact = self.report(self.run_adv(m, p, PORTRAIT, schedule="sa2_8", steps=8))
        self.assertNotIn("schedule", [w["param"] for w in exact["warnings"] if w["code"] == "setting_adjusted"],
                         "the measured eight steps are not adjusted")

    def test_the_plan_leaves_out_what_it_cannot_use_inside_the_run(self):
        m, p, _, _ = self.phased_chain()
        for name, ov, code, param in (
                ("reexpose before the first step", dict(reexpose=True, r_c=0.0, r_k=0.0, r_f=0.0, w_commit=1.0),
                 "darkroom_override_ignored", "reexpose"),
                ("reexpose past the last step", dict(reexpose=True, r_k=0.99, r_f=0.99), "darkroom_override_ignored",
                 "reexpose"),
                ("finish without a step", dict(r_f=1.0), "darkroom_override_ignored", "r_f"),
                ("switch without the second model", dict(switch=True), "darkroom_override_ignored", "switch")):
            with self.subTest(name):
                rep = self.ran(self.run_dk(m, p, PORTRAIT, **ov), code, param)
                if param == "reexpose":
                    self.assertFalse(rep["darkroom"]["settings"]["reexpose"]["enabled"])
                    self.assertEqual(len(rep["darkroom"]["paths"]), 1)
                if param == "switch":
                    self.assertFalse(rep["darkroom"]["settings"]["finish"]["switch"])
                    self.assertEqual({s["model"] for s in rep["darkroom"]["segments"]}, {"main"})

    def test_the_boost_without_a_loader_is_skipped_with_a_warning(self):
        """The core API without a boost loader: nothing can patch the constant character, so the boost is skipped."""
        k_model, k_profile = self.constant_chain()
        out = ss.run(self.tiny, model=k_model, identity_profile=k_profile, preset_id=PORTRAIT, cfg=1.0, node_id=NODE,
                     darkroom=self.darkroom.DarkroomRequest())
        rep = self.ran(out, "darkroom_boost_skipped")
        self.assertEqual(rep["darkroom"]["character"]["boost_skipped"], "no_identity_lora")

    def test_a_distill_and_a_character_are_optional(self):
        d_model, d_profile = self.distill()
        c_model, c_profile, _ = self.lora(self.patcher, "A.safetensors", role="character", strength_model=1.0)
        for recipe in (OFF, PORTRAIT, FULLBODY):
            for name, model, profile in (("no distill", c_model, c_profile), ("no character", d_model, d_profile),
                                         ("no profile", self.patcher, None)):
                with self.subTest(recipe=recipe, case=name):
                    out = self.run_dk(model, profile, recipe)
                    self.assertTrue(self.torch.isfinite(self.samples(out)).all())
                    self.assertEqual(self.report(out)["sampling"]["steps"], 12)
                    self.assertNotIn("missing_lora", ss.codes(self.report(out)))

    def test_a_bad_cfg_widget_is_adjusted_and_the_note_names_the_cfg_widget_not_the_finish_knob(self):
        m, p, _, _ = self.phased_chain()
        for call in (lambda: self.run_simple(m, p, PORTRAIT, cfg=0.5), lambda: self.run_adv(m, p, PORTRAIT, cfg=0.5)):
            rep = self.ran(call(), "setting_adjusted", "cfg")
            self.assertEqual(rep["darkroom"]["settings"]["finish"]["cfg"], 1.0)
            note = next(w for w in rep["warnings"] if w["code"] == "setting_adjusted" and w["param"] == "cfg")
            self.assertNotIn("finish_cfg", note["message"] + note["fix"])
        rep = self.ran(self.run_adv(m, p, PORTRAIT, finish_cfg=0.5), "setting_adjusted", "finish_cfg")
        self.assertEqual(rep["darkroom"]["settings"]["finish"]["cfg"], 1.0)


class NodeEndToEndTest(DarkroomBase):
    def test_both_nodes_run_both_darkroom_presets_on_a_phased_and_a_constant_profile(self):
        m, p, _, _ = self.phased_chain()
        k_model, k_profile = self.constant_chain()
        for preset in (PORTRAIT, FULLBODY):
            for label, model, profile, mode in (("phased", m, p, "phased_adapter"),
                                                ("constant", k_model, k_profile, "constant_profile_extra_patch")):
                for node in (self.run_simple, self.run_adv):
                    with self.subTest(preset=preset, profile=label, node=node.__name__):
                        report = node(model, profile, preset)["result"][1]
                        version = self.presets.catalog()[preset]["preset_version"]
                        self.assertEqual(report["darkroom"]["recipe"], "%s@%s" % (preset, version))
                        self.assertEqual(report["darkroom"]["character"]["mode"], mode)
                        self.assertTrue(report["forwards"]["match"])
                        self.assertEqual(report["label"]["parity_class"], "not_reference")
                        self.assertIn("darkroom_experimental", report["label"]["not_reference_reasons"])
                        self.assertEqual(report["darkroom"]["overrides"], [])

    def test_the_advanced_knobs_reach_the_plan_and_are_named_in_the_report(self):
        m, p, _, _ = self.phased_chain()
        report = self.report(self.run_adv(m, p, PORTRAIT, steps=16, finish_cfg=2.5, w_commit=1.1, r_f=0.3))
        block = report["darkroom"]
        self.assertEqual(block["steps"], 16)
        self.assertEqual(block["settings"]["finish"]["cfg"], 2.5)
        self.assertEqual(block["settings"]["commit"]["w_commit"], 1.1)
        self.assertEqual(block["settings"]["finish"]["r_f"], 0.3)
        self.assertEqual(sorted(block["overrides"]), ["finish_cfg", "r_f", "w_commit"])
        self.assertIn("sampling.steps", [d["path"] for d in report["preset"]["diff"]], "steps is the preset's own override")

    def test_every_knob_alone_overrides_the_preset_value(self):
        m, p, _, _ = self.phased_chain()
        plain = self.report(self.run_adv(m, p, PORTRAIT))["darkroom"]["settings"]
        for widget, value, path in (("compose_cfg", 1.1, ("compose", "cfg")), ("compose_slg", "on", ("compose", "slg")),
                                    ("r_c", 0.05, ("commit", "r_c")), ("r_k", 0.2, ("commit", "r_k")),
                                    ("w_compose", 0.9, ("commit", "w_compose")), ("w_commit", 1.1, ("commit", "w_commit")),
                                    ("reexpose", "on", ("reexpose", "enabled")), ("sigma_r", 0.7, ("reexpose", "sigma_r")),
                                    ("r_f", 0.2, ("finish", "r_f")), ("finish_cfg", 2.5, ("finish", "cfg")),
                                    ("eta", 0.1, ("finish", "eta"))):
            extra = {"reexpose": dict(r_k=0.3, r_f=0.3), "r_k": dict(r_f=0.25)}.get(widget, {})
            with self.subTest(widget=widget):
                got = self.report(self.run_adv(m, p, PORTRAIT, **dict(extra, **{widget: value})))["darkroom"]["settings"]
                self.assertNotEqual(got[path[0]][path[1]], plain[path[0]][path[1]])
                self.assertEqual(got[path[0]][path[1]], {"on": True}.get(value, value))


class FinishModelIgnoredTest(DarkroomBase):
    """MP-67: a finish_model connected while switch_model resolves to off is IGNORED with a warning, never an error."""

    def assert_ignored(self, out, plain):
        report = self.report(out)
        warn = [w for w in report["warnings"] if w["code"] == "finish_model_ignored"]
        self.assertEqual((len(warn), warn[0]["severity"]), (1, "warning"))
        self.assertNotIn("invalid_override", self.codes(report))
        block = report["darkroom"]
        self.assertFalse(block["settings"]["finish"]["switch"])
        self.assertEqual({s_["model"] for s_ in block["segments"]}, {"main"})
        self.assertEqual(report["forwards"]["match"], True)
        self.assertTrue(self.equal(self.samples(out), self.samples(plain)), "the run is the one without the finish_model")

    def codes(self, report):
        return [w["code"] for w in report["warnings"]]

    def test_the_widget_off_ignores_the_finish_model(self):
        m, p, _, _ = self.phased_chain()
        out = self.run_adv(m, p, PORTRAIT, finish_model=self.patcher, switch_model="off")
        self.assert_ignored(out, self.run_adv(m, p, PORTRAIT, switch_model="off"))

    def test_a_preset_without_the_switch_ignores_the_finish_model(self):
        m, p, _, _ = self.phased_chain()
        out = self.run_adv(m, p, PORTRAIT, finish_model=self.patcher)
        self.assert_ignored(out, self.run_adv(m, p, PORTRAIT))

    def test_the_core_run_warns_and_the_ignored_model_is_not_even_probed(self):
        other = tiny_moss.build_tiny_moss(layers=4).patcher  # would be unsupported_model were it looked at
        m, p, _, _ = self.phased_chain()
        out = self.run_dk(m, p, PORTRAIT, finish_model=other)
        self.assert_ignored(out, self.run_dk(m, p, PORTRAIT))

    def test_no_warning_without_a_finish_model_and_none_with_the_switch_on(self):
        m, p, _, _ = self.phased_chain()
        self.assertNotIn("finish_model_ignored", self.codes(self.report(self.run_dk(m, p, PORTRAIT))))
        on = self.run_dk(m, p, PORTRAIT, finish_model=self.patcher, switch=True)
        self.assertNotIn("finish_model_ignored", self.codes(self.report(on)))
        self.assertIn("finish", {s_["model"] for s_ in self.report(on)["darkroom"]["segments"]})


class ConstantProfileBoostTest(DarkroomBase):
    """MP-67: the default recipe on the default (constant) character profile. The commit boost is an EXTRA patch of the same
    LoRA, strength (w_commit - w_compose) on the commit-segment clones only."""

    def test_the_darkroom_presets_run_on_a_constant_profile_in_the_simple_node(self):
        k_model, k_profile = self.constant_chain(1.0)
        for recipe in (PORTRAIT, FULLBODY):
            with self.subTest(recipe=recipe):
                out = self.run_simple(k_model, k_profile, recipe)
                report = out["result"][1]
                block = report["darkroom"]
                self.assertEqual(block["character"], {"mode": "constant_profile_extra_patch", "stack_index": 1,
                                                      "profile_phase_plan_replaced": False, "base_strength": 1.0,
                                                      "boost_skipped": None})
                self.assertNotIn("darkroom_boost_skipped", ss.codes(report))
                self.assertEqual(report["forwards"]["match"], True)
                weights = [s_["weight"] for s_ in block["segments"]]
                self.assertEqual(weights[0], 1.0)
                self.assertEqual(weights[-1], self.block(recipe)["commit"]["w_commit"],
                                 "the finish keeps the committed weight of the preset")
                self.assertEqual(len(weights), 5)
                self.assertEqual(block["clones_built"], 5)
                self.assertEqual(report["label"]["parity_class"], "not_reference")

    def torch_weight(self, patched, key):
        """The weight of ``key`` with every patch of ``patched`` applied, from the original module weight (no loading)."""
        import comfy.lora
        name = key if isinstance(key, str) else key[0]
        out = patched.model.state_dict()[name].detach().clone().float()
        for patch in patched.patches[key]:
            out = comfy.lora.calculate_weight([patch], out, name)
        return out

    def test_constant_plus_extra_equals_the_phased_path_at_the_same_levels(self):
        """Additive LoRA math: 1.0 + (w - 1.0) is w. In float32 the two additions round differently from one addition of w,
        so the weights agree to float rounding, not bit for bit; the latents are compared with a tolerance of 1e-4."""
        torch = self.torch
        m_p, p_p, _, _ = self.phased_chain()
        m_c, p_c = self.constant_chain(1.0)
        phased = self.run_dk(m_p, p_p, PORTRAIT)
        const = self.run_dk(m_c, p_c, PORTRAIT)
        self.assertEqual([(s_["first_step"], s_["last_step"], s_["cfg"], s_["weight"])
                          for s_ in self.report(const)["darkroom"]["segments"]],
                         [(s_["first_step"], s_["last_step"], s_["cfg"], s_["weight"])
                          for s_ in self.report(phased)["darkroom"]["segments"]])
        a, b = self.samples(const), self.samples(phased)
        self.assertLess(float((a - b).abs().max()), 1e-4)
        off = self.run_dk(m_c, p_c, PORTRAIT, w_commit=1.0)
        self.assertGreater(float((a - self.samples(off)).abs().max()), 1e-3, "control: the boost changes the latent")
        self.assertGreater(float((b - self.samples(off)).abs().max()), 1e-3)
        # the patched weights themselves: a level of the phased route against the constant weight plus the extra patch
        adapter = self.lora_phase.adapter_of(m_p)
        for weight in (1.04, 1.12, 1.2):
            level_phased = self.lora_phase.make_level_model(m_p, adapter, weight)
            level_const = self.lora_phase.make_level_model(m_c, adapter, weight - 1.0)
            keys = sorted(k for k in adapter.patch_dict if isinstance(k, str))[:6]
            self.assertGreaterEqual(len(keys), 3)
            for key in keys:
                wp, wc = self.torch_weight(level_phased, key), self.torch_weight(level_const, key)
                self.assertTrue(torch.allclose(wp, wc, atol=1e-6, rtol=1e-6), (weight, key))
                self.assertFalse(torch.allclose(wp, self.torch_weight(m_c, key), atol=1e-6, rtol=0.0),
                                 "control: the level differs from the unboosted weight")

    def segment_weights(self, model, profile, **ov):
        """For every clone ``run_plan`` builds through ``make_level_model``: (the weight asked for, the patched weights of
        a few keys), taken the moment the clone exists. A real run, not a hand-made level."""
        tiny_moss.sample_euler(self.tiny)  # the shared module holds the last run's weights: reload the base first
        seen, real = [], self.lora_phase.make_level_model

        def spy(m, adapter, weight):
            made = real(m, adapter, weight)
            keys = sorted(k for k in adapter.patch_dict if isinstance(k, str))[:6]  # the character's own keys
            seen.append((weight, {k: self.torch_weight(made, k) for k in keys}))
            return made
        with mock.patch.object(self.lora_phase, "make_level_model", spy):
            self.run_dk(model, profile, PORTRAIT, **ov)
        return seen

    def test_the_commit_clones_of_a_constant_profile_carry_the_weights_of_the_phased_profile(self):
        """The ticket's equivalence: constant 1.0 + extra (w_commit - w_compose = 0.2 at the top) is phased 1.0 -> 1.2 in the
        weights every commit-phase clone really holds (float rounding only: two additions against one)."""
        torch = self.torch
        m_p, p_p, _, _ = self.phased_chain()
        m_c, p_c = self.constant_chain(1.0)
        phased = self.segment_weights(m_p, p_p)
        const = self.segment_weights(m_c, p_c)
        self.assertEqual(len(phased), 5, "the phased route builds a level clone for every segment")
        self.assertEqual(len(const), 4, "the constant route skips the plain first segment (weight == w_compose)")
        self.assertAlmostEqual(const[-1][0], 0.2, places=6, msg="the finish clone: the top of the boost, w_commit - w_compose")
        for (w_p, weights_p), (w_c, weights_c) in zip(phased[1:], const):
            self.assertAlmostEqual(w_p - 1.0, w_c, places=6)
            self.assertGreaterEqual(len(weights_c), 3)
            for key, tensor in weights_c.items():
                self.assertTrue(torch.allclose(weights_p[key], tensor, atol=1e-6, rtol=1e-6), (w_p, key))
        base = self.segment_weights(m_p, p_p, w_commit=1.0)  # control: the unboosted weights differ from the boosted ones
        self.assertTrue(any(not torch.allclose(base[-1][1][k], phased[-1][1][k], atol=1e-6, rtol=0.0) for k in base[-1][1]))

    def test_the_extra_patch_never_reaches_the_input_model_or_the_other_segments(self):
        m_c, p_c = self.constant_chain(1.0)
        tiny_moss.sample_euler(self.tiny)  # the shared module holds the last run's weights: reload the base first
        before = NoLeakTest.snapshot(self, m_c)
        patched = []
        real = self.lora_phase.make_level_model

        def spy(model, adapter, weight):
            made = real(model, adapter, weight)
            patched.append((weight, sum(len(v) for v in made.patches.values())))
            return made
        with mock.patch.object(self.lora_phase, "make_level_model", spy):
            self.run_dk(m_c, p_c, PORTRAIT)
        base = sum(len(v) for v in m_c.patches.values())
        self.assertEqual(len(patched), 4, "an extra-patch clone per boosted segment (3 levels + finish); the first is plain")
        self.assertEqual(len({n for _, n in patched}), 1)
        self.assertGreater(patched[0][1], base, "the extra patch adds entries on the segment clone")
        self.assertEqual(sum(len(v) for v in m_c.patches.values()), base)
        tiny_moss.sample_euler(self.tiny)
        self.assertEqual(NoLeakTest.snapshot(self, m_c), before)
        first = self.run_dk(m_c, p_c, PORTRAIT)
        self.assertTrue(self.equal(self.samples(first), self.samples(self.run_dk(m_c, p_c, PORTRAIT))))

    def test_steps_before_the_boost_are_the_unboosted_run(self):
        m_c, p_c = self.constant_chain(1.0)
        seen, patch = self.recorder()
        with patch:
            self.run_dk(m_c, p_c, PORTRAIT, preview=True)
        boosted = [s_[1] for s_ in seen]
        seen, patch = self.recorder()
        with patch:
            self.run_dk(m_c, p_c, PORTRAIT, preview=True, w_commit=1.0)
        plain = [s_[1] for s_ in seen]
        first = next(i for i in range(12) if not self.equal(boosted[i], plain[i]))
        self.assertEqual(first, 5, "steps 0-3 run at 1.0, step 4's boost shows in the input of step 5")

    def test_at_most_one_clone_is_alive_with_the_extra_patch(self):
        m_c, p_c = self.constant_chain(1.0)
        real, refs = self.lora_phase.make_level_model, []

        def level(model, adapter, weight):
            gc.collect()
            self.assertEqual([r() for r in refs], [None] * len(refs))
            made = real(model, adapter, weight)
            refs.append(weakref.ref(made))
            return made
        with mock.patch.object(self.lora_phase, "make_level_model", level):
            self.run_dk(m_c, p_c, PORTRAIT)
        gc.collect()
        self.assertEqual([r() for r in refs], [None] * len(refs))

    def swap_file(self, state_dict):
        """The folder now answers 'A.safetensors' with another file (the original stays memory-mapped on Windows)."""
        import folder_paths
        self.env.write("B.safetensors", state_dict)
        swap = mock.patch.object(folder_paths, "get_full_path_or_raise",
                                 side_effect=lambda kind, name: self.env.path("B.safetensors"))
        swap.start()
        self.addCleanup(swap.stop)

    def assert_skipped(self, out, reason, m_c, p_c):
        report = self.report(out)
        self.assertEqual(report["darkroom"]["character"]["boost_skipped"], reason)
        self.assertEqual(report["darkroom"]["character"]["mode"], "constant_profile")
        warn = [w for w in report["warnings"] if w["code"] == "darkroom_boost_skipped"]
        self.assertEqual((len(warn), warn[0]["severity"]), (1, "warning"))
        self.assertEqual(warn[0]["detail"]["reason"], reason)
        self.assertEqual(report["forwards"]["match"], True)
        self.assertEqual({s_["weight"] for s_ in report["darkroom"]["segments"]}, {None})
        reference = self.run_dk(m_c, p_c, PORTRAIT, w_commit=1.0)
        self.assertTrue(self.equal(self.samples(out), self.samples(reference)), "the run is the unboosted run")

    def test_a_missing_character_file_skips_the_boost_and_the_run_goes_on(self):
        import os
        m_c, p_c = self.constant_chain(1.0)
        os.remove(self.env.path("A.safetensors"))
        self.assert_skipped(self.run_dk(m_c, p_c, PORTRAIT), "file_not_found", m_c, p_c)

    def test_a_changed_character_file_skips_the_boost(self):
        m_c, p_c = self.constant_chain(1.0)
        self.swap_file(lora_synth.make_lora_sd(self.patcher, seed=99, modules=("qkv", "o")))
        self.assert_skipped(self.run_dk(m_c, p_c, PORTRAIT), "sha_differs", m_c, p_c)

    def test_a_resized_character_file_skips_the_boost(self):
        m_c, p_c = self.constant_chain(1.0)
        self.swap_file(lora_synth.make_lora_sd(self.patcher, seed=99, modules=("qkv", "o", "ffn_down")))
        self.assert_skipped(self.run_dk(m_c, p_c, PORTRAIT), "size_differs", m_c, p_c)

    def test_an_unreadable_file_is_a_skip_not_a_crash(self):
        m_c, p_c = self.constant_chain(1.0)

        def unreadable(*args, **kwargs):
            # a new exception per call: a Mock's ``side_effect=RuntimeError(..)`` is ONE instance that keeps the traceback of its
            # last raise (frames with the MODEL) until the Mock's cycle is collected, which left a dead LoadedModel (MP-88)
            raise RuntimeError("unreadable")

        with mock.patch.object(self.lora_core, "load_lora_state", unreadable):
            report = self.report(self.run_dk(m_c, p_c, PORTRAIT))
        self.assertEqual(report["darkroom"]["character"]["boost_skipped"], "load_error")

    def test_the_lora_state_comes_from_the_cache_not_from_a_second_read(self):
        m_c, p_c = self.constant_chain(1.0)
        reads = []
        import comfy.utils
        real = comfy.utils.load_torch_file

        def counting(path, *a, **k):
            reads.append(path)
            return real(path, *a, **k)
        with mock.patch.object(comfy.utils, "load_torch_file", counting):
            self.run_dk(m_c, p_c, PORTRAIT)
        self.assertEqual(reads, [], "the chain's LoRA node already loaded the file into the shared cache")

    def test_a_neutral_plan_on_the_same_constant_chain_stays_bit_equal_to_stock(self):
        m_c, p_c = self.constant_chain(1.0)
        stock = self.run_stock(m_c, p_c, steps=12)
        self.assertTrue(self.equal(self.samples(self.run_dk(m_c, p_c, OFF)), self.samples(stock)))


class PresetValuesTest(DarkroomBase):
    """MP-71: the values come from the preset file and nothing else. A run of the preset (nothing overridden) equals the same
    run with every value of the block passed as an explicit knob, and equals the hand-chained stock segments built from the
    block's numbers (not from constants in this file)."""

    ALL_KNOBS = (("schedule", None), ("compose_slg", ("compose", "slg")), ("r_c", ("commit", "r_c")),
                 ("r_k", ("commit", "r_k")), ("w_compose", ("commit", "w_compose")), ("w_commit", ("commit", "w_commit")),
                 ("reexpose", ("reexpose", "enabled")), ("sigma_r", ("reexpose", "sigma_r")), ("r_f", ("finish", "r_f")),
                 ("finish_cfg", ("finish", "cfg")), ("eta", ("finish", "eta")), ("switch", ("finish", "switch")))

    def explicit(self, preset):
        block = self.block(preset)
        out = {}
        for name, path in self.ALL_KNOBS:
            out[name] = block[name] if path is None else block[path[0]][path[1]]
        return out

    def test_the_preset_run_equals_the_run_with_every_value_given_explicitly(self):
        m, p, _, _ = self.phased_chain()
        for preset in (PORTRAIT, FULLBODY):
            with self.subTest(preset=preset):
                plain = self.run_dk(m, p, preset, pinned=False)
                given = self.run_dk(m, p, preset, pinned=False, **self.explicit(preset))
                self.assertTrue(self.equal(self.samples(plain), self.samples(given)))
                self.assertEqual(self.report(plain)["darkroom"]["settings"], self.report(given)["darkroom"]["settings"])
                self.assertEqual(self.report(plain)["darkroom"]["overrides"], [])
                self.assertEqual(len(self.report(given)["darkroom"]["overrides"]), len(self.ALL_KNOBS))

    def test_the_preset_run_equals_the_hand_chained_stock_segments_of_its_own_numbers(self):
        for preset in (PORTRAIT, FULLBODY):
            with self.subTest(preset=preset):
                block = self.block(preset)
                w0, w1 = block["commit"]["w_compose"], block["commit"]["w_commit"]
                m, p, d_model, d_profile = self.phased_chain()
                result = self.run_dk(m, p, preset, pinned=False)
                weights = textbook_applied(self.sigmas12.tolist(), w0, w1, block["commit"]["r_c"], block["commit"]["r_k"])
                sig = self.sigmas12.tolist()
                r = [(sig[0] - v) / (sig[0] - sig[-1]) for v in sig[:-1]]
                cfgs = [block["compose"]["cfg"] if x < block["finish"]["r_f"] else block["finish"]["cfg"] for x in r]
                runs, start = [], 0
                for k in range(1, 13):
                    if k == 12 or (weights[k], cfgs[k]) != (weights[start], cfgs[start]):
                        runs.append((weights[start], cfgs[start], start, k))
                        start = k
                segments = [(self.lora(d_model, "A.safetensors", role="character", strength_model=w,
                                       profile_in=d_profile)[0], cfg, first, stop) for w, cfg, first, stop in runs]
                expected = HandoverTest.chain(self, segments)
                self.assertTrue(self.equal(self.samples(result), expected.to(self.samples(result).dtype)))

    def test_the_simple_node_cfg_widget_is_the_finish_guidance(self):
        m, p, _, _ = self.phased_chain()
        by_widget = self.run_simple(m, p, PORTRAIT, cfg=3.0)
        by_knob = self.run_dk(m, p, PORTRAIT, pinned=False, finish_cfg=3.0)
        self.assertTrue(self.equal(self.samples(by_widget), self.samples(by_knob)))
        report = self.report(by_widget)
        self.assertEqual(report["sampling"]["cfg"], 1.0, "sampling.cfg is the compose guidance")
        self.assertEqual(report["darkroom"]["settings"]["finish"]["cfg"], 3.0)
        self.assertEqual(report["darkroom"]["settings"]["compose"]["cfg"], 1.0)
        self.assertEqual(report["darkroom"]["overrides"], ["finish_cfg"])
        warn = [w for w in report["warnings"] if w["code"] == "darkroom_experimental"][0]
        self.assertEqual(warn["detail"]["finish_cfg_source"], "cfg_widget")
        self.assertIn("finish guidance from the cfg widget", warn["message"])
        text = by_widget["ui"]["text"][0]
        self.assertIn("finish cfg 3 from the cfg widget", text)
        self.assertIn("compose cfg 1 from the preset", text)
        # MP-75 M2 (design A7): an explicit value that differs from the preset's block customizes the run, with a diff entry
        self.assertTrue(report["label"]["customized"])
        self.assertIn("darkroom_setting_differs", report["label"]["not_reference_reasons"])
        self.assertTrue(report["label"]["ui_label"].endswith("+customized"))
        self.assertEqual([d for d in report["preset"]["diff"] if d["path"].startswith("darkroom.")],
                         [{"path": "darkroom.finish.cfg", "base": 2.0, "effective": 3.0}])
        self.assertIn("customized_run", ss.codes(report))

    def test_the_finish_cfg_knob_wins_over_the_cfg_widget_with_a_warning(self):
        m, p, _, _ = self.phased_chain()
        out = self.run_adv(m, p, PORTRAIT, cfg=3.0, finish_cfg=2.5)
        report = self.report(out)
        self.assertEqual(report["darkroom"]["settings"]["finish"]["cfg"], 2.5)
        warn = [w for w in report["warnings"] if w["code"] == "darkroom_override_ignored"]
        self.assertEqual((len(warn), warn[0]["detail"]["fields"], warn[0]["severity"]), (1, ["cfg"], "warning"))
        detail = [w for w in report["warnings"] if w["code"] == "darkroom_experimental"][0]["detail"]
        self.assertEqual(detail["finish_cfg_source"], "finish_cfg")
        equal = self.report(self.run_adv(m, p, PORTRAIT, cfg=2.5, finish_cfg=2.5))
        self.assertNotIn("darkroom_override_ignored", ss.codes(equal), "the same value twice is no conflict")

    def test_on_a_stock_preset_the_cfg_widget_is_the_sampling_cfg_as_before(self):
        m, p = self.constant_chain()
        out = self.run_simple(m, p, "turbo_identity", cfg=1.5)
        self.assertEqual(self.report(out)["sampling"]["cfg"], 1.5)
        self.assertNotIn("darkroom", self.report(out))
        self.assertIn("cfg 1.5 from the user (widget)", out["ui"]["text"][0])

    def test_the_compose_cfg_knob_is_the_sampling_cfg(self):
        m, p, _, _ = self.phased_chain()
        report = self.report(self.run_adv(m, p, PORTRAIT, compose_cfg=1.1))
        self.assertEqual(report["sampling"]["cfg"], 1.1)
        self.assertEqual(report["darkroom"]["settings"]["compose"]["cfg"], 1.1)
        self.assertIn("compose_cfg", report["darkroom"]["overrides"])
        self.assertIn("sampling.cfg", [d["path"] for d in report["preset"]["diff"]])

    def test_a_connected_sigmas_input_replaces_the_sa2_schedule_with_a_warning(self):
        m, p, _, _ = self.phased_chain()
        sigmas = self.sigmas12.clone()
        report = self.report(self.run_adv(m, p, PORTRAIT, schedule="sa2_8", sigmas=sigmas))
        warn = [w for w in report["warnings"] if w["code"] == "darkroom_override_ignored"]
        self.assertEqual([w["detail"]["fields"] for w in warn], [["schedule"]])
        self.assertEqual(report["darkroom"]["schedule"], "native")
        self.assertEqual(report["darkroom"]["steps"], 12)


class ReviewFixesTest(DarkroomBase):
    """MP-75: the report says what the run did (SLG, customized) and an error names the widget the user has."""

    def diff_paths(self, report):
        return [d["path"] for d in report["preset"]["diff"] if d["path"].startswith("darkroom.")]

    def test_values_equal_to_the_preset_do_not_customize(self):
        m, p, _, _ = self.phased_chain()
        block = self.block(PORTRAIT)
        same = self.report(self.run_adv(m, p, PORTRAIT, finish_cfg=block["finish"]["cfg"], r_f=block["finish"]["r_f"],
                                        w_commit=block["commit"]["w_commit"], eta=0.0, compose_slg="off"))
        self.assertEqual(same["darkroom"]["overrides"], ["compose_slg", "eta", "finish_cfg", "r_f", "w_commit"])
        self.assertEqual((same["label"]["customized"], self.diff_paths(same)), (False, []))
        self.assertNotIn("customized_run", ss.codes(same))
        self.assertNotIn("darkroom_setting_differs", same["label"]["not_reference_reasons"])
        plain = self.report(self.run_simple(m, p, PORTRAIT, cfg=block["finish"]["cfg"]))
        self.assertEqual((plain["label"]["customized"], self.diff_paths(plain)), (False, []))

    def test_every_differing_knob_customizes_with_a_diff_entry_and_a_reason(self):
        m, p, _, _ = self.phased_chain()
        for widgets, path in ((dict(w_commit=1.1), "darkroom.commit.w_commit"), (dict(eta=0.1), "darkroom.finish.eta"),
                              (dict(finish_cfg=2.5), "darkroom.finish.cfg"), (dict(r_f=0.2), "darkroom.finish.r_f"),
                              (dict(compose_slg="on"), "darkroom.compose.slg"),
                              (dict(w_compose=0.9), "darkroom.commit.w_compose")):
            with self.subTest(widgets=widgets):
                report = self.report(self.run_adv(m, p, PORTRAIT, **widgets))
                self.assertTrue(report["label"]["customized"])
                self.assertEqual(self.diff_paths(report), [path])
                self.assertIn("darkroom_setting_differs", report["label"]["not_reference_reasons"])
                self.assertEqual(report["label"]["parity_class"], "not_reference")
        report = self.report(self.run_adv(m, p, PORTRAIT, compose_cfg=1.1))
        self.assertTrue(report["label"]["customized"])
        self.assertIn("sampling.cfg", [d["path"] for d in report["preset"]["diff"]])

    def test_the_sampling_slg_widgets_are_ignored_on_a_darkroom_preset_and_the_report_says_so(self):
        m, p, _, _ = self.phased_chain()
        base = dict(compose_cfg=1.5)
        plain = self.run_adv(m, p, PORTRAIT, **base)
        slg = self.run_adv(m, p, PORTRAIT, slg_mode="uncond_skip", slg_layers="22", **base)
        self.assertTrue(self.equal(self.samples(slg), self.samples(plain)), "the sampling SLG changes nothing in the run")
        report = self.report(slg)
        warn = [w for w in report["warnings"] if w["code"] == "darkroom_override_ignored"]
        self.assertEqual([w["detail"]["fields"] for w in warn], [["slg_layers", "slg_mode"]])
        self.assertEqual(warn[0]["severity"], "warning")
        self.assertNotIn("slg_differs", report["label"]["not_reference_reasons"])
        self.assertNotIn("turbo_slg_experimental", ss.codes(report))
        self.assertEqual(report["sampling"]["slg"]["mode"], "off")
        self.assertEqual(report["label"]["not_reference_reasons"], self.report(plain)["label"]["not_reference_reasons"])
        self.assertEqual(report["forwards"]["slg_skip_calls"], 0)
        # control: the darkroom's own SLG knob does change the latent
        own = self.run_adv(m, p, PORTRAIT, compose_slg="on", **base)
        self.assertFalse(self.equal(self.samples(own), self.samples(plain)))
        self.assertGreater(self.report(own)["forwards"]["slg_skip_calls"], 0)

    def test_the_sampling_slg_widgets_still_work_on_a_stock_turbo_preset(self):
        d_model, d_profile = self.distill()
        out = self.report(self.run_adv(d_model, d_profile, "turbo_8step", cfg=1.5, slg_mode="uncond_skip", slg_layers="22"))
        self.assertIn("turbo_slg_experimental", ss.codes(out))
        self.assertIn("slg_differs", out["label"]["not_reference_reasons"])
        self.assertNotIn("darkroom_override_ignored", ss.codes(out))

class StockPresetIgnoresTheKnobsTest(DarkroomBase):
    """MP-71: a preset without a darkroom block never reaches the engine. A phase knob or a finish_model on it is ignored with a
    warning (never an exception) and the run is bit-equal to the run without them."""

    def test_the_knobs_are_ignored_with_one_warning_naming_the_fields(self):
        m, p = self.constant_chain()
        plain = self.run_adv(m, p, "turbo_identity", cfg=1.0)
        for widgets, fields in ((dict(r_f=0.3), ["r_f"]), (dict(eta=0.2, finish_cfg=2.0), ["eta", "finish_cfg"]),
                                (dict(compose_cfg=1.5, w_commit=1.4, switch_model="on"),
                                 ["compose_cfg", "switch", "w_commit"]),
                                (dict(schedule="sa2_8"), ["schedule"]), (dict(reexpose="on", sigma_r=0.5),
                                                                          ["reexpose", "sigma_r"])):
            with self.subTest(widgets=widgets):
                out = self.run_adv(m, p, "turbo_identity", cfg=1.0, **widgets)
                report = self.report(out)
                warn = [w for w in report["warnings"] if w["code"] == "darkroom_override_ignored"]
                self.assertEqual(len(warn), 1)
                self.assertEqual((warn[0]["severity"], warn[0]["detail"]["fields"]), ("warning", sorted(fields)))
                self.assertTrue(self.equal(self.samples(out), self.samples(plain)), "the run is the one without the knobs")
                self.assertNotIn("darkroom", report)
                self.assertNotIn("darkroom_experimental", ss.codes(report))
                self.assertEqual(report["label"]["parity_class"], self.report(plain)["label"]["parity_class"])
                self.assertEqual(report["label"]["not_reference_reasons"], self.report(plain)["label"]["not_reference_reasons"])
                self.assertEqual(report["forwards"]["match"], True)
        self.assertNotIn("darkroom_override_ignored", ss.codes(self.report(plain)))

    def test_a_finish_model_on_a_stock_preset_is_ignored_and_not_even_probed(self):
        other = tiny_moss.build_tiny_moss(layers=4).patcher  # would be unsupported_model were it looked at
        m, p = self.constant_chain()
        plain = self.run_adv(m, p, "turbo_identity", cfg=1.0)
        out = self.run_adv(m, p, "turbo_identity", cfg=1.0, finish_model=other, switch_model="on")
        report = self.report(out)
        warn = [w for w in report["warnings"] if w["code"] == "finish_model_ignored"]
        self.assertEqual((len(warn), warn[0]["severity"], warn[0]["detail"]), (1, "warning", {"preset": "turbo_identity"}))
        self.assertEqual([w["detail"]["fields"] for w in report["warnings"] if w["code"] == "darkroom_override_ignored"],
                         [["switch"]])
        self.assertTrue(self.equal(self.samples(out), self.samples(plain)))

    def test_every_stock_preset_takes_the_knobs_without_an_exception(self):
        k_model, k_profile = self.constant_chain()
        for preset in ("draft_fast", "turbo_8step", "turbo_identity"):
            with self.subTest(preset=preset):
                model, profile = (k_model, k_profile) if preset == "turbo_identity" else (
                    (self.distill() if preset == "turbo_8step" else (self.patcher, None)))
                self.run_adv(model, profile, preset, r_f=0.3, finish_cfg=2.0, finish_model=self.patcher)

    def test_the_simple_node_sends_a_stock_preset_no_darkroom_request_at_all(self):
        m, p = self.constant_chain()
        seen = []
        real = self.solver.execute

        def spy(req):
            seen.append(req)
            return real(req)
        with mock.patch.object(self.sampler_node.core_solver, "execute", spy):
            self.run_simple(m, p, "turbo_identity")
        self.assertEqual(len(seen), 1)
        self.assertIsNone(seen[0].darkroom)
        self.assertFalse(seen[0].allow_phased_lora)


if __name__ == "__main__":
    unittest.main()
