"""MP-84: a seeded widget fuzz over MossPhotonSampler, MossPhotonSamplerAdvanced, MossPhotonLoRAPhase, the deprecated
MossPhotonLoRAProfile and MossPhotonPromptSplit on the tiny MOSS (CPU), and (MP-91) over MossPhotonHiresFix with a fake face
detector. (MossPhotonIdentityDetailer is not fuzzed: it needs a face detector and stays a refusal node, see the whitelist.)

Owner's rule (2026-11-02): every settings conflict degrades with a warning; a node raises only for what is impossible. The static
half is ``tests/unit/test_raise_whitelist.py`` (every raise site of the source is on a short list). This is the dynamic half: the
widgets are drawn from a stdlib ``random.Random(SEED)`` over their whole declared ranges and the edge values around them (the
sentinels, the bounds, one step outside, a phase order the wrong way round, a LoRA at +-20, a pin that is not the installed one),
and a case must end in one of exactly two ways:

* the run returns a finite latent and a report that validates against the schema, or
* it raises a ``MossPhotonError`` whose code is a legitimate one for THAT case (``FUZZ_RAISES``); any other exception type, and
  any other code, is a failure with the case's widgets in the message (the seed and the case number reproduce it).

A control proves the fuzz is not vacuous: it must reach every degradation the audit list names at least once.
"""
import logging
import random
import unittest

from tests.unit import test_raise_whitelist as guard

from . import lora_synth, test_darkroom_cpu as dk_tests
from . import sampler_support as ss
from .test_hires_cpu import FACE_PX, HiresBase, pin

SEED = 84
CASES = 110
FINGERPRINT = "off"

# The darkroom presets carry most of the settings (phases, weights, LoRA Phase entries): they are drawn three times as often.
PRESETS = ("reference", "photo_safe", "identity_safe", "wardrobe_safe", "scene_safe", "night_photo", "draft_fast", "advanced",
           "turbo_8step", "turbo_identity") + ("turbo_darkroom_portrait", "turbo_darkroom_fullbody") * 3
# Presets whose input contract needs a prompt block this fuzz does not feed: a refusal there is legitimate (wrong conditioning).
BLOCK_PRESETS = ("wardrobe_safe", "scene_safe")
FUZZ_RAISES = {"missing_required_input", "non_finite_output"}
# the degradations of the audit list; a pinned control asserts the fuzz reaches each at least once
AUDIT_CODES = ("setting_adjusted", "override_outside_bounds", "forward_budget_exceeded", "preset_pin_mismatch",
               "phase_plan_ignored", "darkroom_boost_skipped", "darkroom_override_ignored", "phased_lora_demoted",
               "multiple_distill_loras", "distill_lora_non_turbo", "lora_not_recorded")

CFGS = (0.0, 0.0, 0.01, 0.5, 0.99, 1.0, 1.5, 2.0, 3.0, 7.0, 30.0)
STEPS = (0, 0, 1, 2, 3, 4, 6, 9, 17)
PINS = ("", "", "", "reference@1.0.0", "reference@9.9.9", "turbo_8step@1.0.0#00000000", "not a pin", "x@1.0",
        "turbo_darkroom_portrait@0.0.1")
FRACTIONS = (-1.0, -1.0, -1.0, 0.0, 0.06, 0.17, 0.3, 0.5, 0.9, 1.0)
WEIGHTS = (-1.0, -1.0, 0.0, 0.5, 1.0, 1.2, 1.35, 10.0)
SLG_LAYERS = ("", "", "22", "22, 99", "0 1 2", "99", "-1", "abc", "22 22")
LORA_STRENGTHS = (-20.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.5, 20.0)
PHASES = ("all", "composition", "identity", "texture", "composition+identity")
ONOFF = ("from_preset", "off", "on")


def pick(rng, values):
    return values[rng.randrange(len(values))]


class FuzzBase(dk_tests.DarkroomBase):
    def build_model(self, rng, notes):
        """The MODEL of a case: optionally a stock loader, optionally the deprecated profile chain, 0..3 LoRA Phase nodes."""
        from moss_photon.nodes.lora_plan import MossPhotonLoRAPhase
        model, profile = self.patcher, None
        if rng.random() < 0.3:
            model = ss.apply_lora(model, self.a_sd, pick(rng, (0.3, 1.0)))
            notes.append("stock")
        if rng.random() < 0.3:
            for _ in range(rng.randrange(1, 3)):
                kind = pick(rng, ("character", "style", "distill", "distill", "other"))
                phased = rng.random() < 0.4
                model, profile, _clip = lora_synth.run_node(
                    model, pick(rng, ("A.safetensors", "D.safetensors")), role=kind,
                    strength_model=pick(rng, (0.5, 1.0)), profile="phased" if phased else "constant", profile_in=profile,
                    w0=pick(rng, (0.1, 0.5)), w1=pick(rng, (0.9, 1.2)), a=0.1, b=0.6)
                notes.append("profile:%s%s" % (kind, ":phased" if phased else ""))
        for _ in range(pick(rng, (0, 0, 1, 1, 2, 3))):
            name = pick(rng, ("A.safetensors", "D.safetensors", "none"))
            strength, phase = pick(rng, LORA_STRENGTHS), pick(rng, PHASES)
            model = MossPhotonLoRAPhase().apply(model, name, strength, phase)[0]
            notes.append("phase:%s:%g:%s" % (name, strength, phase))
        return model, profile

    def draw_case(self, rng):
        notes = []
        model, profile = self.build_model(rng, notes)
        preset = pick(rng, PRESETS)
        case = {"preset": preset, "cfg": pick(rng, CFGS), "seed": rng.randrange(0, 2 ** 32), "preset_pin": pick(rng, PINS),
                "steps": pick(rng, STEPS), "advanced": rng.random() < 0.7, "model": notes}
        advanced = {}
        if case["advanced"]:
            advanced = dict(
                slg_mode=pick(rng, ("from_preset", "off", "uncond_skip")), slg_layers=pick(rng, SLG_LAYERS),
                slg_start=pick(rng, (-1.0, 0.0, 0.3, 0.8, 1.0)), slg_end=pick(rng, (-1.0, 0.0, 0.3, 0.8, 1.0)),
                noise_policy=pick(rng, ("native", "native", "seed_plus_i")), max_forwards=pick(rng, (0, 0, 0, 1, 10, 100)),
                report_step_times=rng.random() < 0.2, schedule=pick(rng, ("from_preset", "native", "sa2_8")),
                compose_cfg=pick(rng, (0.0, 0.0, 0.5, 1.0, 1.5, 30.0)), compose_slg=pick(rng, ONOFF),
                r_c=pick(rng, FRACTIONS), r_k=pick(rng, FRACTIONS), w_compose=pick(rng, WEIGHTS),
                w_commit=pick(rng, WEIGHTS), reexpose=pick(rng, ONOFF), sigma_r=pick(rng, FRACTIONS),
                r_f=pick(rng, FRACTIONS), finish_cfg=pick(rng, (0.0, 0.0, 0.5, 1.0, 2.0, 30.0)),
                eta=pick(rng, (-1.0, -1.0, 0.0, 0.3, 1.0)), switch_model=pick(rng, ONOFF))
            if rng.random() < 0.3:
                finish, _ = self.build_model(rng, notes)
                advanced["finish_model"] = finish
                case["finish_model"] = True
            if rng.random() < 0.1:
                advanced["sigmas"] = self.torch.tensor([1.0, 0.6, 0.2, 0.0])
                case["sigmas"] = "custom"
        case["widgets"] = {k: v for k, v in advanced.items() if k != "finish_model" and k != "sigmas"}
        return model, profile, case, advanced

    def run_case(self, model, profile, case, advanced):
        from moss_photon.nodes import sampler
        tiny = self.tiny
        common = dict(model=model, positive=tiny.make_cond(23), negative=tiny.make_cond(9),
                      latent={"samples": tiny.latent.clone()}, seed=case["seed"], preset=case["preset"], cfg=case["cfg"],
                      identity_profile=profile, preview=False, preset_pin=case["preset_pin"])
        with ss.known_sft():
            if case["advanced"]:
                widgets = dict(solver="euler", cfg_profile="constant", fingerprint=FINGERPRINT, **advanced)
                return sampler.MossPhotonSamplerAdvanced().sample(steps=case["steps"], **common, **widgets)
            return sampler.MossPhotonSampler().sample(steps=case["steps"], **common)


class WidgetFuzzTest(FuzzBase):
    def test_seeded_widget_fuzz_raises_nothing_outside_the_whitelist(self):
        from moss_photon.core import errors, report
        from tests.unit import test_raise_whitelist as guard
        allowed = guard.ALLOWED_CODES
        rng = random.Random(SEED)
        ran = refused = 0
        seen_codes = set()
        for number in range(CASES):
            model, profile, case, advanced = self.draw_case(rng)
            label = "fuzz seed %d case %d: %r" % (SEED, number, dict(case, advanced_widgets=sorted(advanced)))
            try:
                out = self.run_case(model, profile, case, advanced)
            except errors.MossPhotonError as exc:
                refused += 1
                self.assertIn(exc.code, allowed, "a code outside the no-raise whitelist: %s\n%s" % (exc, label))
                legit = exc.code in FUZZ_RAISES and (exc.code != "missing_required_input" or case["preset"] in BLOCK_PRESETS)
                self.assertTrue(legit, "%s raised for a settings conflict instead of degrading: %s\n%s" % (
                    exc.code, exc, label))
                continue
            except Exception as exc:  # noqa: BLE001 - the point of the fuzz: any other exception type is a defect
                self.fail("%s: %s\n%s" % (type(exc).__name__, exc, label))
            ran += 1
            latent, rep = out["result"]
            self.assertTrue(self.torch.isfinite(latent["samples"]).all(), label)
            self.assertEqual(report.validate_report(rep), [], label)
            self.assertTrue(rep["forwards"]["match"], label)
            seen_codes |= {w["code"] for w in rep["warnings"]}
            for w in rep["warnings"]:
                self.assertNotEqual(w["severity"], "error", label)
        self.assertGreater(ran, CASES * 0.7, "the fuzz must mostly run: %d ran, %d refused" % (ran, refused))
        missed = [code for code in AUDIT_CODES if code not in seen_codes]
        self.assertEqual(missed, [], "the fuzz never reached these degradations (change SEED/CASES or the pools): %s" % missed)


class ProfileWidgetFuzzTest(FuzzBase):
    """The deprecated MossPhotonLoRAProfile: its widgets are drawn over their whole range (strength +-10, windows with a >= b, a
    phased profile on any role and any chain position, the text-encoder flag with no clip, every fingerprint mode, trigger text),
    chains of 1..3 nodes, and about a third of the chains are then sampled. A node call returns, or raises only what is impossible:
    ComfyUI's ``FileNotFoundError`` for a file that is not there, ``incompatible_lora`` for a file that maps nothing."""

    CASES = 60
    FILES = ("A.safetensors", "D.safetensors", "A.safetensors", "D.safetensors", "A.safetensors", "D.safetensors", "none",
             "A.safetensors", "D.safetensors", "none", "bogus.safetensors", "missing.safetensors")
    ROLES = ("character", "style", "wardrobe", "scene", "other", "distill")
    STRENGTHS = (-10.0, -1.0, -0.01, 0.0, 0.01, 0.5, 1.0, 1.35, 10.0)
    UNIT = (0.0, 0.01, 0.2, 0.6, 0.99, 1.0)
    TRIGGERS = ("", "", "zxq", "a, b", "x" * 300)
    SAMPLED = ("turbo_8step", "turbo_identity", "identity_safe", "reference", "turbo_darkroom_portrait")

    def setUp(self):
        super().setUp()
        self.env.write("bogus.safetensors", lora_synth.add_bogus_modules({}, 6))

    def draw_node(self, rng):
        from moss_photon.core import fingerprint
        return dict(
            lora_name=pick(rng, self.FILES), strength_model=pick(rng, self.STRENGTHS), role=pick(rng, self.ROLES),
            profile=pick(rng, ("constant", "phased")), trigger=pick(rng, self.TRIGGERS),
            apply_text_encoder=rng.random() < 0.3, strength_clip=pick(rng, self.STRENGTHS),
            allow_low_coverage=rng.random() < 0.3, fingerprint=pick(rng, fingerprint.HASH_MODES),
            w0=pick(rng, self.STRENGTHS), w1=pick(rng, self.STRENGTHS), a=pick(rng, self.UNIT), b=pick(rng, self.UNIT))

    def chain(self, rng, label):
        """Run 1..3 nodes on one MODEL; returns ``(model, profile, nodes_run)``. A raise must be an impossible one."""
        from moss_photon.core import errors
        from moss_photon.nodes.lora import MossPhotonLoRAProfile
        model, profile, ran = self.patcher, None, 0
        for _ in range(pick(rng, (1, 1, 2, 3))):
            kw = self.draw_node(rng)
            where = "%s node %r" % (label, kw)
            try:
                model, profile, clip = MossPhotonLoRAProfile().apply(model, profile_in=profile, **kw)
            except FileNotFoundError:
                self.assertEqual(kw["lora_name"], "missing.safetensors", where)
                break
            except errors.MossPhotonError as exc:
                self.assertIn(exc.code, guard.ALLOWED_CODES, "a code outside the no-raise whitelist: %s\n%s" % (exc, where))
                self.assertEqual((exc.code, kw["lora_name"]), ("incompatible_lora", "bogus.safetensors"),
                                 "%s raised for a settings conflict instead of degrading: %s\n%s" % (exc.code, exc, where))
                break
            except Exception as exc:  # noqa: BLE001 - any other exception type is a defect
                self.fail("%s: %s\n%s" % (type(exc).__name__, exc, where))
            self.assertIsNone(clip, where)
            ran += 1
        return model, profile, ran

    def test_seeded_profile_widget_fuzz_degrades_or_raises_only_the_impossible(self):
        from moss_photon.core import errors, report
        rng = random.Random(SEED + 1)
        chains = sampled = 0
        seen = set()
        for number in range(self.CASES):
            label = "profile fuzz seed %d case %d" % (SEED + 1, number)
            model, profile, ran = self.chain(rng, label)
            chains += bool(ran)
            if not ran or number % 3:
                continue
            _model, _profile, case, advanced = self.draw_case(rng)
            case["preset"], case["cfg"], case["steps"] = pick(rng, self.SAMPLED), 0.0, 0
            label = "%s: %r" % (label, dict(case, advanced_widgets=sorted(advanced)))
            try:
                out = self.run_case(model, profile, case, advanced)
            except errors.MossPhotonError as exc:
                self.assertIn(exc.code, guard.ALLOWED_CODES, "a code outside the no-raise whitelist: %s\n%s" % (exc, label))
                self.assertIn(exc.code, FUZZ_RAISES, "%s raised for a settings conflict instead of degrading: %s\n%s" % (
                    exc.code, exc, label))
                continue
            except Exception as exc:  # noqa: BLE001
                self.fail("%s: %s\n%s" % (type(exc).__name__, exc, label))
            sampled += 1
            latent, rep = out["result"]
            self.assertTrue(self.torch.isfinite(latent["samples"]).all(), label)
            self.assertEqual(report.validate_report(rep), [], label)
            seen |= {w["code"] for w in rep["warnings"]}
        self.assertGreater(chains, self.CASES * 0.7, "the fuzz must mostly build chains")
        self.assertGreater(sampled, 10, "the fuzz must sample some of them")
        self.assertTrue({"phased_lora_demoted"} & seen, "control: the phased request that cannot phase degraded at least once")


class PromptSplitFuzzTest(unittest.TestCase):
    """MossPhotonPromptSplit never raises (a broken answer degrades to empty fields and a report), whatever text it is given,
    and its report and log name markers only, never the text (R5)."""

    CASES = 300
    PIECES = ("<<<POSITIVE>>>", "<<<negative>>>", "<<< Identity >>>", "<<<FORBIDDEN CHANGES>>>", "<<<END>>>", "<<<UNKNOWN>>>",
              "<<<>>>", "<<<", ">>>", "</think>", "<think>", "```", "```json", "none", "N/A", "-", "\r\n", "\n", " ", "\t", "",
              "caf" + chr(0xE9) + " " + chr(0x4E2D) + chr(0x6587) + " " + chr(0x1F600), "\x00", "x" * 5000)

    def draw(self, rng, number):
        words = []
        for _ in range(rng.randrange(0, 14)):
            words.append(pick(rng, self.PIECES))
            words.append("SECRETWORD%d_%d" % (number, rng.randrange(1000)))
        return "".join(words)

    def test_seeded_text_fuzz_never_raises_and_the_report_names_markers_only(self):
        from moss_photon.nodes.prompt_split import FIELDS, MossPhotonPromptSplit
        rng = random.Random(SEED + 2)
        node = MossPhotonPromptSplit()
        texts = [None, "", " ", "\n", "<<<POSITIVE>>>", "<<<END>>>", "no markers at all"] + [
            self.draw(rng, n) for n in range(self.CASES)]
        for number, text in enumerate(texts):
            with self.assertLogs("MossPhoton", level="DEBUG") as logs:
                logging.getLogger("MossPhoton").debug("anchor")   # assertLogs fails on a run that logs nothing at all
                out = node.split(text)
            fields, report = out["result"][:len(FIELDS)], out["result"][len(FIELDS)]
            label = "prompt split fuzz seed %d case %d" % (SEED + 2, number)
            self.assertEqual(len(out["result"]), len(FIELDS) + 1, label)
            self.assertTrue(all(isinstance(v, str) for v in out["result"]), label)
            self.assertEqual(len(out["ui"]["text"]), 1, label)
            self.assertNotIn("SECRETWORD", report, "the report carries prompt text: " + label)
            self.assertNotIn("SECRETWORD", "\n".join(logs.output), "the log carries prompt text: " + label)
            self.assertNotIn("SECRETWORD", out["ui"]["text"][0], label)
        control = node.split("<<<POSITIVE>>>\nSECRETWORD a woman<<<END>>>")["result"]
        self.assertEqual(control[0], "SECRETWORD a woman", "control: a field is passed through unchanged")
        self.assertNotIn("SECRETWORD", control[len(FIELDS)])


class HiresWidgetFuzzTest(HiresBase):
    """MossPhotonHiresFix (MP-91) with a fake detector: the widgets are drawn over their whole range and the edge values (the
    bounds, the sentinels, a face_sigma on either side of the schedule, every sampler family that changes the schedule), on a plain
    MODEL, a MODEL with an ordinary loader and a MODEL with a LoRA Phase plan of any phase; the detector is present, missing,
    failing, empty or finds two faces. A case ends in exactly one of two ways: a finite latent of the expected size and a summary, or
    a raise that is legitimate for THAT case: ComfyUI's ``FileNotFoundError`` for a character LoRA that is not there and
    ``incompatible_lora`` for one that maps nothing (both only when a boost is asked, the file is not read otherwise), or the
    ``non_finite_output`` guard. Any other exception, and any other code, is a failure with the case's widgets in the message."""

    CASES = 36
    SAMPLERS = ("res_multistep", "res_multistep", "euler", "dpmpp_2m", "dpm_2", "heun", "uni_pc", "ddim")
    SCHEDULERS = ("linear_quadratic", "linear_quadratic", "simple", "sgm_uniform", "karras", "exponential", "beta", "normal",
                  "kl_optimal", "ddim_uniform")
    SCALES = (1.0, 1.05, 1.5, 1.5, 1.95, 2.0)
    STEP_VALUES = (1, 2, 3, 6, 6, 9)
    DENOISE = (0.05, 0.2, 0.32, 0.32, 0.5, 0.99, 1.0)
    CFGS = (0.0, 1.0, 1.0, 1.5, 3.0, 30.0)
    FACE_SIGMAS = (0.0, 0.01, 0.2, 0.5, 0.5, 0.86, 0.9, 1.0)
    BOOSTS = (0.0, 0.0, 0.05, 0.2, 0.2, 1.0, 2.0)
    LORAS = ("none", "none", "none", "A.safetensors", "D.safetensors", "missing.safetensors", "bogus.safetensors")
    DETECTORS = ("ok", "ok", "ok", "ok", "missing", "fails", "empty", "two")
    METHODS = ("latent_bicubic", "latent_bicubic", "pixel_lanczos")
    SECOND_FACE = (16.0, 16.0, 64.0, 80.0, 0.8)
    DEGRADATIONS = ("hires_no_detector", "hires_detector_failed", "hires_no_face", "hires_face_sigma_above_schedule",
                    "hires_face_sigma_below_schedule", "hires_boost_no_target", "hires_lora_phase_skipped",
                    "hires_boost_already_in_plan", "hires_bicubic_not_held")

    def build_model(self, rng, notes):
        model = self.patcher
        if rng.random() < 0.3:
            model = ss.apply_lora(model, self.a_sd, pick(rng, (0.3, 1.0)))
            notes.append("stock")
        for _ in range(pick(rng, (0, 0, 1, 2, 3))):
            name, strength, phase = pick(rng, ("A.safetensors", "D.safetensors")), pick(rng, LORA_STRENGTHS), pick(rng, PHASES)
            model = self.phase_node.apply(model, name, strength, phase)[0]
            notes.append("phase:%s:%g:%s" % (name, strength, phase))
        return pin(model)

    PINNED = (0, 1)

    def pinned_case(self, number, model, detector, widgets, boxes, notes):
        """Case 0: a detector that finds nothing, protection on, a face_sigma inside the schedule. Case 1: a plan entry with phase
        identity and the same file as character_lora."""
        if number == 0:
            return model, "empty", dict(widgets, face_protect=True, face_sigma=0.5, upscale_method="latent_bicubic"), []
        notes[:] = ["phase:A.safetensors:0.8:identity"]
        model = pin(self.phase_node.apply(self.patcher, "A.safetensors", 0.8, "identity")[0])
        return model, "ok", dict(widgets, character_lora="A.safetensors", identity_boost=0.2, face_protect=False), [FACE_PX]

    def test_seeded_widget_fuzz_degrades_or_raises_only_the_impossible(self):
        from moss_photon.core import errors
        from tests.unit import test_raise_whitelist as guard
        import comfy.model_management as mm
        self.env.write("bogus.safetensors", lora_synth.add_bogus_modules({}, 6))
        rng = random.Random(SEED + 3)
        ran = refused = 0
        seen = set()
        for number in range(self.CASES):
            notes = []
            model = self.build_model(rng, notes)
            detector = pick(rng, self.DETECTORS)
            widgets = dict(
                scale=pick(rng, self.SCALES), steps=pick(rng, self.STEP_VALUES), denoise=pick(rng, self.DENOISE),
                cfg=pick(rng, self.CFGS), sampler_name=pick(rng, self.SAMPLERS), scheduler=pick(rng, self.SCHEDULERS),
                face_protect=rng.random() < 0.8, face_sigma=pick(rng, self.FACE_SIGMAS), identity_boost=pick(rng, self.BOOSTS),
                character_lora=pick(rng, self.LORAS), upscale_method=pick(rng, self.METHODS), seed=rng.randrange(0, 2 ** 32))
            boxes = {"fails": RuntimeError("planted detector failure"), "empty": [], "two": [FACE_PX, self.SECOND_FACE]}.get(
                detector, [FACE_PX])
            if number in self.PINNED:     # the cases that reach the rarer degradations on every run, whatever the pools hold
                model, detector, widgets, boxes = self.pinned_case(number, model, detector, widgets, boxes, notes)
            label = "hires fuzz seed %d case %d: %r" % (SEED + 3, number, dict(widgets, model=notes, detector=detector))
            wants_file = widgets["character_lora"] != "none" and widgets["identity_boost"] > 0.0
            self.clones.clear()
            try:
                result, _vae = self.run_node(model=model, boxes=boxes, detector="missing" if detector == "missing" else "ok",
                                             **widgets)
            except mm.InterruptProcessingException:
                self.fail("an interrupt nobody asked for: " + label)
            except FileNotFoundError:
                refused += 1
                self.assertTrue(wants_file and widgets["character_lora"] == "missing.safetensors", label)
                continue
            except errors.MossPhotonError as exc:
                refused += 1
                self.assertIn(exc.code, guard.ALLOWED_CODES, "a code outside the no-raise whitelist: %s\n%s" % (exc, label))
                legit = (exc.code == "incompatible_lora" and wants_file and widgets["character_lora"] == "bogus.safetensors"
                         ) or exc.code == "non_finite_output"
                self.assertTrue(legit, "%s raised for a settings conflict instead of degrading: %s\n%s" % (exc.code, exc, label))
                continue
            except Exception as exc:  # noqa: BLE001 - the point of the fuzz: any other exception type is a defect
                self.fail("%s: %s\n%s" % (type(exc).__name__, exc, label))
            ran += 1
            out = self.samples(result)
            want_hw = round(16 * widgets["scale"])
            self.assertEqual(tuple(out.shape), (1, 128, want_hw, want_hw), label)
            self.assertTrue(bool(self.torch.isfinite(out).all()), label)
            text = self.summary(result)
            self.assertTrue(text.startswith("MossPhoton HiresFix"), label)
            seen |= {code for code in self.DEGRADATIONS if "[%s]" % code in text}
        self.assertGreater(ran, self.CASES * 0.7, "the fuzz must mostly run: %d ran, %d refused" % (ran, refused))
        missed = [code for code in self.DEGRADATIONS if code not in seen]
        self.assertEqual(missed, [], "the fuzz never reached these degradations (change SEED/CASES or the pools): %s" % missed)


class FuzzControlTest(FuzzBase):
    """The fuzz can fail: a planted raise of a code outside the list, and of a plain exception, is caught by it."""

    def test_a_planted_conflict_raise_is_caught(self):
        from unittest import mock
        from moss_photon.core import errors, solver

        def plant(*a, **k):
            errors.raise_first_error([errors.make_issue("budget_exceeded", "n", "p", "planted", "f", severity="error")])

        case = {"preset": "turbo_8step", "cfg": 0.0, "seed": 1, "preset_pin": "", "steps": 0, "advanced": False}
        with mock.patch.object(solver, "plan_run", plant):
            with self.assertRaises(errors.MossPhotonError) as cm:
                self.run_case(self.patcher, None, case, {})
        self.assertEqual(cm.exception.code, "budget_exceeded")
        from tests.unit import test_raise_whitelist as guard
        self.assertNotIn("budget_exceeded", guard.ALLOWED_CODES, "so the fuzz's assertIn(code, allowed) would fail on it")


if __name__ == "__main__":
    unittest.main()
