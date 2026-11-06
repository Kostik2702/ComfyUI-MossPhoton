"""MossPhotonHiresFix on the tiny random MOSS, CPU (MP-91): the node end to end with the REAL upscale and the REAL sampling
(``comfy.sample.sample`` with the late-entry mask of the face), a fake VAE and a fake face detector (no weights, no
ultralytics).

The guarantees of the ticket, each with the test that goes red when it is violated:

(a) with ``face_protect`` off and ``identity_boost`` 0 the node is the stock ``LatentUpscaleBy`` + ``common_ksampler``, bit for bit
    (``torch.equal``), at scale 1.5 and 1.0, for a batch, and in the pixel mode against its hand-built chain -> ``ParityTest``;
(c) a ``face_sigma`` below the schedule leaves the face core of the output equal to the upscaled source while the same zone
    changes when protection is off (so the check is not vacuous); a release inside the schedule makes the zone differ from both
    -> ``FaceProtectTest``;
(d) no detector, a failing detector, no face, a ``face_sigma`` out of range, a boost without a target and a mask that is already
    taken: a warning in the summary and a run, never a raise; an interrupt is not swallowed -> ``DegradationTest``;
(e) the incoming MODEL is not mutated -> ``ModelNotMutatedTest``;
(f) plan entries: composition+identity does not act in the second pass, identity is boosted, a known distill LoRA never is, a
    character LoRA that is also a boosted plan entry is boosted once -> ``LoraPlanTest``.

The references are stock comfy parts (``nodes.LatentUpscaleBy``, ``nodes.common_ksampler``, ``comfy.utils.common_upscale``), never
``finishing/hires.py``. Every run goes through a wrapper of ``comfy.sample.sample`` that keeps the sampling patcher in a
reference cycle, as the server does (``graph_cache``), and the MODELs are pinned like the graph cache pins them, so the strict
leak check of ``tests/run_tests.py`` also covers this node.
"""
import contextlib
import copy
import unittest
from unittest import mock

from . import graph_cache, lora_synth, tiny_moss
from . import test_darkroom_cpu as dk_tests
from .test_detailer_cpu import FakeVae
from .test_input_not_mutated import snapshot

SEED = 7
LATENT_HW = 16                       # 256 px on the fake VAE; the upscaled latent at scale 1.5 is 24 x 24
FACE_PX = (88.0, 80.0, 168.0, 176.0, 0.9)    # x0, y0, x1, y1, conf on the 256 x 256 decode
OWNER_SIGMAS = (0.861, 0.776, 0.668, 0.536, 0.381, 0.202)   # the recipe: 6 steps, linear_quadratic, denoise 0.32
ZONE_TOLERANCE = 1e-7                # see FaceProtectTest.test_a_face_sigma_below_the_schedule_holds_the_core_at_the_source


def pin(model):
    """The graph cache keeps the MODEL a node receives alive for the whole workflow (``graph_cache``)."""
    graph_cache._pinned[id(model)] = model
    return model


class HiresBase(dk_tests.DarkroomBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch
        import comfy.sample
        import comfy.samplers
        import comfy.utils
        import nodes
        from moss_photon.core import consts, errors, lora, lora_plan
        from moss_photon.finishing import deps, hires as pure
        from moss_photon.nodes import hires as node
        from moss_photon.nodes.lora_plan import MossPhotonLoRAPhase
        cls.torch, cls.comfy_sample, cls.comfy_samplers, cls.comfy_utils, cls.nodes = torch, comfy.sample, comfy.samplers, comfy.utils, nodes
        cls.consts, cls.errors, cls.lora_core, cls.lora_plan, cls.deps, cls.pure, cls.node = consts, errors, lora, lora_plan, deps, pure, node
        cls.phase_node = MossPhotonLoRAPhase()
        gen = torch.Generator().manual_seed(5)
        cls.source = torch.randn(1, 128, LATENT_HW, LATENT_HW, generator=gen) * 0.6    # a latent that is not empty
        cls.pos, cls.neg = cls.tiny.make_cond(23), cls.tiny.make_cond(9)
        pin(cls.tiny.patcher)

    def setUp(self):
        super().setUp()
        self.clones = []
        self.detect_calls = []

    # ---- helpers
    def latent(self, batch=1, **extra):
        samples = self.source.repeat(batch, 1, 1, 1).contiguous()
        if batch > 1:
            samples = samples + 0.05 * self.torch.arange(batch).reshape(batch, 1, 1, 1).float()
        return dict({"samples": samples}, **extra)

    def sampler_wrapper(self):
        """``comfy.sample.sample`` that records the sampling patcher and keeps it in a cycle, as the server does."""
        real = self.comfy_sample.sample

        def wrapper(model, *args, **kwargs):
            self.clones.append(model)
            model.emulated_server_cycle = model
            return real(model, *args, **kwargs)

        return mock.patch.object(self.comfy_sample, "sample", wrapper)

    def run_node(self, *, model=None, latent=None, boxes=(), detector="ok", vae=None, patch_sampler=True, **widgets):
        """The node, with a fake detector. ``boxes``: a list of ``(x0, y0, x1, y1, conf)`` per call of the detector, or an
        exception to raise. Returns ``(result dict, vae)``."""
        vae = vae if vae is not None else FakeVae(self.torch)
        args = dict(model=pin(model) if model is not None else self.patcher, positive=self.pos, negative=self.neg, vae=vae,
                    latent=latent if latent is not None else self.latent(), seed=SEED, scale=1.5, steps=6, denoise=0.32,
                    cfg=1.0, sampler_name="res_multistep", scheduler="linear_quadratic", face_protect=True, face_sigma=0.5,
                    identity_boost=0.2, upscale_method="latent_bicubic")
        args.update(widgets)
        if args["upscale_method"] is None:      # None = leave it to the node's own default
            del args["upscale_method"]
        avail = (self.deps.Availability(None, "detector.pt", None, None) if detector == "ok" else
                 self.deps.Availability("python package ultralytics is not installed", None, None, None))

        def fake_detect(avail_, image, conf):
            self.detect_calls.append((tuple(image.shape), str(image.dtype), float(conf)))
            if isinstance(boxes, BaseException):
                raise boxes
            return list(boxes)

        patches = [mock.patch.object(self.node, "_probe", lambda: avail), mock.patch.object(self.node, "_detect_faces", fake_detect)]
        if patch_sampler:
            patches.append(self.sampler_wrapper())
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            return self.node.MossPhotonHiresFix().hires(**args), vae

    def samples(self, result):
        return result["result"][0]["samples"]

    def summary(self, result):
        text = result["result"][1]
        self.assertEqual(result["ui"]["text"], [text])
        return text

    # ---- stock references
    def stock_upscale(self, latent, scale):
        return self.nodes.LatentUpscaleBy().upscale(latent, "bicubic", scale)[0]

    def pixel_upscale(self, latent, scale):
        """decode, Lanczos in pixels, encode: the source the pixel path hands to the sampler (stock comfy parts only)."""
        vae = FakeVae(self.torch)
        side = round(LATENT_HW * scale) * 16
        big = self.comfy_utils.common_upscale(vae.decode(latent["samples"]).movedim(-1, 1), side, side, "lanczos",
                                              "disabled").movedim(1, -1)
        return vae.encode(big[:, :, :, :3])

    def stock_sample(self, model, up, **kw):
        args = dict(steps=6, cfg=1.0, sampler_name="res_multistep", scheduler="linear_quadratic", denoise=0.32)
        args.update(kw)
        return self.nodes.common_ksampler(model, SEED, args["steps"], args["cfg"], args["sampler_name"], args["scheduler"],
                                          self.pos, self.neg, up, denoise=args["denoise"])[0]["samples"]

    def zone(self, shape_hw, px_box=FACE_PX, image_px=LATENT_HW * 16):
        """The face core (release map exactly 0) at the latent size ``shape_hw``, as a bool ``[1, 1, H, W]`` tensor."""
        fractions = self.pure.boxes_to_fractions([px_box], image_px, image_px)
        core = self.pure.release_map(shape_hw[0], shape_hw[1], fractions) == 0.0
        return self.torch.from_numpy(core)[None, None]

    def schedule(self, steps=6, denoise=0.32, sampler="res_multistep", scheduler="linear_quadratic"):
        """The sigmas the sampler runs, as python floats (comfy's own KSampler)."""
        ks = self.comfy_samplers.KSampler(self.patcher, steps=steps, device=self.patcher.load_device, sampler=sampler,
                                          scheduler=scheduler, denoise=denoise, model_options=self.patcher.model_options)
        return [float(s) for s in ks.sigmas.tolist()]

    def strengths(self, model):
        return sorted({round(float(entry[0]), 6) for entries in model.patches.values() for entry in entries})

    def max_diff(self, a, b, mask=None):
        diff = (a - b).abs()
        if mask is not None:
            diff = diff.masked_select(mask.expand_as(diff))
        return float(diff.max())


class RecipeTest(HiresBase):
    def test_the_defaults_run_the_sigmas_of_the_owners_graph(self):
        ks = self.comfy_samplers.KSampler(self.patcher, steps=6, device=self.patcher.load_device, sampler="res_multistep",
                                          scheduler="linear_quadratic", denoise=0.32, model_options=self.patcher.model_options)
        got = [round(float(s), 3) for s in ks.sigmas.tolist()]
        self.assertEqual(got, list(OWNER_SIGMAS) + [0.0])

    def test_the_static_combo_lists_equal_comfys(self):
        self.assertEqual(list(self.node.SAMPLER_NAMES), list(self.comfy_samplers.KSampler.SAMPLERS))
        self.assertEqual(list(self.node.SCHEDULER_NAMES), list(self.comfy_samplers.KSampler.SCHEDULERS))
        spec = self.node.MossPhotonHiresFix.INPUT_TYPES()["required"]
        self.assertEqual(list(spec["sampler_name"][0]), list(self.comfy_samplers.KSampler.SAMPLERS))

    def test_the_summary_states_the_size_the_window_and_the_faces(self):
        result, _ = self.run_node(boxes=[FACE_PX], face_sigma=0.5)
        text = self.summary(result)
        self.assertEqual(tuple(self.samples(result).shape), (1, 128, 24, 24))
        self.assertIn("scale 1.5", text)
        self.assertIn("latent 16x16 -> 24x24", text)
        self.assertIn("pixels 256x256 -> 384x384", text)
        self.assertIn("0.861 .. 0.202", text)
        self.assertIn("6 steps", text)
        self.assertIn("image 0: 1 face", text)
        self.assertIn("face_sigma 0.50", text)


class ParityTest(HiresBase):
    """(a) face_protect off and identity_boost 0: the node is the stock pair of nodes."""

    def parity(self, scale, batch=1, **extra):
        latent = self.latent(batch, **extra)
        up = self.stock_upscale(copy.copy(latent), scale)
        want = self.stock_sample(self.patcher, up)
        result, vae = self.run_node(latent=latent, scale=scale, face_protect=False, identity_boost=0.0)
        self.assertTrue(self.torch.equal(self.samples(result), want), "max diff %g" % self.max_diff(self.samples(result), want))
        self.assertEqual((vae.decodes, vae.encodes), (0, 0), "no VAE work when nothing needs the pixels")
        self.assertEqual(self.detect_calls, [])
        return result, up

    def test_scale_1_5_is_bit_equal_to_the_stock_pair(self):
        result, up = self.parity(1.5)
        self.assertEqual(tuple(up["samples"].shape), (1, 128, 24, 24))

    def test_scale_1_0_is_bit_equal_to_the_stock_pair(self):
        result, up = self.parity(1.0)
        self.assertEqual(tuple(self.samples(result).shape), (1, 128, 16, 16))

    def test_an_odd_scale_rounds_like_the_stock_node(self):
        self.parity(1.35)    # 16 * 1.35 = 21.6 -> 22

    def test_a_batch_is_bit_equal_to_the_stock_pair(self):
        self.parity(1.5, batch=2)

    def test_batch_index_and_other_keys_of_the_latent_are_kept_as_the_stock_pair_keeps_them(self):
        latent = self.latent(2, batch_index=[0, 3], downscale_ratio_spacial=16, extra="keep")
        want_up = self.stock_upscale(copy.copy(latent), 1.5)
        want = self.nodes.common_ksampler(self.patcher, SEED, 6, 1.0, "res_multistep", "linear_quadratic", self.pos, self.neg,
                                          want_up, denoise=0.32)[0]
        result, _ = self.run_node(latent=latent, face_protect=False, identity_boost=0.0)
        out = result["result"][0]
        self.assertTrue(self.torch.equal(out["samples"], want["samples"]))
        self.assertEqual(sorted(out), sorted(want))
        self.assertEqual((out["batch_index"], out["extra"]), ([0, 3], "keep"))

    def test_an_incoming_noise_mask_is_passed_through_as_the_stock_pair_does(self):
        mask = self.torch.zeros(1, 1, LATENT_HW, LATENT_HW)
        mask[..., 4:12, 4:12] = 1.0
        latent = self.latent(noise_mask=mask)
        up = self.stock_upscale(copy.copy(latent), 1.5)
        want = self.stock_sample(self.patcher, up)
        result, _ = self.run_node(latent=latent, face_protect=False, identity_boost=0.0)
        self.assertTrue(self.torch.equal(self.samples(result), want))

    def test_the_pixel_mode_equals_decode_lanczos_encode_and_the_stock_sampler(self):
        latent = self.latent()
        vae = FakeVae(self.torch)
        pixels = vae.decode(latent["samples"])
        big = self.comfy_utils.common_upscale(pixels.movedim(-1, 1), 24 * 16, 24 * 16, "lanczos", "disabled").movedim(1, -1)
        fresh = dict(latent, samples=vae.encode(big[:, :, :, :3]))
        want = self.stock_sample(self.patcher, fresh)
        vae2 = FakeVae(self.torch)
        result, vae2 = self.run_node(latent=latent, vae=vae2, face_protect=False, identity_boost=0.0, upscale_method="pixel_lanczos")
        self.assertTrue(self.torch.equal(self.samples(result), want), "max diff %g" % self.max_diff(self.samples(result), want))
        self.assertEqual((vae2.decodes, vae2.encodes), (1, 1))

    def test_the_pixel_mode_at_scale_1_0_is_legal(self):
        result, vae = self.run_node(face_protect=False, identity_boost=0.0, upscale_method="pixel_lanczos", scale=1.0)
        self.assertEqual(tuple(self.samples(result).shape), (1, 128, 16, 16))
        self.assertTrue(bool(self.torch.isfinite(self.samples(result)).all()))

    def test_the_other_sampler_and_scheduler_are_the_stock_ones_too(self):
        latent = self.latent()
        up = self.stock_upscale(copy.copy(latent), 1.5)
        want = self.stock_sample(self.patcher, up, sampler_name="euler", scheduler="simple", denoise=0.5, steps=4, cfg=2.0)
        result, _ = self.run_node(latent=latent, face_protect=False, identity_boost=0.0, sampler_name="euler", scheduler="simple",
                                  denoise=0.5, steps=4, cfg=2.0)
        self.assertTrue(self.torch.equal(self.samples(result), want))


class FaceProtectTest(HiresBase):
    """(c) the face enters the redraw late."""

    def runs(self, face_sigma, **kw):
        latent = self.latent()
        up = self.pixel_upscale(latent, 1.5)
        protected, _ = self.run_node(latent=latent, boxes=[FACE_PX], face_sigma=face_sigma, identity_boost=0.0, upscale_method="pixel_lanczos", **kw)
        plain, _ = self.run_node(latent=latent, face_protect=False, identity_boost=0.0, upscale_method="pixel_lanczos")
        return up, self.samples(protected), self.samples(plain)

    def test_a_face_sigma_below_the_schedule_holds_the_core_at_the_source(self):
        up, protected, plain = self.runs(0.0)
        zone = self.zone(tuple(up.shape[-2:]))
        self.assertGreater(int(zone.sum()), 20, "control: the core has cells")
        held = self.max_diff(protected, up, zone)
        free = self.max_diff(plain, up, zone)
        # the last step of res_multistep is x + (x - denoised) / sigma * (0 - sigma): one rounding away from `denoised`, so the
        # held core is the source to one float32 rounding (measured: max diff 5.96e-8 = 2**-24), not bit for bit; torch.equal does
        # not hold, 1e-7 is the tightest round tolerance that does. The unprotected run moves the same zone by about 4.
        self.assertLessEqual(held, ZONE_TOLERANCE, "held core differs from the source by %g" % held)
        self.assertGreater(free, 1000 * max(held, 1e-9), "non-vacuous: without protection the same zone moves by %g" % free)
        self.assertGreater(free, 1e-2)
        outside = ~zone
        self.assertGreater(self.max_diff(protected, up, outside), 1e-2, "outside the core the redraw happened")
        self.assertGreater(self.max_diff(protected, plain, outside), 0.0)

    def test_a_release_inside_the_schedule_makes_the_core_differ_from_both(self):
        up, protected, plain = self.runs(0.5)
        zone = self.zone(tuple(up.shape[-2:]))
        self.assertGreater(self.max_diff(protected, up, zone), 1e-2, "the core was redrawn")
        self.assertGreater(self.max_diff(protected, plain, zone), 1e-2, "and not as the unprotected run draws it")

    def test_the_later_the_release_the_closer_the_core_stays_to_the_source(self):
        latent = self.latent()
        up = self.pixel_upscale(latent, 1.5)
        zone = self.zone(tuple(up.shape[-2:]))
        dist = []
        for face_sigma in (0.8, 0.5, 0.3, 0.0):
            result, _ = self.run_node(latent=latent, boxes=[FACE_PX], face_sigma=face_sigma, identity_boost=0.0,
                                      upscale_method="pixel_lanczos")
            dist.append(self.max_diff(self.samples(result), up, zone))
        self.assertEqual(dist, sorted(dist, reverse=True), dist)

    def test_two_faces_are_both_held(self):
        second = (16.0, 16.0, 64.0, 80.0, 0.8)
        latent = self.latent()
        up = self.pixel_upscale(latent, 1.5)
        result, _ = self.run_node(latent=latent, boxes=[FACE_PX, second], face_sigma=0.0, identity_boost=0.0,
                                  upscale_method="pixel_lanczos")
        out = self.samples(result)
        for box in (FACE_PX, second):
            self.assertLessEqual(self.max_diff(out, up, self.zone(tuple(up.shape[-2:]), box)), ZONE_TOLERANCE)
        self.assertIn("image 0: 2 faces", self.summary(result))

    def test_the_mask_and_the_function_are_on_the_clone_only(self):
        result, _ = self.run_node(boxes=[FACE_PX])
        [clone] = self.clones
        self.assertIn("denoise_mask_function", clone.model_options)
        self.assertNotIn("denoise_mask_function", self.patcher.model_options)

    def test_the_detector_sees_the_decoded_input_image(self):
        result, vae = self.run_node(boxes=[FACE_PX])
        [(shape, dtype, conf)] = self.detect_calls
        self.assertEqual((shape, dtype), ((256, 256, 3), "float32"))
        self.assertAlmostEqual(conf, 0.45)     # face_detailer.Settings().det_conf
        self.assertEqual((vae.decodes, vae.encodes), (1, 1), "decoded once for the faces; a held face takes the pixel path")

    def test_a_batch_gets_one_detection_per_image_and_images_without_a_face_are_untouched(self):
        latent = self.latent(2)
        up = self.pixel_upscale(latent, 1.5)
        calls = iter([[FACE_PX], []])
        with mock.patch.object(self.node, "_detect_faces", side_effect=lambda a, i, c: next(calls)) as detect:
            with mock.patch.object(self.node, "_probe", lambda: self.deps.Availability(None, "d.pt", None, None)):
                with self.sampler_wrapper():
                    result = self.node.MossPhotonHiresFix().hires(
                        model=self.patcher, positive=self.pos, negative=self.neg, vae=FakeVae(self.torch), latent=latent,
                        seed=SEED, scale=1.5, steps=6, denoise=0.32, cfg=1.0, sampler_name="res_multistep",
                        scheduler="linear_quadratic", face_protect=True, face_sigma=0.0, identity_boost=0.0,
                        upscale_method="pixel_lanczos")
        self.assertEqual(detect.call_count, 2)
        out = self.samples(result)
        zone = self.zone(tuple(up.shape[-2:]))
        self.assertLessEqual(self.max_diff(out[:1], up[:1], zone), ZONE_TOLERANCE)
        self.assertGreater(self.max_diff(out[1:], up[1:], zone), 1e-2, "image 1 has no face: nothing is held")
        text = self.summary(result)
        self.assertIn("image 0: 1 face", text)
        self.assertIn("image 1: 0 faces", text)


class MethodTest(HiresBase):
    """MP-93: the bicubic latent upscale leaves a doubled image, so a face is never held on it, and the default is the pixel upscale."""

    def assert_pixel_path(self, result, vae, warned):
        """The run decoded once, upscaled in pixels and encoded once, names pixel_lanczos in its first line, and says (or not) why."""
        text = self.summary(result)
        self.assertEqual((vae.decodes, vae.encodes), (1, 1))
        self.assertIn("(pixel_lanczos)", text.splitlines()[0])
        self.assertNotIn("(latent_bicubic)", text.splitlines()[0])
        self.assertEqual("[hires_bicubic_not_held]" in text, warned)

    def test_the_default_run_uses_the_pixel_path(self):
        result, vae = self.run_node(upscale_method=None, boxes=[FACE_PX], identity_boost=0.0)
        self.assert_pixel_path(result, vae, warned=False)
        same, _ = self.run_node(upscale_method="pixel_lanczos", boxes=[FACE_PX], identity_boost=0.0)
        self.assertTrue(self.torch.equal(self.samples(result), self.samples(same)))

    def test_the_default_run_without_protection_uses_the_pixel_path_too(self):
        result, vae = self.run_node(upscale_method=None, face_protect=False, identity_boost=0.0)
        self.assert_pixel_path(result, vae, warned=False)

    def test_bicubic_with_a_found_face_takes_the_pixel_path_and_warns(self):
        result, vae = self.run_node(upscale_method="latent_bicubic", boxes=[FACE_PX], identity_boost=0.0)
        self.assert_pixel_path(result, vae, warned=True)
        text = self.summary(result)
        self.assertIn("doubles the image", text)
        self.assertIn("face_protect to off", text)
        self.assertIn("face protection: on", text)
        pixel, _ = self.run_node(upscale_method="pixel_lanczos", boxes=[FACE_PX], identity_boost=0.0)
        self.assertTrue(self.torch.equal(self.samples(result), self.samples(pixel)), "it is the pixel run, with the same hold")
        [clone] = self.clones[:1]
        self.assertIn("denoise_mask_function", clone.model_options)

    def test_bicubic_with_protection_off_stays_the_stock_recipe_and_does_not_warn(self):
        latent = self.latent()
        result, vae = self.run_node(latent=latent, upscale_method="latent_bicubic", face_protect=False, identity_boost=0.0)
        text = self.summary(result)
        self.assertEqual((vae.decodes, vae.encodes), (0, 0))
        self.assertIn("(latent_bicubic)", text.splitlines()[0])
        self.assertNotIn("hires_bicubic_not_held", text)
        want = self.stock_sample(self.patcher, self.stock_upscale(copy.copy(latent), 1.5))
        self.assertTrue(self.torch.equal(self.samples(result), want))

    def test_bicubic_with_no_face_stays_the_stock_recipe_and_does_not_warn_about_the_method(self):
        latent = self.latent()
        for boxes, detector, code in (([], "ok", "hires_no_face"), ([FACE_PX], "missing", "hires_no_detector")):
            with self.subTest(code=code):
                result, vae = self.run_node(latent=latent, upscale_method="latent_bicubic", boxes=boxes, detector=detector,
                                            identity_boost=0.0)
                text = self.summary(result)
                self.assertIn("[%s]" % code, text)
                self.assertNotIn("hires_bicubic_not_held", text)
                self.assertEqual(vae.encodes, 0, "no pixel upscale")
                self.assertIn("(latent_bicubic)", text.splitlines()[0])
                want = self.stock_sample(self.patcher, self.stock_upscale(copy.copy(latent), 1.5))
                self.assertTrue(self.torch.equal(self.samples(result), want))

    def test_bicubic_with_a_face_sigma_above_the_schedule_builds_no_hold_and_stays_stock(self):
        result, vae = self.run_node(upscale_method="latent_bicubic", boxes=[FACE_PX], face_sigma=1.0, identity_boost=0.0)
        text = self.summary(result)
        self.assertNotIn("hires_bicubic_not_held", text)
        self.assertIn("(latent_bicubic)", text.splitlines()[0])
        self.assertEqual(vae.encodes, 0)

    def test_planted_a_hold_on_the_bicubic_latent_goes_red(self):
        """Invariant proven by breaking it: with the switch removed the face is held on the bicubic latent, and the checks of the
        tests above fail."""
        with mock.patch.object(self.node, "_method_for", lambda upscale_method, holding: upscale_method):
            result, vae = self.run_node(upscale_method="latent_bicubic", boxes=[FACE_PX], identity_boost=0.0)
        self.assertIn("denoise_mask_function", self.clones[0].model_options, "control: the hold was built")
        with self.assertRaises(AssertionError):
            self.assert_pixel_path(result, vae, warned=True)


class DegradationTest(HiresBase):
    """(d) a setting that cannot be honoured is a warning in the summary and a run, never a raise."""

    def unprotected(self, **kw):
        result, _ = self.run_node(face_protect=False, identity_boost=0.0, **kw)
        return self.samples(result)

    def assert_degraded(self, code, **kw):
        result, _ = self.run_node(identity_boost=0.0, **kw)
        text = self.summary(result)
        self.assertIn("[%s]" % code, text)
        self.assertTrue(bool(self.torch.isfinite(self.samples(result)).all()))
        return result

    def test_no_detector_is_a_warning_and_an_unprotected_run(self):
        result = self.assert_degraded("hires_no_detector", detector="missing", boxes=[FACE_PX])
        self.assertIn("ultralytics", self.summary(result))
        self.assertEqual(self.detect_calls, [])
        self.assertTrue(self.torch.equal(self.samples(result), self.unprotected()))
        self.assertNotIn("denoise_mask_function", self.clones[0].model_options)

    def test_a_failing_detector_is_a_warning_and_an_unprotected_run(self):
        result = self.assert_degraded("hires_detector_failed", boxes=RuntimeError("model file is broken"))
        self.assertIn("model file is broken", self.summary(result))
        self.assertTrue(self.torch.equal(self.samples(result), self.unprotected()))

    def test_no_face_is_a_warning_and_an_unprotected_run(self):
        result = self.assert_degraded("hires_no_face", boxes=[])
        self.assertTrue(self.torch.equal(self.samples(result), self.unprotected()))
        self.assertNotIn("denoise_mask_function", self.clones[0].model_options)

    def test_a_face_sigma_at_or_above_the_first_sigma_is_a_warning_with_no_mask_at_all(self):
        for face_sigma in (self.schedule()[0], 0.95, 1.0):
            with self.subTest(face_sigma=face_sigma):
                self.clones.clear()
                result = self.assert_degraded("hires_face_sigma_above_schedule", boxes=[FACE_PX], face_sigma=face_sigma)
                self.assertTrue(self.torch.equal(self.samples(result), self.unprotected()))
                self.assertNotIn("denoise_mask_function", self.clones[0].model_options)

    def test_a_face_sigma_at_or_below_the_last_sigma_is_a_warning_and_the_face_stays_as_in_the_source(self):
        for face_sigma in (self.schedule()[-2], 0.1, 0.0):
            with self.subTest(face_sigma=face_sigma):
                result = self.assert_degraded("hires_face_sigma_below_schedule", boxes=[FACE_PX], face_sigma=face_sigma)
                self.assertIn("stays as in the source", self.summary(result))
                self.assertIn("denoise_mask_function", self.clones[-1].model_options, "the run still goes with protection")

    def test_a_boost_without_a_target_is_a_warning_and_changes_nothing(self):
        result, _ = self.run_node(face_protect=False, identity_boost=0.5)
        self.assertIn("[hires_boost_no_target]", self.summary(result))
        self.assertTrue(self.torch.equal(self.samples(result), self.unprotected()))
        self.assertEqual(self.strengths(self.clones[0]), [])

    def test_no_boost_asked_no_warning(self):
        result, _ = self.run_node(face_protect=False, identity_boost=0.0)
        self.assertNotIn("hires_boost", self.summary(result))

    def test_an_interrupt_in_the_detector_is_not_swallowed(self):
        import comfy.model_management as mm
        with self.assertRaises(mm.InterruptProcessingException):
            self.run_node(boxes=mm.InterruptProcessingException())

    def test_an_interrupt_raised_by_the_polling_call_is_not_swallowed(self):
        import comfy.model_management as mm
        with mock.patch.object(mm, "throw_exception_if_processing_interrupted", side_effect=mm.InterruptProcessingException()):
            with self.assertRaises(mm.InterruptProcessingException):
                self.run_node(boxes=[FACE_PX])

    def test_an_incoming_noise_mask_with_faces_is_not_combined_and_passes_through(self):
        mask = self.torch.zeros(1, 1, LATENT_HW, LATENT_HW)
        mask[..., 2:14, 2:14] = 1.0
        latent = self.latent(noise_mask=mask)
        result = self.assert_degraded("hires_mask_not_combined", boxes=[FACE_PX], latent=latent)
        up = self.stock_upscale(copy.copy(latent), 1.5)
        self.assertTrue(self.torch.equal(self.samples(result), self.stock_sample(self.patcher, up)))
        self.assertNotIn("denoise_mask_function", self.clones[0].model_options)

    def test_a_mask_function_that_is_already_on_the_model_is_not_replaced(self):
        mine = self.patcher.clone()
        mine.set_model_denoise_mask_function(lambda sigma, denoise_mask, extra_options: denoise_mask)
        result = self.assert_degraded("hires_mask_not_combined", boxes=[FACE_PX], model=mine)
        self.assertIs(self.clones[0].model_options["denoise_mask_function"], mine.model_options["denoise_mask_function"])

    def test_a_non_moss_model_is_refused_before_any_work(self):
        fake = mock.MagicMock()
        with mock.patch.object(self.node.adapter, "probe_model") as probe:
            probe.return_value = mock.MagicMock(is_moss=False, class_name="Other")
            with self.assertRaises(self.errors.UnsupportedModel):
                self.run_node(model=fake, boxes=[FACE_PX])
        self.assertEqual(self.detect_calls, [])

    def test_a_latent_with_the_wrong_channels_is_refused(self):
        with self.assertRaises(self.errors.InvalidLatent):
            self.run_node(latent={"samples": self.torch.zeros(1, 4, 8, 8) + 0.1}, face_protect=False)

    def test_a_missing_character_lora_file_is_refused(self):
        with self.assertRaises(FileNotFoundError):
            self.run_node(face_protect=False, character_lora="nowhere.safetensors")

    def test_a_character_lora_that_maps_nothing_is_refused(self):
        self.env.write("bogus.safetensors", lora_synth.add_bogus_modules({}, 6))
        with self.assertRaises(self.errors.IncompatibleLora):
            self.run_node(face_protect=False, character_lora="bogus.safetensors")

    def test_a_non_finite_result_is_refused_not_returned(self):
        def nan_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, **kwargs):
            self.clones.append(model)
            return self.torch.full_like(latent_image, float("nan"))

        with mock.patch.object(self.comfy_sample, "sample", nan_sample):
            with self.assertRaises(self.errors.NonFiniteOutput):
                self.run_node(face_protect=False, identity_boost=0.0, patch_sampler=False)


class SizeWarningTest(HiresBase):
    """The size of the result against the MOSS range (recommended up to 2 MP, extrapolation up to 4 MP). The sampling is stubbed:
    the subject is the note, not the picture."""

    def run_big(self, hw, scale):
        def skip(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, **kwargs):
            return latent_image.clone()

        with mock.patch.object(self.comfy_sample, "sample", skip):
            result, _ = self.run_node(latent={"samples": self.torch.randn(1, 128, hw, hw) * 0.1}, scale=scale,
                                      face_protect=False, identity_boost=0.0, patch_sampler=False)
        return self.summary(result)

    def test_above_two_megapixels_is_an_info(self):
        for hw in (100, 120):          # 1600 px = 2.56 MP, 1920 px = 3.69 MP
            with self.subTest(latent=hw):
                text = self.run_big(hw, 1.0)
                self.assertIn("info: [MossPhoton] MossPhotonHiresFix/scale", text)
                self.assertNotIn("warning: [MossPhoton] MossPhotonHiresFix/scale", text)
                self.assertIn("[megapixels_out_of_documented_range]", text)
                self.assertIn("2 MP", text)

    def test_above_four_megapixels_is_a_warning(self):
        text = self.run_big(134, 1.0)          # 2144 px = 4.60 MP
        self.assertIn("warning: [MossPhoton] MossPhotonHiresFix/scale", text)
        self.assertIn("[megapixels_out_of_documented_range]", text)
        self.assertIn("4 MP", text)

    def test_the_size_after_the_scale_is_what_counts(self):
        self.assertIn("warning: [MossPhoton] MossPhotonHiresFix/scale", self.run_big(100, 1.5))    # 150 x 16 = 2400 px = 5.76 MP

    def test_small_results_say_nothing(self):
        result, _ = self.run_node(face_protect=False, identity_boost=0.0)
        self.assertNotIn("megapixels", self.summary(result))


class ModelNotMutatedTest(HiresBase):
    """(e) the incoming MODEL is the same afterwards, on the success path with every feature on."""

    def test_patches_options_and_attachments_of_the_incoming_model_are_unchanged(self):
        self.env.write("char.safetensors", lora_synth.make_lora_sd(self.patcher, seed=31, modules=("qkv", "o")))
        stocked = ss_apply(self.patcher, self.a_sd, 0.8)
        planned = self.phase_node.apply(stocked, "A.safetensors", 0.8, "identity")[0]
        planned = self.phase_node.apply(planned, "D.safetensors", 0.5, "composition+identity")[0]
        snaps = [snapshot(m) for m in (self.patcher, stocked, planned)]
        uuids = [str(m.patches_uuid) for m in (self.patcher, stocked, planned)]
        plan_before = copy.deepcopy(self.lora_plan.entries_of(planned))
        keys_before = sorted(planned.model_options)
        result, _ = self.run_node(model=planned, boxes=[FACE_PX], character_lora="char.safetensors", identity_boost=0.2)
        for model, snap, uuid in zip((self.patcher, stocked, planned), snaps, uuids):
            after = snapshot(model)
            w_b, w_a = snap.pop("weights"), after.pop("weights")
            self.assertEqual(snap, after)
            self.assertEqual(str(model.patches_uuid), uuid)
            for key, value in w_b.items():
                self.assertTrue(self.torch.equal(value, w_a[key]), key)
        self.assertNotIn("denoise_mask_function", planned.model_options)
        self.assertEqual(sorted(planned.model_options), keys_before)
        self.assertEqual(self.lora_plan.entries_of(planned), plan_before)
        [clone] = self.clones
        self.assertIsNot(clone, planned)
        self.assertIn("denoise_mask_function", clone.model_options)
        self.assertGreater(entry_count(clone), entry_count(planned), "and it did patch: the clone carries the boost")

    def test_the_incoming_latent_and_conditioning_are_not_written(self):
        latent = self.latent()
        before = {"samples": latent["samples"].clone()}
        pos = [(t.clone(), dict(d)) for t, d in self.pos]
        self.run_node(latent=latent, boxes=[FACE_PX])
        self.assertTrue(self.torch.equal(latent["samples"], before["samples"]))
        self.assertEqual(sorted(latent), ["samples"])
        for (t, d), (t0, d0) in zip(self.pos, pos):
            self.assertTrue(self.torch.equal(t, t0))
            self.assertEqual(d, d0)


def ss_apply(model, sd, strength):
    from . import sampler_support as ss
    return ss.apply_lora(model, sd, strength)


def entry_count(model):
    return sum(len(entries) for entries in model.patches.values())


class LoraPlanTest(HiresBase):
    """(f) the second pass counts as the finish phase."""

    def run_plan(self, model, **kw):
        kw.setdefault("face_protect", False)
        result, _ = self.run_node(model=model, **kw)
        return result, self.clones[-1]

    def test_composition_plus_identity_does_not_act_and_identity_is_boosted(self):
        model = self.phase_node.apply(self.patcher, "A.safetensors", 0.5, "composition+identity")[0]
        model = self.phase_node.apply(model, "D.safetensors", 0.8, "identity")[0]
        result, clone = self.run_plan(model, identity_boost=0.2)
        self.assertEqual(self.strengths(clone), [1.0], "D acts at 0.8 + 0.2, A is not in the clone")
        text = self.summary(result)
        self.assertIn("[hires_lora_phase_skipped]", text)
        self.assertIn("A.safetensors", text)
        self.assertNotIn("hires_boost_no_target", text)
        self.assertIn("D.safetensors", text)
        self.assertIn("+0.20", text)

    def test_without_a_boost_the_identity_entry_acts_at_its_strength(self):
        model = self.phase_node.apply(self.patcher, "D.safetensors", 0.8, "identity")[0]
        result, clone = self.run_plan(model, identity_boost=0.0)
        self.assertEqual(self.strengths(clone), [0.8])

    def test_texture_acts_and_all_acts_but_neither_takes_the_boost(self):
        model = self.phase_node.apply(self.patcher, "A.safetensors", 0.6, "texture")[0]
        model = self.phase_node.apply(model, "D.safetensors", 0.7, "all")[0]
        result, clone = self.run_plan(model, identity_boost=0.2)
        self.assertEqual(self.strengths(clone), [0.6, 0.7])
        self.assertIn("[hires_boost_no_target]", self.summary(result))

    def test_composition_is_skipped_by_phase(self):
        model = self.phase_node.apply(self.patcher, "A.safetensors", 0.6, "composition")[0]
        result, clone = self.run_plan(model, identity_boost=0.0)
        self.assertEqual(self.strengths(clone), [])
        self.assertIn("[hires_lora_phase_skipped]", self.summary(result))

    def test_the_plan_is_resolved_in_plan_order_on_top_of_an_ordinary_loader(self):
        stocked = ss_apply(self.patcher, self.a_sd, 0.4)
        model = self.phase_node.apply(stocked, "D.safetensors", 0.8, "identity")[0]
        result, clone = self.run_plan(model, identity_boost=0.2)
        self.assertEqual(self.strengths(clone), [0.4, 1.0])

    def test_a_known_distill_entry_is_never_boosted(self):
        sha = self.env.write("dmd.safetensors", lora_synth.make_lora_sd(self.patcher, seed=41, modules=("qkv", "o")))
        row = dict(self.consts.KNOWN_DISTILL_LORAS["7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2"])
        with mock.patch.dict(self.consts.KNOWN_DISTILL_LORAS, {sha: row}):
            model = self.phase_node.apply(self.patcher, "dmd.safetensors", 0.9, "identity")[0]
            result, clone = self.run_plan(model, identity_boost=0.2)
        self.assertEqual(self.strengths(clone), [0.9])
        self.assertIn("[hires_boost_no_target]", self.summary(result))

    def test_a_known_distill_as_character_lora_gets_no_boost_and_a_warning(self):
        sha = self.env.write("dmd.safetensors", lora_synth.make_lora_sd(self.patcher, seed=41, modules=("qkv", "o")))
        row = dict(self.consts.KNOWN_DISTILL_LORAS["7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2"])
        with mock.patch.dict(self.consts.KNOWN_DISTILL_LORAS, {sha: row}):
            result, clone = self.run_plan(self.patcher, character_lora="dmd.safetensors", identity_boost=0.2)
        self.assertEqual(self.strengths(clone), [])
        text = self.summary(result)
        self.assertIn("[hires_boost_distill_skipped]", text)
        self.assertIn("[hires_boost_no_target]", text)

    def test_a_plain_loader_model_is_boosted_through_the_character_lora(self):
        stocked = ss_apply(self.patcher, self.a_sd, 0.8)
        result, clone = self.run_plan(stocked, character_lora="A.safetensors", identity_boost=0.2)
        self.assertEqual(self.strengths(clone), [0.2, 0.8])
        self.assertNotIn("hires_boost_no_target", self.summary(result))
        self.assertIn("A.safetensors", self.summary(result))
        self.assertIn("+0.20", self.summary(result))
        plain, _ = self.run_node(model=stocked, face_protect=False, identity_boost=0.0)
        self.assertFalse(self.torch.equal(self.samples(plain), self.samples(result)), "the boost changes the picture")

    def test_a_zero_boost_loads_nothing_and_patches_nothing(self):
        result, clone = self.run_plan(self.patcher, character_lora="A.safetensors", identity_boost=0.0)
        self.assertEqual(self.strengths(clone), [])

    def test_a_character_lora_that_is_a_boosted_plan_entry_is_boosted_once(self):
        model = self.phase_node.apply(self.patcher, "A.safetensors", 0.8, "identity")[0]
        result, clone = self.run_plan(model, character_lora="A.safetensors", identity_boost=0.2)
        self.assertEqual(self.strengths(clone), [1.0])
        self.assertEqual(max(len(entries) for entries in clone.patches.values()), 1, "one patch per key, not two")
        self.assertIn("[hires_boost_already_in_plan]", self.summary(result))
        twice, clone2 = self.run_plan(model, identity_boost=0.2)
        self.assertTrue(self.torch.equal(self.samples(result), self.samples(twice)))

    def test_a_character_lora_that_is_a_plan_entry_outside_the_finish_is_boosted_by_itself(self):
        model = self.phase_node.apply(self.patcher, "A.safetensors", 0.5, "composition+identity")[0]
        result, clone = self.run_plan(model, character_lora="A.safetensors", identity_boost=0.2)
        self.assertEqual(self.strengths(clone), [0.2])

    def test_the_clone_is_a_direct_clone_of_the_incoming_model(self):
        model = self.phase_node.apply(self.patcher, "A.safetensors", 0.8, "identity")[0]
        self.run_plan(model, identity_boost=0.2)
        self.assertIs(self.clones[0].parent, model)


if __name__ == "__main__":
    unittest.main()
