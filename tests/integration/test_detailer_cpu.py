"""MossPhotonIdentityDetailer on the tiny random MOSS, CPU (MP-58): the node end to end with the REAL redraw
(VAE encode, turbo-style sampling at denoise 0.25 through comfy.sample, VAE decode) on a clone of the MODEL, a fake VAE
and fake detector/ArcFace (no weights, no ultralytics, no insightface).

(f) the input MODEL is not mutated; plus the node-level rules: profile checks, consent, accept rule through the node,
outside-mask equality on the real tensors, the report section, batch handling."""
import copy
import dataclasses
import unittest
from unittest import mock

from . import _comfy_env, tiny_moss
from . import sampler_support as ss
from .test_input_not_mutated import snapshot


class FakeVae:
    """16x average-pool "encoder" and nearest-neighbour "decoder" with the MOSS latent shape (128 channels, ratio 16)."""

    def __init__(self, torch):
        self.torch, self.encodes, self.decodes = torch, 0, 0

    def encode(self, pixels):
        t = self.torch
        self.encodes += 1
        self.last_encoded_shape = None
        x = (pixels[..., :3].permute(0, 3, 1, 2).float() - 0.5) * 2.0
        x = t.nn.functional.avg_pool2d(x, 16)
        out = x.repeat(1, 43, 1, 1)[:, :128].contiguous()
        self.last_encoded_shape = tuple(out.shape)
        return out

    def decode(self, latent):
        t = self.torch
        self.decodes += 1
        x = t.nn.functional.interpolate(latent[:, :3].float(), scale_factor=16, mode="nearest")
        return (x / 2.0 + 0.5).clamp(0.0, 1.0).permute(0, 2, 3, 1).contiguous()


class FakeClip:
    def __init__(self, tiny):
        self.tiny, self.texts = tiny, []

    def tokenize(self, text):
        self.texts.append(text)
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return self.tiny.make_cond(3 if tokens == "" else 7)


class IntegrationBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.core import canon, consts, errors, report as report_mod
        from moss_photon.finishing import deps, face_detailer as fd, geometry
        from moss_photon.nodes import detailer
        from tests.unit.detailer import fakes
        cls.torch, cls.report_mod, cls.deps, cls.fd, cls.geometry = torch, report_mod, deps, fd, geometry
        cls.canon, cls.consts, cls.errors = canon, consts, errors
        cls.detailer, cls.fakes = detailer, fakes
        cls.base = cls.tiny.patcher
        cls.chain = ss.apply_lora(cls.base, ss.make_lora_dict(cls.base, seed=11), 1.0)
        cls.profile = ss.make_profile(cls.chain, base_model=cls.base, role="distill", name="distill.safetensors")
        character = copy.deepcopy(cls.profile["stack"][0])
        character["role"] = "character"
        character["trigger_hints"] = ["MyTrigger"]
        cls.profile["stack"].append(character)
        cls.report = ss.read_contract("report.json")

    def avail(self):
        return self.deps.Availability(None, "detector.pt", None, "/arcface")

    def scene(self, faces, batch=1):
        img = self.fakes.make_scene(240, 320, faces)
        t = self.torch.from_numpy(img)[None]
        return t.repeat(batch, 1, 1, 1).contiguous()

    def photos(self, n=3, person="A"):
        """Reference photos: a 96x96 frame whose centre has the person's tone and sharp texture."""
        imgs = []
        for k in range(n):
            imgs.append(self.torch.from_numpy(self.fakes.make_scene(96, 96, [(person, 8, 8, 88, 88)], seed=100 + k)))
        return self.torch.stack(imgs)

    WORK_PX = 128  # the real redraw runs at 128 px (8x8 latent) to keep the suite fast; one test runs the default 1024

    def make_hybrid(self, detections, score_before, score_after, captured, work_px=None):
        """Backend factory for ``detailer._make_backend``: fakes for the detector/ArcFace, the REAL redraw on the clone."""
        fakes, backends, tiny = self.fakes, None, self.tiny
        from moss_photon.finishing import backends as backends_mod

        def factory(avail, settings, model, vae, positive, negative):
            captured.update(model=model, vae=vae, positive=positive, negative=negative, settings=settings)
            small = dataclasses.replace(settings, work_px=work_px or self.WORK_PX)
            real = backends_mod.ComfyBackend(avail, small, model=model, vae=vae, positive=positive, negative=negative)

            class Hybrid(fakes.FakeBackend):
                def read_faces(inner, rgb_u8):
                    readings = super().read_faces(rgb_u8)
                    if rgb_u8.shape[0] == 96:  # a reference photo: person A at sharp quality
                        std = fakes.SHARP_STD
                    else:                      # a crop: the quality before / after the REAL redraw is scripted
                        std = score_before if inner.redraw_calls == 0 else score_after
                    return [fakes.fd.FaceReading(fakes.embed("A", std), r.det_score, r.bbox, r.kps) for r in readings]

                def redraw(inner, crop, seed):
                    inner.redraw_calls += 1
                    inner.redraw_seeds.append(seed)
                    return real.redraw(crop, seed)

            hybrid = Hybrid(detections)
            captured["backend"] = hybrid
            return hybrid

        return factory

    def run_node(self, image, *, model=None, profile=None, detections, score_before=0.05, score_after=0.25,
                 work_px=None, **kwargs):
        captured = {}
        node = self.detailer.MossPhotonIdentityDetailer()
        clip = FakeClip(self.tiny)
        vae = FakeVae(self.torch)
        args = dict(image=image, model=model if model is not None else self.chain, clip=clip, vae=vae,
                    report=copy.deepcopy(self.report), identity_profile=profile if profile is not None else self.profile,
                    seed=11, size_gate_px=110, denoise=0.25, accept_without_references=True)
        args.update(kwargs)
        with mock.patch.object(self.detailer, "_probe", self.avail), \
                mock.patch.object(self.detailer, "_make_backend",
                                  self.make_hybrid(detections, score_before, score_after, captured, work_px)):
            result = node.detail(**args)
        captured.update(clip=clip, vae=vae, args=args)
        return result, captured


FACE_A = ("A", 130, 90, 180, 150)
FACE_BIG = ("A", 200, 40, 312, 200)


def det(face):
    return (float(face[1]), float(face[2]), float(face[3]), float(face[4]), 0.9)


class DetailerNodeCpuTest(IntegrationBase):
    def test_no_references_the_redraw_is_kept_unscorable_and_outside_the_mask_is_bit_equal(self):
        image = self.scene([FACE_A])
        before = image.clone()
        result, cap = self.run_node(image, detections=[det(FACE_A)])
        out, report, summary = result["result"]
        self.assertTrue(self.torch.equal(image, before))                 # the input tensor is not written
        self.assertEqual((out.shape, out.dtype), (image.shape, image.dtype))
        sec = report["detailer"]
        self.assertEqual(self.report_mod.validate_report(report), [])
        self.assertEqual([f["decision"] for f in sec["faces"]], ["accepted_unscorable"])
        self.assertTrue(sec["faces"][0]["unscorable"])
        rec = sec["faces"][0]
        self.assertEqual((rec["denoise"], rec["steps"]), (0.25, 8))
        full = self.fakes.full_mask_of((240, 320), rec)
        outside = self.torch.from_numpy(full <= 0.0)
        self.assertTrue(self.torch.equal(out[0][outside], image[0][outside]))      # bit-equal outside the mask
        self.assertFalse(self.torch.equal(out, image))                             # control: the face changed
        self.assertEqual(cap["vae"].encodes, 1)
        self.assertEqual(cap["vae"].decodes, 1)
        self.assertIn("IdentityDetailer: 1 face(s): 1 accepted", summary)
        self.assertEqual(result["ui"]["text"], [summary])

    def test_the_default_1024_px_redraw_runs_end_to_end(self):
        """One run at the real default (crop upscaled to 1024 px, 8 steps at denoise 0.25) on the tiny model."""
        image = self.scene([FACE_A])
        result, cap = self.run_node(image, detections=[det(FACE_A)], work_px=1024)
        sec = result["result"][1]["detailer"]
        self.assertEqual(sec["settings"]["work_px"], 1024)
        self.assertEqual(sec["faces"][0]["decision"], "accepted_unscorable")
        self.assertEqual(cap["vae"].encodes, 1)
        enc_shape = cap["vae"].last_encoded_shape
        self.assertEqual(enc_shape, (1, 128, 64, 64))  # 1024 px / 16

    def test_f_the_input_model_is_not_mutated(self):
        """(f) patches, options, attachments, wrappers and weights of the incoming MODEL are the same afterwards."""
        snaps = [snapshot(m) for m in (self.chain, self.base)]
        image = self.scene([FACE_A])
        result, cap = self.run_node(image, detections=[det(FACE_A)])
        self.assertIsNot(cap["model"], self.chain)                      # sampling ran on a clone
        for model, snap in zip((self.chain, self.base), snaps):
            after = snapshot(model)
            w_b, w_a = snap.pop("weights"), after.pop("weights")
            self.assertEqual(snap, after)
            for k, v in w_b.items():
                self.assertTrue(self.torch.equal(v, w_a[k]), k)
        self.assertNotIn("wrappers", self.chain.model_options.get("transformer_options", {}))
        self.assertEqual(self.chain.model_options, {"transformer_options": {}})
        # and it did sample: the clone carries the same LoRA entries as the incoming MODEL
        self.assertEqual(ss.entries(cap["model"]), ss.entries(self.chain))

    def test_the_prompt_is_the_face_and_the_trigger_and_never_reaches_the_report(self):
        image = self.scene([FACE_A])
        result, cap = self.run_node(image, detections=[det(FACE_A)])
        self.assertEqual(cap["clip"].texts, ["close-up photo of MyTrigger's face, natural skin texture", ""])
        text = repr(result["result"][1]["detailer"]) + result["result"][2]
        self.assertNotIn("MyTrigger", text)
        sec = result["result"][1]["detailer"]
        self.assertTrue(sec["prompt"]["trigger_used"])
        self.assertEqual(sec["prompt"]["sha256"], self.canon.sha256_hex("close-up photo of MyTrigger's face, natural skin texture"))

    def test_without_a_trigger_the_prompt_names_no_one(self):
        profile = copy.deepcopy(self.profile)
        profile["stack"][1]["trigger_hints"] = []
        result, cap = self.run_node(self.scene([FACE_A]), profile=profile, detections=[det(FACE_A)])
        self.assertEqual(cap["clip"].texts[0], "close-up photo of a person's face, natural skin texture")
        self.assertFalse(result["result"][1]["detailer"]["prompt"]["trigger_used"])

    def test_a_large_face_is_skipped_and_nothing_is_sampled(self):
        image = self.scene([FACE_BIG])
        result, cap = self.run_node(image, detections=[det(FACE_BIG)])
        out, report, _ = result["result"]
        self.assertTrue(self.torch.equal(out, image))
        self.assertEqual(cap["vae"].encodes, 0)
        self.assertEqual(report["detailer"]["faces"][0]["decision"], "skipped_large")
        self.assertIn("face_large_skipped", [w["code"] for w in report["detailer"]["warnings"]])

    def test_the_size_gate_widget_is_honoured(self):
        mid = ("A", 100, 50, 220, 180)  # 120 x 130 px: above the default gate, below 200
        image = self.scene([mid])
        result, cap = self.run_node(image, detections=[det(mid)], size_gate_px=200)
        self.assertEqual(result["result"][1]["detailer"]["faces"][0]["decision"], "accepted_unscorable")
        self.assertEqual(result["result"][1]["detailer"]["settings"]["size_gate_px"], 200.0)

    def test_references_with_consent_a_better_face_is_accepted(self):
        image = self.scene([FACE_A])
        result, cap = self.run_node(image, detections=[det(FACE_A)], score_before=0.05, score_after=0.25,
                                    reference_images=self.photos(), consent_reference_photos=True)
        out, report, _ = result["result"]
        rec = report["detailer"]["faces"][0]
        self.assertEqual(rec["decision"], "accepted")
        self.assertGreater(rec["score_after"], rec["score_before"])
        self.assertFalse(rec["unscorable"])
        self.assertEqual(report["detailer"]["references"], {
            "connected": True, "consent": True, "used": True, "n_images": 3, "n_with_face": 3, "n_characters": 1})
        self.assertFalse(self.torch.equal(out, image))

    def test_references_with_consent_a_worse_face_returns_the_original_crop(self):
        image = self.scene([FACE_A])
        result, cap = self.run_node(image, detections=[det(FACE_A)], score_before=0.25, score_after=0.05,
                                    reference_images=self.photos(), consent_reference_photos=True)
        out, report, _ = result["result"]
        rec = report["detailer"]["faces"][0]
        self.assertEqual(rec["decision"], "rejected")
        self.assertLess(rec["score_after"], rec["score_before"])
        self.assertTrue(self.torch.equal(out, image))                    # the whole frame is the input, bit for bit
        self.assertEqual(cap["vae"].encodes, 1)                          # it did try
        self.assertIn("detailer_rejected", [w["code"] for w in report["detailer"]["warnings"]])
        self.assertEqual(report["detailer"]["counts"]["rejected"], 1)

    def test_references_without_consent_are_not_read(self):
        photos = self.photos()
        result, cap = self.run_node(self.scene([FACE_A]), detections=[det(FACE_A)], reference_images=photos)
        report = result["result"][1]
        self.assertEqual(report["detailer"]["references"]["used"], False)
        self.assertEqual(report["detailer"]["references"]["n_images"], 0)
        self.assertIn("reference_photos_ignored", [w["code"] for w in report["detailer"]["warnings"]])
        self.assertEqual(report["detailer"]["faces"][0]["decision"], "accepted_unscorable")
        self.assertEqual(cap["backend"].read_calls, 2)  # the crop before and after the redraw; none of the photos

    def test_n04_without_references_the_default_is_pass_through_and_nothing_is_sampled(self):
        image = self.scene([FACE_A])
        result, cap = self.run_node(image, detections=[det(FACE_A)], accept_without_references=False)
        out, report, _ = result["result"]
        self.assertTrue(self.torch.equal(out, image))
        self.assertEqual(cap["vae"].encodes, 0)
        face = report["detailer"]["faces"][0]
        self.assertEqual((face["decision"], face["unscorable_reason"]), ("skipped_unscorable", "no_references"))
        self.assertFalse(report["detailer"]["settings"]["accept_without_references"])

    def test_n04_a_non_finite_image_is_refused(self):
        image = self.scene([FACE_A])
        image[0, 5, 5, 0] = float("nan")
        with self.assertRaises(self.errors.OutputError):
            self.run_node(image, detections=[det(FACE_A)])

    def test_the_report_carries_the_detailer_recipe(self):
        image = self.scene([FACE_A])
        result, cap = self.run_node(image, detections=[det(FACE_A)])
        sec = result["result"][1]["detailer"]
        recipe = sec["recipe"]
        self.assertEqual(recipe["input_sha256"], [self.canon.sha256_hex(image[0].numpy().tobytes())])
        self.assertEqual([x["role"] for x in recipe["lora"]], ["distill", "character"])
        self.assertEqual(recipe["lora"][0]["basename"], "distill.safetensors")
        self.assertIsNone(recipe["reference_sha256"])
        self.assertIsNone(recipe["vae_sha256"])  # the fake VAE has no weights to hash
        self.assertEqual(recipe["settings"], sec["settings"])
        self.assertRegex(recipe["recipe_hash"], "^[0-9a-f]{64}$")
        self.assertNotEqual(recipe["recipe_hash"], result["result"][1]["recipe_hash"])
        again = self.run_node(image, detections=[det(FACE_A)])[0]["result"][1]["detailer"]["recipe"]
        self.assertEqual(again["recipe_hash"], recipe["recipe_hash"])
        other_seed = self.run_node(image, detections=[det(FACE_A)], seed=12)[0]["result"][1]["detailer"]["recipe"]
        self.assertNotEqual(other_seed["recipe_hash"], recipe["recipe_hash"])
        refs = self.run_node(image, detections=[det(FACE_A)], reference_images=self.photos(),
                             consent_reference_photos=True)[0]["result"][1]["detailer"]["recipe"]
        self.assertRegex(refs["reference_sha256"], "^[0-9a-f]{64}$")

    def test_references_without_any_face_are_refused(self):
        blank = self.torch.full((2, 64, 64, 3), 0.5)
        with mock.patch.object(self.fakes.FakeBackend, "read_faces", lambda self_, rgb: []):
            with self.assertRaises(self.errors.MissingRequiredInput):
                self.run_node(self.scene([FACE_A]), detections=[det(FACE_A)], reference_images=blank,
                              consent_reference_photos=True)

    def test_a_batch_is_processed_image_by_image_with_different_seeds(self):
        image = self.scene([FACE_A], batch=2)
        result, cap = self.run_node(image, detections=[det(FACE_A)])
        out, report, _ = result["result"]
        self.assertEqual(out.shape[0], 2)
        faces = report["detailer"]["faces"]
        self.assertEqual([(f["image_index"], f["index"]) for f in faces], [(0, 0), (1, 0)])
        self.assertNotEqual(faces[0]["seed"], faces[1]["seed"])
        self.assertEqual(report["detailer"]["images"], 2)
        # identical inputs, different seeds -> different redraws
        self.assertFalse(self.torch.equal(out[0], out[1]))

    def test_the_same_inputs_give_the_same_pixels(self):
        image = self.scene([FACE_A])
        a = self.run_node(image, detections=[det(FACE_A)])[0]["result"][0]
        b = self.run_node(image, detections=[det(FACE_A)])[0]["result"][0]
        self.assertTrue(self.torch.equal(a, b))

    def test_the_incoming_report_is_returned_with_a_section_and_is_not_mutated(self):
        image = self.scene([FACE_A])
        report_in = copy.deepcopy(self.report)
        result, cap = self.run_node(image, detections=[det(FACE_A)], report=report_in)
        self.assertEqual(report_in, self.report)
        out_report = result["result"][1]
        self.assertEqual(out_report["recipe_hash"], self.report["recipe_hash"])
        self.assertIsNot(out_report, report_in)
        self.assertEqual({k for k in out_report} - {k for k in self.report}, {"detailer"})

    def test_non_float32_or_malformed_images_are_refused(self):
        with self.assertRaises(self.errors.OutputError):
            self.run_node(self.torch.zeros(1, 8, 8, 2), detections=[])

    def test_a_non_moss_model_is_refused_before_any_work(self):
        image = self.scene([FACE_A])
        fake_model = mock.MagicMock()
        with mock.patch.object(self.detailer.adapter, "probe_model") as probe:
            probe.return_value = mock.MagicMock(is_moss=False, class_name="Other")
            with self.assertRaises(self.errors.UnsupportedModel):
                self.run_node(image, model=fake_model, detections=[det(FACE_A)])


class DetailerProfileChecksCpuTest(IntegrationBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # The MODEL a node receives is held by the graph cache for longer than one run. Built inside the test it would die with
        # the test, in the same gc pass as the redraw clone that ``run_node``'s Hybrid backend keeps alive through its class
        # cycle: a dead LoadedModel that the test, not the node, caused (MP-88).
        cls.tampered = ss.apply_lora(cls.chain, ss.make_lora_dict(cls.chain, seed=5), 0.3)
        cls.tampered.set_attachments(cls.consts.ATTACH_LORA_STACK, cls.chain.get_attachment(cls.consts.ATTACH_LORA_STACK))

    def setUp(self):
        self.image = self.scene([FACE_A])

    def test_no_profile_is_missing_lora(self):
        node = self.detailer.MossPhotonIdentityDetailer()
        with mock.patch.object(self.detailer, "_probe", self.avail):
            with self.assertRaises(self.errors.MissingLora):
                node.detail(self.image, self.chain, FakeClip(self.tiny), FakeVae(self.torch),
                            copy.deepcopy(self.report), None, 1, 110, 0.25)

    def test_a_profile_of_another_chain_is_incompatible(self):
        other = ss.apply_lora(self.base, ss.make_lora_dict(self.base, seed=99), 0.5)
        foreign_profile = ss.make_profile(other, base_model=self.base, role="distill", name="other.safetensors")
        with self.assertRaises(self.errors.IncompatibleLora):
            self.run_node(self.image, profile=foreign_profile, detections=[det(FACE_A)])

    def test_a_chain_without_a_distill_adapter_is_refused(self):
        plain = ss.apply_lora(self.base, ss.make_lora_dict(self.base, seed=12), 0.8)
        profile = ss.make_profile(plain, base_model=self.base, role="character", name="char.safetensors")
        with self.assertRaises(self.errors.UnsupportedFeature) as ctx:
            self.run_node(self.image, model=plain, profile=profile, detections=[det(FACE_A)])
        self.assertIn("distill", str(ctx.exception))

    def test_foreign_patches_on_the_model_are_a_warning_in_the_section(self):
        result, _ = self.run_node(self.image, model=self.tampered, detections=[det(FACE_A)])
        codes = [w["code"] for w in result["result"][1]["detailer"]["warnings"]]
        self.assertIn("foreign_patches", codes)


if __name__ == "__main__":
    unittest.main()
