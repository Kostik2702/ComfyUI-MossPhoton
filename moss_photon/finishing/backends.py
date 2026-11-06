"""The real backend of MossPhotonIdentityDetailer: YOLO face detector, ArcFace scorer, diffusion redraw.

Every heavy import (ultralytics, insightface, torch, comfy) sits inside a function, so importing this module costs
nothing and works without any of them. The detector and ArcFace models are loaded once per process and cached.

``redraw`` is the MP-56 recipe: upscale the crop to ``work_px``, VAE-encode, sample ``steps`` turbo steps at ``denoise``
(the MODEL already carries the distill and character LoRAs; this module only ever samples on the clone it was given),
decode, downscale back. The crop is cut and pasted by ``face_detailer``; nothing here touches the original image.
"""
import numpy as np

from . import deps
from .face_detailer import FaceReading, to_u8

import hashlib
import weakref

_YOLO = {}      # model path -> ultralytics.YOLO
_ARCFACE = {}   # root -> insightface FaceAnalysis


def _yolo(path):
    model = _YOLO.get(path)
    if model is None:
        from ultralytics import YOLO
        model = YOLO(path)
        _YOLO[path] = model
    return model


def _arcface(root):
    app = _ARCFACE.get(root)
    if app is None:
        from insightface.app import FaceAnalysis
        # detection + recognition only: the landmark and genderage heads add CPU time and nothing this node uses
        app = FaceAnalysis(name=deps.ARCFACE_PACK, root=root, providers=["CPUExecutionProvider"],
                           allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _ARCFACE[root] = app
    return app


def detect_faces(img, detector_path, conf):
    """``[(x0, y0, x1, y1, conf)]`` in pixels: every face the YOLO model at ``detector_path`` finds in ``img`` (float 0..1
    ``[H, W, C]``) with a confidence of at least ``conf``. The one place that calls the detector (the detailer and the hires fix)."""
    bgr = to_u8(img)[..., ::-1].copy()
    results = _yolo(detector_path).predict(bgr, conf=float(conf), verbose=False)
    out = []
    for r in results:
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.detach().float().cpu().numpy()
        scores = r.boxes.conf.detach().float().cpu().numpy()
        out.extend((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)) for b, c in zip(xyxy, scores))
    return out


def _resize(pixels, size):
    """``[B, H, W, C]`` torch -> ``[B, size, size, C]`` with ComfyUI's Lanczos (the MP-56 resampler)."""
    import comfy.utils
    if int(pixels.shape[1]) == size and int(pixels.shape[2]) == size:
        return pixels
    return comfy.utils.common_upscale(pixels.movedim(-1, 1), size, size, "lanczos", "disabled").movedim(1, -1)


_VAE_HASHES = {}  # id(first_stage_model) -> (weakref, sha256)


def vae_sha256(vae):
    """sha256 over the VAE weights (names and raw bytes, sorted by name), cached per VAE object; ``None`` when the object
    has no ``first_stage_model`` (a stand-in) or hashing fails. Costs one pass over the weights the first time."""
    try:
        import torch
        model = vae.first_stage_model
        hit = _VAE_HASHES.get(id(model))
        if hit is not None and hit[0]() is model:
            return hit[1]
        h = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            h.update(name.encode("utf-8"))
            h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        digest = h.hexdigest()
        _VAE_HASHES[id(model)] = (weakref.ref(model), digest)
        return digest
    except Exception:  # noqa: BLE001 - a fingerprint is a label, never a gate
        return None


class ComfyBackend:
    """Detector + ArcFace + redraw for one node run. ``model`` must be a clone (the caller clones; this never mutates)."""

    def __init__(self, availability, settings, *, model, vae, positive, negative):
        self.availability = availability
        self.settings = settings
        self.model, self.vae, self.positive, self.negative = model, vae, positive, negative

    @property
    def arcface_available(self):
        return self.availability.arcface_ok

    def detect(self, img):
        return detect_faces(img, self.availability.detector_path, float(self.settings.det_conf))

    def read_faces(self, rgb_u8):
        """Every face ArcFace finds in an ``[S, S, 3]`` uint8 RGB crop, as ``FaceReading`` (box, landmarks, embedding)."""
        found = _arcface(self.availability.arcface_root).get(np.ascontiguousarray(rgb_u8[..., ::-1]))
        out = []
        for f in found:
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            emb = emb / max(float(np.linalg.norm(emb)), 1e-8)
            kps = getattr(f, "kps", None)
            out.append(FaceReading(emb, float(f.det_score), tuple(float(v) for v in f.bbox[:4]),
                                   None if kps is None else np.asarray(kps, dtype=np.float64)))
        return out

    def redraw(self, crop, seed):
        import torch
        import comfy.model_management
        import comfy.sample
        import comfy.utils
        s = self.settings
        comfy.model_management.throw_exception_if_processing_interrupted()
        side = int(crop.shape[0])
        pixels = _resize(torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32))[None], int(s.work_px))
        latent = self.vae.encode(pixels[:, :, :, :3])
        latent = comfy.sample.fix_empty_latent_channels(self.model, latent)
        noise = comfy.sample.prepare_noise(latent, int(seed))
        samples = comfy.sample.sample(
            self.model, noise, int(s.steps), float(s.cfg), s.sampler, s.scheduler, self.positive, self.negative, latent,
            denoise=float(s.denoise), seed=int(seed), disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED)
        if samples.is_nested:
            samples = samples.unbind()[0]
        decoded = self.vae.decode(samples)
        if len(decoded.shape) == 5:  # video VAEs: fold the frame axis into the batch, as the stock VAEDecode does
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        out = _resize(decoded[:1, :, :, :3].float().cpu(), side)
        return out[0].clamp(0.0, 1.0).numpy().astype(np.float32)
