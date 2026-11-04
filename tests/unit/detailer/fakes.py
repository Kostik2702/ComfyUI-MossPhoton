"""Synthetic scenes and a fake backend for the detailer tests (MP-58): no model, no detector, no ArcFace.

A scene is a grey float32 image with textured rectangles ("faces"), each with its own flat tone. The fake ArcFace reads
a crop like this: the person is the face tone nearest to the crop centre, and the quality of the face is its local
contrast at the centre. So a redraw that raises the contrast raises the score, one that flattens lowers it, and two people
are two directions the cosine tells apart.
"""
import numpy as np

from moss_photon.finishing import face_detailer as fd

# tone of the flat part of each synthetic face, by person
TONES = {"A": 0.78, "B": 0.22}
# direction of each person in the fake embedding space [A, B, quality]
QUALITY_GAIN = 3.0
SHARP_STD = 0.25


def make_scene(height, width, faces, seed=7):
    """float32 ``[H, W, 3]``: noisy grey background and one textured rectangle per ``(person, x0, y0, x1, y1)``."""
    rng = np.random.default_rng(seed)
    img = (0.5 + 0.02 * rng.standard_normal((height, width, 3))).astype(np.float32)
    for person, x0, y0, x1, y1 in faces:
        tone = TONES[person]
        patch = tone + 0.04 * rng.standard_normal((y1 - y0, x1 - x0, 3))
        img[y0:y1, x0:x1] = patch.astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def center_patch(rgb, half=8):
    h, w = rgb.shape[:2]
    return rgb[h // 2 - half:h // 2 + half, w // 2 - half:w // 2 + half].astype(np.float32)


def embed(person, std):
    v = np.array([1.0 if person == "A" else 0.0, 1.0 if person == "B" else 0.0, QUALITY_GAIN * float(std)])
    return (v / np.linalg.norm(v)).astype(np.float32)


def reference(person, n=3, std=SHARP_STD):
    """``[n, 3]`` reference matrix of a person at SHARP quality (identical rows are fine for the mean cosine)."""
    return np.stack([embed(person, std)] * n)


def fake_kps(face_box, offset=0.0, roll_deg=0.0, dx=0.0):
    """Five landmarks of a frontal face in ``face_box`` (eye, eye, nose, mouth, mouth). ``offset`` moves the nose sideways
    in inter-eye distances (0 = frontal), ``roll_deg`` turns the whole set about the box centre, ``dx`` shifts it."""
    x0, y0, x1, y1 = face_box
    cx, cy, w, h = (x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0
    ioc = 0.4 * w
    pts = np.array([[cx - ioc / 2, cy - 0.1 * h], [cx + ioc / 2, cy - 0.1 * h], [cx + offset * ioc, cy + 0.1 * h],
                    [cx - 0.15 * w, cy + 0.3 * h], [cx + 0.15 * w, cy + 0.3 * h]])
    t = np.deg2rad(roll_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (pts - [cx, cy]) @ rot.T + [cx, cy] + [dx, 0.0]


class FakeBackend:
    """Duck-typed backend (see ``face_detailer``). Counts calls; ``redraw_fn(crop) -> crop`` decides the redraw.

    ``read_faces`` reports ONE face for the crop: a centred square of ``side / 1.8`` (where the pipeline's crop puts the
    detector's face), the person and the quality read from the centre patch. ``before`` / ``after`` are dicts that shape
    the reading before the first redraw and after it: ``yaw`` (nose offset), ``roll`` (degrees), ``det_score``,
    ``missing`` (no face at all), ``kps_dx`` (landmarks shifted sideways, px), ``extra_face`` (a second, LARGER face with
    a perfect match to person A at sharp quality, away from the real one), ``no_kps``."""

    def __init__(self, detections, redraw_fn=None, arcface_available=True, yaw=0.0, det_score=0.9, no_face_for=(),
                 before=None, after=None):
        self.detections = list(detections)
        self.redraw_fn = redraw_fn or (lambda crop: crop)
        self.arcface_available = arcface_available
        self.before = dict({"yaw": yaw, "det_score": det_score}, **(before or {}))
        self.after = dict(self.before, **(after or {}))
        self.no_face_for = set(no_face_for)
        self.redraw_calls, self.read_calls, self.redraw_seeds = 0, 0, []

    def detect(self, img):
        return list(self.detections)

    def read_faces(self, rgb_u8):
        self.read_calls += 1
        cfg = self.before if self.redraw_calls == 0 else self.after
        if cfg.get("missing"):
            return []
        rgb = rgb_u8.astype(np.float32) / 255.0
        patch = center_patch(rgb)
        tone = float(patch.mean())
        person = min(TONES, key=lambda p: abs(TONES[p] - tone))
        if person in self.no_face_for:
            return []
        side = rgb_u8.shape[0]
        half = side / 1.8 / 2.0
        c = side / 2.0
        box = (c - half, c - half, c + half, c + half)
        kps = None if cfg.get("no_kps") else fake_kps(box, cfg.get("yaw", 0.0), cfg.get("roll", 0.0), cfg.get("kps_dx", 0.0))
        out = [fd.FaceReading(embed(person, patch.std()), cfg.get("det_score", 0.9), box, kps)]
        if cfg.get("extra_face"):
            big = (0.0, 0.0, side * 0.62, side * 0.62)  # larger than the real face (0.56 side), IoU with it ~0.29
            out.append(fd.FaceReading(embed("A", SHARP_STD), 0.99, big, fake_kps(big)))
        return out

    def redraw(self, crop, seed):
        self.redraw_calls += 1
        self.redraw_seeds.append(seed)
        return self.redraw_fn(crop.copy())


def box_blur(crop, radius=2):
    """Mean of the (2r+1)^2 neighbourhood, edge-replicated; keeps the flat tone of a face and removes its texture."""
    pad = np.pad(crop, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    acc = np.zeros_like(crop, dtype=np.float64)
    n = 2 * radius + 1
    for dy in range(n):
        for dx in range(n):
            acc += pad[dy:dy + crop.shape[0], dx:dx + crop.shape[1]]
    return (acc / (n * n)).astype(np.float32)


def sharpen(crop, gain=3.0):
    """More texture contrast at the same tone: the fake ArcFace score of a face goes up."""
    return np.clip(crop + gain * (crop - box_blur(crop)), 0.0, 1.0).astype(np.float32)


def flatten(crop, keep=0.1):
    """Less texture contrast at the same tone: the score goes down."""
    blur = box_blur(crop)
    return (blur + keep * (crop - blur)).astype(np.float32)


def full_mask_of(shape_hw, record, settings=None):
    """The BASE paste mask a processed face used (before the seam factor, which can only lower it), placed in the frame."""
    from moss_photon.finishing import geometry
    settings = settings or fd.Settings()
    full = np.zeros(shape_hw, dtype=np.float32)
    c = record["crop"]
    x0, y0, x1, y1 = record["bbox"]
    face_box = (x0 - c["x"], y0 - c["y"], x1 - c["x"], y1 - c["y"])
    full[c["y"]:c["y"] + c["side"], c["x"]:c["x"] + c["side"]] = geometry.ellipse_mask(
        c["side"], face_box, settings.mask_dilate, settings.feather)
    return full
