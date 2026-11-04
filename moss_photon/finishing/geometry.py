"""Face selection, crop box, feather mask and masked composite of MossPhotonIdentityDetailer (SPEC 4.2, 5).

numpy only; no torch, no comfy (unit-testable on any machine). Images are float arrays ``[H, W, C]`` (ComfyUI IMAGE
0..1), boxes are ``(x0, y0, x1, y1)`` in pixels (x along the width).

The one safety property of the node lives in ``composite_region``: a pixel whose mask value is exactly 0 is COPIED from
the source, never recomputed, so the result outside the mask is bit-equal to the input (SPEC 5, "Ограниченная правка
по маске"). ``assert_outside_mask_equal`` is the independent check of that property.
"""
import numpy as np

from ..core import errors

NODE = "MossPhotonIdentityDetailer"


def short_side(bbox):
    """Short side of a ``(x0, y0, x1, y1)`` box in pixels (``min(width, height)``); 0 for a degenerate box."""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return max(0.0, min(x1 - x0, y1 - y0))


def select_faces(detections, *, max_faces, min_conf):
    """Detections ``[(x0, y0, x1, y1, conf)]`` with ``conf >= min_conf``, largest bbox area first, at most ``max_faces``.

    The sort is stable (equal areas keep the detector's order). Pure; ``max_faces`` below 1 returns ``[]``.
    """
    kept = []
    for det in detections:
        x0, y0, x1, y1, conf = [float(v) for v in det]
        if conf < float(min_conf):
            continue
        kept.append(((x0, y0, x1, y1, conf), max(0.0, x1 - x0) * max(0.0, y1 - y0)))
    kept.sort(key=lambda item: -item[1])
    return [det for det, _ in kept[:max(0, int(max_faces))]]


def crop_box(bbox, crop_factor, width, height):
    """Square crop ``(bx, by, side)`` centred on the face: ``side = round(crop_factor * max(w, h))``.

    The square is capped at the shorter image side and SHIFTED (never shrunk) to stay inside the image, so the face is
    not centred when it sits at the border. ``side`` is at least 1.
    """
    x0, y0, x1, y1 = [float(v) for v in bbox]
    side = int(round(float(crop_factor) * max(x1 - x0, y1 - y0)))
    side = max(1, min(side, int(width), int(height)))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bx = int(round(min(max(cx - side / 2.0, 0.0), width - side)))
    by = int(round(min(max(cy - side / 2.0, 0.0), height - side)))
    return bx, by, side


def feather_mask(side, feather):
    """``[side, side]`` float32 mask: the inscribed ellipse (a circle) with a smoothstep ramp.

    The ramp runs INWARD from the ellipse edge over ``feather * side`` pixels (``smoothstep(t) = 3t^2 - 2t^3``); the
    centre is 1.0. It is exactly 0.0 on and beyond the ellipse edge, including all four corners of the square.
    """
    side = int(side)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float64)
    c = (side - 1) / 2.0
    a = side / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / a
    width_px = max(float(feather) * side, 1e-6)
    t = np.clip((1.0 - r) * a / width_px, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def composite_region(src_region, new_region, mask):
    """``src * (1 - m) + new * m`` where ``m > 0``; every pixel with ``m == 0`` is copied unchanged from ``src_region``.

    ``src_region`` / ``new_region`` are ``[S, S, C]`` float32, ``mask`` is ``[S, S]`` float32 in 0..1. Returns a new
    array; neither input is modified. Channels beyond the first three (alpha) are copied from the source.
    """
    out = src_region.copy()
    sel = mask > 0.0
    m = mask[sel][:, None]
    blended = src_region[sel][:, :3] * (1.0 - m) + new_region[sel][:, :3] * m
    out[sel, :3] = blended.astype(out.dtype, copy=False)
    return out


def assert_outside_mask_equal(src, out, full_mask):
    """Raise ``output_error`` unless ``out`` equals ``src`` on every pixel where ``full_mask == 0``, bit for bit."""
    outside = full_mask <= 0.0
    if src.shape != out.shape or not np.array_equal(src[outside], out[outside]):
        bad = int((src[outside] != out[outside]).any(axis=-1).sum()) if src.shape == out.shape else -1
        raise errors.OutputError(errors.make_issue(
            "output_error", NODE, "image",
            "pixels outside the face mask changed (%d pixel(s)); the result is discarded" % bad,
            "report this as a bug of the composite step", detail={"changed_pixels": bad}))


# --------------------------------------------------------------------------- landmarks and the face-shaped masks
def yaw_proxy(kps):
    """Roll-invariant lateral offset of the nose, from the five insightface landmarks (eye, eye, nose, mouth, mouth).

    The vector from the midpoint of the eyes to the nose is projected onto the NORMALISED inter-eye axis and divided by
    the Euclidean inter-eye distance, so turning the head or the frame in the image plane (roll) changes nothing:
    0 is frontal and the value grows as the head turns away. It is a proxy for yaw, not a measured angle. Returns
    ``None`` when the landmarks are missing or the eyes coincide.
    """
    if kps is None:
        return None
    k = np.asarray(kps, dtype=np.float64)
    if k.shape[0] < 3 or k.shape[1] != 2:
        return None
    axis = k[1] - k[0]
    ioc = float(np.linalg.norm(axis))
    if ioc < 1e-6:
        return None
    mid = (k[0] + k[1]) / 2.0
    return abs(float(np.dot(k[2] - mid, axis / ioc))) / ioc


def bbox_iou(a, b):
    """Intersection over union of two ``(x0, y0, x1, y1)`` boxes; 0 when they do not overlap."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def kps_shift(a, b):
    """Largest landmark displacement between two readings of the SAME face, in inter-eye distances of ``a``.

    ``inf`` when either landmark set is missing or the eyes of ``a`` coincide."""
    if a is None or b is None:
        return float("inf")
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ioc = float(np.linalg.norm(a[1] - a[0]))
    if a.shape != b.shape or ioc < 1e-6:
        return float("inf")
    return float(np.linalg.norm(a - b, axis=1).max() / ioc)


def ellipse_mask(side, face_box, dilate, feather):
    """``[side, side]`` float32 paste mask shaped like the FACE, not like the crop.

    The ellipse is inscribed in ``face_box`` (``(x0, y0, x1, y1)`` in crop pixels, the detector's face box) grown by
    ``dilate`` around its centre; the smoothstep ramp runs INWARD from the ellipse edge over ``feather * side`` pixels.
    Exactly 0.0 on and beyond the ellipse edge, whatever the box (the box may stick out of the crop: the mask is clipped
    by the array)."""
    side = int(side)
    x0, y0, x1, y1 = [float(v) for v in face_box]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = max((x1 - x0) * float(dilate) / 2.0, 1.0), max((y1 - y0) * float(dilate) / 2.0, 1.0)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float64)
    r = np.sqrt(((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2)
    width_px = max(float(feather) * side, 1e-6)
    t = np.clip((1.0 - r) * min(rx, ry) / width_px, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def skin_mask(side, face_box, kps, inner=0.85):
    """``[side, side]`` float32 0/1 region of FACE SKIN used for the colour statistics.

    The ellipse inscribed in the face box shrunk to ``inner`` of its size (so hair, ears and background at the rim are
    out), minus discs around both eyes and a capsule over the mouth when landmarks (crop pixels) are given: eyes and lips
    are not skin and have their own colours. Never empty: if the exclusions would remove everything the plain ellipse is
    returned."""
    side = int(side)
    x0, y0, x1, y1 = [float(v) for v in face_box]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = max((x1 - x0) * inner / 2.0, 1.0), max((y1 - y0) * inner / 2.0, 1.0)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float64)
    px, py = xx + 0.5, yy + 0.5
    region = (((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2) < 1.0
    keep = region.copy()
    if kps is not None and np.asarray(kps).shape[0] >= 5:
        k = np.asarray(kps, dtype=np.float64)
        ioc = float(np.linalg.norm(k[1] - k[0]))
        if ioc > 1e-6:
            for eye in (k[0], k[1]):
                keep &= np.hypot(px - eye[0], py - eye[1]) > 0.32 * ioc
            a, b = k[3], k[4]
            ab = b - a
            denom = max(float(ab @ ab), 1e-9)
            t = np.clip(((px - a[0]) * ab[0] + (py - a[1]) * ab[1]) / denom, 0.0, 1.0)
            keep &= np.hypot(px - (a[0] + t * ab[0]), py - (a[1] + t * ab[1])) > 0.25 * ioc
    out = keep if keep.any() else region
    return out.astype(np.float32)
