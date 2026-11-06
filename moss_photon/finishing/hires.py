"""The pure side of MossPhotonHiresFix (MP-91): the release map of the face, the late-entry mask formula and the judgement of
``face_sigma`` against the schedule. numpy only; no torch, no comfy (unit-testable on any machine).

The idea. The second pass of a hires fix redraws the whole frame from a noisy start; the face is redrawn with it and its features
drift away from the source. Here the face is held at the source (re-noised to the sigma of the step) while the rest of the frame
is redrawn, and enters the redraw only when the schedule reaches ``face_sigma``: its coarse structure, which is set at the high
sigmas, then comes from the source; only the fine detail is drawn.

* ``release_map`` is ``v``: 1.0 outside the faces, 0.0 in the core of a face, a smoothstep ramp on an ellipse between. It is the
  ``noise_mask`` the sampler is given, so ComfyUI resizes it to the latent and moves it to the device itself.
* ``mask_value`` turns ``v`` into the mask the sampler blends with at one sigma (``denoise_mask_function``). A pixel with
  ``v == 1`` gets exactly 1.0 at every sigma of the run, so the pixels outside the faces are not touched by the blending at all
  (``KSamplerX0Inpaint``: ``x * mask + source * (1 - mask)``); a pixel with ``v == 0`` is 0 (held) while the sigma is above
  ``face_sigma + BAND`` and 1 (free) from ``face_sigma`` down.
"""
import numpy as np

# Width of the sigma ramp in which a held pixel is handed over to the redraw. 0.05 of a schedule that runs 0.86 -> 0.2 is about
# one step wide: the hand-over is not a cut, and not a slow cross-fade either.
BAND = 0.05
# The grown ellipse around a face box (x 1.3 of the ellipse inscribed in the box) and the share of it that is a ramp.
MARGIN = 1.3
FEATHER = 0.35

# judge_face_sigma
INSIDE, ABOVE, BELOW, NO_WINDOW = "inside", "above", "below", "none"


def boxes_to_fractions(boxes, width, height):
    """``[(x0, y0, x1, y1, ...)]`` in pixels of a ``width`` x ``height`` image -> ``[(x0, y0, x1, y1)]`` as fractions of it,
    clipped to 0..1. A box that is empty after the clip (or an inverted one), and any box of an image with no area, is dropped.
    The fractions are what makes the release map independent of the latent size."""
    if width <= 0 or height <= 0:
        return []
    out = []
    for box in boxes:
        x0, y0, x1, y1 = [float(v) for v in box[:4]]
        x0, x1 = min(max(x0 / width, 0.0), 1.0), min(max(x1 / width, 0.0), 1.0)
        y0, y1 = min(max(y0 / height, 0.0), 1.0), min(max(y1 / height, 0.0), 1.0)
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1, y1))
    return out


def release_map(height, width, boxes, margin=MARGIN, feather=FEATHER):
    """``[height, width]`` float32: 1.0 outside the faces, 0.0 in the core of every face, a smoothstep ramp between.

    ``boxes`` are ``(x0, y0, x1, y1)`` FRACTIONS of the image. Per box the ellipse inscribed in it is grown by ``margin``; with
    ``r`` the elliptical radius (1.0 on the grown edge) the map is ``smoothstep((r - (1 - feather)) / feather)``, so the inner
    ``1 - feather`` of the grown ellipse is exactly 0.0 and the edge and everything beyond it exactly 1.0. Several faces combine
    with ``min``. Pixel centres are used, so the map has the same shape at any resolution. An empty or inverted box is ignored.
    """
    out = np.ones((int(height), int(width)), dtype=np.float64)
    margin, feather = max(float(margin), 1e-6), min(max(float(feather), 1e-6), 1.0)
    ys = ((np.arange(int(height), dtype=np.float64) + 0.5) / max(int(height), 1))[:, None]
    xs = ((np.arange(int(width), dtype=np.float64) + 0.5) / max(int(width), 1))[None, :]
    for x0, y0, x1, y1 in boxes:
        rx, ry = (x1 - x0) / 2.0 * margin, (y1 - y0) / 2.0 * margin
        if rx <= 0.0 or ry <= 0.0:
            continue
        r = np.sqrt(((xs - (x0 + x1) / 2.0) / rx) ** 2 + ((ys - (y0 + y1) / 2.0) / ry) ** 2)
        t = np.clip((r - (1.0 - feather)) / feather, 0.0, 1.0)
        out = np.minimum(out, t * t * (3.0 - 2.0 * t))
    return out.astype(np.float32)


def _clamp01(x):
    """``x`` limited to 0..1; a torch tensor (``clamp``), a numpy array (``clip``) or a plain number."""
    if hasattr(x, "clamp"):
        return x.clamp(0.0, 1.0)
    if hasattr(x, "clip"):
        return x.clip(0.0, 1.0)
    return min(max(x, 0.0), 1.0)


def mask_value(v, sigma, s0, face_sigma, band=BAND):
    """The mask the sampler blends with at ``sigma``, from the release value ``v`` (a number, a numpy array or a torch tensor).

    ``release = face_sigma + (s0 - face_sigma) * v`` is the sigma at which a pixel is handed over to the redraw, and
    ``mask = clamp((release - sigma) / band + 1, 0, 1)``: 0 while ``sigma >= release + band`` (held at the source), 1 from
    ``sigma <= release`` down, linear in between. ``s0`` is the first sigma of the run. The release is written as the weighted
    mean ``(1 - v) * face_sigma + v * s0``: the same number, but exact at both ends, so ``v == 1`` gives ``release == s0`` to the
    last bit and the mask is exactly 1.0 at every ``sigma <= s0``. ``sigma`` must broadcast with ``v``.
    """
    release = (1.0 - v) * face_sigma + v * s0
    return _clamp01((release - sigma) / band + 1.0)


def sigma_window(sigmas):
    """``(first, last)``: the first and the last NON-ZERO sigma of a schedule (the sigmas a model is called at), or
    ``(None, None)`` when there is none."""
    live = [float(s) for s in sigmas if float(s) > 0.0]
    return (live[0], live[-1]) if live else (None, None)


def judge_face_sigma(face_sigma, sigmas):
    """How ``face_sigma`` sits in the schedule: ``ABOVE`` (at or above the first sigma: every pixel is free from the start, the
    protection does nothing), ``BELOW`` (at or below the last non-zero sigma: the face is held to the end and stays as in the
    source), ``INSIDE``, or ``NO_WINDOW`` for a schedule without a non-zero sigma."""
    first, last = sigma_window(sigmas)
    if first is None:
        return NO_WINDOW
    if face_sigma >= first:
        return ABOVE
    if face_sigma <= last:
        return BELOW
    return INSIDE
