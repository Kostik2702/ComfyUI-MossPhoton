"""LAB mean/std colour match on the SKIN of the face (MP-56: the redraw drifted warm; SPEC 7.4 colour check; N05).

numpy only. sRGB (D65, 0..1) -> CIE L*a*b* -> per-channel mean/std transfer -> sRGB. The statistics are taken over a SKIN
mask only (face ellipse without eyes, lips, hair and background; ``geometry.skin_mask``), so a dark hairline or a bright
background does not leak into the face colour. The transfer is a gain and an offset per channel:
``lab' = (lab - mean_new) * k + mean_ref`` with ``k = clip(std_ref / std_new, GAIN_MIN, GAIN_MAX)``. The gain range is
narrow on purpose: the match is there to remove a colour DRIFT (MP-56: warm), not to rewrite the skin contrast. Every
clipped gain is reported. ``seam_factor`` lowers the paste weight on the feather ring where the redraw still disagrees
with its surroundings.
"""
import numpy as np

GAIN_MIN = 0.85
GAIN_MAX = 1.15
# Seam guard: CIE76 distance from which the paste weight on the feather ring starts to fall, and where it reaches 0.
SEAM_DE_LO = 8.0
SEAM_DE_HI = 30.0

_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = _M_RGB2XYZ @ np.ones(3)  # D65 white of the matrix above
_DELTA = 6.0 / 29.0


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _f(t):
    return np.where(t > _DELTA ** 3, np.cbrt(t), t / (3.0 * _DELTA ** 2) + 4.0 / 29.0)


def _f_inv(t):
    return np.where(t > _DELTA, t ** 3, 3.0 * _DELTA ** 2 * (t - 4.0 / 29.0))


def rgb_to_lab(rgb):
    """``[..., 3]`` sRGB in 0..1 -> L* (0..100), a*, b* as float64."""
    xyz = _srgb_to_linear(np.asarray(rgb, dtype=np.float64)) @ _M_RGB2XYZ.T
    fx, fy, fz = [_f(xyz[..., i] / _WHITE[i]) for i in range(3)]
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def lab_to_rgb(lab):
    """Inverse of ``rgb_to_lab``; the result is clipped to 0..1."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    xyz = np.stack([_f_inv(fx) * _WHITE[0], _f_inv(fy) * _WHITE[1], _f_inv(fz) * _WHITE[2]], axis=-1)
    return np.clip(_linear_to_srgb(xyz @ _M_XYZ2RGB.T), 0.0, 1.0)


def _weighted_stats(lab, weights):
    w = weights.astype(np.float64)
    total = float(w.sum())
    mean = (lab * w[..., None]).sum(axis=(0, 1)) / total
    var = (((lab - mean) ** 2) * w[..., None]).sum(axis=(0, 1)) / total
    return mean, np.sqrt(var)


def delta_e_of_means(a, b):
    """CIE76 distance between two L*a*b* mean vectors."""
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).sum()))


def lab_match(new_rgb, ref_rgb, stats_mask):
    """Move the LAB mean/std of ``new_rgb`` to those of ``ref_rgb``, both measured over ``stats_mask`` only.

    ``new_rgb`` and ``ref_rgb`` are ``[S, S, 3]`` float in 0..1 (the redraw and the original crop), ``stats_mask`` is
    ``[S, S]`` (the skin region, weights 0..1). The transfer is applied to every pixel of the crop (the paste mask
    weights it afterwards). Returns ``(matched float32 [S, S, 3], stats)``; ``stats`` holds the skin pixel count, the mean
    shift, the gains used, which gains were clipped to ``[GAIN_MIN, GAIN_MAX]``, and the distance of the two LAB means
    before and after (``delta_e_before`` / ``delta_e_after``, CIE76 of the MEANS over the skin; after the transfer it is
    small by construction, so it is a report of what was corrected, not an independent check), all JSON-safe.
    """
    weights = np.asarray(stats_mask, dtype=np.float64)
    if float(weights.sum()) <= 0.0:
        raise ValueError("lab_match needs a skin mask with positive weight")
    lab_new = rgb_to_lab(new_rgb)
    lab_ref = rgb_to_lab(ref_rgb)
    mean_new, std_new = _weighted_stats(lab_new, weights)
    mean_ref, std_ref = _weighted_stats(lab_ref, weights)
    raw_gain = std_ref / np.maximum(std_new, 1e-6)
    gain = np.clip(raw_gain, GAIN_MIN, GAIN_MAX)
    matched = lab_to_rgb((lab_new - mean_new) * gain + mean_ref)
    mean_after, _ = _weighted_stats(rgb_to_lab(matched), weights)
    stats = {
        "skin_px": int((weights > 0.5).sum()),
        "delta_e_before": round(delta_e_of_means(mean_new, mean_ref), 4),
        "delta_e_after": round(delta_e_of_means(mean_after, mean_ref), 4),
        "mean_shift_lab": [round(float(v), 4) for v in (mean_ref - mean_new)],
        "std_gain": [round(float(v), 4) for v in gain],
        "gain_clipped": [bool(c) for c in (raw_gain != gain)],
    }
    return matched.astype(np.float32), stats


def _box_blur(a, radius):
    pad = np.pad(a, radius, mode="edge")
    n = 2 * radius + 1
    out = np.zeros_like(a, dtype=np.float64)
    for dy in range(n):
        for dx in range(n):
            out += pad[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out / (n * n)


def seam_factor(src_rgb, new_rgb, mask):
    """``[S, S]`` factor in 0..1 to multiply into the paste mask: 1 inside the core and outside the ring, and on the
    feather ring (``0 < mask < 1``) it falls from 1 to 0 as the CIE76 difference between the original and the redraw
    goes from ``SEAM_DE_LO`` to ``SEAM_DE_HI``, smoothed over 2 px. Where the redraw still disagrees strongly with what is
    around the face, the original is kept: an edge-aware feather. It can only LOWER the mask, so it never widens the area
    that changes."""
    ring = (mask > 0.0) & (mask < 1.0)
    if not ring.any():
        return np.ones(mask.shape, dtype=np.float32)
    de = np.sqrt(((rgb_to_lab(src_rgb) - rgb_to_lab(new_rgb)) ** 2).sum(axis=-1))
    factor = np.clip(1.0 - (de - SEAM_DE_LO) / (SEAM_DE_HI - SEAM_DE_LO), 0.0, 1.0)
    factor = np.minimum(_box_blur(factor, 2), 1.0)
    out = np.where(ring, factor, 1.0)
    return out.astype(np.float32)
