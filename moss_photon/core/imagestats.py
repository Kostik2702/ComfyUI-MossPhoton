"""Image and mask measurements for ``MossPhotonDiagnostics`` (design 4.7).

Imports: ``errors`` only at module level; numpy is imported inside the functions (R1, 7.3). Inputs are torch
tensors or numpy arrays and are only read. Everything returned is a JSON-safe dict. These are MEASUREMENTS: there is
no pass/fail and no threshold here (SPEC 7: no universal threshold in code).
"""
from . import errors

__all__ = ["HIST_BINS", "image_stats", "mask_coverage", "batch_image_stats", "batch_mask_coverage"]

NODE_ID = "MossPhotonDiagnostics"
HIST_BINS = 1024
_LUMA = (0.2126, 0.7152, 0.0722)  # Rec.709 weights on the stored (non-linearised) values
_BLACK = 1.0 / 255.0
_WHITE = 254.0 / 255.0


def _bad(param, message):
    return errors.OutputError(errors.make_issue(
        "output_error", NODE_ID, param, message, "connect an IMAGE [B,H,W,C] or a MASK [B,H,W]"))


def _as_array(x, param):
    import numpy as np
    arr = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    if arr.dtype.kind not in "fiub":
        raise _bad(param, "expected a numeric array, got dtype %s" % arr.dtype)
    return arr


def _percentile(hist, total, q):
    """Centre of the first bin whose cumulative count reaches ``q * total`` (``q`` in 0..1)."""
    import numpy as np
    cum = np.cumsum(hist)
    idx = int(np.searchsorted(cum, q * total, side="left"))
    idx = min(idx, HIST_BINS - 1)
    return (idx + 0.5) / HIST_BINS


def image_stats(image):
    """Measurements of ONE image ``[H, W, C]`` (float, 0..1 nominal) as a dict.

    Keys: ``width height channels mean_luma std_luma p01_luma p99_luma clipped_black_frac clipped_white_frac
    laplacian_var nonfinite_count``.

    * ``luma = 0.2126 R + 0.7152 G + 0.0722 B`` on the stored values (no linearisation); one channel = that
      channel, two channels = the first (ASSUMED); channels beyond the third (alpha) are ignored.
    * ``std_luma`` is the population standard deviation. Percentiles come from a 1024-bin histogram of luma
      clamped to 0..1 and are the centre of the bin that holds the quantile (resolution 1/1024).
    * ``clipped_black_frac`` = share of pixels with ``luma <= 1/255``, ``clipped_white_frac`` with
      ``luma >= 254/255``.
    * ``laplacian_var`` = population variance of the 4-neighbour Laplacian of ``255 * luma`` over interior pixels
      (None when a side is shorter than 3).
    * ``nonfinite_count`` counts NaN/Inf elements over all channels. Pixels whose luma is not finite are left out
      of every statistic above; if none is finite the statistics are None.
    """
    import numpy as np
    arr = _as_array(image, "image")
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[1] < 1 or arr.shape[2] < 1:
        raise _bad("image", "expected one image [H,W,C], got shape %s" % (tuple(arr.shape),))
    height, width, channels = (int(n) for n in arr.shape)
    arr = arr.astype(np.float64, copy=False)
    nonfinite = int(arr.size - np.count_nonzero(np.isfinite(arr)))
    if channels >= 3:
        luma = _LUMA[0] * arr[:, :, 0] + _LUMA[1] * arr[:, :, 1] + _LUMA[2] * arr[:, :, 2]
    else:
        luma = arr[:, :, 0]
    finite = np.isfinite(luma)
    values = luma[finite]
    out = {"width": width, "height": height, "channels": channels, "mean_luma": None, "std_luma": None,
           "p01_luma": None, "p99_luma": None, "clipped_black_frac": None, "clipped_white_frac": None,
           "laplacian_var": None, "nonfinite_count": nonfinite}
    if values.size == 0:
        return out
    out["mean_luma"] = float(values.mean())
    out["std_luma"] = float(values.std())
    bins = np.clip((np.clip(values, 0.0, 1.0) * HIST_BINS).astype(np.int64), 0, HIST_BINS - 1)
    hist = np.bincount(bins, minlength=HIST_BINS)
    out["p01_luma"] = _percentile(hist, values.size, 0.01)
    out["p99_luma"] = _percentile(hist, values.size, 0.99)
    out["clipped_black_frac"] = float(np.count_nonzero(values <= _BLACK)) / values.size
    out["clipped_white_frac"] = float(np.count_nonzero(values >= _WHITE)) / values.size
    if height >= 3 and width >= 3:
        y = 255.0 * luma
        lap = y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:] - 4.0 * y[1:-1, 1:-1]
        lap = lap[np.isfinite(lap)]
        if lap.size:
            out["laplacian_var"] = float(lap.var())
    return out


def mask_coverage(mask):
    """Measurements of ONE mask ``[H, W]`` as a dict: ``width height mean coverage_frac nonfinite_count``.

    ``coverage_frac`` = share of pixels with value > 0.5 (ASSUMED threshold: a MASK is 0..1, 0.5 splits it);
    ``mean`` is the mean over finite pixels; both are None when no pixel is finite.
    """
    import numpy as np
    arr = _as_array(mask, "masks")
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise _bad("masks", "expected one mask [H,W], got shape %s" % (tuple(arr.shape),))
    arr = arr.astype(np.float64, copy=False)
    finite = np.isfinite(arr)
    values = arr[finite]
    out = {"width": int(arr.shape[1]), "height": int(arr.shape[0]), "mean": None, "coverage_frac": None,
           "nonfinite_count": int(arr.size - values.size)}
    if values.size:
        out["mean"] = float(values.mean())
        out["coverage_frac"] = float(np.count_nonzero(values > 0.5)) / values.size
    return out


def batch_image_stats(images):
    """``image_stats`` for every item of a batch ``[B, H, W, C]``; a single ``[H, W, C]`` counts as one item."""
    arr = _as_array(images, "image")
    if arr.ndim == 3:
        return [image_stats(arr)]
    if arr.ndim != 4:
        raise _bad("image", "expected IMAGE [B,H,W,C], got shape %s" % (tuple(arr.shape),))
    return [image_stats(arr[i]) for i in range(arr.shape[0])]


def batch_mask_coverage(masks):
    """``mask_coverage`` for every item of a batch ``[B, H, W]``; a single ``[H, W]`` counts as one item."""
    arr = _as_array(masks, "masks")
    if arr.ndim == 2:
        return [mask_coverage(arr)]
    if arr.ndim != 3:
        raise _bad("masks", "expected MASK [B,H,W], got shape %s" % (tuple(arr.shape),))
    return [mask_coverage(arr[i]) for i in range(arr.shape[0])]
