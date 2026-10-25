"""MossPhotonEmptyLatent (design 4.1): an empty MOSS latent that refuses sizes the model cannot represent."""
import logging

from ..core import adapter, consts, errors

_log = logging.getLogger("MossPhoton")

NODE_ID = "MossPhotonEmptyLatent"
MP_MIN, MP_MAX = 0.25, 4.0  # documented megapixel range (USERGUIDE, INV 5.3)


def _size_input(tooltip):
    return ("INT", {"default": 1024, "min": adapter.MIN_SIDE, "max": adapter.MAX_SIDE,
                    "step": adapter.SPATIAL_RATIO, "tooltip": tooltip})


class MossPhotonEmptyLatent:
    """Empty ``[B, 128, H/16, W/16]`` latent. Width and height must be multiples of 16: the node refuses instead
    of flooring (the native EmptyFlux2LatentImage floors silently)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": _size_input("Image width in pixels; must be a multiple of 16."),
            "height": _size_input("Image height in pixels; must be a multiple of 16."),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": adapter.MAX_BATCH,
                                   "tooltip": "Number of images (latents) in the batch."}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_TOOLTIPS = ("Empty 128-channel latent at 1/16 of the pixel size, float32.",)
    FUNCTION = "generate"
    CATEGORY = consts.CAT_SETUP
    DESCRIPTION = ("Creates an empty latent for MOSS-Image (128 channels, 16x downscale). Sizes that are not "
                   "multiples of 16 are refused, never rounded.")

    def generate(self, width, height, batch_size):
        """Raises ``invalid_dimensions`` for a side that is not a multiple of 16 or is out of range."""
        errors.raise_first_error(adapter.validate_dimensions(width, height, batch_size, node=NODE_ID))
        channels, ratio = adapter.latent_format_constants()
        latent = adapter.make_empty_latent(width, height, batch_size)
        h, w = height // ratio, width // ratio
        megapixels = width * height / 1e6
        mib = batch_size * channels * h * w * 4 / (1024 * 1024)
        text = "%dx%d -> latent [%d,%d,%d,%d], %d image tokens, %.1f MiB latent, %.2f MP" % (
            width, height, batch_size, channels, h, w, h * w, mib, megapixels)
        if not MP_MIN <= megapixels <= MP_MAX:
            issue = errors.make_issue(
                "megapixels_out_of_documented_range", NODE_ID, "width/height",
                "%.2f MP is outside the documented %g..%g MP range" % (megapixels, MP_MIN, MP_MAX),
                "stay within %g..%g MP for the verified quality" % (MP_MIN, MP_MAX))
            _log.warning(issue.format())
            text += "\n" + issue.format()
        return {"ui": {"text": [text]}, "result": (latent,)}


NODE_CLASS_MAPPINGS = {NODE_ID: MossPhotonEmptyLatent}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "Moss Photon Empty Latent"}
