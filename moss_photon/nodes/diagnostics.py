"""MossPhotonDiagnostics (design 4.7): measurements and warnings added to a PHOTON_REPORT, nothing else.

The node has no IMAGE output, so it cannot change an image; the tensors it receives are only read.
"""
import logging

from ..core import consts, errors, imagestats
from ..core import report as report_mod

_log = logging.getLogger("MossPhoton")

NODE_ID = "MossPhotonDiagnostics"


def _require_valid_report(rep):
    """Raise ``unsupported_feature`` for a newer schema, ``output_error`` for an invalid report."""
    if isinstance(rep, dict):
        report_mod.redact(rep)  # version gate (raises unsupported_feature); the copy is dropped
    problems = report_mod.validate_report(rep)
    if problems:
        raise errors.OutputError(errors.make_issue(
            "output_error", NODE_ID, "report", "the report is not a valid PHOTON_REPORT: " + "; ".join(problems[:5]),
            "connect the report output of a MossPhoton sampler"))


def _batch_size(tensor):
    shape = getattr(tensor, "shape", None)
    return int(shape[0]) if shape is not None and len(shape) else 0


def _line_image(i, s):
    def f(v):
        return "-" if v is None else "%.4g" % v
    return ("image[%d]: %dx%d, luma mean %s std %s p01 %s p99 %s, clipped black %s white %s, laplacian_var %s, "
            "nonfinite %d" % (i, s["width"], s["height"], f(s["mean_luma"]), f(s["std_luma"]), f(s["p01_luma"]),
                              f(s["p99_luma"]), f(s["clipped_black_frac"]), f(s["clipped_white_frac"]),
                              f(s["laplacian_var"]), s["nonfinite_count"]))


class MossPhotonDiagnostics:
    """Measures the image and masks and attaches the result to a copy of the report (``diagnostics`` block)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "report": ("PHOTON_REPORT", {"tooltip": "Report from a MossPhoton sampler."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Generated image(s) to measure. Never modified."}),
                "reference_images": ("IMAGE", {
                    "tooltip": "1..5 reference photos. Ignored unless consent_reference_photos is on; "
                               "only their count is kept."}),
                "masks": ("MASK", {"tooltip": "Masks to measure coverage of."}),
                "consent_reference_photos": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow the reference photos to be counted. They are never stored or logged."}),
            },
        }

    RETURN_TYPES = ("PHOTON_REPORT", "STRING")
    RETURN_NAMES = ("report", "summary")
    OUTPUT_TOOLTIPS = ("A copy of the report with a diagnostics block.", "Text summary of the report and metrics.")
    FUNCTION = "inspect"
    OUTPUT_NODE = True
    CATEGORY = consts.CAT_DIAGNOSTICS
    DESCRIPTION = ("Shows run facts, warnings and image metrics (luma, clipping, sharpness, mask coverage). "
                   "Measurements only: it does not judge and does not change the image.")

    def inspect(self, report, image=None, reference_images=None, masks=None, consent_reference_photos=False):
        """Returns ``(report_copy, summary)``. Raises ``unsupported_feature`` (newer report schema) or
        ``output_error`` (invalid report or a tensor of the wrong rank). The inputs are never mutated."""
        _require_valid_report(report)
        block = {}
        warnings = []
        if image is not None:
            block["image"] = {"items": imagestats.batch_image_stats(image)}
        if masks is not None:
            block["masks"] = {"items": imagestats.batch_mask_coverage(masks)}
        if reference_images is not None:
            if consent_reference_photos:
                block["identity"] = {"status": "not_run", "reason": "missing_guard_dependency",
                                     "n_reference_images": _batch_size(reference_images)}
            else:
                warnings.append(errors.make_issue(
                    "reference_photos_ignored", NODE_ID, "reference_images",
                    "reference photos are connected but consent_reference_photos is off; they were not read",
                    "turn on consent_reference_photos on MossPhotonDiagnostics to count them"))
        if warnings:
            block["warnings"] = [w.to_dict() for w in warnings]
        out = report_mod.with_diagnostics(report, block)
        lines = [report_mod.format_summary(report)]
        lines += [_line_image(i, s) for i, s in enumerate(block.get("image", {}).get("items", []))]
        for i, m in enumerate(block.get("masks", {}).get("items", [])):
            cov = "-" if m["coverage_frac"] is None else "%.4g" % m["coverage_frac"]
            lines.append("mask[%d]: %dx%d, coverage %s, nonfinite %d" % (i, m["width"], m["height"], cov,
                                                                        m["nonfinite_count"]))
        if "identity" in block:
            ident = block["identity"]
            lines.append("identity: %s (%s), %d reference image(s) counted" % (
                ident["status"], ident["reason"], ident["n_reference_images"]))
        for w in warnings:
            _log.warning(w.format())
            lines.append(w.format())
        summary = "\n".join(lines)
        return {"ui": {"text": [summary]}, "result": (out, summary)}


NODE_CLASS_MAPPINGS = {NODE_ID: MossPhotonDiagnostics}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "Moss Photon Diagnostics"}
