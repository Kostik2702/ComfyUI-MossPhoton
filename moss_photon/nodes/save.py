"""MossPhotonSaveImage (design 4.8): save the image with its run facts, never overwriting a file.

PNG gets one tEXt chunk (``moss_photon``) and/or a ``<name>.json`` sidecar; jpeg and webp carry no embedded metadata
(LD10), their facts go to the sidecar. Prompts and the workflow are written only when ``embed_workflow`` is on.
"""
import logging
import os

from ..core import consts, errors, saveio
from ..core import report as report_mod

_log = logging.getLogger("MossPhoton")

NODE_ID = "MossPhotonSaveImage"


def _folder_paths():
    import folder_paths  # lazy: a ComfyUI module (R1)
    return folder_paths


def _disable_metadata():
    """True when ComfyUI runs with ``--disable-metadata`` (suppresses embedding only)."""
    from comfy.cli_args import args  # lazy (R1)
    return bool(args.disable_metadata)


def _fail(message, fix, param="filename_prefix"):
    return errors.OutputError(errors.make_issue("output_error", NODE_ID, param, message, fix))


def _require_valid_report(rep):
    """Redacted deep copy of ``rep``; ``unsupported_feature`` for a newer schema, ``output_error`` if invalid."""
    redacted = report_mod.redact(rep) if isinstance(rep, dict) else None
    problems = report_mod.validate_report(rep)
    if problems:
        raise _fail("the report is not a valid PHOTON_REPORT: " + "; ".join(problems[:5]),
                    "connect the report output of a MossPhoton sampler", "report")
    return redacted


class MossPhotonSaveImage:
    """Writes the images of a batch plus the facts of the run; existing files are never replaced."""

    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The images to save."}),
                "report": ("PHOTON_REPORT", {"tooltip": "Report from a MossPhoton sampler (its facts are saved)."}),
                "filename_prefix": ("STRING", {
                    "default": "MossPhoton/img",
                    "tooltip": "Relative path prefix inside the output folder; no drive letter, leading slash or "
                               "'..'. Accepts %width% and %date% style tokens like the stock node."}),
                "format": (list(saveio.FORMATS), {
                    "default": "png", "tooltip": "png is lossless; jpeg and webp are lossy."}),
                "metadata": (list(saveio.METADATA_MODES), {
                    "default": "both",
                    "tooltip": "both = PNG text chunk and JSON sidecar; embedded = PNG chunk; sidecar = JSON file; "
                               "none = pixels only. jpeg/webp never embed (sidecar instead)."}),
            },
            "optional": {
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "tooltip": "jpeg / webp quality."}),
                "embed_workflow": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also write the ComfyUI prompt and workflow into the PNG. They contain your prompt "
                               "texts; off by default."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = consts.CAT_OUTPUT
    DESCRIPTION = ("Saves images without ever overwriting a file, with the run's preset, parameters, fingerprints "
                   "and recipe hash in a PNG text chunk and/or a JSON sidecar. Prompts are not written unless "
                   "embed_workflow is on.")

    def save(self, image, report, filename_prefix="MossPhoton/img", format="png", metadata="both", quality=95,
             embed_workflow=False, prompt=None, extra_pnginfo=None):
        """Returns ``{"ui": {"images": [...]}, "result": ()}``. Raises ``output_error`` (bad prefix, unusable
        folder, no free name, bad shape or invalid report) or ``unsupported_feature`` (newer report schema)."""
        redacted = _require_valid_report(report)
        if format not in saveio.FORMATS:
            raise _fail("unknown format %r" % (format,), "choose one of %s" % ", ".join(saveio.FORMATS), "format")
        if metadata not in saveio.METADATA_MODES:
            raise _fail("unknown metadata mode %r" % (metadata,),
                        "choose one of %s" % ", ".join(saveio.METADATA_MODES), "metadata")
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) != 4 or shape[0] < 1:
            raise _fail("expected IMAGE [B,H,W,C], got shape %s" % (tuple(shape) if shape is not None else None,),
                        "connect an IMAGE output", "image")
        prefix = saveio.sanitize_prefix(filename_prefix)
        warnings = []
        if metadata in ("embedded", "both") and format != "png":
            warnings.append(errors.make_issue(
                "embedded_unsupported_for_format", NODE_ID, "metadata",
                "%s files carry no embedded metadata; the facts are written to the JSON sidecar instead" % format,
                "use format png to embed, or metadata sidecar to silence this"))
            for issue in warnings:
                _log.warning(issue.format())
            seen = {(w.get("code"), w.get("param")) for w in redacted.get("warnings", [])}
            for issue in warnings:
                if (issue.code, issue.param) not in seen:
                    redacted.setdefault("warnings", []).append(issue.to_dict())
        sidecar_on = saveio.wants_sidecar(format, metadata)
        embed_on = saveio.wants_embedded(format, metadata)
        disable_meta = _disable_metadata() if embed_on else False

        height, width = int(image.shape[1]), int(image.shape[2])
        fp = _folder_paths()
        try:
            folder, filename, counter, subfolder, _ = fp.get_save_image_path(
                prefix, fp.get_output_directory(), width, height)
        except errors.MossPhotonError:
            raise
        except Exception as exc:  # folder_paths raises a bare Exception for a path outside the output folder
            raise _fail("cannot resolve the output folder: %s" % exc,
                        "use a relative prefix inside the output folder") from exc

        ext = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}[format]
        exts = (ext, ".json") if sidecar_on else (ext,)
        results = []
        names = []
        for batch_index in range(int(image.shape[0])):
            stem = filename.replace("%batch_num%", str(batch_index))
            paths, counter = saveio.next_free_path(folder, stem, exts, counter)
            try:
                name = os.path.basename(paths[0])
                info = {"name": name, "format": format, "width": width, "height": height,
                        "lossless": saveio.is_lossless(format)}
                chunks = saveio.png_text_chunks(
                    redacted, metadata, prompt, extra_pnginfo, embed_workflow, file_info=info,
                    batch_index=batch_index, disable_metadata=disable_meta) if embed_on else {}
                workflow_embedded = any(k != saveio.EMBED_KEY for k in chunks)
                data = saveio.encode_image(saveio.to_uint8(image[batch_index]), format, quality=quality,
                                           text_chunks=chunks)
                with open(paths[0], "wb") as f:
                    f.write(data)
                if sidecar_on:
                    sidecar = saveio.build_sidecar(
                        redacted, file_info=saveio.file_facts(name, format, width, height, data),
                        batch_index=batch_index, workflow_embedded=workflow_embedded)
                    with open(paths[1], "wb") as f:
                        f.write(saveio.sidecar_bytes(sidecar))
            except BaseException as exc:
                for path in paths:  # the empty/partial files are ours (exclusive create), nobody else's
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                if isinstance(exc, OSError):
                    raise _fail("cannot write %s: %s" % (os.path.basename(paths[0]), exc),
                                "check that the output folder is writable and has free space") from exc
                raise
            results.append({"filename": name, "subfolder": subfolder, "type": self.type})
            names.extend(os.path.basename(p) for p in paths)
            counter += 1
        text = "saved " + ", ".join(names)
        if warnings:
            text += "\n" + "\n".join(w.format() for w in warnings)
        return {"ui": {"images": results, "text": [text]}, "result": ()}


NODE_CLASS_MAPPINGS = {NODE_ID: MossPhotonSaveImage}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "Moss Photon Save Image"}
