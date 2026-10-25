"""Pure file-output helpers for ``MossPhotonSaveImage`` (design 1.7, 4.8).

Imports: stdlib, ``canon`` and ``errors`` at module level; numpy and PIL only inside the functions that need them
(rules R1, 7.3). Nothing here touches comfy, torch or ``folder_paths``.

Two promises live here: a file that already exists is never overwritten (``next_free_path`` creates with
``O_CREAT | O_EXCL``), and ``to_uint8`` is the exact expression of the stock ``SaveImage`` so a lossless save is
pixel-equal to it.
"""
import io
import json
import os
import re
import time

from . import canon, errors

__all__ = [
    "SIDECAR_KIND", "SIDECAR_SCHEMA", "EMBED_KEY", "FORMATS", "METADATA_MODES", "PNG_COMPRESS_LEVEL",
    "sanitize_prefix", "next_free_path", "to_uint8", "encode_image", "png_text_chunks", "build_sidecar",
    "sidecar_bytes", "wants_sidecar", "wants_embedded", "is_lossless",
]

NODE_ID = "MossPhotonSaveImage"
SIDECAR_SCHEMA = 1
SIDECAR_KIND = "photon_sidecar"
EMBED_KEY = "moss_photon"  # the one PNG tEXt key the pack writes
FORMATS = ("png", "jpeg", "webp")
METADATA_MODES = ("both", "embedded", "sidecar", "none")
PNG_COMPRESS_LEVEL = 4  # the stock SaveImage value (CUI/nodes.py:1674)
MAX_COUNTER_TRIES = 100000
_DRIVE = re.compile(r"^[A-Za-z]:")
_SEPARATORS = re.compile(r"[\\/]")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _output_error(param, message, fix):
    return errors.OutputError(errors.make_issue("output_error", NODE_ID, param, message, fix))


def sanitize_prefix(prefix):
    """Validated ``filename_prefix`` (unchanged text, subfolders kept) or ``output_error``.

    Rejects: not a string / empty after stripping, control characters (ASSUMED), a drive letter (``C:``), a leading
    slash or backslash (absolute or UNC path), and any ``..`` component. ``%width%``-style tokens pass through;
    ComfyUI expands them later (``folder_paths.get_save_image_path``), and they can only become digits.
    """
    fix = "use a relative prefix such as MossPhoton/img (no drive letter, no leading slash, no '..')"
    if not isinstance(prefix, str) or not prefix.strip():
        raise _output_error("filename_prefix", "filename_prefix is empty", fix)
    if _CONTROL.search(prefix):
        raise _output_error("filename_prefix", "filename_prefix contains a control character", fix)
    if _DRIVE.match(prefix) or prefix[0] in "/\\":
        raise _output_error("filename_prefix", "filename_prefix %r is an absolute path" % prefix, fix)
    if ".." in _SEPARATORS.split(prefix):
        raise _output_error("filename_prefix", "filename_prefix %r contains a '..' component" % prefix, fix)
    return prefix


def next_free_path(directory, stem, exts, counter_start):
    """Reserve one unused name per extension and return ``(paths, counter)``; never overwrites.

    ``exts`` is one extension (``".png"``) or a sequence of them (``(".png", ".json")``): all of them share a
    counter, so an image and its sidecar always carry the same number. Names follow the stock node:
    ``<stem>_<counter:05>_<ext>``. Every name is created EMPTY with ``O_CREAT | O_EXCL`` (atomic, so concurrent
    savers get distinct names); the caller then writes into its own reserved files. When any name of the set is
    taken (a stale sidecar included) the files this call created are removed again and the counter advances.
    ``counter_start`` is the first number tried (>= 0). Raises ``output_error`` when the directory is not writable
    or no free number is found within ``MAX_COUNTER_TRIES``.
    """
    if isinstance(exts, str):
        exts = (exts,)
    exts = tuple(exts)
    if not exts:
        raise ValueError("next_free_path needs at least one extension")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    counter = counter_start
    for _ in range(MAX_COUNTER_TRIES):
        paths = tuple(os.path.join(directory, "%s_%05d_%s" % (stem, counter, ext)) for ext in exts)
        created = []
        clash = False
        try:
            for path in paths:
                try:
                    os.close(os.open(path, flags))
                except FileExistsError:
                    clash = True
                    break
                created.append(path)
        except OSError as exc:
            _remove_all(created)
            raise _output_error("filename_prefix", "cannot create %s: %s" % (directory, exc),
                                "check that the output folder exists and is writable") from exc
        if not clash:
            return paths, counter
        _remove_all(created)
        counter += 1
    raise _output_error("filename_prefix", "no free file number for %r after %d tries" % (stem, MAX_COUNTER_TRIES),
                        "use another filename_prefix or clean up the output folder")


def _remove_all(paths):
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def to_uint8(image):
    """``[H, W, C]`` float image (0..1 nominal) -> uint8 array, exactly the stock SaveImage expression.

    ``np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)`` (``CUI/nodes.py`` 1707-1708): values below
    0 or above 1 saturate; NaN follows numpy's cast. Accepts a torch tensor (anything with ``cpu``) or an array.
    """
    import numpy as np
    arr = image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)
    i = 255. * arr
    return np.clip(i, 0, 255).astype(np.uint8)


def is_lossless(fmt):
    """True for ``png`` only (jpeg is lossy; webp is written lossy, ASSUMED)."""
    return fmt == "png"


def wants_sidecar(fmt, mode):
    """A sidecar is written for ``sidecar`` and ``both``, and for ``embedded`` on a format that cannot embed."""
    if mode in ("sidecar", "both"):
        return True
    return mode == "embedded" and fmt != "png"


def wants_embedded(fmt, mode):
    """PNG text chunks are written for png with ``embedded`` or ``both``; nothing is embedded in jpeg/webp (LD10)."""
    return fmt == "png" and mode in ("embedded", "both")


def encode_image(arr, fmt, *, quality=95, text_chunks=None):
    """Encode a uint8 ``[H, W, C]`` array to ``bytes``. No EXIF, no ICC, no other metadata is ever added.

    png: ``compress_level 4`` like the stock node, ``text_chunks`` (str -> str) become tEXt chunks.
    jpeg / webp: ``quality`` 1..100, ``text_chunks`` are ignored (no embedded metadata, LD10); a 4-channel array is
    flattened to RGB for jpeg. Raises ``output_error`` for an unknown format.
    """
    import numpy as np
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    if fmt not in FORMATS:
        raise _output_error("format", "unknown format %r" % (fmt,), "choose one of %s" % ", ".join(FORMATS))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    img = Image.fromarray(np.ascontiguousarray(arr))
    buf = io.BytesIO()
    if fmt == "png":
        info = None
        if text_chunks:
            info = PngInfo()
            for key, value in text_chunks.items():
                info.add_text(key, value)
        img.save(buf, format="PNG", pnginfo=info, compress_level=PNG_COMPRESS_LEVEL)
    elif fmt == "jpeg":
        img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    else:
        img.save(buf, format="WEBP", quality=int(quality))
    return buf.getvalue()


def _json_ascii(obj):
    """Compact sorted JSON, ASCII only (so PIL writes a plain tEXt chunk, never iTXt)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def build_sidecar(report, *, file_info, batch_index, saved_utc=None, workflow_embedded=False):
    """The sidecar object ``{schema_version, kind, file, report, saved_utc}`` (design 4.8).

    ``report`` must already be redacted by the caller (``report.redact``; this module cannot import it, 7.3).
    ``file_info`` supplies ``name, format, width, height, lossless`` and, when the file exists already, ``bytes``
    and ``sha256``; ``batch_index`` and ``workflow_embedded`` (the user opted into the prompt/workflow chunks) are
    added to ``file``. The embedded PNG chunk is the same object without ``bytes``/``sha256`` (a file cannot hold
    its own hash; ASSUMED). The input dicts are not mutated.
    """
    file_block = dict(file_info)
    file_block["batch_index"] = batch_index
    file_block["workflow_embedded"] = bool(workflow_embedded)
    if saved_utc is None:
        saved_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "schema_version": SIDECAR_SCHEMA,
        "kind": SIDECAR_KIND,
        "file": file_block,
        "report": report,
        "saved_utc": saved_utc,
    }


def sidecar_bytes(sidecar):
    """UTF-8 pretty JSON (sorted keys, 2-space indent, trailing newline) for the ``.json`` file."""
    return (json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def png_text_chunks(report, mode, prompt, extra_pnginfo, embed_workflow, *, file_info=None, batch_index=0,
                    saved_utc=None, disable_metadata=False):
    """PNG tEXt chunks as ``{key: str}``; ``{}`` when nothing may be embedded.

    ``report`` is the already redacted report. ``mode`` is the node's ``metadata`` value: only ``embedded`` and
    ``both`` embed, and ``disable_metadata`` (ComfyUI ``--disable-metadata``) empties the result. The key
    ``moss_photon`` holds the sidecar object without ``bytes``/``sha256``. ``prompt`` and ``extra_pnginfo`` (the
    workflow; both contain the prompt texts) are written, as the stock node does, ONLY when ``embed_workflow`` is
    true. ``extra_pnginfo`` keys named ``moss_photon`` are skipped (ours wins).
    """
    if disable_metadata or mode not in ("embedded", "both"):
        return {}
    info = {k: v for k, v in (file_info or {}).items() if k not in ("bytes", "sha256")}
    chunks = {}
    want_workflow = bool(embed_workflow) and (prompt is not None or bool(extra_pnginfo))
    sidecar = build_sidecar(report, file_info=info, batch_index=batch_index, saved_utc=saved_utc,
                            workflow_embedded=want_workflow)
    chunks[EMBED_KEY] = _json_ascii(sidecar)
    if embed_workflow:
        if prompt is not None:
            chunks["prompt"] = json.dumps(prompt)
        for key in (extra_pnginfo or {}):
            if key != EMBED_KEY:
                chunks[key] = json.dumps(extra_pnginfo[key])
    return chunks


def file_facts(name, fmt, width, height, data):
    """``file_info`` for a written file: name, format, size, lossless flag, byte count and sha256 of ``data``."""
    return {"name": name, "format": fmt, "width": width, "height": height, "lossless": is_lossless(fmt),
            "bytes": len(data), "sha256": canon.sha256_hex(data)}
