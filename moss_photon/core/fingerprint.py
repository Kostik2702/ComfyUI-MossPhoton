"""Checkpoint / text-encoder identity: known-hash tables, a small hash cache and guarded path discovery (design 1.2, 1.8).

Imports: stdlib and ``errors`` only (design 7.3); ``folder_paths`` is imported inside ``default_cache``.

Identity is a label, never a gate: a path that cannot be found, a disabled fingerprint or a hash that is not in the
table all end as ``unknown`` plus a warning (D5), they never stop a run. Reading a file never touches RNG.
"""
import hashlib
import json
import logging
import os
import threading
import uuid

from . import errors

__all__ = [
    "KNOWN_CHECKPOINTS", "KNOWN_TEXT_ENCODERS", "HASH_MODES", "CACHE_FILE_NAME", "JsonCache", "default_cache",
    "sha256_file", "cache_key", "checkpoint_path_of", "text_encoder_path_of", "file_basic_info", "classify_hash",
    "identify_checkpoint",
]

_log = logging.getLogger("MossPhoton")

HASH_MODES = ("cached_or_compute", "cached_only", "off")
CACHE_FILE_NAME = "hash_cache.json"
_CACHE_VERSION = 1

# Design 1.8 (source: INV 7 and research/spike_mp2/web/hf_tree.json). The SFT bf16 digest equals
# consts.SFT_BF16_SHA256 (a unit test pins that; this module may not import consts, design 7.3).
KNOWN_CHECKPOINTS = {
    "f96738d8116d78b7316b57c663d8c98591ddafab7488f48e3099b7655e34ab6f": {
        "family": "moss-image-sft", "variant": "bf16", "official": True, "verified_matrix": True,
        "name": "moss_image_sft_bf16", "source": "INV 7 (hashed locally, equals the HF LFS oid)"},
    "cacd158728a1fb9e57c408a922b54752ec71fceb566b3821269927964e62b41f": {
        "family": "moss-image-base", "variant": "bf16", "official": True, "verified_matrix": False,
        "name": "moss_image_base_bf16", "source": "HF LFS oid; file not present locally; not verified in v1"},
    "b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f": {
        "family": "moss-image-sft", "variant": "int8", "official": True, "verified_matrix": False,
        "name": "moss_image_sft_int8_convrot", "source": "HF LFS oid; quantized (SPEC 3 separate matrix)"},
    "f601c1e1161915fb3fea003da0c1782e374dd960fbb3978bc0e4e3fa1f2d8e6f": {
        "family": "moss-image-base", "variant": "int8", "official": True, "verified_matrix": False,
        "name": "moss_image_base_int8_convrot", "source": "HF LFS oid; quantized"},
}

# The reference TE used for the project's own runs (a community qwen3.5-9b bf16 variant, OD4) is NOT official and not in this table.
KNOWN_TEXT_ENCODERS = {
    "7e6e9f08d598f829cb940e60ac0c698e1f1c27a47daffd7e598cd78c78b4cc53": {
        "family": "qwen3.5-9b", "variant": "bf16", "official": True, "name": "qwen3.5_9b_bf16",
        "source": "design 1.8"},
    "32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a": {
        "family": "qwen3.5-9b", "variant": "int8_convrot", "official": True, "name": "qwen3.5_9b_int8_convrot",
        "source": "design 1.8"},
    "4cc6a1543ad2061806da46f35f90cadb4801dcba246c44782cf42ec0feb6407c": {
        "family": "qwen3.5-9b", "variant": "w4a8_gptq", "official": True, "name": "qwen3.5_9b_w4a8_gptq",
        "source": "design 1.8"},
}

# ComfyUI loader functions whose cached_patcher_init carries a real file path (CUI/comfy/sd.py:2449, 2182).
_CHECKPOINT_LOADERS = ("load_diffusion_model", "load_checkpoint_guess_config")
# CUI/comfy/sd.py:1629 (standalone CLIP) and :2188 (CLIP bundled in a checkpoint).
_TEXT_ENCODER_LOADERS = ("load_clip_model_patcher", "load_checkpoint_clip_patcher")


class JsonCache:
    """Tiny key -> dict store persisted as ``{"version": 1, "entries": {key: value}}``.

    ``path=None`` keeps everything in memory. The file is read lazily, written atomically (temp file +
    ``os.replace``) under a lock; every IO failure is swallowed and logged once, the cache then keeps working in
    memory. It holds hash entries only (design m13).
    """

    def __init__(self, path):
        self._path = path
        self._lock = threading.Lock()
        self._entries = None
        self._warned = False

    @property
    def path(self):
        return self._path

    def _warn_once(self, what, exc):
        if not self._warned:
            self._warned = True
            _log.warning("[MossPhoton] hash cache %s (%s): %s; continuing without persistence", what, self._path, exc)

    def _load(self):
        if self._entries is not None:
            return
        self._entries = {}
        if self._path is None or not os.path.isfile(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries") if isinstance(data, dict) else None
            if data.get("version") != _CACHE_VERSION or not isinstance(entries, dict):
                raise ValueError("unknown cache layout")
            self._entries = {k: v for k, v in entries.items() if isinstance(k, str) and isinstance(v, dict)}
        except Exception as exc:  # noqa: BLE001 - a corrupt cache must never stop a run
            self._warn_once("unreadable", exc)

    def get(self, key):
        """Return a copy of the entry for ``key`` or ``None``."""
        with self._lock:
            self._load()
            value = self._entries.get(key)
            return dict(value) if value is not None else None

    def put(self, key, value):
        """Store ``value`` (a dict) under ``key`` and persist when a path is set. Never raises."""
        with self._lock:
            self._load()
            self._entries[key] = dict(value)
            if self._path is None:
                return
            # unique per write and per process: two ComfyUI processes may share the user directory (m14)
            tmp = "%s.%d.%s.tmp" % (self._path, os.getpid(), uuid.uuid4().hex)
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
                with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                    json.dump({"version": _CACHE_VERSION, "entries": self._entries}, f, sort_keys=True, indent=1)
                os.replace(tmp, self._path)
            except Exception as exc:  # noqa: BLE001
                self._warn_once("not writable", exc)
                try:
                    os.remove(tmp)
                except OSError:
                    pass


_default_cache = None
_default_cache_lock = threading.Lock()


def default_cache():
    """Process-wide cache at ``<get_system_user_directory("moss_photon")>/hash_cache.json`` (A17).

    Falls back to an in-memory cache when ``folder_paths`` is unavailable (unit tests, no ComfyUI).
    """
    global _default_cache
    with _default_cache_lock:
        if _default_cache is None:
            path = None
            try:
                import folder_paths
                path = os.path.join(folder_paths.get_system_user_directory("moss_photon"), CACHE_FILE_NAME)
            except Exception as exc:  # noqa: BLE001 - no ComfyUI: in-memory is the documented fallback
                _log.debug("[MossPhoton] no persistent hash cache: %s", exc)
            _default_cache = JsonCache(path)
        return _default_cache


def file_basic_info(path):
    """``{"basename", "size", "mtime_ns"}`` of an existing file (bytes, nanoseconds)."""
    st = os.stat(path)
    return {"basename": os.path.basename(path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def cache_key(path):
    """Hash-cache key of a file: ``realpath`` + ``normcase``, so a junction and the real path share one entry (m14)."""
    return os.path.normcase(os.path.realpath(path))


def _hash_with_source(path, cache, mode, progress, interrupt_check, chunk):
    """Return ``(sha256 or None, source)``; source is ``cache``, ``computed`` or ``unavailable``."""
    if mode not in HASH_MODES:
        raise ValueError("mode must be one of %s, got %r" % (HASH_MODES, mode))
    if mode == "off" or not path:
        return None, "unavailable"
    try:
        info = file_basic_info(path)
    except OSError:
        return None, "unavailable"
    key = cache_key(path)
    if cache is None:
        cache = default_cache()
    hit = cache.get(key)
    if (hit and hit.get("size") == info["size"] and hit.get("mtime_ns") == info["mtime_ns"]
            and isinstance(hit.get("sha256"), str)):
        return hit["sha256"], "cache"
    if mode == "cached_only":
        return None, "unavailable"
    digest = hashlib.sha256()
    done = 0
    try:
        with open(path, "rb") as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                digest.update(block)
                done += len(block)
                if progress is not None:
                    progress(done, info["size"])
                if interrupt_check is not None:
                    interrupt_check()
    except OSError as exc:
        _log.warning("[MossPhoton] cannot hash %s: %s", os.path.basename(path), exc)
        return None, "unavailable"
    sha = digest.hexdigest()
    cache.put(key, {"size": info["size"], "mtime_ns": info["mtime_ns"], "sha256": sha})
    return sha, "computed"


def sha256_file(path, *, cache=None, mode="cached_or_compute", progress=None, interrupt_check=None, chunk=16 << 20):
    """sha256 hex of a file, or ``None`` when it cannot be had.

    ``mode``: ``cached_or_compute`` (hash when the (path, size, mtime) cache misses; about 21 s per 10 GB),
    ``cached_only`` (cache or ``None``), ``off`` (always ``None``). ``progress(done_bytes, total_bytes)`` and
    ``interrupt_check()`` run after every chunk of ``chunk`` bytes; an exception from ``interrupt_check`` (ComfyUI's
    ``throw_exception_if_processing_interrupted``) propagates. A missing or unreadable file gives ``None``.
    """
    return _hash_with_source(path, cache, mode, progress, interrupt_check, chunk)[0]


_init_missing_warned = False
_init_missing_lock = threading.Lock()


def _comfyui_version():
    try:
        import comfyui_version
        return str(comfyui_version.__version__)
    except Exception:  # noqa: BLE001 - the version is only named in a warning
        return "unknown"


def _warn_init_missing_once():
    """One warning per process when a patcher has no ``cached_patcher_init`` at all (m16): a renamed attribute in a
    newer ComfyUI would otherwise turn every run into ``fingerprint_unavailable`` without a word."""
    global _init_missing_warned
    with _init_missing_lock:
        if _init_missing_warned:
            return
        _init_missing_warned = True
    _log.warning("[MossPhoton] ComfyUI %s: ModelPatcher has no cached_patcher_init, so the checkpoint and "
                 "text-encoder file paths cannot be found; identity falls back to fingerprint_unavailable and "
                 "runs are labelled reference_unverified. MossPhoton was tested with ComfyUI 0.42.0; update "
                 "MossPhoton for this ComfyUI version.", _comfyui_version())


def _loader_args(holder, loader_names):
    """``cached_patcher_init`` of ``holder`` when it is ``(callable, args, ...)`` with a known loader, else None."""
    if holder is not None and not hasattr(holder, "cached_patcher_init"):
        _warn_init_missing_once()
        return None
    init = getattr(holder, "cached_patcher_init", None)
    if not isinstance(init, (tuple, list)) or len(init) < 2:
        return None
    fn, args = init[0], init[1]
    if not callable(fn) or getattr(fn, "__name__", None) not in loader_names:
        return None
    if not isinstance(args, (tuple, list)) or not args:
        return None
    return args


def checkpoint_path_of(patcher):
    """Path of the checkpoint file a MODEL was loaded from, or ``None`` (guarded ``cached_patcher_init``, D5).

    Accepted only for the two core loaders ``load_diffusion_model`` / ``load_checkpoint_guess_config`` and an
    existing file; ``ModelPatcher.clone()`` copies the attribute, so LoRA/SLG clones still resolve it.
    """
    args = _loader_args(patcher, _CHECKPOINT_LOADERS)
    if args is None:
        return None
    path = args[0]
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    if isinstance(path, str) and os.path.isfile(path):
        return path
    return None


def text_encoder_path_of(clip):
    """Path of the text-encoder file of a CLIP object, or ``None``.

    ``[1][0]`` is a ONE-element list for a standalone CLIP (CUI/comfy/sd.py:1629) and a plain ``str`` for a CLIP
    bundled in a checkpoint (CUI/comfy/sd.py:2188, m14). A multi-file list is ambiguous and gives ``None``.
    """
    args = _loader_args(getattr(clip, "patcher", None), _TEXT_ENCODER_LOADERS)
    if args is None:
        return None
    first = args[0]
    if isinstance(first, (list, tuple)):
        if len(first) != 1:
            return None
        first = first[0]
    if isinstance(first, os.PathLike):
        first = os.fspath(first)
    if isinstance(first, str) and os.path.isfile(first):
        return first
    return None


def classify_hash(sha256, table):
    """Copy of the table entry for ``sha256``, or ``None`` (also for ``sha256=None``)."""
    if not sha256:
        return None
    entry = table.get(sha256)
    return dict(entry) if entry is not None else None


def identify_checkpoint(patcher, *, mode="cached_or_compute", cache=None, node="MossPhotonSampler",
                        progress=None, interrupt_check=None):
    """Identify the checkpoint behind a MODEL. Never raises for a missing path or hash.

    Returns ``{"identity": {...}, "family": str, "warnings": [Issue, ...]}`` (ASSUMED shape: the design names
    the ``model.identity`` report block and the family / warnings the test plan expects, in one return value).
    ``identity`` is exactly the report block ``model.identity``; ``family`` is ``moss-image-sft``,
    ``moss-image-base`` or ``unknown-moss``. When no file path exists (another loader) or the hash is unavailable,
    ``basename`` is ``"unknown"`` and ``size`` is ``0`` (ASSUMED: the report schema forbids nulls there).

    Warnings: ``fingerprint_unavailable`` (no path, mode ``off``, ``cached_only`` miss or unreadable file);
    ``unverified_checkpoint`` (hash known to us as nothing); ``family_not_verified`` (known, ``verified_matrix``
    false); ``quantized_checkpoint_unverified`` (known int8 variant).
    """
    path = checkpoint_path_of(patcher)
    sha, source = _hash_with_source(path, cache, mode, progress, interrupt_check, 16 << 20)
    info = file_basic_info(path) if path and os.path.isfile(path) else None
    known = classify_hash(sha, KNOWN_CHECKPOINTS)
    identity = {
        "sha256": sha,
        "size": info["size"] if info else 0,
        "basename": info["basename"] if info else "unknown",
        "hash_source": source,
        "classified_by": "known_table" if known else "none",
        "official": known["official"] if known else None,
        "variant": known["variant"] if known else None,
        "verified_matrix": known["verified_matrix"] if known else None,
    }
    family = known["family"] if known else "unknown-moss"
    warnings = []
    if sha is None:
        if path is None:
            why = "the MODEL carries no checkpoint file path (loaded by a custom loader?)"
        elif mode == "off":
            why = "fingerprinting is switched off"
        else:
            why = "the sha256 of %s is not cached or the file is unreadable (mode %s)" % (identity["basename"], mode)
        warnings.append(errors.make_issue(
            "fingerprint_unavailable", node, "model", "checkpoint identity unknown: " + why,
            "load the model with UNETLoader / Load Diffusion Model and keep fingerprint mode cached_or_compute"))
    elif known is None:
        warnings.append(errors.make_issue(
            "unverified_checkpoint", node, "model",
            "sha256 %s... is not a known MOSS checkpoint; run is not marked calibrated" % sha[:8],
            "use the official moss_image_sft_bf16.safetensors for the verified reference"))
    else:
        if not known["verified_matrix"]:
            warnings.append(errors.make_issue(
                "family_not_verified", node, "model",
                "%s (%s) is recognised but outside the verified v1 matrix" % (known["name"], known["variant"]),
                "use moss_image_sft_bf16 for the verified reference"))
        if str(known["variant"]).startswith("int8"):
            warnings.append(errors.make_issue(
                "quantized_checkpoint_unverified", node, "model",
                "%s is an int8 variant, outside the BF16 matrix" % known["name"],
                "use moss_image_sft_bf16 for the verified reference"))
    return {"identity": identity, "family": family, "warnings": warnings}
