"""Prepare ComfyUI for the integration suite: on sys.path, CPU only, 8 threads, no weights, no server (design 6.0).

Call ``setup()`` (idempotent) before importing anything from ``comfy``. If ComfyUI cannot be found it raises
``ComfyNotAvailable``, a ``unittest.SkipTest``, so the test class is reported as skipped with the reason.

ComfyUI root: env ``MOSS_PHOTON_COMFY_ROOT`` or the default portable install below. Threads: ``THREADS`` (8; the
owner's machine is shared, CLAUDE.md CPU-load rule).
"""
import os
import sys
import unittest

COMFY_ROOT_ENV = "MOSS_PHOTON_COMFY_ROOT"
DEFAULT_COMFY_ROOT = "E:/CUI portable/ComfyUI-torch2.9-cu130-cp313-v1.2/ComfyUI"
THREADS = 8

_ready = False


class ComfyNotAvailable(unittest.SkipTest):
    pass


def comfy_root():
    """Return the ComfyUI root directory (it contains ``comfy/model_base.py``) or ``None``."""
    root = os.environ.get(COMFY_ROOT_ENV) or DEFAULT_COMFY_ROOT
    if os.path.isfile(os.path.join(root, "comfy", "model_base.py")):
        return os.path.abspath(root)
    return None


def setup():
    """Put ComfyUI on ``sys.path`` and import it in CPU mode. Returns the root. Idempotent."""
    global _ready
    root = comfy_root()
    if root is None:
        raise ComfyNotAvailable(
            "ComfyUI not found (set %s or install it at %s); integration tests skipped"
            % (COMFY_ROOT_ENV, DEFAULT_COMFY_ROOT))
    if _ready:
        return root
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
    if root not in sys.path:
        sys.path.insert(0, root)
    saved_argv = sys.argv
    sys.argv = [saved_argv[0], "--cpu"]
    try:
        # ComfyUI ignores argv unless parsing is enabled before comfy.cli_args is imported.
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy.cli_args
    finally:
        sys.argv = saved_argv
    if not comfy.cli_args.args.cpu:
        raise RuntimeError("comfy.cli_args was imported before _comfy_env.setup(); cannot force --cpu")
    import torch
    torch.set_num_threads(THREADS)
    import comfy.model_management
    device = comfy.model_management.get_torch_device()
    if device.type != "cpu":
        raise RuntimeError("integration suite must run on CPU, ComfyUI picked %s" % (device,))
    _ready = True
    return root
