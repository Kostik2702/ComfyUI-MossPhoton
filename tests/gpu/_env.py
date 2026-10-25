"""Opt-in switch and shared session of the GPU gates (MP-12; design 6.7, docs/03-turbo.md 5.6).

* ``MOSS_PHOTON_GPU=1``  runs the gates; without it every gate module skips in ``setUpModule``.
* ``MOSS_PHOTON_GPU_SIZE``  ``smoke`` (default: C01 only, a few repeats) or ``full`` (SPEC 10 sizes).
* The harness lives in ``<repo>/research/parity``; a standalone pack install has no such folder and skips.

This module imports nothing heavy at import time (no torch, no comfy), so discovery without the switch is free.
"""
import os
import sys
import unittest

GPU_ENV = "MOSS_PHOTON_GPU"
SIZE_ENV = "MOSS_PHOTON_GPU_SIZE"

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(HERE))
REPO_ROOT = os.path.dirname(PKG_ROOT)
PARITY_DIR = os.path.join(REPO_ROOT, "research", "parity")


def parity_available():
    return os.path.isfile(os.path.join(PARITY_DIR, "harness.py"))


def require_parity_dir():
    """``SkipTest`` unless the harness folder exists (the pure verdict tests need only this)."""
    if not parity_available():
        raise unittest.SkipTest("research/parity not found next to the pack (%s)" % PARITY_DIR)
    if PARITY_DIR not in sys.path:
        sys.path.insert(0, PARITY_DIR)


def gpu_enabled():
    return os.environ.get(GPU_ENV) == "1"


def require_gpu():
    """``SkipTest`` unless ``MOSS_PHOTON_GPU=1`` and the harness folder exists. Call from ``setUpModule``."""
    if not gpu_enabled():
        raise unittest.SkipTest("GPU gate is opt-in: set %s=1 (see research/parity)" % GPU_ENV)
    require_parity_dir()


def size():
    value = os.environ.get(SIZE_ENV, "smoke")
    if value not in ("smoke", "full"):
        raise ValueError("%s must be smoke or full, got %r" % (SIZE_ENV, value))
    return value


class Session:
    """Models loaded once per test process: ``harness``, ``cases``, ``assets``, ``machine``, ``out_dir``."""

    def __init__(self):
        import atexit
        import gallery
        import harness
        self.harness = harness
        self.gallery = gallery.Gallery(os.path.join(harness.RESULTS, harness.today()), gallery.new_run_id("tests"),
                                       "tests").register()
        atexit.register(self.gallery.close, "finished")
        self.out_dir = self.gallery.run_dir
        self.cases = harness.load_cases()
        self.gallery.stage("machine gate")
        self.machine = harness.ensure_machine_free()
        self.gallery.stage("loading ComfyUI and models")
        harness.init_comfy()
        self.assets = harness.Assets(self.cases)
        texts = sorted(set(self.cases["prompts"].values()) | {self.cases["negative_prompt"]})
        self.assets.preencode(texts)


_session = None


def session():
    """The shared ``Session`` (created on first use). Raises ``SkipTest`` when the gate is off."""
    global _session
    require_gpu()
    if _session is None:
        _session = Session()
    return _session
