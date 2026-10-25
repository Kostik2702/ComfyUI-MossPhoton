"""Opt-in config for the ``local`` suite: tests that read the owner's real model files (MP-40).

The suite runs only with env ``MOSS_PHOTON_LOCAL=1``. The real file names are NOT in the repository: they come from
``tests/local/local_models.json`` (untracked, see ``local_models.example.json``) or from the file named by env
``MOSS_PHOTON_LOCAL_MODELS``. With the flag set a missing or incomplete config is a FAILURE with a clear message.
"""
import json
import os
import unittest

FLAG = "MOSS_PHOTON_LOCAL"
CONFIG_ENV = "MOSS_PHOTON_LOCAL_MODELS"
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_models.json")
REQUIRED_FILES = ("split_qkv_full", "fused_qkv_gap22", "kohya_a", "lokr", "fused_qkv_b", "kohya_distill_sft",
                  "foreign_a", "foreign_b", "distill_8_dmd2", "distill_8_sft")


class LocalConfigError(RuntimeError):
    pass


def enabled():
    return os.environ.get(FLAG) == "1"


def require_flag():
    """Skip the test class unless the opt-in flag is set."""
    if not enabled():
        raise unittest.SkipTest("local suite is opt-in: set %s=1" % FLAG)


def config_path():
    return os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG


def load():
    """Return the parsed config; raise ``LocalConfigError`` (a test error, not a skip) when it is missing or incomplete."""
    path = config_path()
    if not os.path.isfile(path):
        raise LocalConfigError(
            "%s=1 but the local models config is missing: %s. Copy tests/local/local_models.example.json to "
            "tests/local/local_models.json and fill in your real file names (or set %s)." % (FLAG, path, CONFIG_ENV))
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    files = cfg.get("files") or {}
    missing = [k for k in REQUIRED_FILES if not files.get(k)]
    if missing:
        raise LocalConfigError("local models config %s lacks file entries: %s" % (path, ", ".join(missing)))
    return cfg
