"""Shared fixtures of the darkroom unit tests (MP-71): the preset a settings object is built from.

The phase values live in the ``darkroom`` block of ``presets/turbo_darkroom_*.json`` and nowhere else; ``OFF`` is a preset
dict (the portrait preset with an all-neutral block) written out here, so a test of the neutral plan does not depend on a
shipped preset.
"""
import copy

from moss_photon.core import darkroom, presets

PORTRAIT = "turbo_darkroom_portrait"
FULLBODY = "turbo_darkroom_fullbody"
OFF = "neutral_block"   # not a catalog id: the portrait preset with every phase neutral
MP57 = "mp57_values"    # not a catalog id: the portrait preset with the values of the MP-57 recipes, pinned here so a
#                         calibration of the shipped presets (MP-72 bumped w_commit) does not move the plan tests
DARKROOM_IDS = (PORTRAIT, FULLBODY)

MP57_BLOCK = {
    "schedule": "native",
    "compose": {"cfg": 1.0, "slg": False},
    "commit": {"r_c": 0.06, "r_k": 0.17, "w_compose": 1.0, "w_commit": 1.2},
    "reexpose": {"enabled": False, "sigma_r": 0.8},
    "finish": {"r_f": 0.17, "cfg": 2.0, "eta": 0.0, "switch": False},
}

NEUTRAL_BLOCK = {
    "schedule": "native",
    "compose": {"cfg": 1.0, "slg": False},
    "commit": {"r_c": 0.06, "r_k": 0.17, "w_compose": 1.0, "w_commit": 1.0},
    "reexpose": {"enabled": False, "sigma_r": 0.8},
    "finish": {"r_f": 0.17, "cfg": 1.0, "eta": 0.0, "switch": False},
}


def preset_of(pid=PORTRAIT):
    """The catalog preset ``pid`` (a private copy); ``OFF`` / ``MP57`` = the portrait preset with ``NEUTRAL_BLOCK`` /
    ``MP57_BLOCK``."""
    if pid == OFF:
        return dict(presets.catalog()[PORTRAIT], darkroom=copy.deepcopy(NEUTRAL_BLOCK))
    if pid == MP57:
        return dict(presets.catalog()[PORTRAIT], darkroom=copy.deepcopy(MP57_BLOCK))
    return presets.catalog()[pid]


def settings_of(pid=PORTRAIT, overrides=None, **kw):
    """``darkroom.resolve_settings`` of ``preset_of(pid)``."""
    return darkroom.resolve_settings(preset_of(pid), overrides, **kw)
