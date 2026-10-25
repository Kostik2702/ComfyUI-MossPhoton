"""Finite-output check and the report's guard items (design 1.5, A9).

v1 has no detectors: ``guard_items`` reports every guard of a preset as ``not_run``. Imports: ``errors`` only; the
tensor is used through its own methods, so this module never imports torch.
"""
from . import errors

__all__ = ["check_finite", "guard_items", "required_guards"]

_NO_DEPENDENCY = (None, "", "none")


def check_finite(tensor, *, node, where):
    """Raise ``non_finite_output`` when ``tensor`` holds a NaN or an Inf; one reduction, no copy.

    ``where`` names the place for the message (``"step 3"``, ``"final latent"``). ``tensor`` is anything with
    ``isfinite()`` returning something with ``all()`` (a torch tensor). Returns ``None`` when every value is finite.
    """
    if not bool(tensor.isfinite().all()):
        errors.raise_first_error([errors.make_issue(
            "non_finite_output", node, "model", "NaN or Inf in the %s" % where,
            "try another attention backend (ModelAttentionBackend node), another seed, or report the checkpoint hash",
            severity="error", detail={"where": where})])


def guard_items(preset):
    """Report block ``guards.items``: one ``{"id", "status": "not_run", "reason"}`` per guard of ``preset``.

    ``reason`` is ``missing_guard_dependency`` when the guard declares a dependency (``requires_dependency`` other
    than ``none``) that v1 cannot provide, else ``not_implemented_v1``. Pure; the preset is not modified.
    """
    items = []
    for guard in preset.get("guards") or []:
        needs = guard.get("requires_dependency")
        reason = "not_implemented_v1" if needs in _NO_DEPENDENCY else "missing_guard_dependency"
        items.append({"id": guard["id"], "status": "not_run", "reason": reason})
    return items


def required_guards(preset):
    """Ids of the guards of ``preset`` marked ``required`` (v1 cannot run any, so the sampler refuses these)."""
    return [g["id"] for g in preset.get("guards") or [] if g.get("required")]
