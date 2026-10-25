"""Phased LoRA weight: extension point ONLY (design 1.5, LD10).

The smoothstep / progress-fraction / weight math of SPEC 5 and its tests belong to MP-16, which adds a function that
turns a plan into per-segment patched clones and one hook in ``solver.execute``. Until then a profile that carries a
``phase_plan`` is refused before sampling. Imports: ``errors`` only.
"""
from . import errors

__all__ = ["reject_phase_plan"]


def reject_phase_plan(profile, *, node):
    """Raise ``unsupported_feature`` when ``profile["phase_plan"]`` is set; return ``None`` otherwise.

    ``profile`` is a PHOTON_IDENTITY_PROFILE dict or ``None`` (no profile connected). A missing ``phase_plan`` key
    counts as ``None``.
    """
    if profile is not None and profile.get("phase_plan") is not None:
        errors.raise_first_error([errors.make_issue(
            "unsupported_feature", node, "identity_profile",
            "the identity profile carries a phase_plan (phased LoRA weight), which this version does not run",
            "phased LoRA weight arrives with MP-16; use a constant-weight profile (MossPhotonLoRAProfile "
            "profile=constant)", severity="error", detail={"feature": "phase_plan"})])
    return None
