"""Phased LoRA weight, the comfy-facing part (MP-16, EXPERIMENTAL; the math and the plan are in ``phase.py``).

Mechanism: clone-per-level. The phased LoRA is NOT applied to the MODEL. ``MossPhotonLoRAProfile profile=phased`` returns a
clone that carries the mapped patch dict as a ``PhaseAdapter`` attachment; the Advanced sampler builds one DIRECT clone of
the incoming MODEL per weight level (``add_patches`` at that level, exactly what the constant path does), runs the steps of
that level as one ordinary ``CFGGuider.sample`` and carries the latent ``x`` over to the next segment. The euler step is a
pure function of ``(x, sigma_i, sigma_i+1)``, so the segments chain bit-exactly; a run with one level is the stock run. The
incoming MODEL, the shared module weights and the cached LoRA state are never written to: every patch lives on a level
clone that is dropped when its segment ends (ComfyUI unloads or re-patches the module through its own model management,
and a cancelled run is cleaned by ``CFGGuider.outer_sample``'s own ``finally``).

Why not ComfyUI's weight hooks (``comfy/hooks.py``, keyframes by ``start_percent``)? They express the per-step weight
exactly (research/phase_lora/hooks_probe.py), but they apply a LoRA in float32 with stochastic rounding, while the
dynamic-VRAM loader (the default on NVIDIA) applies it in bf16 on the fly: 1.5-2.4 % of the bf16 weights differ
(research/phase_lora/numerics_probe.json), so a constant weight through hooks is not the stock result, and hooks need a
second, non-dynamic copy of the model in memory. See research/phase_lora/NOTES.md.

Imports: stdlib, ``errors`` and ``phase``; comfy only inside functions (R1).
"""
import dataclasses

from . import errors, phase

__all__ = ["ATTACH_PHASE_LORA", "PhaseAdapter", "attach_adapter", "adapter_of", "check_adapter", "make_level_model",
           "run_segments"]

ATTACH_PHASE_LORA = "moss_photon.phase_lora"   # ModelPatcher attachment key of the phased adapter


@dataclasses.dataclass(frozen=True)
class PhaseAdapter:
    """The phased LoRA as the MODEL carries it. ``patch_dict`` is the native mapper's output for the file (shared with the
    LoRA state cache, never modified); the identity fields repeat the profile entry so the sampler can prove that the MODEL
    and the profile belong together."""

    stack_index: int
    basename: str
    size: int
    sha256: object
    patch_dict: dict
    metadata: object
    patch_entries: int


def attach_adapter(model, adapter):
    """Store ``adapter`` on ``model`` (a clone the caller owns, never a shared MODEL)."""
    model.set_attachments(ATTACH_PHASE_LORA, adapter)


def adapter_of(model):
    """The ``PhaseAdapter`` attached to ``model`` or ``None``."""
    adapter = model.get_attachment(ATTACH_PHASE_LORA)
    return adapter if isinstance(adapter, PhaseAdapter) else None


def check_adapter(adapter, entry, plan, *, node):
    """``incompatible_lora`` issues when the MODEL does not carry the phased adapter the profile describes."""
    def bad(message):
        return [errors.make_issue(
            "incompatible_lora", node, "model", message,
            "connect the MODEL and the identity_profile outputs of the same MossPhotonLoRAProfile chain",
            severity="error", detail={"feature": "phase_plan", "stack_index": plan.stack_index})]

    if adapter is None:
        return bad("the identity_profile has a phased LoRA (stack[%d]) but the MODEL does not carry it" % plan.stack_index)
    lora = entry["lora"]
    if (adapter.stack_index, adapter.basename, adapter.size, adapter.sha256) != (
            plan.stack_index, lora["basename"], lora["size"], lora["sha256"]):
        return bad("the phased LoRA on the MODEL (%s) is not the one the identity_profile describes (%s)"
                   % (adapter.basename, lora["basename"]))
    return []


def make_level_model(model, adapter, weight):
    """A DIRECT clone of ``model`` with the phased LoRA added at ``weight``; ``model`` is not modified.

    The same two calls as ``lora.apply_patches`` makes for a constant weight, so a level equals the constant run of that
    weight bit for bit.
    """
    level = model.clone()
    level.add_patches(adapter.patch_dict, float(weight))
    if adapter.metadata:
        level.set_attachments("lora_metadata", adapter.metadata)
    return level


class _Carrier:
    """The raw latent ``x`` at the end of the previous segment (before ComfyUI's ``inverse_noise_scaling``)."""

    x = None


def _resume_sampler(real, carrier):
    """A ``KSAMPLER`` that runs ``real``'s sampler function but starts every segment after the first from the carried
    ``x`` instead of the freshly noise-scaled one, and remembers the segment's result."""
    import comfy.samplers

    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
        start = x if carrier.x is None else carrier.x
        carrier.x = real.sampler_function(model, start, sigmas, extra_args=extra_args, callback=callback,
                                          disable=disable, **options)
        return carrier.x

    return comfy.samplers.KSAMPLER(sample, real.extra_options, real.inpaint_options)


def _global_steps(callback, first, total):
    """``callback(step, x0, x, total)`` with the step and total of the whole run instead of the segment's."""
    if callback is None:
        return None

    def shifted(step, x0, x, _segment_total):
        return callback(first + step, x0, x, total)
    return shifted


def run_segments(schedule, model, adapter, *, make_clone, make_guider, sampler, noise, latent_image, sigmas, callback,
                 disable_pbar, seed, poll):
    """Sample ``sigmas`` segment by segment; returns ``(samples, clones_built)``.

    ``schedule`` is a ``phase.StepSchedule`` for exactly these sigmas. For each segment: ``make_clone(level_model)`` turns
    the level clone into the model of the run (SLG, forward counter), ``make_guider(clone)`` builds the plain CFGGuider,
    one ``guider.sample`` runs the segment's steps. At most one level clone is alive at a time and the clone count is at
    most ``schedule.plan.levels``. ``poll`` raises on an interrupt and runs between segments; ``callback`` sees the step
    index and total of the whole run. Exceptions (an interrupt among them) propagate after the segment's own cleanup.
    """
    if len(schedule.segments) > schedule.plan.levels:
        errors.raise_first_error([errors.make_issue(
            "unsupported_feature", "(graph)", "identity_profile",
            "the phase schedule has %d segments for %d levels" % (len(schedule.segments), schedule.plan.levels),
            "report this: the weight steps are not monotone", severity="error")])
    total = len(schedule.applied)
    carrier = _Carrier()
    resume = _resume_sampler(sampler, carrier)
    samples, built = None, 0
    try:
        for seg in schedule.segments:
            poll()
            clone = guider = None
            try:
                clone = make_clone(make_level_model(model, adapter, seg.weight))
                built += 1
                guider = make_guider(clone)
                samples = guider.sample(noise, latent_image.clone(), resume, sigmas[seg.first:seg.stop + 1],
                                        callback=_global_steps(callback, seg.first, total),
                                        disable_pbar=disable_pbar, seed=seed)
            finally:
                clone = guider = None
    finally:
        carrier.x = None
    return samples, built
