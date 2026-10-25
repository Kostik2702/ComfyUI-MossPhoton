"""The only door to ComfyUI model / latent / conditioning / VAE internals for sampling (design 1.2).

Imports: stdlib and ``errors`` / ``canon`` / ``fingerprint`` / ``consts``; comfy and torch only inside functions
(R1), so ``validate_*`` and ``detect_foreign_modifiers`` run in a plain interpreter on fake facts. ``probe_*``
touch comfy and return frozen dataclasses of plain facts (R7); ``validate_*`` are pure and return ``list[Issue]``.

Preflight order of a sampler (M8, LD9): ``fix_empty_latent`` first, then ``probe_latent`` on the repaired tensor,
then ``validate_latent_facts``. Nothing here allocates a GPU tensor except ``make_empty_latent`` (a zero latent on
the intermediate device, like the native node).
"""
import dataclasses
import hashlib
import os
import platform
import threading
from typing import Optional

from . import canon, consts, errors, fingerprint

__all__ = [
    "LATENT_CHANNELS", "SPATIAL_RATIO", "MIN_SIDE", "MAX_SIDE", "MAX_BATCH", "ModelFacts", "LatentFacts", "CondFacts",
    "VaeFacts", "detect_foreign_modifiers", "probe_model", "validate_model_facts", "fix_empty_latent",
    "probe_latent", "validate_latent_facts", "probe_conditioning", "validate_cond_facts", "probe_vae",
    "validate_vae_facts", "native_sigmas", "shifted_sigmas", "validate_dimensions", "make_empty_latent",
    "to_intermediate", "env_info", "latent_format_constants", "COND_MODIFIER",
]

# Modifier code of a conditioning that carries area / mask / hooks / control / a timestep range (m4); it joins the
# model-side codes of ``detect_foreign_modifiers`` in ``label.modifiers``.
COND_MODIFIER = "cond_extras"

# The Flux 2 latent format (CUI/comfy/latent_formats.py:197-199); latent_format_constants() reads the live values.
LATENT_CHANNELS = 128
SPATIAL_RATIO = 16
# MossPhotonEmptyLatent bounds (design 4.1).
MIN_SIDE, MAX_SIDE, MAX_BATCH = 64, 8192, 64

_NODE_DEFAULT = "MossPhotonSampler"


# --------------------------------------------------------------------------- model
@dataclasses.dataclass(frozen=True)
class ModelFacts:
    is_moss: bool
    class_name: str
    num_layers: Optional[int]
    dim: Optional[int]
    text_dim: Optional[int]
    model_sampling_class: str
    is_flow: bool
    is_const: bool
    shift: Optional[float]
    multiplier: Optional[float]
    sigma_max: Optional[float]
    latent_channels: Optional[int]
    spacial_downscale_ratio: Optional[int]
    dtype: str                        # ASSUMED: the weight dtype, same value as weight_dtype
    load_device: str
    is_dynamic: bool
    weight_dtype: str                 # str(patcher.model_dtype()), first parameter dtype if the loader left it None
    manual_cast_dtype: Optional[str]  # CUI/comfy/model_base.py:177
    foreign_modifiers: tuple          # sorted codes, exact rules of design 1.2 (LD3)
    patches_uuid: str
    patch_entries: int                # sum(len(v) for v in patcher.patches.values())
    checkpoint_path: Optional[str]
    lora_stack_attachment: Optional[str]  # canonical-JSON string under consts.ATTACH_LORA_STACK, or None


# model_options top-level key -> code (design 1.2 table); any other key -> "model_option:<key>".
_OPTION_CODES = {
    "sampler_cfg_function": "cfg_function",
    "sampler_post_cfg_function": "post_cfg_function",
    "sampler_pre_cfg_function": "pre_cfg_function",
    "sampler_calc_cond_batch_function": "cond_batch_function",
    "model_function_wrapper": "model_function_wrapper",
    "denoise_mask_function": "denoise_mask_function",
    "disable_cfg1_optimization": "disable_cfg1_optimization",
}
_PATCHER_ATTRS = ("hook_patches", "injections", "additional_models", "weight_wrapper_patches", "wrappers", "callbacks")


def _nonempty(value):
    """Not None / False and, for containers, holding at least one non-empty element (recursive)."""
    if value is None or value is False:
        return False
    if isinstance(value, dict):
        return any(_nonempty(v) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_nonempty(v) for v in value)
    return True


def detect_foreign_modifiers(patcher, *, weight_dtype, manual_cast_dtype, reference_dtypes):
    """Sorted tuple of foreign sampling-modifier codes of a ModelPatcher-like object (design 1.2, LD3, M2).

    Pure attribute reads (works on fakes). ``weight_dtype`` / ``manual_cast_dtype`` are strings or None;
    ``reference_dtypes`` are the allowed weight dtypes. Weight patches, attachments and ``patches_uuid`` are not
    modifiers. A clean ModelPatcher (``model_options == {"transformer_options": {}}``) gives ``()``.
    """
    out = set()
    options = getattr(patcher, "model_options", None) or {}
    for key, value in options.items():
        if key != "transformer_options" and _nonempty(value):
            out.add(_OPTION_CODES.get(key, "model_option:" + str(key)))
    for key, value in (options.get("transformer_options") or {}).items():
        if _nonempty(value):
            out.add("attention_override" if key == "optimized_attention_override" else "transformer_option:" + str(key))
    for key in getattr(patcher, "object_patches", None) or {}:
        if key != "model_sampling":  # the AuraFlow shift object patch is judged by the live shift / multiplier
            out.add("object_patch:" + str(key))
    for name in _PATCHER_ATTRS:
        if _nonempty(getattr(patcher, name, None)):
            out.add(name)
    if getattr(patcher, "forced_hooks", None) is not None or getattr(patcher, "current_hooks", None) is not None:
        out.add("hooks")
    if weight_dtype not in reference_dtypes:
        out.add("weight_dtype:%s" % weight_dtype)
    if manual_cast_dtype is not None:
        out.add("manual_cast:%s" % manual_cast_dtype)
    return tuple(sorted(out))


def _try(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - probing a foreign object: any failure means "not available"
        return default


def probe_model(model, *, reference_dtypes=None):
    """Facts of a MODEL (``ModelPatcher``); no GPU work. ``reference_dtypes=None`` means ``REFERENCE_WEIGHT_DTYPES``.

    Never raises for an unfamiliar model: unreadable facts become ``None`` / defaults and ``is_moss`` is false when
    ``comfy.model_base.MOSSImage`` is missing (ComfyUI older than commit e087d8f6) or the model is another class.
    """
    import comfy.model_base
    if reference_dtypes is None:
        reference_dtypes = consts.REFERENCE_WEIGHT_DTYPES
    inner = getattr(model, "model", None)
    moss_cls = getattr(comfy.model_base, "MOSSImage", None)
    is_moss = moss_cls is not None and isinstance(inner, moss_cls)
    dit = getattr(inner, "diffusion_model", None)
    ms = _try(lambda: model.get_model_object("model_sampling"))
    lf = _try(lambda: model.get_model_object("latent_format"))
    weight_dtype = _try(model.model_dtype)
    if weight_dtype is None:  # fixtures built without a loader: fall back to the first parameter
        weight_dtype = _try(lambda: next(iter(dit.parameters())).dtype)
    weight_dtype = str(weight_dtype)
    cast = getattr(inner, "manual_cast_dtype", None)
    cast = None if cast is None else str(cast)
    flow = comfy_sampling_class("ModelSamplingDiscreteFlow")
    const = comfy_sampling_class("CONST")
    bases = _try(lambda: [b.__name__ for b in type(ms).__bases__ if b is not object], [])
    attachment = _try(lambda: model.get_attachment(consts.ATTACH_LORA_STACK))
    return ModelFacts(
        is_moss=bool(is_moss),
        class_name=type(inner).__name__,
        num_layers=_try(lambda: len(dit.layers)),
        dim=_try(lambda: int(dit.text_embedder.proj.out_features)),
        text_dim=_try(lambda: int(dit.text_embedder.proj.in_features)),
        model_sampling_class="+".join(bases) if bases else type(ms).__name__,
        is_flow=bool(flow is not None and isinstance(ms, flow)),
        is_const=bool(const is not None and isinstance(ms, const)),
        shift=_try(lambda: float(ms.shift)),
        multiplier=_try(lambda: float(ms.multiplier)),
        sigma_max=_try(lambda: float(ms.sigma_max)),
        latent_channels=_try(lambda: int(lf.latent_channels)),
        spacial_downscale_ratio=_try(lambda: int(lf.spacial_downscale_ratio)),
        dtype=weight_dtype,
        load_device=str(getattr(model, "load_device", "unknown")),
        is_dynamic=bool(_try(model.is_dynamic, False)),
        weight_dtype=weight_dtype,
        manual_cast_dtype=cast,
        foreign_modifiers=detect_foreign_modifiers(
            model, weight_dtype=weight_dtype, manual_cast_dtype=cast, reference_dtypes=tuple(reference_dtypes)),
        patches_uuid=str(getattr(model, "patches_uuid", "")),
        patch_entries=int(_try(lambda: sum(len(v) for v in model.patches.values()), 0)),
        checkpoint_path=fingerprint.checkpoint_path_of(model),
        lora_stack_attachment=attachment if isinstance(attachment, str) else None,
    )


def comfy_sampling_class(name):
    """Class ``comfy.model_sampling.<name>`` or ``None``."""
    import comfy.model_sampling
    return getattr(comfy.model_sampling, name, None)


def validate_model_facts(f, *, node, slg_requested):
    """Pure checks of design 1.2 / D4; returns ``list[Issue]`` (warnings and infos included).

    Errors: ``unsupported_model`` (not MOSS or MOSSImage missing; latent channels / ratio differ from 128 / 16),
    ``invalid_model_sampling`` (not flow + CONST; ``multiplier != 1.0`` with the ModelSamplingAuraFlow hint),
    ``conflicting_patch`` (a foreign cond-batch hook while SLG is requested). Warnings: ``external_calc_cond_batch``
    (same hook, SLG off), ``foreign_modifiers`` (any modifier; the run is allowed, the label says so),
    info ``non_default_attention_backend`` (attention override).
    """
    mk = errors.make_issue
    if not f.is_moss:
        return [mk("unsupported_model", node, "model",
                   "the MODEL is %s, not comfy.model_base.MOSSImage (or this ComfyUI has no MOSSImage)" % f.class_name,
                   "connect a model loaded by UNETLoader / Load Diffusion Model from moss_image_*.safetensors; "
                   "ComfyUI must contain commit e087d8f6 or newer",
                   detail={"class_name": f.class_name})]
    out = []
    if not (f.is_flow and f.is_const):
        out.append(mk("invalid_model_sampling", node, "model",
                      "model_sampling is %s, expected a flow model with CONST sampling" % f.model_sampling_class,
                      "use the MOSS model as loaded, or ModelSamplingAuraFlow for another shift",
                      detail={"model_sampling_class": f.model_sampling_class}))
    if f.multiplier is None or f.multiplier != 1.0:
        out.append(mk("invalid_model_sampling", node, "model",
                      "model_sampling.multiplier = %s, expected 1.0" % (f.multiplier,),
                      "insert ModelSamplingAuraFlow (multiplier 1.0) instead of ModelSamplingSD3 "
                      "(it sets multiplier 1000 and the native MOSS model then sees a 1000x timestep)",
                      detail={"multiplier": f.multiplier, "expected": 1.0}))
    if f.latent_channels != LATENT_CHANNELS or f.spacial_downscale_ratio != SPATIAL_RATIO:
        out.append(mk("unsupported_model", node, "model",
                      "latent format is %s channels at ratio %s, expected %d at %d"
                      % (f.latent_channels, f.spacial_downscale_ratio, LATENT_CHANNELS, SPATIAL_RATIO),
                      "connect a model loaded from moss_image_*.safetensors",
                      detail={"latent_channels": f.latent_channels, "spacial_downscale_ratio": f.spacial_downscale_ratio}))
    mods = list(f.foreign_modifiers)
    if "cond_batch_function" in mods:
        if slg_requested:
            out.append(mk("conflicting_patch", node, "model",
                          "the MODEL already carries a sampler_calc_cond_batch_function and SLG is requested",
                          "remove the upstream SkipLayerGuidance* node; MossPhoton applies SLG itself from the preset"))
        else:
            out.append(mk("external_calc_cond_batch", node, "model",
                          "the MODEL carries a sampler_calc_cond_batch_function (SLG is off)",
                          "remove the upstream SkipLayerGuidance* node if you do not want it"))
    if mods:
        out.append(mk("foreign_modifiers", node, "model",
                      "sampling modifiers on the MODEL: %s" % ", ".join(mods),
                      "label is reference_modified; remove the upstream patches for the verified reference",
                      detail={"modifiers": mods}))
    if "attention_override" in mods:
        out.append(mk("non_default_attention_backend", node, "model",
                      "an attention backend override is set on the MODEL (it changes bits)",
                      "remove the ModelAttentionBackend node for the verified reference"))
    return out


# --------------------------------------------------------------------------- latent
@dataclasses.dataclass(frozen=True)
class LatentFacts:
    """Facts of the latent AFTER the native repair (M8). ``ndim`` is an addition to the design fields."""

    batch: int
    channels: int
    height: int
    width: int
    dtype: str
    device: str
    is_nested: bool
    is_empty: bool
    has_noise_mask: bool
    has_batch_index: bool
    raw_shape: tuple
    raw_downscale_ratio_spacial: Optional[float]
    rescaled_by_native_fix: bool
    pixel_width: int
    pixel_height: int
    ndim: int = 4


def fix_empty_latent(model, latent, *, node=_NODE_DEFAULT):
    """The native repair, exactly as the stock KSampler (CUI/nodes.py:1582-1583); runs BEFORE ``probe_latent``.

    ``comfy.sample.fix_empty_latent_channels`` turns an EMPTY (all-zero) latent of any channel count / ratio into
    ``[B, 128, H/16, W/16]``; a non-empty latent is returned unchanged. A latent dict without a tensor
    ``samples`` raises ``invalid_latent``.
    """
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None or not hasattr(samples, "shape"):
        errors.raise_first_error([errors.make_issue(
            "invalid_latent", node, "latent.samples", "the latent has no tensor under 'samples'",
            "use MossPhotonEmptyLatent, EmptyFlux2LatentImage or the stock EmptyLatentImage (empty latents are "
            "repaired by ComfyUI); a non-empty latent must already have 128 channels", severity="error")])
    import comfy.sample
    return comfy.sample.fix_empty_latent_channels(
        model, samples, latent.get("downscale_ratio_spacial"), latent.get("downscale_ratio_temporal"))


def probe_latent(latent, fixed):
    """Facts of ``fixed`` (the output of ``fix_empty_latent``) plus how the raw latent differed."""
    raw = latent["samples"]
    raw_shape = tuple(int(x) for x in getattr(raw, "shape", ()))
    nested = bool(getattr(fixed, "is_nested", False))
    ratio = latent.get("downscale_ratio_spacial")
    if nested:
        batch = channels = height = width = 0
        ndim, empty, dtype, device = 0, False, str(fixed.dtype), str(fixed.device)
        shape = ()
    else:
        shape = tuple(int(x) for x in fixed.shape)
        ndim = len(shape)
        batch, channels, height, width = shape if ndim == 4 else (shape[0] if shape else 0, 0, 0, 0)
        empty = bool(fixed.count_nonzero() == 0)
        dtype, device = str(fixed.dtype), str(fixed.device)
    return LatentFacts(
        batch=batch, channels=channels, height=height, width=width, dtype=dtype, device=device, is_nested=nested,
        is_empty=empty, has_noise_mask="noise_mask" in latent, has_batch_index="batch_index" in latent,
        raw_shape=raw_shape, raw_downscale_ratio_spacial=float(ratio) if ratio is not None else None,
        rescaled_by_native_fix=bool(not nested and raw_shape != shape),
        pixel_width=width * SPATIAL_RATIO, pixel_height=height * SPATIAL_RATIO, ndim=ndim)


def validate_latent_facts(f, *, channels, node):
    """Pure checks on the REPAIRED latent (M8); returns ``list[Issue]``.

    ``invalid_latent``: nested, not 4-D, channels != ``channels`` (only an empty latent is repaired by ComfyUI,
    so this is a non-empty latent with the wrong channel count), or a non-empty latent that declares a spatial
    ratio other than 16. ``unsupported_source_input``: a ``noise_mask`` (masked refine is second queue). Info
    ``latent_rescaled_by_native_fix`` when ComfyUI's own repair changed the shape (stock ``EmptyLatentImage``).
    """
    mk = errors.make_issue
    fix = ("use MossPhotonEmptyLatent, EmptyFlux2LatentImage or the stock EmptyLatentImage (empty latents are "
           "repaired by ComfyUI); a non-empty latent must already have 128 channels")
    param = "latent.samples"
    if f.is_nested:
        return [mk("invalid_latent", node, param, "the latent is a nested tensor", fix)]
    out = []
    if f.ndim != 4:
        return [mk("invalid_latent", node, param, "latent has %d dimensions, expected 4 ([B, C, H, W])" % f.ndim, fix)]
    if f.channels != channels:
        out.append(mk("invalid_latent", node, param, "channels = %d, expected %d" % (f.channels, channels), fix,
                      detail={"channels": f.channels, "expected": channels, "raw_shape": list(f.raw_shape)}))
    ratio = f.raw_downscale_ratio_spacial
    if not f.is_empty and ratio is not None and ratio != SPATIAL_RATIO:
        out.append(mk("invalid_latent", node, "latent.downscale_ratio_spacial",
                      "a non-empty latent declares spatial ratio %s, expected %d" % (ratio, SPATIAL_RATIO), fix,
                      detail={"downscale_ratio_spacial": ratio, "expected": SPATIAL_RATIO}))
    if f.has_noise_mask:
        out.append(mk("unsupported_source_input", node, "latent.noise_mask",
                      "the latent carries a noise_mask",
                      "source-based editing arrives with MossPhotonProtectedRefine (second queue); "
                      "feed MossPhotonEmptyLatent"))
    if f.rescaled_by_native_fix and not out:
        out.append(mk("latent_rescaled_by_native_fix", node, param,
                      "latent %s was repaired by ComfyUI to %dx%d pixels"
                      % (list(f.raw_shape), f.pixel_width, f.pixel_height),
                      "none needed; MossPhotonEmptyLatent avoids the repair",
                      detail={"raw_shape": list(f.raw_shape), "pixel_width": f.pixel_width,
                              "pixel_height": f.pixel_height}))
    return out


# --------------------------------------------------------------------------- conditioning
@dataclasses.dataclass(frozen=True)
class CondFacts:
    entries: int
    token_counts: tuple
    dim: Optional[int]
    dtype: str
    sha256: str
    has_area_or_hooks: bool
    all_tensors: bool = True  # addition to the design fields


# ASSUMED: entry-dict keys that make a conditioning more than a plain text embedding.
_COND_EXTRAS = ("area", "mask", "hooks", "control", "timestep_start", "timestep_end")


def _f32_bytes(t):
    return t.detach().cpu().float().contiguous().numpy().tobytes()


def probe_conditioning(cond, name):
    """Facts of a CONDITIONING list ``[[tensor, dict], ...]``; the hash covers float32 bytes of every entry tensor
    in order (CPU, no RNG). ``name`` is kept for messages by the caller.
    """
    entries = list(cond) if isinstance(cond, (list, tuple)) else []
    tensors = [e[0] for e in entries if isinstance(e, (list, tuple)) and e and hasattr(e[0], "shape")]
    ok = bool(entries) and len(tensors) == len(entries)
    digest = hashlib.sha256()
    for t in tensors:
        digest.update(_f32_bytes(t))
    extras = any(isinstance(e, (list, tuple)) and len(e) > 1 and isinstance(e[1], dict)
                 and any(k in e[1] for k in _COND_EXTRAS) for e in entries)
    return CondFacts(
        entries=len(entries),
        token_counts=tuple(int(t.shape[1]) if len(t.shape) > 1 else 0 for t in tensors),
        dim=int(tensors[0].shape[-1]) if tensors else None,
        dtype=str(tensors[0].dtype) if tensors else "none",
        sha256=digest.hexdigest(), has_area_or_hooks=extras, all_tensors=ok)


def validate_cond_facts(f, *, name, text_dim, node):
    """Pure checks: ``invalid_conditioning`` for an empty list, a non-tensor entry or an embedding dim that is not
    the model's ``text_dim``; warning ``multiple_cond_entries`` for more than one entry.
    """
    mk = errors.make_issue
    fix = "encode with a CLIP of type moss_image (Qwen3.5-9B) through CLIPTextEncode or MossPhotonPromptLayout"
    if f.entries == 0:
        return [mk("invalid_conditioning", node, name, "the %s conditioning is empty" % name, fix)]
    if not f.all_tensors:
        return [mk("invalid_conditioning", node, name, "the %s conditioning has an entry that is not a tensor" % name, fix)]
    out = []
    if text_dim is not None and f.dim != text_dim:
        out.append(mk("invalid_conditioning", node, name,
                      "%s embedding dim = %s, expected %d" % (name, f.dim, text_dim), fix,
                      detail={"dim": f.dim, "expected": text_dim}))
    if f.entries > 1:
        out.append(mk("multiple_cond_entries", node, name,
                      "%s has %d conditioning entries; batch composition may differ from the stock single-entry case"
                      % (name, f.entries), "combine the prompt into one entry for the verified reference"))
    if f.has_area_or_hooks:
        out.append(mk("foreign_modifiers", node, name,
                      "the %s conditioning carries area, mask, hooks, control or a timestep range" % name,
                      "label is reference_modified; use a plain text conditioning for the verified reference",
                      detail={"modifiers": [COND_MODIFIER]}))
    return out


# --------------------------------------------------------------------------- VAE
@dataclasses.dataclass(frozen=True)
class VaeFacts:
    latent_channels: Optional[int]
    downscale_ratio: Optional[int]


def probe_vae(vae):
    """``vae.latent_channels`` / ``vae.downscale_ratio`` (128 / 16 for the Flux 2 VAE, CUI/comfy/sd.py:624-625).

    A non-integer ratio (other VAE families use tuples or callables) is reported as ``None``.
    """
    channels = getattr(vae, "latent_channels", None)
    ratio = getattr(vae, "downscale_ratio", None)
    return VaeFacts(
        channels if isinstance(channels, int) and not isinstance(channels, bool) else None,
        ratio if isinstance(ratio, int) and not isinstance(ratio, bool) else None)


def validate_vae_facts(f, *, channels, ratio, node):
    """``invalid_vae`` when the VAE is not ``channels`` x ``ratio`` (128 x 16)."""
    if f.latent_channels != channels or f.downscale_ratio != ratio:
        return [errors.make_issue(
            "invalid_vae", node, "vae",
            "VAE has %s latent channels at ratio %s, expected %d at %d"
            % (f.latent_channels, f.downscale_ratio, channels, ratio),
            "load flux2-vae.safetensors (or flux2-vae_small) with VAELoader",
            detail={"latent_channels": f.latent_channels, "downscale_ratio": f.downscale_ratio})]
    return []


# --------------------------------------------------------------------------- sigmas, latents, environment
def native_sigmas(model, scheduler, steps):
    """``comfy.samplers.calculate_sigmas`` on the live model sampling: the function behind KSampler and
    BasicScheduler (CUI/comfy/samplers.py:1378-1386). Returns a float32 CPU tensor of ``steps + 1`` values.
    """
    import comfy.samplers
    return comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)


def shifted_sigmas(model, scheduler, steps, shift):
    """Schedule for another ``shift`` at multiplier 1.0: deepcopy of the live model sampling +
    ``set_parameters`` (bit-equal to the ModelSamplingAuraFlow route, check C3). The live model is not touched.
    """
    import copy
    import comfy.samplers
    ms = copy.deepcopy(model.get_model_object("model_sampling"))
    ms.set_parameters(shift=shift, multiplier=1.0)
    return comfy.samplers.calculate_sigmas(ms, scheduler, steps)


def validate_dimensions(width, height, batch, *, node, check_bounds=True):
    """Pure checks for ``MossPhotonEmptyLatent``; returns ``list[Issue]`` (``invalid_dimensions``).

    Each side must be a multiple of 16 (never floored) and, when ``check_bounds``, within
    ``MIN_SIDE..MAX_SIDE``; ``batch`` within ``1..MAX_BATCH``. Pixels.
    """
    out = []
    for side, value in (("width", width), ("height", height)):
        if not isinstance(value, int) or isinstance(value, bool):
            out.append(errors.make_issue("invalid_dimensions", node, side, "%s = %r is not an integer" % (side, value),
                                         "use a multiple of 16"))
            continue
        if value % SPATIAL_RATIO:
            lo = value // SPATIAL_RATIO * SPATIAL_RATIO
            out.append(errors.make_issue(
                "invalid_dimensions", node, side,
                "%s = %d is not a multiple of %d" % (side, value, SPATIAL_RATIO),
                "use a multiple of 16 (nearest: %d or %d)" % (lo, lo + SPATIAL_RATIO),
                detail={"value": value, "floor": lo, "ceil": lo + SPATIAL_RATIO}))
        elif value <= 0 or (check_bounds and not MIN_SIDE <= value <= MAX_SIDE):
            out.append(errors.make_issue(
                "invalid_dimensions", node, side,
                "%s = %d is outside %d..%d" % (side, value, MIN_SIDE, MAX_SIDE),
                "use a multiple of 16 between %d and %d" % (MIN_SIDE, MAX_SIDE), detail={"value": value}))
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1 or (check_bounds and batch > MAX_BATCH):
        out.append(errors.make_issue("invalid_dimensions", node, "batch_size",
                                     "batch_size = %r is outside 1..%d" % (batch, MAX_BATCH),
                                     "use a batch size between 1 and %d" % MAX_BATCH))
    return out


def latent_format_constants():
    """``(latent_channels, spacial_downscale_ratio)`` of ``comfy.latent_formats.Flux2`` (128, 16)."""
    import comfy.latent_formats
    fmt = comfy.latent_formats.Flux2
    return int(fmt.latent_channels), int(fmt.spacial_downscale_ratio)


def make_empty_latent(width, height, batch, *, node="MossPhotonEmptyLatent"):
    """``{"samples": zeros [B, 128, H/16, W/16]}`` on ``intermediate_device()``, float32, like ``EmptyFlux2LatentImage``
    (CUI/comfy_extras/nodes_flux.py:38-58) but never flooring: a side that is not a multiple of 16 raises
    ``invalid_dimensions``. Sizes in pixels.
    """
    errors.raise_first_error(validate_dimensions(width, height, batch, node=node, check_bounds=False))
    import torch
    import comfy.model_management
    channels, ratio = latent_format_constants()
    samples = torch.zeros([batch, channels, height // ratio, width // ratio],
                          device=comfy.model_management.intermediate_device())
    return {"samples": samples}


def to_intermediate(samples):
    """``samples`` on ``intermediate_device()`` in ``intermediate_dtype()`` (CUI/comfy/sample.py:79-84)."""
    import comfy.model_management
    return samples.to(comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())


_env_lock = threading.Lock()
_env_base = None


def _git_commit(root):
    """Commit hash of a git checkout by plain file reads (no subprocess); ``None`` when not a checkout."""
    git = os.path.join(root, ".git")
    try:
        with open(os.path.join(git, "HEAD"), "r", encoding="utf-8") as f:
            head = f.read().strip()
        if not head.startswith("ref: "):
            return head or None
        ref = head[5:].strip()
        ref_file = os.path.join(git, *ref.split("/"))
        if os.path.isfile(ref_file):
            with open(ref_file, "r", encoding="utf-8") as f:
                return f.read().strip() or None
        with open(os.path.join(git, "packed-refs"), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        pass
    return None


def _attention_global():
    import comfy.ldm.modules.attention as att
    current = att.optimized_attention
    for attr, label in (("attention_sage", "sage"), ("attention_flash", "flash"), ("attention_xformers", "xformers"),
                        ("attention_pytorch", "pytorch"), ("attention_split", "split"),
                        ("attention_sub_quad", "sub_quad"), ("attention_comfy_kitchen_int8", "comfy_kitchen")):
        if current is getattr(att, attr, None):
            return label
    # A backend this table does not know (a newer ComfyUI) must not pass for pytorch: it changes bits and feeds
    # recipe_hash. The function name keeps the report honest; the enum accepts the ``unknown:`` prefix (m15).
    return "unknown:%s" % getattr(current, "__name__", type(current).__name__)


def _env_base_info():
    global _env_base
    with _env_lock:
        if _env_base is None:
            import torch
            import comfy.cli_args
            import comfy.model_management
            version = _try(lambda: __import__("comfyui_version").__version__, "unknown")
            root = _try(lambda: os.path.dirname(os.path.abspath(__import__("comfyui_version").__file__)))
            device = comfy.model_management.get_torch_device()
            _env_base = {
                "comfyui_version": str(version),
                "comfyui_commit": _git_commit(root) if root else None,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "device": str(device),
                "device_name": _try(lambda: torch.cuda.get_device_name(device)) if device.type == "cuda" else None,
                "attention": {"global": _attention_global(), "model_override": None},
                "deterministic_flag": bool(comfy.cli_args.args.deterministic),
            }
        return _env_base


def env_info(model=None):
    """The ``env`` block of the report (design 2.3); the static part is memoised, a copy is returned.

    ``model`` (a ModelPatcher) adds the per-model fields ``weights_streaming``, ``inference_dtype`` and
    ``attention.model_override``; without it they are ``False`` / ``"unknown"`` / ``None``.
    """
    import copy
    info = copy.deepcopy(_env_base_info())
    info["weights_streaming"] = False
    info["inference_dtype"] = "unknown"
    if model is not None:
        info["weights_streaming"] = bool(_try(model.is_dynamic, False))
        dtype = _try(model.model_dtype)
        if dtype is None:
            dtype = _try(lambda: next(iter(model.model.diffusion_model.parameters())).dtype)
        info["inference_dtype"] = str(dtype)
        override = ((getattr(model, "model_options", None) or {}).get("transformer_options") or {}).get(
            "optimized_attention_override")
        if override is not None:
            # set_model_optimized_attention wraps the backend in a closure named after the option; the backend's
            # own name survives only as container_function, when it has one (CUI/comfy/model_patcher.py:690-696)
            named = getattr(override, "container_function", None) or override
            info["attention"]["model_override"] = str(getattr(named, "__name__", type(named).__name__))
    return info
