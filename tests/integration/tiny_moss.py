"""A tiny random MOSS for CPU tests: 24 single blocks, seeded, no checkpoint (design 6.0, LD4/M3).

Why 24 layers: the shipped ``reference`` preset skips layer 22 (SLG), which must EXIST for the preset to run
unmodified on CPU. On the 4-layer spike model SLG 22 equalled no-SLG bit for bit (MP-5 R5), i.e. a silent no-op.

Ported from research/design_mp4/design_checks.py (M3 section) and research/spike_mp2/tiny_moss_route_test.py.
``build_tiny_moss()`` returns a ``TinyMoss`` named tuple that also unpacks as
``(patcher, model, make_cond, latent, sigmas)``.
"""
from typing import NamedTuple

from . import _comfy_env

DIM = 128
FFN_DIM = 256
HEADS = 2
LAYERS = 24
TXT = 64  # text_dim
LATENT_CHANNELS = 128
LATENT_HW = (8, 8)  # 128 px at the 16x ratio
WEIGHT_SEED = 1234
STEPS = 6
SEED = 42
CFG = 3.0


class TinyMoss(NamedTuple):
    patcher: object      # comfy.model_patcher.ModelPatcher on CPU
    model: object        # comfy.model_base.MOSSImage
    make_cond: object    # make_cond(n) -> CONDITIONING with one entry of n tokens, seeded by n
    latent: object       # zeros (1, 128, H, W), float32
    sigmas: object       # native sgm_uniform schedule of the model, STEPS steps, float32 CPU


def build_tiny_moss(layers=LAYERS, weight_seed=WEIGHT_SEED, steps=STEPS):
    """Build the random MOSS. Same arguments give bit-identical weights. Skips the test when ComfyUI is missing."""
    _comfy_env.setup()
    import torch
    import comfy.model_base
    import comfy.model_patcher
    import comfy.samplers
    import comfy.supported_models

    if not hasattr(comfy.model_base, "MOSSImage"):
        raise _comfy_env.ComfyNotAvailable("this ComfyUI has no MOSSImage (needs commit e087d8f6 or newer)")

    ucfg = {"image_model": "moss_image", "dim": DIM, "ffn_dim": FFN_DIM, "num_heads": HEADS, "num_layers": layers,
            "text_dim": TXT, "latent_channels": LATENT_CHANNELS, "fused_qkv": True}
    model = comfy.model_base.MOSSImage(comfy.supported_models.MOSSImage(ucfg), device="cpu")
    g = torch.Generator().manual_seed(weight_seed)
    with torch.no_grad():
        for name, p in model.diffusion_model.named_parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
            elif "norm" in name:
                p.copy_(torch.ones(p.shape) + 0.01 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.01)
    model.eval()
    patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))

    def make_cond(n):
        gg = torch.Generator().manual_seed(100 + n)
        return [[torch.randn(1, n, TXT, generator=gg), {"pooled_output": None}]]

    latent = torch.zeros(1, LATENT_CHANNELS, LATENT_HW[0], LATENT_HW[1])
    sigmas = comfy.samplers.calculate_sigmas(patcher.get_model_object("model_sampling"), "sgm_uniform", steps)
    return TinyMoss(patcher, model, make_cond, latent, sigmas)


def sample_euler(tiny, model_patcher=None, seed=SEED, cfg=CFG, positive_tokens=23, negative_tokens=9, callback=None):
    """Run the stock CFGGuider + euler route (design D2) on the tiny model; returns the final latent tensor.

    ``model_patcher`` defaults to ``tiny.patcher``; pass a clone (e.g. with SLG) to compare runs.
    """
    import torch
    import comfy.model_management
    import comfy.sample
    import comfy.samplers

    patcher = tiny.patcher if model_patcher is None else model_patcher
    noise = comfy.sample.prepare_noise(tiny.latent, seed)
    guider = comfy.samplers.CFGGuider(patcher)
    guider.set_conds(tiny.make_cond(positive_tokens), tiny.make_cond(negative_tokens))
    guider.set_cfg(cfg)
    sampler = comfy.samplers.sampler_object("euler")
    with torch.inference_mode():
        out = guider.sample(noise, tiny.latent.clone(), sampler, tiny.sigmas.clone(), callback=callback, seed=seed)
    return out.to(comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
