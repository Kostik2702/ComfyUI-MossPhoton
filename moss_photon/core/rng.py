"""Noise policies (design 1.2, A3, D2). The only module of the pack that may consume a random generator (R8).

Imports: stdlib and ``errors``; torch / comfy only inside the functions that need them.

* ``native``: ``comfy.sample.prepare_noise`` verbatim, one CPU stream for the whole batch. It reseeds the global CPU
  RNG exactly like the stock KSampler (documented, not hidden).
* ``seed_plus_i``: one own CPU ``torch.Generator`` per batch item seeded ``seed + i``; never touches the global RNG.
"""
from . import errors

__all__ = [
    "NOISE_POLICIES", "SEED_MASK", "subseed", "subseeds", "seed_warnings", "make_noise", "own_generator_noise",
    "tensor_sha256",
]

NOISE_POLICIES = ("native", "seed_plus_i")
SEED_MASK = 0xFFFFFFFFFFFFFFFF  # 2**64 - 1, the seed widget maximum (same as the stock KSampler)


def _check_policy(policy):
    if policy not in NOISE_POLICIES:
        raise ValueError("noise policy must be one of %s, got %r" % (NOISE_POLICIES, policy))


def subseed(seed, i):
    """``(seed + i) & SEED_MASK``: the seed of batch item ``i`` under ``seed_plus_i`` (wraps at 2**64)."""
    return (int(seed) + int(i)) & SEED_MASK


def subseeds(seed, batch, policy):
    """Seeds actually used: ``[seed]`` for ``native`` (one stream), ``batch`` sub-seeds for ``seed_plus_i``."""
    _check_policy(policy)
    if policy == "native":
        return [int(seed)]
    return [subseed(seed, i) for i in range(int(batch))]


def seed_warnings(seed, batch, policy, *, node="MossPhotonSamplerAdvanced"):
    """``[Issue seed_wrapped]`` when ``seed + i`` exceeded 2**64 - 1 for some item of a ``seed_plus_i`` batch."""
    _check_policy(policy)
    if policy == "seed_plus_i" and int(seed) + int(batch) - 1 > SEED_MASK:
        return [errors.make_issue(
            "seed_wrapped", node, "seed",
            "seed + i exceeded 2**64 - 1 for batch %d and wrapped around" % int(batch),
            "use a seed at least %d below 2**64 to avoid the wrap" % int(batch))]
    return []


def own_generator_noise(shape, dtype, seed):
    """Gaussian noise from a private CPU generator: float32 ``randn`` then ``.to(dtype)``.

    Equals ``comfy.sample.prepare_noise`` for the same shape and seed (check C4), without touching the global RNG.
    Used by ``seed_plus_i`` and as the test oracle of ``native``.
    """
    import torch
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randn(tuple(shape), dtype=torch.float32, generator=gen, device="cpu").to(dtype)


def make_noise(latent_image, seed, policy="native", batch_index=None, *, node="MossPhotonSamplerAdvanced"):
    """Initial noise for ``latent_image`` (shape ``[B, C, H, W]``, CPU float32 result cast to the latent dtype).

    ``native`` delegates to ``comfy.sample.prepare_noise(latent_image, seed, batch_index)``. ``seed_plus_i`` builds
    item ``i`` from ``seed + i`` and rejects ``batch_index`` and nested latents with ``unsupported_feature``.
    Unknown policy: ``ValueError`` (the widget is a combo, so this is a programmer error).
    """
    _check_policy(policy)
    if policy == "native":
        import comfy.sample
        return comfy.sample.prepare_noise(latent_image, seed, batch_index)
    if batch_index is not None or getattr(latent_image, "is_nested", False):
        what = "a batch_index" if batch_index is not None else "a nested latent"
        errors.raise_first_error([errors.make_issue(
            "unsupported_feature", node, "noise_policy", "noise policy seed_plus_i does not support " + what,
            "use noise policy native for this latent", severity="error")])
    import torch
    items = [own_generator_noise((1,) + tuple(latent_image.shape[1:]), latent_image.dtype, subseed(seed, i))
             for i in range(int(latent_image.shape[0]))]
    return torch.cat(items, dim=0)


def tensor_sha256(t):
    """sha256 hex of the float32 CPU bytes of a tensor (no RNG; bf16/fp16 are widened first)."""
    import hashlib
    import torch
    data = t.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()
