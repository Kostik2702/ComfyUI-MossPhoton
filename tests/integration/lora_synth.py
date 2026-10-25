"""Synthetic LoRAs for the tiny 24-layer MOSS (design 6.5): files on disk, ``folder_paths`` pointed at them.

Keys are built from the tiny model's own state dict, so the native mapper maps them exactly as it maps a real MOSS
LoRA: ``diffusion_model.layers.N.proj.{qkv,o,...}`` with ``lora_A/lora_B`` (PEFT, fused qkv) or
``lora_down/lora_up/alpha`` (kohya suffixes, split q/k/v like the 8-step distill files). Weights are seeded random,
so a LoRA changes the result and the same seed gives the same file.

``install(testcase)`` registers every cleanup on the test case: the temp folder, the ``folder_paths`` patch and an
in-memory hash cache (no file is written into ComfyUI's user directory).
"""
import hashlib
import os
import tempfile
import types
from unittest import mock

from . import _comfy_env

PEFT_FUSED = ("qkv", "o")
KOHYA_SPLIT = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


def _weight_shape(state_dict, layer, module):
    """(out, in) of the model weight a LoRA module acts on; q/k/v are the thirds of the fused qkv weight."""
    if module in ("q", "k", "v"):
        out, inp = state_dict["diffusion_model.layers.%d.proj.qkv.weight" % layer].shape
        return out // 3, inp
    out, inp = state_dict["diffusion_model.layers.%d.proj.%s.weight" % (layer, module)].shape
    return out, inp


def make_lora_sd(patcher, *, fmt="peft", modules=PEFT_FUSED, skip_layers=(), rank=4, seed=11, scale=0.5):
    """LoRA state dict for the tiny model. ``fmt``: ``peft`` (lora_A/lora_B) or ``kohya`` (lora_down/lora_up/alpha)."""
    import torch
    sd = patcher.model.state_dict()
    layers = len(patcher.model.diffusion_model.layers)
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for n in range(layers):
        if n in skip_layers:
            continue
        for m in modules:
            o, i = _weight_shape(sd, n, m)
            base = "diffusion_model.layers.%d.proj.%s" % (n, m)
            a = torch.randn(rank, i, generator=gen) * scale
            b = torch.randn(o, rank, generator=gen) * scale
            if fmt == "peft":
                out[base + ".lora_A.weight"], out[base + ".lora_B.weight"] = a, b
            else:
                out[base + ".lora_down.weight"], out[base + ".lora_up.weight"] = a, b
                out[base + ".alpha"] = torch.tensor(float(rank))
    return out


def add_bogus_modules(sd, count, rank=4):
    """``count`` extra modules the native mapper cannot place (a layer that does not exist)."""
    import torch
    for k in range(count):
        base = "diffusion_model.layers.%d.proj.qkv" % (900 + k)
        sd[base + ".lora_A.weight"] = torch.zeros(rank, 8)
        sd[base + ".lora_B.weight"] = torch.zeros(8, rank)
    return sd


def te_lora_sd(prefix="text_encoders.enc.proj", rank=2, dim=16, seed=5):
    """A text-encoder LoRA part for ``FakeClip`` (module ``enc.proj``)."""
    import torch
    gen = torch.Generator().manual_seed(seed)
    return {prefix + ".lora_A.weight": torch.randn(rank, dim, generator=gen),
            prefix + ".lora_B.weight": torch.randn(dim, rank, generator=gen)}


def sha256_of_tensors(sd):
    """Stable digest of a state dict (used to register a synthetic file in the known-distill table)."""
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode("utf-8"))
        h.update(sd[k].numpy().tobytes())
    return h.hexdigest()


class LoraEnv:
    """A temp loras folder plus the patched ``folder_paths`` and an in-memory hash cache."""

    def __init__(self, folder):
        self.folder = folder
        self.written = {}

    def write(self, name, sd, metadata=None):
        """Write ``sd`` as ``<folder>/<name>`` (safetensors, string metadata) and return the file's real sha256."""
        from safetensors.torch import save_file
        path = os.path.join(self.folder, name)
        save_file(sd, path, metadata={k: str(v) for k, v in (metadata or {}).items()})
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        self.written[name] = digest
        return digest

    def path(self, name):
        return os.path.join(self.folder, name)

    def resolve(self, kind, name):
        if kind != "loras":
            raise AssertionError("only the loras folder is faked, got %r" % (kind,))
        path = os.path.join(self.folder, name)
        if not os.path.isfile(path):
            raise FileNotFoundError("Model in folder 'loras' with filename '%s' not found." % name)
        return path


def install(testcase):
    """Point ``folder_paths`` at a fresh temp folder for one test case; everything is undone by the case's cleanups."""
    _comfy_env.setup()
    import folder_paths
    from moss_photon.core import fingerprint
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    env = LoraEnv(tmp.name)
    patcher = mock.patch.object(folder_paths, "get_full_path_or_raise", side_effect=env.resolve)
    patcher.start()
    testcase.addCleanup(patcher.stop)
    cache = mock.patch.object(fingerprint, "_default_cache", fingerprint.JsonCache(None))
    cache.start()
    testcase.addCleanup(cache.stop)
    return env


class FakeClip:
    """A CLIP stand-in: one linear module ``enc.proj`` wrapped in a real ModelPatcher, like ``comfy.sd.CLIP``."""

    def __init__(self):
        import torch
        import comfy.model_patcher

        class Enc(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = torch.nn.ModuleDict({"proj": torch.nn.Linear(16, 16)})

        self.cond_stage_model = Enc()
        self.patcher = comfy.model_patcher.ModelPatcher(
            self.cond_stage_model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))

    def clone(self):
        n = FakeClip.__new__(FakeClip)
        n.cond_stage_model = self.cond_stage_model
        n.patcher = self.patcher.clone()
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)


def run_node(model, lora_name, **kw):
    """Call the real node with defaults; returns ``(model, profile, clip)``."""
    from moss_photon.nodes import lora as node
    args = dict(lora_name=lora_name, strength_model=1.0, role="character", profile="constant")
    args.update(kw)
    return node.MossPhotonLoRAProfile().apply(model, **args)


def namespace(**kw):
    return types.SimpleNamespace(**kw)
