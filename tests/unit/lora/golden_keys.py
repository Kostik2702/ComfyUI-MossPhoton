"""Golden LoRA key lists (names only) for the formats of the 8 real MOSS LoRA headers plus the 8-step distill files.

The counts are the ones measured on the real headers (design 1.9, check C6; docs/03-turbo.md 2): every generator
documents its tensor count and its weight-module count, and ``test_lora_real_headers.py`` checks the real files
against the same numbers.
"""

MODULES6 = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
MODULES4 = ("qkv", "o", "ffn_up_gate", "ffn_down")
ALL42 = tuple(range(42))
NO22 = tuple(i for i in range(42) if i != 22)  # 41 layers: the layer-22-less LoRAs (17 of 35 local files)


def peft_split():
    """character_a (split qkv): 504 tensors, 252 weight modules, no alpha."""
    return ["diffusion_model.layers.%d.proj.%s.lora_%s.weight" % (n, m, ab)
            for n in ALL42 for m in MODULES6 for ab in "AB"]


def peft_fused():
    """character_b_fused_qkv / lenovo_moss: fused qkv, layer 22 absent: 328 tensors, 164 modules."""
    return ["diffusion_model.layers.%d.proj.%s.lora_%s.weight" % (n, m, ab)
            for n in NO22 for m in MODULES4 for ab in "AB"]


def kohya():
    """MOSS_LUISAC_REALOGIC: lora_unet_ prefix, 41 layers x 6 modules x (alpha, down, up) = 738 tensors, 246 modules."""
    keys = []
    for n in NO22:
        for m in MODULES6:
            base = "lora_unet_layers_%d_proj_%s" % (n, m)
            keys += [base + ".alpha", base + ".lora_down.weight", base + ".lora_up.weight"]
    return keys


def lokr():
    """MOSSRealisticSnapshotV1: 574 tensors, 164 weight modules + 123 alpha-only modules (stray q/k/v .alpha)."""
    keys = []
    for n in NO22:
        for m in MODULES4:
            base = "diffusion_model.layers.%d.proj.%s" % (n, m)
            keys += [base + ".lokr_w1", base + ".lokr_w2"]
            if m != "qkv":
                keys.append(base + ".alpha")
        for m in ("q", "k", "v"):
            keys.append("diffusion_model.layers.%d.proj.%s.alpha" % (n, m))
    return keys


def diffusion_kohya():
    """The 4-step LoRA and both 8-step distill files: diffusion_model.<module>.{alpha,lora_down.weight,lora_up.weight},
    42 layers x 6 split modules: 756 tensors, 252 modules."""
    keys = []
    for n in ALL42:
        for m in MODULES6:
            base = "diffusion_model.layers.%d.proj.%s" % (n, m)
            keys += [base + ".alpha", base + ".lora_down.weight", base + ".lora_up.weight"]
    return keys


def other_architecture():
    """mossCINEMA-like (Z-Image layout): lora_A/B on attention.to_* and feed_forward.w*."""
    keys = []
    for n in range(30):
        for m in ("attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0",
                  "feed_forward.w1", "feed_forward.w2", "feed_forward.w3", "adaLN_modulation.1"):
            keys += ["diffusion_model.layers.%d.%s.lora_%s.weight" % (n, m, ab) for ab in "AB"]
    return keys


# name -> (generator, tensors, weight modules, key_format)
GOLDEN = {
    "character_a": (peft_split, 504, 252, "peft_split_qkv"),
    "character_b_fused_qkv": (peft_fused, 328, 164, "peft_fused_qkv"),
    "LUISAC": (kohya, 738, 246, "kohya"),
    "SnapshotV1": (lokr, 574, 164, "lokr"),
    "sft_4_steps": (diffusion_kohya, 756, 252, "kohya"),
    "mossCINEMA": (other_architecture, 480, 240, "peft_split_qkv"),
}
