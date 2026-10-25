"""Builders for identity-profile tests without ComfyUI: a full-coverage mapping and stack entries."""
from moss_photon.core import lora

from . import golden_keys as gk

SHA_A = "a" * 64
SHA_B = "b" * 64
DMD2 = "7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2"
SFT_DMD = "4abb14e39c1a59018907581e609f43616ec6810126144ec2b1d2e6f8969628fe"
SFT_BF16 = "f96738d8116d78b7316b57c663d8c98591ddafab7488f48e3099b7655e34ab6f"


def full_mapping():
    """Split-qkv PEFT-shaped: 504 tensors, 252 modules all mapped, 168 distinct keys, 42/42 layers."""
    keys = gk.peft_split()
    accepted = []
    for n in gk.ALL42:
        accepted += [("diffusion_model.layers.%d.proj.qkv.weight" % n, (0, o * 8, 8)) for o in range(3)]
        accepted += ["diffusion_model.layers.%d.proj.%s.weight" % (n, t) for t in ("o", "ffn_up_gate", "ffn_down")]
    return lora.summarize_mapping(keys, accepted_model=accepted, accepted_te=[], mapper_names=set(lora.weight_modules(lora.group_lora_modules(keys))),
                                  num_layers=42, te_active=False)


def entry(index=0, role="character", basename="character_a.safetensors", sha=SHA_A, strength=0.8, size=148697664,
          verified=None, trained_on="unknown", base_match=None, hints=(), warnings=(), mapping=None, sidecar=None,
          strength_clip=0.0, metadata=None, allow_low=False):
    mapping = mapping or full_mapping()
    if sidecar is None:
        sidecar = lora.sidecar_block(None)
        if verified is not None:
            sidecar = dict(sidecar, verified_strength=dict(verified), found=True, curated=True)
    return lora.build_stack_entry(
        index=index, role=role, lora={"basename": basename, "size": size, "sha256": sha, "metadata": metadata or {}},
        strength_model=strength, strength_clip=strength_clip, mapping=mapping,
        verdict=lora.coverage_verdict(mapping.modules_total, mapping.modules_mapped, allow_low=allow_low), trained_on=trained_on,
        base_match=base_match, sidecar=sidecar, trigger_hints=list(hints), warnings=list(warnings))


def profile_of(stack, *, uuid="u-1", baseline=0, model_identity=None, total=None):
    entries = [lora.attachment_entry_of(e) for e in stack]
    if total is None:
        total = baseline + sum(e["patch_entries"] for e in stack)
    att = lora.build_attachment(entries, patches_uuid=uuid, patch_entries_total=total, baseline_entries=baseline)
    return lora.build_identity_profile(stack, attachment=att, model_identity=model_identity)
