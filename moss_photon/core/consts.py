"""Shared constants (design 1.1, Wave 0).

Leaf module: no imports at all (design 7.3). ``REASON_CODES`` lives in ``errors.py``.
"""

PACK_NAME = "ComfyUI-MossPhoton"
# A unit test asserts equality with pyproject.toml.
PACK_VERSION = "0.2.0"

REPORT_SCHEMA = 1
PRESET_SCHEMA = 1
SCHEDULE_INFO_SCHEMA = 1
CONSTRAINTS_SCHEMA = 1
IDENTITY_PROFILE_SCHEMA = 1
COMPILER_VERSION = 1

# The eight stable node IDs, SPEC 4.1 order.
NODE_IDS = (
    "MossPhotonSampler",
    "MossPhotonSamplerAdvanced",
    "MossPhotonSchedule",
    "MossPhotonLoRAProfile",
    "MossPhotonPromptLayout",
    "MossPhotonEmptyLatent",
    "MossPhotonDiagnostics",
    "MossPhotonSaveImage",
)

CAT_SETUP = "MossPhoton/1 Setup"
CAT_SAMPLING = "MossPhoton/2 Sampling"
CAT_DIAGNOSTICS = "MossPhoton/3 Diagnostics"
CAT_OUTPUT = "MossPhoton/4 Output"

# ModelPatcher attachment key (design 7.2, rows MP-10).
ATTACH_LORA_STACK = "moss_photon.lora_stack"

# SPEC 6.1 order.
PRESET_IDS = (
    "reference",
    "photo_safe",
    "identity_safe",
    "wardrobe_safe",
    "scene_safe",
    "night_photo",
    "draft_fast",
    "advanced",
)

# Turbo presets (docs/03-turbo.md A1); PRESET_IDS stays the 8 of SPEC 6.1.
TURBO_PRESET_IDS = (
    "turbo_8step",
    "turbo_identity",
)

# The only weight dtype of the verified reference; CPU tests override it per call.
REFERENCE_WEIGHT_DTYPES = ("torch.bfloat16",)

# ASSUMED names (the design gives the values, not the identifiers).
# sha256 of the float32 little-endian array of the native 50-step sgm_uniform schedule at shift 5 (INV 1.3, design 2.6).
REFERENCE_SIGMAS_SHA256 = "9b2c66b3fb335d998fc03e73ddbafe997fe2d3cf3b187f51629bdf0332f5a536"
# sha256 of the official moss_image_sft_bf16.safetensors (INV 7, design 1.8; equals the HF LFS oid).
SFT_BF16_SHA256 = "f96738d8116d78b7316b57c663d8c98591ddafab7488f48e3099b7655e34ab6f"

# Known distill LoRAs, keyed by full sha256 (docs/03-turbo.md 5.4, research/spike_turbo/distill_headers.json).
# ``status``: verified_recipe (supported by the card recipe, gate pending) | incompatible (refuse with ``reason_code``).
KNOWN_DISTILL_LORAS = {
    "7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2": {
        "name": "lodestones-8-steps-dmd2-lora",
        "role": "distill",
        "status": "verified_recipe",
        "trained_on": "base",
        "recipe": {"steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "shift": 5.0, "strength": 1.0},
    },
    "4abb14e39c1a59018907581e609f43616ec6810126144ec2b1d2e6f8969628fe": {
        "name": "MOSS_Image_SFT_8_steps_dmd_v0.1",
        "role": "distill",
        "status": "incompatible",
        "trained_on": "sft",
        "reason": "data-space DMD with own sampler loop",
        "reason_code": "incompatible_lora",
    },
}
