"""The file names the shipped /object_info snapshot may contain (MP-37; stdlib only).

The snapshot is recorded from a running ComfyUI, whose model COMBO lists are the file lists of that machine. They must
not ship: ``research/refresh_object_info_snapshot.py`` replaces every file-list COMBO below with the allowlist, and
``test_workflows.py`` fails when a snapshot holds any other file name. This module is the one place the lists live.
The example workflows may use exactly these values (a LoRA/text-encoder name that is not on the user's disk is shown
as "value not in list" until the user picks theirs; ``docs/INSTALL.md``, section 3).
"""

# (node type, widget) -> the only values the snapshot may list.
FILE_WIDGET_ALLOWLIST = {
    ("UNETLoader", "unet_name"): ("moss_image_sft_bf16.safetensors",),
    ("CLIPLoader", "clip_name"): ("qwen3.5_9b_bf16.safetensors",),
    ("VAELoader", "vae_name"): ("flux2-vae.safetensors",),
    ("MossPhotonLoRAProfile", "lora_name"): (
        "none", "lodestones-8-steps-dmd2-lora.safetensors", "your_character_lora.safetensors",
        "your_second_lora.safetensors"),
}

# Any other COMBO holding a value that looks like a model file is a leak of a list nobody sanitized.
MODEL_FILE_SUFFIXES = (".safetensors", ".sft", ".pt", ".pth", ".ckpt", ".bin", ".gguf", ".onnx")


def looks_like_model_file(value):
    return isinstance(value, str) and value.lower().endswith(MODEL_FILE_SUFFIXES)


def combo_values(spec):
    """The choice list of a widget spec in either /object_info shape, or None when it is not a COMBO."""
    if not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options", [])
    return None


def leaks(snapshot):
    """[(node, widget, value)] of every file-list value outside the allowlist, in a snapshot dict."""
    found = []
    for node, info in snapshot["nodes"].items():
        for section in info["input"].values():
            for widget, spec in section.items():
                values = combo_values(spec)
                if values is None:
                    continue
                allowed = FILE_WIDGET_ALLOWLIST.get((node, widget))
                for value in values:
                    if allowed is not None:
                        if value not in allowed:
                            found.append((node, widget, value))
                    elif looks_like_model_file(value):
                        found.append((node, widget, value))
    return found


def sanitize(snapshot):
    """Replace each allowlisted file-list COMBO in place by its allowlist; returns the snapshot."""
    for (node, widget), allowed in FILE_WIDGET_ALLOWLIST.items():
        info = snapshot["nodes"].get(node)
        if info is None:
            continue
        for section in info["input"].values():
            spec = section.get(widget)
            if spec is None:
                continue
            if isinstance(spec[0], list):
                spec[0] = list(allowed)
            elif spec[0] == "COMBO":
                spec[1]["options"] = list(allowed)
    return snapshot
