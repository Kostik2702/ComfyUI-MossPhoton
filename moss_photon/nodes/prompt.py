"""MossPhotonPromptLayout: structured prompt fields + optional MOSS layout -> CONDITIONING x2 + PHOTON_CONSTRAINTS
(design 4.3).

No comfy import at all: the ``clip`` object is only called (``tokenize``, ``encode_from_tokens_scheduled``), exactly
as ``CLIPTextEncode.encode`` does (``CUI/nodes.py:75-79``), so the conditioning of a prompt without blocks and
layout is the stock node's.
"""
import os
import sys

from ..core import constraints
from ..core.consts import CAT_SETUP
from ..core.errors import MissingRequiredInput, make_issue
from ..core.fingerprint import sha256_file, text_encoder_path_of

_NODE = constraints.NODE

# ASSUMED: values of text_encoder.fingerprint_source (the design names the key, not the values).
SOURCE_PATH_ONLY = "path_only"
SOURCE_SHA256 = "sha256_file"


def _interrupt_check():
    """ComfyUI's interrupt poll when ComfyUI is running, else ``None``; never imports comfy (unit tests, R1)."""
    mm = sys.modules.get("comfy.model_management")
    return getattr(mm, "throw_exception_if_processing_interrupted", None)


def _text_encoder_info(clip, fingerprint):
    path = text_encoder_path_of(clip)
    if path is None:
        return None
    info = {"basename": os.path.basename(path), "size": os.path.getsize(path), "sha256": None,
            "official": None, "fingerprint_source": SOURCE_PATH_ONLY}
    if fingerprint:
        info["sha256"] = sha256_file(path, interrupt_check=_interrupt_check())
        if info["sha256"] is not None:
            info["fingerprint_source"] = SOURCE_SHA256
    return info


def _encode(clip, text):
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


def _summary(c):
    """UI line without any prompt text (R5): counts and attribute names only."""
    blocks = ",".join(n for n in constraints.BLOCK_NAMES if c["blocks_present"][n]) or "none"
    layout = c["layout"]["items"] if c["layout"] else "no"
    names = ",".join(x["attribute"] for x in c["conflicts"]) or "none"
    return "blocks: %s | layout: %s | conflicts: %s | compiled length: %d" % (
        blocks, layout, names, len(c["compiled_positive"]))


class MossPhotonPromptLayout:
    """Compile structured fields into the MOSS prompt (user text first and verbatim) and carry
    ``forbidden_changes`` only in PHOTON_CONSTRAINTS (OD1)."""

    CATEGORY = CAT_SETUP
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "PHOTON_CONSTRAINTS")
    RETURN_NAMES = ("positive", "negative", "constraints")
    DESCRIPTION = (
        "Encodes the prompt with the native text-encode path. Your text comes first, verbatim; filled blocks "
        "(identity, wardrobe, scene, pose, camera, lighting) follow as labelled lines in a fixed order and nothing "
        "else is ever added. forbidden_changes is NOT put into the prompt: it only travels in PHOTON_CONSTRAINTS. "
        "layout_json (experimental) takes MOSS boxes in abs_centralized pixels, yxyx.")

    OUTPUT_TOOLTIPS = (
        "Positive conditioning: your text first, then the filled blocks as labelled lines, encoded by the native path.",
        "Negative conditioning: the negative text encoded as is.",
        "Private constraints for the sampler: the texts, the compiled prompt, the block flags, conflicts and the "
        "text-encoder identity. Never written to files; connect it to the sampler only.")

    @classmethod
    def INPUT_TYPES(cls):
        def text(tooltip):
            return {"multiline": True, "default": "", "tooltip": tooltip}

        def block(name, line):
            return text("%s block. Added after your text as a labelled line when filled; %s" % (name, line))

        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The MOSS text encoder (CLIP loader, type moss_image)."}),
                "positive": ("STRING", text("Your prompt. It comes first and is sent verbatim.")),
                "negative": ("STRING", text("Negative prompt, encoded as is (no block or layout is added).")),
            },
            "optional": {
                "identity": ("STRING", block("Identity", "who or what the subject is.")),
                "wardrobe": ("STRING", block("Wardrobe", "clothing; a colour that differs from your text "
                                                          "is reported as a conflict.")),
                "scene": ("STRING", block("Scene", "place and surroundings.")),
                "pose": ("STRING", block("Pose", "body position and expression.")),
                "camera": ("STRING", block("Camera", "framing, lens and angle.")),
                "lighting": ("STRING", block("Lighting", "light direction, quality and mood.")),
                "forbidden_changes": ("STRING", text(
                    "Things that must not change. NOT added to the prompt: it only travels in the constraints "
                    "output for the sampler's checks.")),
                "layout_json": ("STRING", text(
                    "Experimental. MOSS layout JSON with boxes in abs_centralized pixels, yxyx. Replaces the "
                    "compiled prompt when it has items.")),
                "fingerprint_text_encoder": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also hash the text-encoder file (sha256, slow on a cache miss, cancellable) so the "
                               "report can name it. Off = file name and size only."}),
            },
        }

    def encode(self, clip, positive, negative, identity="", wardrobe="", scene="", pose="", camera="", lighting="",
               forbidden_changes="", layout_json="", fingerprint_text_encoder=False):
        """Returns ``{"ui": {"text": [summary]}, "result": (positive, negative, constraints)}``.

        Raises ``MissingRequiredInput`` for a missing clip and ``InvalidLayout`` (naming the field path) for a bad
        ``layout_json``.
        """
        if clip is None:
            raise MissingRequiredInput(make_issue(
                "missing_required_input", _NODE, "clip", "clip input is None",
                "connect a CLIP from a text encoder loader (type moss_image)"))
        positive = positive or ""
        negative = negative or ""
        blocks = {"identity": identity, "wardrobe": wardrobe, "scene": scene, "pose": pose, "camera": camera,
                  "lighting": lighting, "forbidden_changes": forbidden_changes}
        blocks = {name: (text or "") for name, text in blocks.items()}

        compiled = constraints.compile_prompt(positive, blocks)
        layout = None
        if layout_json and layout_json.strip():
            layout = constraints.parse_layout(layout_json, compiled_text=compiled)

        # OD1: forbidden_changes is not a conflict source, it is not compiled either.
        sources = {"text": positive}
        sources.update({n: blocks[n] for n in constraints.COMPILED_BLOCKS if blocks[n].strip()})
        found = constraints.detect_conflicts(sources)

        te_info = _text_encoder_info(clip, bool(fingerprint_text_encoder))
        result = constraints.build_constraints(positive, negative, blocks, layout, found, te_info)
        if te_info is not None and fingerprint_text_encoder and te_info["sha256"] is None:
            result["warnings"].append(make_issue(
                "fingerprint_unavailable", _NODE, "text_encoder",
                "sha256 of %s could not be computed; text_encoder is reported by path only" % te_info["basename"],
                "check that the text encoder file is readable, then rerun").to_dict())
        pos_cond = _encode(clip, result["compiled_positive"])
        neg_cond = _encode(clip, negative)
        return {"ui": {"text": [_summary(result)]}, "result": (pos_cond, neg_cond, result)}


NODE_CLASS_MAPPINGS = {"MossPhotonPromptLayout": MossPhotonPromptLayout}
NODE_DISPLAY_NAME_MAPPINGS = {"MossPhotonPromptLayout": "Moss Photon Prompt & Layout"}
