"""MossPhotonPromptSplit: the marker-delimited answer of a prompt enhancer -> the nine text inputs of
MossPhotonPromptLayout (format: ``docs/prompt-enhancer-system-prompt.md`` of the repository).

The enhancer writes ``<<<POSITIVE>>>``, ``<<<NEGATIVE>>>``, ``<<<IDENTITY>>>`` ... ``<<<FORBIDDEN_CHANGES>>>`` and a
closing ``<<<END>>>``, each followed by the field text. A broken answer is not an error (owner 2026-10-25: degrade and
warn): a missing marker gives an empty field, an answer without any marker goes to ``positive`` whole, and every such
fallback is listed in the ``report`` output. The report and the log name markers only, never prompt text (R5).
"""
import logging
import re

from ..core.consts import CAT_SETUP

_log = logging.getLogger("MossPhoton")

NODE_ID = "MossPhotonPromptSplit"

# Output order = input order of MossPhotonPromptLayout (positive, negative, then the blocks).
FIELDS = ("positive", "negative", "identity", "wardrobe", "scene", "pose", "camera", "lighting", "forbidden_changes")
END = "END"

# Tolerant on purpose: any case, spaces inside the brackets, a space instead of the underscore.
_MARKER_RE = re.compile(r"<<<[ \t]*([A-Za-z][A-Za-z_ ]*?)[ \t]*>>>")
_FENCE_RE = re.compile(r"^[ \t]*```[^\n]*$")
# The placeholders the system prompt forbids; a small LLM still writes them.
_PLACEHOLDERS = frozenset(("none", "n/a", "na", "-", "--", "empty", "null"))


def _drop_reasoning(text):
    """Text after the last ``</think>`` (reasoning models; the opening tag is sometimes omitted by the chat template),
    else the text unchanged. The reasoning may quote markers, so it must not be parsed."""
    cut = text.lower().rfind("</think>")
    return text if cut < 0 else text[cut + len("</think>"):]


def _clean(value):
    """Trim the value and drop code-fence lines at its ends; inner lines are kept verbatim."""
    lines = value.strip().split("\n")
    while lines and _FENCE_RE.match(lines[0]):
        lines.pop(0)
    while lines and _FENCE_RE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def split_fields(text):
    """Split an enhancer answer into ``FIELDS``.

    Returns ``(fields, warnings)``: ``fields`` maps every name of ``FIELDS`` to a string (``""`` when absent);
    ``warnings`` is a list of strings naming markers only. Text before the first marker and after ``<<<END>>>`` is
    ignored; a missing ``<<<END>>>`` is fine (the last field runs to the end). Never raises.
    """
    fields = {name: "" for name in FIELDS}
    warnings = []
    text = _drop_reasoning((text or "").replace("\r\n", "\n"))
    if not text.strip():
        return fields, warnings

    matches = list(_MARKER_RE.finditer(text))
    seen = set()
    for i, m in enumerate(matches):
        name = re.sub(r"[ _]+", "_", m.group(1).strip()).upper()
        if name == END:
            break
        key = name.lower()
        if key not in fields:
            warnings.append("unknown marker <<<%s>>> ignored" % name)
            continue
        if key in seen:
            warnings.append("marker <<<%s>>> repeated: the first one is used" % name)
            continue
        seen.add(key)
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = _clean(text[m.end():stop])
        if value.lower().strip(" .") in _PLACEHOLDERS:
            warnings.append("<<<%s>>> held a placeholder, left empty" % name)
            value = ""
        fields[key] = value

    if not seen:
        fields["positive"] = _clean(text)
        warnings.append("no field markers found: the whole text is sent as positive")
        return fields, warnings
    for key in FIELDS:
        if key not in seen:
            warnings.append("marker <<<%s>>> missing, left empty" % key.upper())
    return fields, warnings


def _summary(fields, warnings):
    filled = ",".join(n for n in FIELDS if fields[n]) or "none"
    return "filled: %s | warnings: %d" % (filled, len(warnings))


class MossPhotonPromptSplit:
    """Split a prompt-enhancer answer into the text inputs of Moss Photon Prompt & Layout."""

    CATEGORY = CAT_SETUP
    FUNCTION = "split"
    RETURN_TYPES = ("STRING",) * (len(FIELDS) + 1)
    RETURN_NAMES = FIELDS + ("report",)
    DESCRIPTION = (
        "Splits the answer of a prompt enhancer into the fields of Moss Photon Prompt & Layout. The answer marks each "
        "field with a line <<<POSITIVE>>>, <<<NEGATIVE>>>, <<<IDENTITY>>>, <<<WARDROBE>>>, <<<SCENE>>>, <<<POSE>>>, "
        "<<<CAMERA>>>, <<<LIGHTING>>>, <<<FORBIDDEN_CHANGES>>> and ends with <<<END>>>. Field text is passed "
        "unchanged; a missing marker gives an empty field and an answer without markers goes to positive whole, both "
        "listed in report.")

    OUTPUT_TOOLTIPS = tuple(
        "The %s field; connect it to the %s input of Moss Photon Prompt & Layout." % (n, n) for n in FIELDS) + (
        "What was repaired: missing, repeated or unknown markers, placeholders, no markers at all. 'ok' when nothing. "
        "Names markers only, never prompt text.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "The enhancer answer with <<<FIELD>>> markers. Reasoning before </think> is skipped."}),
            },
        }

    def split(self, text):
        """Returns ``{"ui": {"text": [summary]}, "result": (*fields in FIELDS order, report)}``."""
        fields, warnings = split_fields(text)
        for w in warnings:
            _log.warning("[%s] %s", NODE_ID, w)
        report = "\n".join(warnings) if warnings else "ok"
        return {"ui": {"text": [_summary(fields, warnings)]},
                "result": tuple(fields[n] for n in FIELDS) + (report,)}


NODE_CLASS_MAPPINGS = {NODE_ID: MossPhotonPromptSplit}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "Moss Photon Prompt Split"}
