"""Typed errors, reason codes and the Issue record (design section 3, Wave 0).

Leaf module: stdlib only, no comfy, no torch (design 7.3).

One hierarchy: every user-facing failure is a ``MossPhotonError`` carrying an ``Issue``. ``REASONS`` is the
registry of every code of 3.1 (errors, each with an exception class) and 3.3 (warnings/infos/statuses, no class,
never raisable). Codes are never renamed; new codes are appended by design amendment only (design 7.2).
"""
import dataclasses
from typing import NamedTuple, Optional

__all__ = [
    "SEVERITIES", "MAX_MESSAGE_CHARS", "Issue", "ReasonInfo", "REASONS", "REASON_CODES",
    "MossPhotonError", "ModelError", "UnsupportedModel", "InvalidModelSampling", "ConflictingModelPatch",
    "InputError", "InvalidLatent", "InvalidDimensions", "InvalidConditioning", "InvalidVae", "InvalidLayout",
    "MissingRequiredInput", "UnsupportedSourceInput", "ScheduleError", "InvalidSchedule", "LoraError",
    "MissingLora", "IncompatibleLora", "PresetError", "UnknownPreset", "InvalidPresetFile",
    "PresetVersionMismatch", "InvalidOverride", "CalibrationRequired", "BudgetExceeded", "DependencyError",
    "MissingGuardDependency", "SamplingError", "NonFiniteOutput", "UnsupportedFeature", "OutputError",
    "make_issue", "raise_first_error", "split_issues",
]

SEVERITIES = ("error", "warning", "info")

# Design 3.2: Issue.format() is one line of at most 400 characters.
MAX_MESSAGE_CHARS = 400

_PREFIX = "[MossPhoton] "


def _one_line(text):
    return " ".join(str(text).split())


@dataclasses.dataclass(frozen=True)
class Issue:
    """One problem or note. ``severity`` is ``error``, ``warning`` or ``info``.

    ``message`` states observed vs expected values; ``fix`` names a node or widget and an action;
    ``detail`` (JSON-safe dict, optional) carries machine-readable data. ``param = "(graph)"`` for graph-level
    problems.
    """

    code: str
    severity: str
    node: str
    param: str
    message: str
    fix: str
    detail: Optional[dict] = None

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError("Issue.severity must be one of %s, got %r" % (SEVERITIES, self.severity))

    def to_dict(self):
        """JSON-safe dict ``{code, severity, node, param, message, fix}``; ``detail`` only when set."""
        d = {"code": self.code, "severity": self.severity, "node": self.node, "param": self.param,
             "message": self.message, "fix": self.fix}
        if self.detail is not None:
            d["detail"] = self.detail
        return d

    def format(self):
        """``[MossPhoton] <node>/<param>: <message> -> <fix> [<code>]`` on one line, at most 400 characters.

        Whitespace runs (including newlines) collapse to one space. When the line is too long the message is
        shortened first (the fix text is the part the user acts on); if the fix alone does not fit, the body is cut.
        """
        head = "%s%s/%s: " % (_PREFIX, _one_line(self.node), _one_line(self.param))
        tail = " [%s]" % self.code
        message, fix = _one_line(self.message), _one_line(self.fix)
        body = "%s -> %s" % (message, fix)
        room = MAX_MESSAGE_CHARS - len(head) - len(tail)
        if len(body) > room:
            msg_room = room - len(" -> ") - len(fix)
            if msg_room >= 20:
                body = "%s... -> %s" % (message[:msg_room - 3], fix)
            else:
                body = body[:room - 3] + "..."
        return head + body + tail


class ReasonInfo(NamedTuple):
    """Registry row: ``(exception class or None, default severity, in SPEC 6.3)``."""

    exc: Optional[type]
    severity: str
    in_spec_6_3: bool


class MossPhotonError(Exception):
    """Base of every user-facing failure. ``str(e) == issue.format()``."""

    CODE = None

    def __init__(self, issue):
        if not isinstance(issue, Issue):
            raise TypeError("%s needs an Issue, got %s" % (type(self).__name__, type(issue).__name__))
        super().__init__(issue.format())
        self.issue = issue

    @property
    def code(self):
        return self.issue.code

    def to_dict(self):
        return self.issue.to_dict()


class ModelError(MossPhotonError):
    pass


class UnsupportedModel(ModelError):
    CODE = "unsupported_model"


class InvalidModelSampling(ModelError):
    CODE = "invalid_model_sampling"


class ConflictingModelPatch(ModelError):
    CODE = "conflicting_patch"


class InputError(MossPhotonError):
    pass


class InvalidLatent(InputError):
    CODE = "invalid_latent"


class InvalidDimensions(InputError):
    CODE = "invalid_dimensions"


class InvalidConditioning(InputError):
    CODE = "invalid_conditioning"


class InvalidVae(InputError):
    CODE = "invalid_vae"


class InvalidLayout(InputError):
    CODE = "invalid_layout"


class MissingRequiredInput(InputError):
    CODE = "missing_required_input"


class UnsupportedSourceInput(InputError):
    CODE = "unsupported_source_input"


class ScheduleError(MossPhotonError):
    pass


class InvalidSchedule(ScheduleError):
    CODE = "invalid_schedule"


class LoraError(MossPhotonError):
    pass


class MissingLora(LoraError):
    CODE = "missing_lora"


class IncompatibleLora(LoraError):
    CODE = "incompatible_lora"


class PresetError(MossPhotonError):
    pass


class UnknownPreset(PresetError):
    CODE = "invalid_preset"


class InvalidPresetFile(PresetError):
    CODE = "invalid_preset_file"


class PresetVersionMismatch(PresetError):
    CODE = "preset_version_mismatch"


class InvalidOverride(PresetError):
    CODE = "invalid_override"


class CalibrationRequired(PresetError):
    CODE = "calibration_required"


class BudgetExceeded(MossPhotonError):
    CODE = "budget_exceeded"


class DependencyError(MossPhotonError):
    pass


class MissingGuardDependency(DependencyError):
    CODE = "missing_guard_dependency"


class SamplingError(MossPhotonError):
    pass


class NonFiniteOutput(SamplingError):
    CODE = "non_finite_output"


class UnsupportedFeature(MossPhotonError):
    CODE = "unsupported_feature"


class OutputError(MossPhotonError):
    CODE = "output_error"


# SPEC 6.3 typed reasons (seven): the three of them that stop the run are flagged by the design table too.
_SPEC_6_3 = frozenset({
    "unsupported_model", "missing_lora", "incompatible_lora", "missing_guard_dependency",
    "unscorable_face", "budget_exceeded", "invalid_schedule",
})

_ERROR_CLASSES = (
    UnsupportedModel, InvalidModelSampling, ConflictingModelPatch,
    InvalidLatent, InvalidDimensions, InvalidConditioning, InvalidVae, InvalidLayout,
    MissingRequiredInput, UnsupportedSourceInput,
    InvalidSchedule, MissingLora, IncompatibleLora,
    UnknownPreset, InvalidPresetFile, PresetVersionMismatch, InvalidOverride, CalibrationRequired,
    BudgetExceeded, MissingGuardDependency, NonFiniteOutput, UnsupportedFeature, OutputError,
)

# Design 3.3: code -> default severity. These codes have no exception class and can never be raised.
_STATUS_SEVERITY = (
    ("unscorable_face", "info"), ("guard_not_run", "info"), ("unverified_checkpoint", "warning"),
    ("family_not_verified", "info"), ("quantized_checkpoint_unverified", "warning"),
    ("fingerprint_unavailable", "warning"), ("manifest_missing", "warning"), ("customized_run", "info"),
    ("shift_differs_from_preset", "warning"), ("custom_sigmas", "warning"), ("sigma_step_ratio", "warning"),
    ("multiple_cond_entries", "warning"), ("latent_content_ignored", "info"),
    ("img2img_not_verified", "warning"), ("external_calc_cond_batch", "warning"), ("seed_wrapped", "info"),
    ("forward_count_mismatch", "warning"), ("megapixels_out_of_documented_range", "warning"),
    ("partial_lora_coverage", "warning"), ("lora_low_coverage_forced", "warning"),
    ("lora_base_mismatch", "warning"), ("lora_layers_missing", "info"),
    ("lora_strength_outside_verified_range", "warning"), ("te_keys_ignored", "warning"),
    ("te_lora_applied", "warning"), ("foreign_patches", "warning"), ("foreign_modifiers", "warning"),
    ("preset_unpinned", "warning"), ("latent_rescaled_by_native_fix", "info"),
    ("multiple_character_loras", "warning"), ("duplicate_lora", "warning"),
    ("trigger_not_in_prompt", "warning"), ("layout_experimental", "info"), ("attribute_conflict", "warning"),
    ("preview_only_preset", "info"), ("non_default_attention_backend", "info"),
    ("shift_input_ignored", "info"), ("zero_strength_lora_skipped", "info"),
    ("reference_photos_ignored", "warning"), ("embedded_unsupported_for_format", "warning"),
    # Turbo amendments (docs/03-turbo.md 5.3, MP-24). lora_base_mismatch above is reused: it is emitted with
    # severity="info" through make_issue on an SFT model with a Base-trained distill LoRA.
    ("turbo_setting_outside_verified", "warning"), ("distill_lora_non_turbo", "warning"),
    ("preset_lora_roles_exceeded", "warning"), ("negative_unused_cfg1", "info"),
    ("slg_ineffective_cfg1", "info"), ("unverified_distill_lora", "warning"),
    # MP-32 (MP-30 review): M6 a recommends_input constraint was reported with the unregistered literal code "none";
    # M7 the preset's model_families gate; M2 SLG on a turbo preset (MossPhotonSamplerAdvanced only).
    ("reference_photos_recommended", "info"), ("model_family_unverified", "warning"),
    ("turbo_slg_experimental", "warning"),
    # MP-16: phased LoRA weight by sigma progress (MossPhotonSamplerAdvanced only), experimental.
    ("phase_lora_experimental", "warning"),
    # MP-50: a file the pack knows as a distill LoRA (consts.KNOWN_DISTILL_LORAS) loaded with another role.
    ("distill_lora_role_mismatch", "warning"),
)


def _build_registry():
    reg = {}
    for cls in _ERROR_CLASSES:
        reg[cls.CODE] = ReasonInfo(cls, "error", cls.CODE in _SPEC_6_3)
    for code, severity in _STATUS_SEVERITY:
        reg[code] = ReasonInfo(None, severity, code in _SPEC_6_3)
    return reg


REASONS = _build_registry()

# Keys of REASONS (consts.py stays a leaf and does not carry this).
REASON_CODES = tuple(REASONS)


def make_issue(code, node, param, message, fix, *, severity=None, detail=None):
    """Build an ``Issue`` for a registered code.

    ``severity`` defaults to the registry value. Raises ``ValueError`` (programmer error) for an unknown code, an
    unknown severity, or ``severity="error"`` on a code that has no exception class (design 3.1).
    """
    info = REASONS.get(code)
    if info is None:
        raise ValueError("unknown reason code %r" % (code,))
    if severity is None:
        severity = info.severity
    if severity not in SEVERITIES:
        raise ValueError("severity must be one of %s, got %r" % (SEVERITIES, severity))
    if severity == "error" and info.exc is None:
        raise ValueError("code %r is a status code and cannot have severity 'error'" % (code,))
    return Issue(code, severity, node, param, message, fix, detail)


def raise_first_error(issues):
    """Raise the registered exception for the first ``severity == "error"`` issue (list order); else return None."""
    for issue in issues:
        if issue.severity == "error":
            info = REASONS.get(issue.code)
            cls = info.exc if info is not None and info.exc is not None else MossPhotonError
            raise cls(issue)
    return None


def split_issues(issues):
    """Return ``(errors, others)`` as lists, each in input order."""
    errs, others = [], []
    for issue in issues:
        (errs if issue.severity == "error" else others).append(issue)
    return errs, others
