"""LoRA plan by sampling phase (MP-84, v2): the data side of MossPhotonLoRAPhase; pure, no comfy, no torch.

The MossPhotonLoRAPhase node patches NOTHING. It clones the MODEL and appends one entry ``{lora_name, strength, phase,
basename, size, sha256, warnings}`` to a plan kept in ``model_options`` (``ModelPatcher.clone`` deep-copies the options, so a
plan never leaks into the MODEL it came from, and the entries are small: no tensor lives in them). The samplers read the
plan with ``entries_of``, load and map each file through the injected loader (``core/lora.plan_patches``) and apply the
resolved entries on DIRECT clones of the incoming MODEL (``apply_to``).

Phase -> darkroom phases (``core/darkroom.py`` phases by sigma progress ``r``; the boundaries are the preset's ``r_c`` /
``r_k`` / ``r_f``, this module invents none):

    all                    compose + commit + hold + finish   (what an ordinary LoRA loader does)
    composition            compose
    identity               compose + commit + hold + finish   (the character phase; takes the commit boost)
    texture                finish
    composition+identity   compose + commit + hold            (no boost: the place of style and slider LoRAs)

``identity`` starts at the first step on MOSS (KreaPhoton starts it at sigma 0.85): MP-85 showed that the compose segment sets
hair and silhouette, MP-53 that relaxing the character late costs identity. Only ``identity`` takes the commit boost; ``all``
is exactly an ordinary loader and ``composition+identity`` takes none either: the boost is ADDITIVE (``w_commit - w_compose``),
right for a character LoRA near 1.0 and wrong for a slider (-2 would become -1.65) or a style LoRA (.65 would become 1.0).

A MODEL patched by a stock ``LoraLoader`` carries no plan entry: it is phase ``all`` by construction. A preset without
Darkroom phases applies every entry for the whole run at its constant strength (an entry that does not cover the whole run is named
in the warning ``phase_plan_ignored``; ``identity`` acts in every segment, so it is not named). The Darkroom commit boost (``w_compose -> w_commit``) is added to the strength of the
entries whose phase is in ``BOOSTED_PHASES`` (``takes_boost``) in the segments it acts on; a known distill LoRA
(``consts.KNOWN_DISTILL_LORAS``) never takes it.

Imports: stdlib and ``consts`` / ``errors``.
"""
import dataclasses
import math

from . import consts, errors

__all__ = ["PLAN_KEY", "PHASES", "PHASE_SEGMENTS", "BOOSTED_PHASES", "STRENGTH_LIMIT", "Resolved", "make_entry",
           "add_to_plan", "entries_of", "entries_before_patches", "active", "takes_boost", "boost_indices", "negative_boost_indices", "apply_to",
           "covers_whole_run"]

PLAN_KEY = consts.LORA_PLAN_KEY      # ModelPatcher.model_options key of the plan (a list of entry dicts)
PHASES = consts.LORA_PHASES
STRENGTH_LIMIT = 20.0                # the bound of the node's strength widget (the same as KreaPhoton's)
NODE = "MossPhotonLoRAPhase"

# phase -> darkroom phase names (``darkroom.PHASE_NAMES``); a unit test cross-checks the names.
PHASE_SEGMENTS = {
    "all": ("compose", "commit", "hold", "finish"),
    "composition": ("compose",),
    "identity": ("compose", "commit", "hold", "finish"),
    "texture": ("finish",),
    "composition+identity": ("compose", "commit", "hold"),
}
# the phases whose entries take the Darkroom commit boost; each must act in the commit (a unit test cross-checks it).
# composition+identity is out: the boost is additive and that phase is where style and slider LoRAs go (A/B research/v2_normalize_ab).
BOOSTED_PHASES = ("identity",)

@dataclasses.dataclass(frozen=True)
class Resolved:
    """A plan entry with its mapped patches, as the samplers apply it. ``patch_dict`` is the shared mapper output (never
    modified); ``index`` is the position in the plan, which is the apply order."""

    index: int
    basename: str
    size: int
    sha256: object
    strength: float
    phase: str
    patch_dict: dict
    metadata: object = None

    def describe(self, applied_in, model="main"):
        """The ``lora.plan[]`` report object; ``applied_in`` lists the segments the entry ran in (``["run"]`` when constant);
        ``model`` is the MODEL whose plan the entry came from (``main``, or ``finish`` for the finish_model)."""
        return {"basename": self.basename, "size": self.size, "sha256": self.sha256, "strength": self.strength,
                "phase": self.phase, "applied_in": list(applied_in), "model": model}


def covers_whole_run(phase):
    """True when ``phase`` acts in every segment of a run (``all`` and ``identity``): read from ``PHASE_SEGMENTS``."""
    return set(PHASE_SEGMENTS[phase]) == set(PHASE_SEGMENTS["all"])


def make_entry(lora_name, strength, phase, basename, size, sha256, warnings=()):
    """A validated plan entry (a plain JSON-safe dict). Raises ``invalid_override`` for a phase outside ``PHASES`` and for a
    strength that is not a finite number within +-``STRENGTH_LIMIT`` (a combo and a bounded widget cannot produce either)."""
    if phase not in PHASE_SEGMENTS:
        errors.raise_first_error([errors.make_issue(
            "invalid_override", NODE, "phase", "phase %r is not one of %s" % (phase, ", ".join(PHASES)),
            "choose one of: %s" % ", ".join(PHASES), severity="error")])
    if (isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength)
            or abs(strength) > STRENGTH_LIMIT):
        errors.raise_first_error([errors.make_issue(
            "invalid_override", NODE, "strength", "strength %r is not a finite number within +-%g" % (strength, STRENGTH_LIMIT),
            "set strength between -%g and %g" % (STRENGTH_LIMIT, STRENGTH_LIMIT), severity="error")])
    return {"lora_name": str(lora_name), "strength": float(strength), "phase": str(phase), "basename": str(basename),
            "size": int(size), "sha256": sha256, "warnings": [dict(w) for w in warnings]}


def entries_of(model):
    """The plan of ``model`` as a tuple of entry dicts (copies); ``()`` for a MODEL without a plan or without options."""
    options = getattr(model, "model_options", None) or {}
    plan = options.get(PLAN_KEY) or ()
    return tuple(dict(entry, warnings=[dict(w) for w in entry.get("warnings") or ()]) for entry in plan)


def add_to_plan(model, entry):
    """A CLONE of ``model`` whose plan is the plan of ``model`` plus ``entry``; ``model`` is not modified.

    ``clone()`` deep-copies ``model_options``, so the list written here belongs to the clone alone."""
    clone = model.clone()
    plan = list(clone.model_options.get(PLAN_KEY) or ())
    # the patch set the MODEL had when the entry was recorded: the sampler applies every entry on top of ALL patches of the
    # MODEL, so a loader placed after this node in the graph is applied BEFORE the entry (``entries_before_patches``)
    plan.append(dict(entry, patches_uuid=str(getattr(clone, "patches_uuid", ""))))
    clone.model_options[PLAN_KEY] = plan
    return clone


def entries_before_patches(model, entries):
    """Positions in ``entries`` (``entries_of(model)``) that were recorded when the MODEL carried other patches than it carries
    now: an ordinary loader ran after the node, so the entry is applied after that loader, not before it as in the graph.
    An entry without the record (built by hand) is never reported."""
    current = str(getattr(model, "patches_uuid", ""))
    return tuple(i for i, entry in enumerate(entries) if entry.get("patches_uuid") not in (None, current))


def active(entries, darkroom_phase):
    """Positions in ``entries`` (resolved or plain) that act in the darkroom phase ``darkroom_phase``, in apply order. An
    entry with strength 0 never acts."""
    return tuple(i for i, entry in enumerate(entries)
                 if _strength(entry) != 0.0 and darkroom_phase in PHASE_SEGMENTS[_phase(entry)])


def takes_boost(entry):
    """True when the Darkroom commit boost is added to ``entry`` (resolved or plain): its strength is positive (the boost is
    additive: -2 would become -1.65 and -0.2 would flip to +0.15, and ``identity`` is the node's default phase), its phase is
    in ``BOOSTED_PHASES`` (identity) and it is not a known distill LoRA (the boost would change the distill recipe)."""
    if _strength(entry) <= 0.0 or _phase(entry) not in BOOSTED_PHASES:
        return False
    return not _is_known_distill(entry)


def boost_indices(entries):
    """Positions of the entries that take the Darkroom commit boost (``takes_boost``)."""
    return tuple(i for i, entry in enumerate(entries) if takes_boost(entry))


def negative_boost_indices(entries):
    """Positions of the entries in a boosted phase that take no boost only because their strength is negative (the run names
    them in a ``darkroom_boost_skipped`` warning). A known distill LoRA is not one of them: it never takes the boost."""
    return tuple(i for i, entry in enumerate(entries)
                 if _strength(entry) < 0.0 and _phase(entry) in BOOSTED_PHASES and not _is_known_distill(entry))


def apply_to(model, entries, darkroom_phase=None, boost=0.0):
    """Add the entries that act in ``darkroom_phase`` to ``model`` (a clone the caller owns), in plan order, with comfy's
    ordinary ``add_patches`` (the call ``LoraLoaderModelOnly`` makes). ``darkroom_phase=None`` = every entry (a constant run).
    ``boost`` is added to the strength of the entries that ``takes_boost``. Returns ``model``."""
    for entry in entries:
        if _strength(entry) == 0.0:
            continue
        if darkroom_phase is not None and darkroom_phase not in PHASE_SEGMENTS[_phase(entry)]:
            continue
        strength = entry.strength + (boost if takes_boost(entry) else 0.0)
        model.add_patches(entry.patch_dict, strength)
        if entry.metadata:
            model.set_attachments("lora_metadata", entry.metadata)
    return model


def _strength(entry):
    return float(entry["strength"] if isinstance(entry, dict) else entry.strength)


def _phase(entry):
    return entry["phase"] if isinstance(entry, dict) else entry.phase


def _is_known_distill(entry):
    sha256 = entry.get("sha256") if isinstance(entry, dict) else entry.sha256
    return isinstance(sha256, str) and sha256 in consts.KNOWN_DISTILL_LORAS
