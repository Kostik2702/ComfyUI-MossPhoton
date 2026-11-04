"""Generate docs/presets/<id>.md from presets/<id>.json (SPEC 6.4). Lives inside the package so
``--suite quality`` runs on a copy of the package alone; the development repository has a thin wrapper script.

    <embedded python> tests/preset_docs.py

Every number in a preset doc (effective parameters, bounds, budget, forwards, constraints, guards, pin) is read from the
preset file or from ``presets.effective_view``; the prose per preset lives in ``TEXT`` below. ``render(id, preset)`` is
pure and is also called by ``tests/quality/test_workflows.py``, which compares the files on disk with its output, so a
hand edit of a generated doc or a changed preset file without a re-run goes red.
Exit code 0 on success.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT_DIR = os.path.join(PKG, "docs", "presets")
sys.path.insert(0, PKG)

from moss_photon.core import consts  # noqa: E402
from moss_photon.core import presets as presets_mod  # noqa: E402

PENDING = "pending calibration"
EQUAL_SENTENCE = (
    "The numeric parameters of this preset are numerically equal to reference (same steps, sampler, scheduler, shift, "
    "CFG and SLG). What differs is the input contract and the report; a different result in quality is not claimed."
)

# purpose, inputs (extra to the five basic ones), limitations, status note. Plain statements with a source in SPEC 6.1/6.3,
# the maintainers' design documents; nothing here claims a gain.
TEXT = {
    "reference": {
        "purpose": "Regression and environment check, and the baseline every other preset is compared with. It "
                   "reproduces the official OpenMOSS ComfyUI recipe on the native path (SPEC 6.1, 3).",
        "inputs": ["A native MOSS model from `UNETLoader` (an adapter check refuses anything else: `unsupported_model`).",
                   "Optional: a LoRA profile (`use_explicit_user_profile_or_none`); with LoRAs the run is labelled as such "
                   "in the report."],
        "limits": ["Parity with the native ComfyUI graph is the design target (SPEC 3, 10); the GPU parity run has "
                   "not been done yet (" + PENDING + ").",
                   "Reference-only environment facts are recorded, not assumed: the text encoder file, the attention "
                   "backend, the weight dtype. Quantised weights are labelled `reference_modified` and are not part "
                   "of the verified matrix (DV21).",
                   "50 steps with a negative branch: two model forwards per step (100 forwards)."],
        "status": "Status `stable` is the catalog label of the baseline recipe. It does not mean that a parity gate has "
                  "passed: that gate is pending (" + PENDING + ").",
    },
    "photo_safe": {
        "purpose": "General photography with a plain prompt. In v1 it exists for the contract (an artifact-diagnostics "
                   "guard slot) and as the place where later, separately confirmed changes may go (SPEC 6.1).",
        "inputs": ["Nothing beyond the basic inputs. Camera and light wording is only what the user types."],
        "limits": ["The guard `photo_artifacts` is declared for the second queue and is not implemented in v1: nothing "
                   "checks banding, plastic skin or oversharpening.",
                   "No blind photo gate has been run (SPEC 10): no statement about naturalness is made."],
        "status": "Status `candidate` until the A/B gates of SPEC 10 pass. " + EQUAL_SENTENCE,
        "equal": True,
    },
    "identity_safe": {
        "purpose": "One character with a compatible character LoRA (SPEC 6.1). A character LoRA is expected; since 0.3.0 "
                   "none is required.",
        "inputs": ["A character LoRA: an ordinary LoRA loader (it acts in every phase), `MossPhotonLoRAPhase`, or the "
                   "deprecated `MossPhotonLoRAProfile` (role `character`) connected to the sampler's `identity_profile`. "
                   "A role nobody recorded gives the note `lora_not_recorded`, never an error.",
                   "Optional: reference photos at the Diagnostics node (info only; no detector exists in v1)."],
        "limits": ["LoRA strength is not set by the preset; no verified strength exists for any character LoRA. The "
                   "weight is constant (a phased weight exists only as the experimental `phased` profile of the deprecated "
                   "`MossPhotonLoRAProfile`, since 0.2.0).",
                   "No identity score exists in v1 (no detector or ArcFace dependency): the report can not say "
                   "that identity was preserved. The identity gate (SPEC 10) has not been run.",
                   "The guard `identity_score` is declared for the second queue and not implemented."],
        "status": "Status `candidate`. " + EQUAL_SENTENCE,
        "equal": True,
    },
    "wardrobe_safe": {
        "purpose": "A character with precisely specified clothing, text-to-image only (SPEC 6.1). It requires the "
                   "wardrobe block to be filled.",
        "inputs": ["`MossPhotonPromptLayout` with the `wardrobe` block filled, its `constraints` output connected to the "
                   "sampler (empty block: error `missing_required_input`).",
                   "A character LoRA is optional (maintainers' decision)."],
        "limits": ["A source image or latent is refused (`unsupported_source_input`): the source-based branch needs "
                   "`MossPhotonProtectedRefine`, which is not in v1.",
                   "Nothing compares the generated clothing with the text: the guard `clothing_attributes` is declared "
                   "for the second queue and not implemented. A conflicting colour word between the free text and the "
                   "block is reported as a warning only (`attribute_conflict`).",
                   "`forbidden_changes` is not compiled into the prompt (DV6)."],
        "status": "Status `candidate`. " + EQUAL_SENTENCE,
        "equal": True,
    },
    "scene_safe": {
        "purpose": "A scene with objects and a background, text-to-image only (SPEC 6.1). It requires the scene block "
                   "to be filled.",
        "inputs": ["`MossPhotonPromptLayout` with the `scene` block filled, its `constraints` output connected to the "
                   "sampler (empty block: error `missing_required_input`).",
                   "Optional `layout_json` (experimental, vendor-marked; boxes in `abs_centralized`, `yxyx`)."],
        "limits": ["A source image or latent is refused (`unsupported_source_input`).",
                   "Object counts and positions are not checked: the guard `object_count` is declared for the second "
                   "queue and not implemented.",
                   "The layout feature is experimental (the node adds the warning `layout_experimental`)."],
        "status": "Status `candidate`. " + EQUAL_SENTENCE,
        "equal": True,
    },
    "night_photo": {
        "purpose": "Low light, dark skin or clothing, backlight (SPEC 6.1). A placeholder that carries the numbers of "
                   "reference until a low-light calibration exists.",
        "inputs": ["Nothing beyond the basic inputs."],
        "limits": ["Experimental and not a recommended final preset: the separate low-light set of SPEC 6.4 does not "
                   "exist yet.",
                   "The guard `shadow_banding` is declared for the second queue and not implemented."],
        "status": "Status `experimental`. " + EQUAL_SENTENCE,
        "equal": True,
    },
    "draft_fast": {
        "purpose": "A quick look at a seed: half the steps of reference (SPEC 6.1). The chosen seed is meant to be "
                   "rendered again with a final preset.",
        "inputs": ["Nothing beyond the basic inputs."],
        "limits": ["Preview only. The step count 25 is a candidate value that may change after measurements.",
                   "That a 25-step composition resembles the 50-step one is not validated (SPEC 6.4 asks for a "
                   "measured selection accuracy; none exists)."],
        "status": "Status `preview_only` (label `draft`). The sampling numbers equal reference except `steps`.",
    },
    "advanced": {
        "purpose": "Manual research (SPEC 6.1). Starts as a copy of reference; every explicit override is reported and "
                   "marks the run `customized`.",
        "inputs": ["Use `MossPhotonSamplerAdvanced` for steps, SLG, sigmas and noise policy; the simple sampler only "
                   "overrides `cfg`.",
                   "Optional: a LoRA profile. A distill-role profile gives the warning `distill_lora_non_turbo`."],
        "limits": ["Experimental by definition: it does not become stable automatically.",
                   "`solver` and `cfg_profile` offer one value each in v1; restart, ancestral noise, second order "
                   "and guard parameters do not exist."],
        "status": "Status `experimental`. The numeric parameters are the ones of reference until an override is given.",
    },
    "turbo_8step": {
        "purpose": "8-step DMD2 turbo on SFT with the distill LoRA only (maintainers' decision 2026-10-25). A deliberate "
                   "deviation from SPEC 1 (see DEVIATIONS.md).",
        "inputs": ["The distill LoRA, strength 1.0: an ordinary LoRA loader (a merged turbo checkpoint needs none; the "
                   "preset, not the LoRA, decides that the run is a turbo run), `MossPhotonLoRAPhase`, or the deprecated "
                   "`MossPhotonLoRAProfile` (role `distill`). A distill role nobody recorded gives the note "
                   "`lora_not_recorded`, never an error. Recognised file: `lodestones-8-steps-dmd2-lora` "
                   "(sha256 `7683fe1f...e6f2`).",
                   "An additional character LoRA gives the warning `preset_lora_roles_exceeded` (use "
                   "`turbo_identity` for that stack)."],
        "limits": ["Own gate (the maintainers' turbo design), not run yet (" + PENDING + "). It is not a reference and "
                   "`recommended_final` is false.",
                   "The DMD2 LoRA was trained on MOSS-Image-Base; its author's card does not cover SFT, so the "
                   "combination is unproven (info `lora_base_mismatch`).",
                   "Verified-by-source range is steps 8 and CFG 1.0 to 1.1. Steps 9 to 16 and CFG above 1.1 have no "
                   "source (maintainers' practice); they run with the warning `turbo_setting_outside_verified`.",
                   "At CFG 1.0 the negative prompt has no effect (info `negative_unused_cfg1`). SLG is off; an SLG "
                   "override is dropped in the turbo presets and, with the preset `advanced` at "
                   "CFG 1.0, has no effect (warning `slg_ineffective_cfg1`).",
                   "A `steps` or `cfg` override outside the bounds runs as given with the warning "
                   "`override_outside_bounds` (it was `invalid_override`); an SLG override outside the Advanced node is "
                   "dropped with the warning `setting_adjusted`.",
                   "`MOSS_Image_SFT_8_steps_dmd_v0.1` is a different distillation and is refused "
                   "(`incompatible_lora`). An unknown distill file runs with `unverified_distill_lora`.",
                   "The fp32 delta is merged into the bf16 weights by ComfyUI's loader (the cards warn about that); "
                   "the merge fidelity is not studied in v1."],
        "status": "Status `candidate` with its own gate; the quality of the turbo path is unmeasured.",
    },
    "turbo_identity": {
        "purpose": "The 8-step DMD2 turbo stacked with a character LoRA (maintainers' decision 2026-10-25). Order: distill "
                   "first, then the character LoRA.",
        "inputs": ["The distill LoRA (strength 1.0) first, then the character LoRA: ordinary LoRA loaders chained by their "
                   "MODEL sockets, `MossPhotonLoRAPhase` nodes, or the deprecated `MossPhotonLoRAProfile` chain (role "
                   "`distill`, role `character`, `profile_in` from the first). A role nobody recorded gives the note "
                   "`lora_not_recorded`, never an error.",
                   "Optional: reference photos at the Diagnostics node (info only)."],
        "limits": ["Own gate (the maintainers' turbo design), not run yet (" + PENDING + "); `recommended_final` is false.",
                   "Identity under the distill LoRA is unmeasured and no identity score exists in v1. The character "
                   "strength comes from the LoRA profile; no verified value exists.",
                   "The order of the two LoRAs changes the result in the last bits and is recorded in the report.",
                   "The same unproven points as turbo_8step: DMD2 on SFT, steps 9 to 16, CFG above 1.1."],
        "status": "Status `candidate` with its own gate; identity under distill is unmeasured.",
    },
    "turbo_darkroom_portrait": {
        "purpose": "The darkroom phases (compose, commit, finish) on the 8-step turbo stack with the portrait values: "
                   "guidance 1.0 until the face locks, the character LoRA weight raised over the commit window, then a "
                   "finish with the negative prompt. Of its numbers only w_commit 1.35 comes from the maintainers' calibration; the "
                   "others are the maintainers' recipe values.",
        "inputs": ["No LoRA is required. Ordinary LoRA loaders act in every phase; `MossPhotonLoRAPhase` nodes act in the "
                   "phases they name (composition = compose; identity = every phase from the first step; texture = "
                   "finish; composition+identity = compose, commit, hold; all = every phase, exactly an ordinary "
                   "loader), and the commit weight boost is added to the strength of the `identity` entries "
                   "(never `composition+identity`, never `all`, never a known distill LoRA; no such entry: the boost is "
                   "skipped with the warning `darkroom_boost_skipped`). The deprecated `MossPhotonLoRAProfile` still works: a constant "
                   "character profile gets the boost as an extra patch of the same LoRA file, a phased one has the "
                   "preset's weights instead of its own phase plan.",
                   "The `cfg` widget of the sampler sets the FINISH guidance on this preset (0 = the preset's "
                   "`darkroom.finish.cfg`); the compose guidance is `sampling.cfg`. `MossPhotonSamplerAdvanced` has the "
                   "phase knobs (`schedule`, `compose_cfg`, `compose_slg`, `r_c`, `r_k`, `w_compose`, `w_commit`, "
                   "`reexpose`, `sigma_r`, `r_f`, `finish_cfg`, `eta`, `switch_model`) and the optional `finish_model`; "
                   "each at its sentinel means the preset's value."],
        "limits": ["Experimental: the report label is always `not_reference` with the reason `darkroom_experimental`, "
                   "and `recommended_final` is false. No identity score exists in v1.",
                   "The finish phase runs two model forwards per step (cfg above 1.0): the cost of the run is the "
                   "report's `forwards`, not `steps x 1`.",
                   "`schedule` sa2_8 was measured for 8 steps only: with another step count its knots are resampled "
                   "(warning `setting_adjusted`). A `finish_model` needs `switch_model` on; otherwise it is ignored with "
                   "the warning `finish_model_ignored`. A phase order the wrong way round is sorted, a finish phase with "
                   "no step is skipped, both with a warning.",
                   "Re-expose and noise injection are off in this preset (in the maintainers' runs re-expose smoothed the result on MOSS)."],
        "status": "Status `candidate`, `calibration.state` uncalibrated: w_commit 1.35 is the maintainers' calibration pick (3 seeds, top of "
                  "the tested range, not a proven optimum); the other values are the maintainers' recipe; no gate has been run.",
        "ab": "Pending calibration. The only measurements are the maintainers' own runs: they compared "
              "w_commit on three seeds (result files are in their development repository, not shipped: paired dSim "
              "+0.022 [+0.012, +0.032] against 1.2, texture within noise); the tests in this package run on a tiny CPU "
              "model.",
    },
    "turbo_darkroom_fullbody": {
        "purpose": "The darkroom phases (compose, commit, finish) on the 8-step turbo stack with the full-body values. "
                   "Today the numbers equal the maintainers' recipe `darkroom_turbo_fullbody`; the id is separate so the two "
                   "can diverge after calibration.",
        "inputs": ["No LoRA is required; the LoRA inputs are those of `turbo_darkroom_portrait` (ordinary loaders, "
                   "`MossPhotonLoRAPhase`, or the deprecated profile chain).",
                   "The `cfg` widget sets the FINISH guidance on this preset; the Advanced node has the phase knobs "
                   "(see `turbo_darkroom_portrait`)."],
        "limits": ["Experimental: the report label is always `not_reference` with the reason `darkroom_experimental`, "
                   "and `recommended_final` is false.",
                   "The finish phase runs two model forwards per step (cfg above 1.0).",
                   "Not calibrated: the values are the maintainers' recipe values. Their comparison of w_commit, finish_cfg, r_f, "
                   "steps and eta on full-body prompts found no robust gain, and the values were kept."],
        "status": "Status `candidate`, uncalibrated: values kept after the maintainers' comparison found no robust gain.",
        "ab": "Pending calibration. The only measurements are the maintainers' own runs (no robust "
              "arm on full-body; result files are in their development repository, not shipped); the tests in this "
              "package run on a tiny CPU model.",
    },
}


def _fmt(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "(none)"
    if value is None:
        return "(none)"
    return str(value)


def _constraint_row(c):
    extra = c.get("role") or c.get("block") or c.get("input") or ""
    if c.get("roles"):
        extra = ", ".join(c["roles"])
    return "| `%s` | %s | %s | %s | %s |" % (c["id"], c["kind"], c["severity"], c.get("reason", ""), extra or "")


def render(pid, preset):
    view = presets_mod.effective_view(pid)
    smp = preset["sampling"]
    budget = preset["budget"]
    ui = preset["ui"]
    fw = view["forwards"]
    text = TEXT[pid]
    out = []
    w = out.append
    w("# Preset `%s`" % pid)
    w("")
    w("<!-- Generated by tests/preset_docs.py from presets/%s.json. Do not edit by hand. -->" % pid)
    w("")
    w("Status: %s | Label: %s | Group: %s | Recommended final: %s" % (
        preset["status"], ui["label"], ui["group"], _fmt(ui["recommended_final"])))
    w("")
    w("Preset version %s, pin `%s`, schema version %s. Catalog summary: %s" % (
        preset["preset_version"], presets_mod.pin_of(preset), preset["schema_version"], ui["summary"]))
    w("")
    w("## Purpose")
    w("")
    w(text["purpose"])
    w("")
    w("File description: %s" % preset["description"])
    w("")
    w("## Status")
    w("")
    w(text["status"])
    w("")
    if text.get("equal"):
        w("This preset is numerically equal to reference; the parameter table below shows the same numbers.")
        w("")
    w("## Effective parameters (SFT)")
    w("")
    w("Read from `presets/%s.json`. These are the values the sampler uses when the `cfg` widget is 0.0 and no "
      "override is given (an explicit `cfg` above 0 wins and marks the run customized). Model families listed by the "
      "preset: %s. Only SFT is in scope; Base: not verified in v1 (the preset lists it where the file does, the "
      "numbers are the same, no run exists)." % (pid, _fmt(preset["model_families"])))
    w("")
    w("| Parameter | Value |")
    w("|---|---|")
    for key in ("steps", "sampler", "scheduler", "shift", "cfg", "slg_mode", "slg_layers", "slg_start", "slg_end",
                "restart", "ancestral_noise"):
        w("| sampling.%s | %s |" % (key, _fmt(smp[key])))
    w("| budget.nominal_steps | %s |" % _fmt(budget["nominal_steps"]))
    w("| budget.extra_forwards | %s |" % _fmt(budget["extra_forwards"]))
    w("| budget.max_forwards | %s |" % _fmt(budget["max_forwards"]))
    w("| checkpoint_policy | %s |" % preset["checkpoint_policy"])
    w("| lora_policy | %s |" % preset["lora_policy"])
    w("| fallback | %s |" % preset["fallback"])
    w("| postprocess | %s |" % _fmt([p if isinstance(p, str) else p.get("id", "?") for p in preset["postprocess"]]))
    w("")
    bounds = smp.get("override_bounds")
    if bounds:
        w("Override bounds (explicit overrides outside the verified range run with a warning):")
        w("")
        if preset.get("darkroom"):
            w("The `cfg` bounds are the bounds of the compose guidance (`sampling.cfg`, the `compose_cfg` knob); the `cfg` "
              "widget of this preset is the finish guidance and is limited to 1 to 30 only.")
            w("")
        w("| Parameter | Allowed | Verified by a source | Source kind |")
        w("|---|---|---|---|")
        for key in sorted(bounds):
            b = bounds[key]
            w("| %s | %s to %s | %s to %s | %s |" % (
                key, _fmt(b["min"]), _fmt(b["max"]), _fmt(b["verified_min"]), _fmt(b["verified_max"]),
                b["source"]["kind"]))
        w("")
    block = preset.get("darkroom")
    if block:
        w("## Darkroom phases")
        w("")
        w("Read from the `darkroom` block of `presets/%s.json`; the run goes through the phase engine (compose, commit, "
          "optional re-expose, finish). Phase bounds are sigma progress r = (sigma_0 - sigma) / (sigma_0 - sigma_last). "
          "A step belongs to the phase its start r falls in. The `cfg` widget of the sampler overrides `finish.cfg`; "
          "`MossPhotonSamplerAdvanced` has a knob for every row (the sentinel is the value below). A value that differs "
          "from the one below marks the run customized and is listed in the report's diff as `darkroom.*`. The "
          "sampling SLG widgets are not applied on this preset (warning `darkroom_override_ignored`); the darkroom SLG is "
          "`compose.slg`." % pid)
        w("")
        w("| Parameter | Value |")
        w("|---|---|")
        w("| darkroom.schedule | %s |" % block["schedule"])
        w("| darkroom.compose.cfg | %s |" % _fmt(block["compose"]["cfg"]))
        w("| darkroom.compose.slg | %s |" % _fmt(block["compose"]["slg"]))
        for key in ("r_c", "r_k", "w_compose", "w_commit"):
            w("| darkroom.commit.%s | %s |" % (key, _fmt(block["commit"][key])))
        w("| darkroom.reexpose.enabled | %s |" % _fmt(block["reexpose"]["enabled"]))
        w("| darkroom.reexpose.sigma_r | %s |" % _fmt(block["reexpose"]["sigma_r"]))
        for key in ("r_f", "cfg", "eta", "switch"):
            w("| darkroom.finish.%s | %s |" % (key, _fmt(block["finish"][key])))
        w("")
    w("## Required inputs")
    w("")
    w("Always: `model`, `positive`, `negative`, `latent`, `seed`. Then:")
    w("")
    for line in text["inputs"]:
        w("- " + line)
    w("")
    if preset["constraints"]:
        w("Constraints evaluated before sampling (from the preset file):")
        w("")
        w("| Id | Kind | Severity | Reason code | Role / block / input |")
        w("|---|---|---|---|---|")
        for c in preset["constraints"]:
            w(_constraint_row(c))
        w("")
    else:
        w("The preset file declares no constraint.")
        w("")
    if preset["guards"]:
        w("Guards declared by the preset (none of them runs in v1):")
        w("")
        w("| Guard | Required | Dependency | Implemented in |")
        w("|---|---|---|---|")
        for g in preset["guards"]:
            w("| `%s` | %s | %s | %s |" % (g["id"], _fmt(g["required"]), g["requires_dependency"], g["implemented_in"]))
        w("")
    w("## Cost in model forwards")
    w("")
    w("- Branches per step: %d (%s)." % (
        fw["branches_per_step"], "CFG 1.0 drops the negative branch" if fw["branches_per_step"] == 1
        else "positive and negative branch"))
    w("- Expected forwards for the preset numbers: %d (= %s steps x %d)." % (
        fw["expected_branch_forwards"], _fmt(smp["steps"]), fw["branches_per_step"]))
    w("- Budget cap in the preset file: %s." % _fmt(budget["max_forwards"]) + (
        " A plan above the cap is the warning `forward_budget_exceeded` and the run goes on." if budget["max_forwards"] else ""))
    if block:
        w("- Darkroom: the count above is the compose guidance only. The finish phase (from r = %s on) runs two branches "
          "per step because %s > 1.0; the report states the exact count of the run." % (
              _fmt(block["finish"]["r_f"]), "finish.cfg" if block["finish"]["cfg"] > 1.0 else "its guidance"))
    if bounds:
        top = bounds["steps"]["max"] * 2
        w("- Upper corner inside the bounds: %d steps at CFG above 1.0 = %d forwards." % (bounds["steps"]["max"], top))
    w("- Time: not estimated in v1 (the report field is empty; DV10).")
    w("")
    w("## Calibration")
    w("")
    w("- Reference of this preset: `%s`." % preset["calibration"]["reference"])
    w("- State: %s; manifest: %s." % (preset["calibration"]["state"], _fmt(preset["calibration"]["manifest"])))
    w("- Reference checkpoint: official `moss_image_sft_bf16.safetensors`, sha256 `%s`." % consts.SFT_BF16_SHA256)
    w("")
    w("## Limitations")
    w("")
    for line in text["limits"]:
        w("- " + line)
    w("")
    w("## A/B report")
    w("")
    w(text.get("ab") or "Pending calibration. Nothing has been measured for this preset beyond what the tests in "
                        "the repository check on a tiny CPU model.")
    w("")
    w("### Examples of successes")
    w("")
    w(PENDING + ".")
    w("")
    w("### Examples of failures and worst cases")
    w("")
    w(PENDING + ".")
    w("")
    return "\n".join(out)


def main():
    catalog = presets_mod.catalog()
    os.makedirs(OUT_DIR, exist_ok=True)
    for pid in presets_mod.ALL_PRESET_IDS:
        body = render(pid, catalog[pid])
        path = os.path.join(OUT_DIR, pid + ".md")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
        print("gen_preset_docs: wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
