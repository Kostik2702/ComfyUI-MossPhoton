# Deviations from SPEC (v1)

This is where the shipped pack differs from `SPEC.md` (v1.0, 2026-10-25), and why. Part A is the table DV1 to DV21 of
design v1.1 section 8.1 (`docs/02-design.md` in the repository). Part B lists the deviations decided after that design:
the turbo presets (B1), the `forbidden_changes` field restated, the `cfg` sentinel, the size of the catalog, the Advanced-only
SLG override on turbo (B6), the privacy of conflict warnings (B7) and the neutral example workflows (B8).

"Decided by" names who took the decision; "owner" decisions are the ones the owner stated himself in the ticket feed of
MP-1 on 2026-10-25.

## A. Design v1.1, section 8.1

| # | SPEC | What v1 does instead | Reason | Decided by |
|---|---|---|---|---|
| DV1 | 4.1 sampler inputs | both samplers have one extra optional input `constraints` (PHOTON_CONSTRAINTS) | SPEC 6.3 demands errors for an empty wardrobe/scene block before sampling; the constraints object is the only carrier | architect A8, accepted by lead (Q8) |
| DV2 | 6.2 "serialized workflow contains preset id AND version" | extra STRING input `preset_pin`, filled automatically by the web extension, checked on the server | a ComfyUI workflow stores widget values only; without the pin the version cannot be in the file | lead LD6 |
| DV3 | 4.1 LoRAProfile "paths/strength of adapters" | ONE adapter per node, chained through optional `profile_in`; extra optional inputs `clip`, `apply_text_encoder`, `strength_clip`, `allow_low_coverage`, `fingerprint`; third output `CLIP` | stack order is visible in the graph; text-encoder LoRA only with an explicit flag (SPEC 6, D6) | architect A10, accepted by lead (Q2, Q8) |
| DV4 | 4.1 sampler inputs | extra optional `preview` BOOLEAN | lets a user skip preview decoding | accepted by lead (Q8) |
| DV5 | 4.1 sampler `guard?` input | the socket is NOT registered in v1 | no producer exists without detectors; adding a socket later is non-breaking, removing it is breaking | accepted by lead (Q1) |
| DV6 | 7.1 all fields compile into the prompt | `forbidden_changes` is stored in `PHOTON_CONSTRAINTS` only and is never compiled into the positive or negative prompt | a negation inside a positive MOSS prompt may backfire; the owner chose to keep the field for diagnostics | **DECIDED by owner 2026-10-25** (OD1) |
| DV7 | 4.1 LoRAProfile input `profile` | it is the weight-profile mode (`constant` in v1); curated metadata (verified range, dataset, recommended prompt) comes from an optional `<lora_stem>.photon.json` sidecar | SPEC leaves the meaning open; phased weight is MP-16 | accepted by lead (Q2) |
| DV8 | 4.1 sampler `vae?` | v1 uses it only for a compatibility check (`invalid_vae`); previews use ComfyUI's latent previewer | decoding with the VAE mid-sampling can evict the diffusion model from VRAM | architect |
| DV9 | 4.1 Advanced: "LoRA weights by sigma, guard parameters, solver mode, CFG profile" | LoRA weight by sigma = MP-16, now the experimental `phased` profile (B11); guard parameters absent (DV5); `solver` and `cfg_profile` exist as single-valued extension widgets (`euler`, `constant`) | SPEC 5/6.4: only calibrated mechanisms; widgets are positional in saved workflows, so they exist from day one | lead (cuts), architect |
| DV10 | 4.1 "estimate of calls and time" | the forward count is exact before the run (GET route `/mossphoton/presets/{id}/resolved`) and in the report; the time estimate is `null` | a time estimate needs run history; the cache was cut | **lead LD10**; accepted (Q10) |
| DV11 | 4.1 EmptyLatent "memory estimate" | the node reports latent bytes, token count and megapixels; no activation-memory estimate | a guessed formula would be an unverified claim (m12) | architect |
| DV12 | 4.1 Diagnostics "metrics" / 7 identity | v1 computes image statistics as measurements without thresholds; reference photos give `identity.status = not_run` (`missing_guard_dependency`); output STRING `summary` added | no detector/ArcFace dependency in v1 (SPEC 8: optional extras) | architect |
| DV13 | 4.1 Save "format, metadata/sidecar" | PNG text chunk + JSON sidecar only; jpeg/webp get the sidecar only | EXIF handling cut for proportionality | lead LD10 |
| DV14 | 6.2 minimal preset example | the schema also REQUIRES `description`, `sources`, `slg_start`, `slg_end`, `budget.max_forwards`, `calibration.state`, `ui.label`, `ui.summary` | SPEC 6.2 demands unit/bounds/source for every number and D1 adds the SLG window; SPEC's own example therefore does not validate (m11) | lead D1, architect |
| DV15 | 7.1 user text verbatim | on the layout-JSON path only, `summary.caption` is `compiled_text.strip()` | same as the native node (`prompt.strip()`); whitespace at the two ends only (m8) | architect A24 |
| DV16 | 6.3 first three reasons stop "if the invariant cannot be met" | `budget_exceeded` and `incompatible_lora` ALWAYS stop; foreign modifiers/patches never stop (they are labels) | stricter is safe; the label is the product | architect (3.4) |
| DV17 | 6.3 `candidate_count > 1` | no such widget in v1 | multi-seed selection is the second-queue `MossPhotonSeedDrafts` | accepted by lead (Q11) |
| DV18 | 3 / 10 Base and SFT | SFT verified; Base is recognised by hash but "not verified" | no Base checkpoint locally; verification = MP-19 | owner scope 2026-10-25 |
| DV19 | 6.3 `unscorable_face` | reserved code, never produced in v1 | no detector in v1 | architect |
| DV20 | 3 official BF16 models | the parity reference text encoder is a non-official BF16 Qwen3.5-9B text encoder file (identified by its sha256 in the MP-12 results), not the official Qwen TE; parity is stated "given the recorded TE file" | owner's reference choice (the exact file name is kept in the research results, not in the package). The example workflows and `INSTALL.md` use the official `qwen3.5_9b_bf16.safetensors` instead (owner, 2026-10-25, MP-30 M4); the report records the file that really ran | **DECIDED by owner 2026-10-25** (OD4) |
| DV21 | 10 "Compatibility" gate (quantised) | not claimed in v1; int8/fp8 weights are labelled `reference_modified` | SPEC 3: quantised is a separate matrix | architect |

Reading note for DV18: "SFT verified" means SFT is the only family in scope. The GPU parity run (MP-12) was done at
reduced size on 2026-10-25 (B10): SFT is bit-exact against the native graph on 10 cases; see `CALIBRATION.md`.

## B. Decided after design v1.1

### B1. Turbo presets: a deviation from SPEC section 1

SPEC 1 says that third-party distilled 4 to 8 step LoRAs are not part of the base quality matrix. The owner decided on
2026-10-25 (MP-1 feed, 11:55 and 12:15) to add two presets anyway:

| Item | Shipped behaviour | Source |
|---|---|---|
| Presets | `turbo_8step` (distill LoRA only) and `turbo_identity` (distill LoRA, then a character LoRA); both status `candidate`, `recommended_final` false, each with its OWN gate in MP-12 (run at reduced size, B10; quality part G5 not run) | owner 2026-10-25; lead Q9 |
| Distill file | only the 8-step DMD2 LoRA `lodestones-8-steps-dmd2-lora` (sha256 `7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2`) is recognised. Trained on MOSS-Image-Base by its author; used on SFT here, which the author's card does not cover | docs/03-turbo.md 1.3 |
| Refused file | `MOSS_Image_SFT_8_steps_dmd_v0.1` (a data-space DMD with its own sampler loop) gives `incompatible_lora` | owner Q1 |
| Sampling | 8 Euler steps, scheduler `simple` (SPEC 3 names `sgm_uniform` for the reference chain), native shift 5, CFG 1.0 (one forward per step, the negative prompt is not evaluated), SLG off at any CFG (an SLG override is refused by `MossPhotonSampler` and stands only in `MossPhotonSamplerAdvanced`, see B6) | docs/03-turbo.md 3, Q6 |
| Bounds | steps 8 to 16, CFG 1 to 2. Source-verified range: steps 8, CFG 1.0 to 1.1. A value outside it runs with the warning `turbo_setting_outside_verified`. The values above 8 steps and 1.1 CFG come from the owner's practice; no card or community source was found | docs/03-turbo.md 1.4, Q7 |
| LoRA roles | new role `distill` in `MossPhotonLoRAProfile`; a distill file in a non-turbo preset is `incompatible_lora`, in `advanced` the warning `distill_lora_non_turbo`; `turbo_8step` with a character LoRA gives `preset_lora_roles_exceeded`; an unknown distill hash gives `unverified_distill_lora` | lead Q2, Q4, Q5 |
| Families | the presets list `moss-image-sft` only. A checkpoint of another identified family (Base, a recognised fine-tune) runs with the warning `model_family_unverified` (MP-32); the label is `reference_unverified`, for a turbo preset `not_reference`. An unidentified checkpoint keeps the `unverified_checkpoint` path (a merge that is already distilled gets no special handling) | lead Q3; MP-30 M7 |
| Label | runs of turbo presets carry `parity_class = native_turbo`: the comparison basis is the native graph `LoraLoaderModelOnly` + `KSampler` (euler, simple, 8, CFG 1). A customized steps or CFG value keeps `native_turbo` (the run equals the native graph with the same settings) but sets `label.customized` and lists `steps_differ` / `cfg_differs` / `turbo_setting_outside_verified` as descriptive reasons; custom sigmas, `seed_plus_i`, another shift, an applied SLG and an unlisted family make it `not_reference` | lead Q14; MP-30 M1 (MP-32) |
| Identity | no identity threshold in code (SPEC 7); the character strength comes from the LoRA profile, the distill strength is 1.0 | owner/lead Q11 |
| Merge | the parity basis is ComfyUI's native LoRA patch path; the fp32 delta is merged into bf16 weights, which the DMD2 cards advise against. A merge-fidelity study is outside v1 | lead Q12 |

Unproven and unmeasured: the quality of the turbo path on SFT, identity under the distill LoRA, steps above 8, CFG
above 1.1. No document of this pack claims otherwise.

### B2. `forbidden_changes` is not compiled into the prompt

SPEC 7.1 lists `forbidden_changes` among the fields that compile into the prompt. The shipped `MossPhotonPromptLayout`
keeps the field only in `PHOTON_CONSTRAINTS` (for diagnostics and later guards); it is never added to the positive or
the negative prompt. This is DV6, restated here because it changes what a filled field does: it does **not** steer
generation. Decided by the owner on 2026-10-25 (OD1).

### B3. The `cfg` widget is a sentinel: 0.0 means "use the preset"

Design 4.5 had a `cfg` widget with default 3.0 and minimum 1.0. With that widget every run was an explicit override,
so `turbo_8step` with the default failed `invalid_override`. Since MP-28 the default is 0.0 (minimum 0.0): 0.0 takes
the CFG of the preset; a value above 0 is an explicit override, shown as a diff against the preset and marking the run
`customized` (a value equal to the preset's is not customized). The Advanced sampler has the same widget. Source: commit
`f7a17ca`, node tooltip.

### B4. The catalog has ten presets, not eight

SPEC 6.1 lists eight presets. The shipped catalog also contains `turbo_8step` and `turbo_identity` (B1), so the
sampler's `preset` list has ten entries and `docs/presets/` has ten files.

### B5. Not in v1 at all

Out of scope by owner decision 2026-10-25 (design DV5, DV9, DV17, DV18 and the ticket scope): calibration stages B to D;
the second-queue nodes of SPEC 4.2 (`MossPhotonSeedDrafts`, `MossPhotonPreserveUpscale`, `MossPhotonIdentityDetailer`,
`MossPhotonProtectedRefine`); the masked-detailer example workflow of SPEC 8; Base verification (MP-19); the `guard`
socket; phased LoRA weight (MP-16). SPEC 8 asks for example workflows for Base and for a masked detailer: neither is
shipped (Base: not verified in v1; the masked detailer node does not exist). Design 6.8 expected a Base workflow marked
"not verified"; the MP-13 brief lists five workflows without it.

### B6. SLG override on a turbo preset: only in the Advanced sampler

The owner's turbo decision (MP-1 feed, 12:15) says SLG is off in turbo at any CFG and an override exists "only in
Advanced". The lead's reading (MP-30 M2, 2026-10-25) is that "Advanced" is the node `MossPhotonSamplerAdvanced`, not the
preset `advanced`. Shipped: `MossPhotonSampler` refuses an SLG override on a turbo preset (`invalid_override`);
`MossPhotonSamplerAdvanced` accepts it as an experimental override: warning `turbo_slg_experimental`, parity class
`not_reference` (reason `slg_differs`), and at CFG 1.0 the info `slg_ineffective_cfg1` (no negative branch to skip).
Commit `3f8396c`.

### B7. Privacy of conflict warnings: field names only

The pack warns `attribute_conflict` when two sources give different colours to the same garment noun. Owner decision 2026-10-25 (MP-30): a warning that reaches the report, the
PNG chunk or the sidecar carries the attribute and the field names only (`<attribute> colour conflict between <field>
and <field>`); the colour words stay in the private `PHOTON_CONSTRAINTS` and appear in the node's own UI text only. The
Save node has no switch that writes prompts into the metadata; `embed_workflow` is the one opt-in that embeds the
workflow (and so the prompts) in the PNG. Commit `a9d8b76`.

### B8. Example workflows use neutral file names

SPEC 8 asks for example workflows; the owner's personal LoRA and text-encoder variants are not shipped (MP-30 M4,
decided 2026-10-25). The workflows name `your_character_lora.safetensors` and `your_second_lora.safetensors`
(placeholders to replace), the official `qwen3.5_9b_bf16.safetensors` and the public
`lodestones-8-steps-dmd2-lora.safetensors`. A fresh install shows "value not in list" for the files it does not have;
`INSTALL.md`, section 3 and the Note in each workflow say what to download or rename. The offline quality test accepts
exactly these names as documented placeholders (they are absent from the recorded `/object_info`) and rejects any other
file name in the loader widgets.

### B9. D6 refined: how LoRA patches are applied

LoRA patches are applied via the native mapper output and `ModelPatcher.add_patches` (`core/lora.py`, `apply_patches`),
identical to `comfy.sd.load_lora_for_models` on ComfyUI 0.42.0 (design D6 named the latter call directly). Equality is
pinned by `tests/integration/test_lora_apply_equals_native.py` and the GPU parity cases C19 and C20 (MP-12).
Re-verify both after any ComfyUI upgrade.

### B10. GPU gate sizes are smaller than SPEC section 10

Decided by the owner on 2026-10-25 (MP-1 feed), before the full results were known, to keep the GPU phase to about 35 to
40 minutes. The run is recorded as a pre-registered deviation (`plan` field of each `summary.json`).

| Item | SPEC section 10 / design 6.7 | What was run |
|---|---|---|
| Reference parity | 20 fixed cases | 10 cases (C01 C02 C03 C04 C07 C09 C13 C17 C19 C20), mode exact, F = 0 |
| Determinism | 3 repeats of 10 cases | 10 cases x 3 repeats (turbo: 12 cases x 3 with two 16-step CFG 2 corner cases) |
| Stability | 100 jobs at 50 steps | 30 jobs at 10 steps |
| Turbo gate | G0 to G5, D2 | G1 (10 cases) and G1 corner cases (3), G2, G3 (30 jobs), G4 (5 mechanics checks); **G5 and D2 not run** |

Results: all parity, determinism and stability checks passed on ComfyUI 0.42.0 (`8f04fd6ec`), torch 2.9.0+cu130,
CUDA 13.0, RTX 5090, pack commit `5b33c0e` from a clean tree; files in the development repository
(`research/parity/results/2026-10-25/`: `reference-153349`, `turbo-152509`).

Caveats:
- **VRAM growth, device level (measured 2026-10-25).** The torch allocator counter stays flat under ComfyUI's dynamic
  VRAM and proves nothing about leaks, so the process's own device memory was measured instead: the Windows "GPU Process
  Memory" counter (Dedicated Usage) for the ComfyUI pid, read after `unload_all_models` + `soft_empty_cache` + `gc`
  (NVML and `nvidia-smi` report N/A per process under WDDM). `reference`, 30 jobs: 0.00 % growth between job 1 and job 30.
  `turbo_8step`/`turbo_identity` stability gate G3, 30 jobs: 2.63 % (a one-time +32 MiB by job 10, then flat). Limit: 5 %.
  Results are in the development repository (`research/parity/results/2026-10-25/stability-vram-164517/`).
  Caveats: this is the post-unload residual, not the peak; the counter exists on Windows only; job counts are reduced
  (30, not 100, B10).
- In the earlier turbo stability run the number of live model patchers went from 3 (job 1) to 4 (jobs 10 to 30). The
  cause was the test harness holding an exception traceback (a LoRA-patched clone kept alive by a stored
  `incompatible_lora` error), not the pack; with the harness fixed the count is 3 at every check.
- Bit parity is shown for one machine, one GPU and one ComfyUI build, with the recorded text encoder file; it is not
  evidence about image quality, identity or Base.

### B11. Phased LoRA weight (experimental, since 0.2.0): clone per level, no native hooks

SPEC 5 asks for a LoRA weight that changes with the sampling progress and for "snapshot/clone" isolation. Shipped in
0.2.0 (MP-16): the weight is quantised to 6 levels and each run segment is sampled on its own clone of the
model patched at that level (`core/lora_phase.py`, `core/phase.py`); the clone-per-level design satisfies the SPEC's
clone requirement. Only `MossPhotonSamplerAdvanced` runs it; label `not_reference` / `lora_phased`, warning
`phase_lora_experimental`.

ComfyUI's native weight hooks (`comfy/hooks.py`, keyframes by `start_percent`) were tried first and **not used**. They
express a per-step weight exactly, but a hook applies the LoRA in float32 with stochastic rounding, while the
dynamic-VRAM loader (the default on NVIDIA, active on the test machine) applies it in bf16 on the fly. On three layers of
the real SFT checkpoint 1.5 % to 2.4 % of the bf16 weights differed, so a constant weight through hooks is not the stock
LoRA result, and the pack's `reference_with_lora` claim is a bit-equality claim. Hooks are also not implemented for the
dynamic model patcher, which falls back to a second, non-dynamic copy of the model in memory. Evidence:
`research/phase_lora/NOTES.md`.

Known limits: weights between the 6 levels are quantised (error at most `|w1 - w0| / 10`); segment hand-over is exact for
the Euler solver only (the only solver in v1); the feature is outside every verification and calibration.
Smoke results (GPU, 2026-10-25, one case, one seed): constant 0.8 through the phased path equals the stock loader bit for
bit; a full-body run with 1.0 to 0.6 completed with unchanged forward counts (100 calls, 50 skip).
