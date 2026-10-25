# Usage

Version 0.1.0 on ComfyUI 0.42.0 (commit `8f04fd6ec`).

## 0. Read this first

- **SFT only.** Base: not verified in v1. A Base checkpoint is recognised by its hash and runs with the label
  `reference_unverified`, but no parity run, gate or workflow exists for it (MP-19).
- **Nothing here is calibrated.** The GPU parity run (MP-12) measured bit-exact equality with the native graph on 10
  fixed cases (reduced size, `DEVIATIONS.md` B10; `CALIBRATION.md`); that is a statement about equality, not about
  quality. Calibration stages B to D of SPEC 9 have not been done. The numbers of every preset are the recipe of the official OpenMOSS ComfyUI
  graph or, for the turbo presets, the recipe of the DMD2 card.
- **Status words.** `stable` = the baseline recipe (`reference`); its parity gate passed at reduced size on 10 cases. `candidate` = a preset
  whose gate has not passed; the four `*_safe` presets are numerically equal to reference, and the turbo presets have
  their own gate, whose quality part (G5) was not run. `experimental` = research. `preview_only` = a draft. A status word is not a measurement of
  quality, and the pack makes no claim of a gain in quality, identity or realism.
- **Not in v1:** `MossPhotonSeedDrafts`, `MossPhotonPreserveUpscale`, `MossPhotonIdentityDetailer`,
  `MossPhotonProtectedRefine` (SPEC 4.2); the `guard` socket; phased LoRA weight; an identity score; img2img, inpaint
  and upscaling; a Base workflow; a masked-detailer workflow. A source image or latent is refused by `wardrobe_safe` and
  `scene_safe` (`unsupported_source_input`).

## 1. The eight nodes at a glance

```text
UNETLoader ---------------------------------------> MossPhotonLoRAProfile (0..n, chained) --+
CLIPLoader --> MossPhotonPromptLayout --positive/negative/constraints--+                     |
MossPhotonEmptyLatent ------------------------------latent-------------+--> MossPhotonSampler --> VAEDecode
VAELoader ------------------------------------------vae----------------+        |  report          |
                                                                                v                  v
                                                                      MossPhotonDiagnostics --> MossPhotonSaveImage
```

`MossPhotonSchedule` feeds the Advanced sampler's `sigmas` input only.

## 2. Example workflows (`workflows/`)

All five use the native loaders `UNETLoader`, `CLIPLoader` (type `moss_image`), `VAELoader`, `VAEDecode`, and end in
Diagnostics and Save. Each has a Note with versions, weights, preset status and the experimental features.

| File | Chain | Preset (status) | Needs |
|---|---|---|---|
| `sft_reference.json` | loaders, Prompt Layout, Sampler | `reference` (stable; bit-exact on 10 cases) | SFT, text encoder, VAE |
| `character_lora.json` | + one LoRA Profile (role character, 0.8) | `identity_safe` (candidate) | + `your_character_lora` (placeholder name) |
| `multi_lora.json` | + two chained LoRA Profiles (character 0.8, other 0.5) | `identity_safe` (candidate) | + `your_character_lora`, `your_second_lora` (placeholder names) |
| `turbo_8step.json` | + one LoRA Profile (role distill, 1.0) | `turbo_8step` (candidate; quality gate G5 not run) | + `lodestones-8-steps-dmd2-lora` (public) |
| `turbo_identity.json` | + distill profile, then character profile | `turbo_identity` (candidate; quality gate G5 not run) | + the distill LoRA and `your_character_lora` |

The file names `your_character_lora` and `your_second_lora` are placeholders: put your own LoRA under that name or
pick it in the combo (`INSTALL.md`, section 3 says which files to download). The text encoder is the official
`qwen3.5_9b_bf16.safetensors`. The strengths in the character workflows (0.8, 0.5) are placeholders; no verified strength exists for any character LoRA.
The seed is 42 with `control after generate` set to `fixed`, so a queued graph repeats exactly. Change the widget to
`randomize` yourself if you want a new seed per run.

## 3. Nodes, one by one

### 3.1 Moss Photon Empty Latent (`MossPhotonEmptyLatent`, 1 Setup)

`width`, `height` (64 to 8192, multiples of 16), `batch_size` (1 to 64). Output: an empty 128-channel latent at 1/16 of
the pixel size, float32, like the native `EmptyFlux2LatentImage`. A size that is not a multiple of 16 is **refused** (the
native node floors silently). The node's text states exact facts only: latent shape, token count, megapixels. Outside
0.25 to 4.0 megapixels it adds the warning `megapixels_out_of_documented_range`. The stock `EmptyLatentImage` also works
with the samplers (ComfyUI's own repair is applied first).

### 3.2 Moss Photon Prompt & Layout (`MossPhotonPromptLayout`, 1 Setup)

Inputs: `clip` (type `moss_image`), `positive`, `negative`, optional blocks `identity`, `wardrobe`, `scene`, `pose`,
`camera`, `lighting`, `forbidden_changes`, `layout_json`, `fingerprint_text_encoder`. Outputs: `positive`, `negative`
(CONDITIONING) and `constraints` (PHOTON_CONSTRAINTS).

- Your `positive` text comes first and is never altered. Filled blocks follow as labelled lines in a fixed order
  (Subject, Clothing, Scene, Pose, Camera, Lighting). Nothing else is ever added. With no block and no layout the
  conditioning equals `CLIPTextEncode`.
- **`forbidden_changes` is never put into the prompt** (DV6). It only travels in `constraints` for diagnostics.
- Two sources giving different colours to the same garment noun give a warning `attribute_conflict`. The warning names the
  attribute and the fields only (`<attribute> colour conflict between <field> and <field>`); the colour words
  stay in your prompt and in the private constraints and are never copied into the report. Text is never edited.
- `layout_json` is **experimental**: MOSS boxes in `abs_centralized` pixels, format `yxyx`; each item needs a label
  (`text`, `character`, `object`), a caption and an integer `bbox`. A problem is reported with its path. The node always
  adds the warning `layout_experimental`.
- Connect `constraints` to the sampler: `wardrobe_safe` and `scene_safe` read their required block from it.

### 3.3 Moss Photon LoRA Profile (`MossPhotonLoRAProfile`, 1 Setup)

One adapter per node. Inputs: `model`, `lora_name` (`none` passes the model through), `strength_model` (constant),
`role` (`character`, `style`, `wardrobe`, `scene`, `other`, `distill`), `profile` (`constant` only), optional `trigger`
(recorded privately, never injected into a prompt or written to a report), `profile_in`, `clip`, `apply_text_encoder`,
`strength_clip`, `allow_low_coverage`, `fingerprint`. Outputs: `model`, `identity_profile`, `clip`.

- The LoRA is applied through ComfyUI's own loader on a **clone**; the input model is never modified, so nothing leaks
  between jobs.
- To stack adapters, chain nodes: connect the previous node's `model` to `model` **and** its `identity_profile` to
  `profile_in`. The node order is the patch order and changes the result in the last bits; the stack is recorded.
- The node maps the file onto MOSS before applying it: a file that maps nothing is `incompatible_lora`; below 50 %
  coverage it is refused unless `allow_low_coverage`; layers a file does not cover (e.g. layer 22) are reported
  (`lora_layers_missing`).
- `apply_text_encoder` is **experimental** and off: only the transformer part is applied by default.
- Strength: there is no verified value for a character LoRA. The distill LoRA wants 1.0 (its card).

### 3.4 Moss Photon Schedule (`MossPhotonSchedule`, 2 Sampling)

Builds and validates a sigma schedule: inputs `model`, `latent`, `steps`, `shift`, `mode`. Mode `reference` equals the
native KSampler schedule bit for bit (`sgm_uniform`, the model's live shift); `shifted` recomputes it for `shift` and marks
the run not reference. The `scheduler` widget (`sgm_uniform`, the default and the base presets' scheduler, or `simple`, the scheduler of the
8-step turbo presets) picks the native KSampler spacing that `reference` reproduces; `simple` with 8 steps equals the
native turbo schedule. Outputs `sigmas` and `schedule_info`. Wire `sigmas` into the **Advanced** sampler only.

### 3.5 Moss Photon Sampler (`MossPhotonSampler`, 2 Sampling)

Inputs: `model`, `positive`, `negative`, `latent`, `seed`, `preset`, `cfg`, optional `identity_profile`, `constraints`,
`vae`, `preview`, `preset_pin`. Outputs: `latent` and `report` (PHOTON_REPORT).

- `preset` selects a versioned recipe (section 4). Everything the recipe fixes is shown in the web panel before the
  run and in the report after it.
- **`cfg`: 0.0 means "use the preset's CFG".** A value above 0 is an explicit override of the preset: the run is marked
  `customized` and the report lists the diff against the preset. A value equal to the preset's own is not customized.
  CFG 1.0 runs one forward per step (no negative branch). The turbo presets declare bounds (steps 8 to 16, CFG 1 to 2);
  an override outside them is `invalid_override`. On a turbo preset a different `cfg` (or steps in the Advanced sampler)
  keeps the parity class `native_turbo` (the run still equals the native turbo graph with the same settings) but sets
  `label.customized` and lists `cfg_differs` / `steps_differ` (and `turbo_setting_outside_verified` beyond 8 steps or
  CFG 1.1) in `not_reference_reasons`; there these reasons describe the run, they do not demote it.
- This sampler refuses any SLG override on a turbo preset (`invalid_override`). An SLG override on a turbo preset exists
  only in the Advanced sampler (3.6).
- The preset also lists the checkpoint families it was written for. A checkpoint of another identified family runs with
  the warning `model_family_unverified` and the label `reference_unverified` (a turbo preset: `not_reference`).
- `vae` is a compatibility check only (128 channels, ratio 16); previews use ComfyUI's latent previewer.
- The node's text summarises: preset id and version, family, steps, sampler, scheduler, shift, CFG, SLG, forwards
  (expected / observed), LoRA stack, seed and noise policy, seconds, warning count and the **parity class**.
- Errors stop the run before the first forward (typed codes such as `missing_lora`, `unsupported_model`,
  `invalid_schedule`, `incompatible_lora`); warnings are in the report and never hidden.
- Seeds: the default noise policy `native` is the stock KSampler noise. The Advanced sampler's
  `noise_policy = seed_plus_i` is reported and marks the run `not_reference`.

### 3.6 Moss Photon Sampler (Advanced) (`MossPhotonSamplerAdvanced`, 2 Sampling)

Everything of the simple sampler plus: `sigmas` (optional; a schedule that is not bit-equal to the native one gives
`custom_sigmas` and `not_reference`), `steps` (0 = preset), `slg_mode` / `slg_layers` / `slg_start` / `slg_end`
(sentinels: `from_preset`, empty, -1), `solver` (`euler` only), `cfg_profile` (`constant` only), `noise_policy`,
`max_forwards` (0 = unlimited; a plan above the limit stops before sampling), `fingerprint`, `report_step_times`. A
widget set to a value that **differs** from the preset makes the run `customized`; a value equal to the preset's own, or
a sentinel, does not. Experimental.

This is the only place where an SLG override may stand on a turbo preset (`turbo_8step`, `turbo_identity`). It is
**experimental**: the run warns `turbo_slg_experimental` and is labelled `not_reference` (reason `slg_differs`); the DMD2
recipe has SLG off, and at CFG 1.0 a forced SLG has no effect (info `slg_ineffective_cfg1`). The simple sampler refuses it.

Reserved and absent in v1: LoRA weight by sigma, guard parameters, restart, ancestral noise, second order.

### 3.7 Moss Photon Diagnostics (`MossPhotonDiagnostics`, 3 Diagnostics)

Inputs: `report`, optional `image`, `reference_images` (1 to 5), `masks`, `consent_reference_photos`. Outputs a copy of
the report with a diagnostics block and a text `summary`. It measures: luma mean / spread / percentiles, clipped black
and white fractions, a sharpness figure (Laplacian variance), non-finite pixel count, mask coverage. It applies **no
threshold and no pass/fail** and has no image output, so it can not change the picture. Reference photos are ignored
without consent; with consent only their count is kept and the identity result is `not_run`
(`missing_guard_dependency`): there is no detector in v1. Photos and masks are never stored or logged.

### 3.8 Moss Photon Save Image (`MossPhotonSaveImage`, 4 Output)

Inputs: `image`, `report`, `filename_prefix` (relative; no drive letter, leading slash or `..`), `format` (`png`, `jpeg`,
`webp`), `metadata` (`both`, `embedded`, `sidecar`, `none`), `quality`, `embed_workflow`. It never overwrites an existing
file (exclusive create, the counter advances). PNG pixels equal the stock SaveImage of the same tensor. `embedded` is a
PNG text chunk `moss_photon`; a JSON sidecar `<name>.json` carries the run facts (preset id and version, effective
sampling, fingerprints, `recipe_hash`) with the report redacted. JPEG and WebP get the sidecar only. Prompts are not
written unless `embed_workflow` is on.

## 4. Presets

Details and generated parameter tables: `presets/<id>.md` in this folder. All numbers below come from the preset files.

| Preset | Status | Steps | Scheduler | CFG | SLG | Forwards | Needs |
|---|---|---|---|---|---|---|---|
| `reference` | stable | 50 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 100 | - |
| `photo_safe` | candidate | 50 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 100 | - |
| `identity_safe` | candidate | 50 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 100 | character LoRA profile |
| `wardrobe_safe` | candidate | 50 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 100 | wardrobe block |
| `scene_safe` | candidate | 50 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 100 | scene block |
| `night_photo` | experimental | 50 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 100 | - |
| `draft_fast` | preview_only | 25 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 50 | - |
| `advanced` | experimental | 50 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 100 | - |
| `turbo_8step` | candidate (own gate) | 8 | simple | 1.0 | off | 8 | distill profile |
| `turbo_identity` | candidate (own gate) | 8 | simple | 1.0 | off | 8 | distill + character profiles |

All presets use the Euler sampler and the model's native shift 5. The `*_safe` presets and `night_photo` are numerically
equal to reference; a difference between them is the input contract and the report, not the image maths.

## 5. `preset_pin` and the web panel

A saved ComfyUI workflow stores widget values only, so the pack adds the string widget `preset_pin`
(`<id>@<version>#<sha8>`, e.g. `reference@1.0.0#1936562e`). The web extension (`web/mossphoton.js`) fills it when the
node is created and when the preset changes. On the server an empty pin runs the installed version with the warning
`preset_unpinned`; a pin that differs from the installed preset gives `preset_version_mismatch`. A pin you type yourself is
never overwritten. The pack works without the extension.

The same extension draws a read-only panel on both samplers: it asks `GET /mossphoton/presets/<id>/resolved` and
shows the effective steps, sampler, scheduler, shift, CFG, SLG, the expected number of forwards, whether the settings are
customized, the preset's requirements and its warnings, before you queue anything. It computes nothing itself. The time
is not estimated (shown empty).

## 6. Reading the report

The `label.parity_class` in the report says what kind of run it was, from the worst to the best:
`not_reference` (any of: another step count / CFG / SLG than the reference, custom sigmas, another shift, `seed_plus_i`,
a draft preset, a customized run of a non-turbo preset; the reasons are listed), `reference_modified` (foreign modifiers on the model, or weights
outside the reference dtype such as int8 / fp8), `reference_with_foreign_patches` (other LoRA/patches not made by this pack), `reference_unverified` (family
or checkpoint not verified, e.g. Base; also a checkpoint family the preset does not list, with the warning
`model_family_unverified`), `reference_with_lora`, `reference_exact`; turbo presets are compared with the
native turbo graph instead and end as `native_turbo` (a customized steps or CFG value keeps this class and is shown by
`label.customized` and the descriptive reasons; custom sigmas, `seed_plus_i`, another shift, an applied SLG or an
unlisted checkpoint family make it `not_reference`). This is a description of the recipe, not a quality verdict.

## 7. The turbo workflow

`turbo_8step.json` and `turbo_identity.json` follow the 8-step DMD2 recipe of the LoRA's card: Euler, scheduler `simple`,
native shift 5, **CFG 1.0**, strength 1.0, no SLG. At CFG 1.0 the model runs once per step (8 forwards) and the negative
prompt is not used (info `negative_unused_cfg1`).

- Both are `candidate`. Their own gate ran at reduced size: bit-exact against the native turbo graph on 10 cases plus 3
  corner cases, determinism and stability passed; the quality sheets (G5) were **not run**. `recommended_final` is false.
- Required profiles: role `distill` (and `character` for `turbo_identity`, chained distill first). A missing one is
  `missing_lora` before sampling; the data-space DMD file `MOSS_Image_SFT_8_steps_dmd_v0.1` is refused
  (`incompatible_lora`); an unknown distill file runs with `unverified_distill_lora`; a distill LoRA in another preset is
  `incompatible_lora` (`advanced`: warning `distill_lora_non_turbo`).
- The DMD2 LoRA was trained on Base by its author; on SFT it is unproven (info `lora_base_mismatch`).
- Experimental extras: steps up to 16 (Advanced sampler) and CFG up to 2 are allowed by the bounds but have no source;
  anything beyond 8 steps or CFG 1.1 gives `turbo_setting_outside_verified`. Raising CFG above 1.0 brings the negative
  branch back (2 forwards per step: up to 32).
- SLG is off in the turbo presets. The simple sampler refuses an SLG override on them (`invalid_override`); the Advanced
  sampler lets it stand as an **experimental** override (warning `turbo_slg_experimental`, label `not_reference`), and at
  CFG 1.0 a forced SLG has no effect (`slg_ineffective_cfg1`). With the preset `advanced` a forced SLG at CFG 1.0 is
  reported with the info `slg_ineffective_cfg1`.
- The turbo run's quality and its effect on identity are unmeasured.

## 8. Typical stops

| You see | Meaning | Fix |
|---|---|---|
| `unsupported_model` | the MODEL is not a native MOSS Image model, or ComfyUI lacks MOSS support | load `moss_image_*.safetensors` with `UNETLoader`; update ComfyUI to commit `e087d8f6` or later |
| `missing_lora` | the preset needs a LoRA role that is not in the profile | add the LoRA Profile (character / distill) and connect `identity_profile` |
| `missing_required_input` | `wardrobe_safe` / `scene_safe` without its block | fill the block in Prompt Layout and connect `constraints` |
| `incompatible_lora` | the file maps nothing onto MOSS, is the refused DMD file, or a distill file in a non-turbo preset | use a MOSS LoRA; remove the distill LoRA from this preset |
| `invalid_override` | an explicit value outside the preset's bounds, or an SLG override on a turbo preset in the simple sampler | set `cfg` to 0.0 or inside the bounds; use the Advanced sampler for the experimental SLG override |
| `invalid_dimensions` | width or height not a multiple of 16 | use multiples of 16 |
| `budget_exceeded` | the plan is above `max_forwards` | raise the limit or reduce steps |
| `non_finite_output` | NaN/Inf in the result | the run stops with this error; read the message (precision, LoRA, model file) |

## 9. Out of scope in v1 (summary)

Base verification (MP-19); calibration stages B to D; phased LoRA weight (MP-16); the nodes `MossPhotonSeedDrafts`,
`MossPhotonPreserveUpscale`, `MossPhotonIdentityDetailer`, `MossPhotonProtectedRefine`; identity and clothing scoring;
quantised-checkpoint matrix; the `guard` socket. See `DEVIATIONS.md` for the reasons.
