# Usage

Version 0.3.0 on ComfyUI 0.42.0 (commit `8f04fd6ec`).

## 0. Read this first

- **SFT only.** Base: not verified in v1. A Base checkpoint is recognised by its hash and runs with the label
  `reference_unverified`, but no parity run, gate or workflow exists for it.
- **Nothing here is calibrated.** The GPU parity run measured bit-exact equality with the native graph on 10
  fixed cases (reduced size, `DEVIATIONS.md` B10; `CALIBRATION.md`); that is a statement about equality, not about
  quality. Calibration stages B to D of SPEC 9 have not been done. The numbers of every preset are the recipe of the official OpenMOSS ComfyUI
  graph or, for the turbo presets, the recipe of the DMD2 card.
- **Status words.** `stable` = the baseline recipe (`reference`); its parity gate passed at reduced size on 10 cases. `candidate` = a preset
  whose gate has not passed; the four `*_safe` presets are numerically equal to reference, and the turbo presets have
  their own gate, whose quality part (G5) was not run. `experimental` = research. `preview_only` = a draft. A status word is not a measurement of
  quality, and the pack makes no claim of a gain in quality, identity or realism.
- **Not in v1:** `MossPhotonSeedDrafts`, `MossPhotonPreserveUpscale`, `MossPhotonIdentityDetailer`,
  `MossPhotonProtectedRefine` (SPEC 4.2); the `guard` socket; phased LoRA weight as a verified feature (it exists as the experimental `phased` profile since 0.2.0, 3.3.1); an identity score; img2img, inpaint
  and upscaling; a Base workflow; a masked-detailer workflow. A source image or latent is refused by `wardrobe_safe` and
  `scene_safe` (`unsupported_source_input`).

## 1. The nodes at a glance

```text
UNETLoader --> LoraLoaderModelOnly (0..n) --> MossPhotonLoRAPhase (0..n, chained) ----------+
CLIPLoader --> MossPhotonPromptLayout --positive/negative/constraints--+                     |
MossPhotonEmptyLatent ------------------------------latent-------------+--> MossPhotonSampler --> VAEDecode
VAELoader ------------------------------------------vae----------------+        |  report          |
                                                                                v                  v
                                                                      MossPhotonDiagnostics --> MossPhotonSaveImage
```

Both LoRA stages are optional and chain by their MODEL sockets (section 3.3). `MossPhotonSchedule` feeds the Advanced
sampler's `sigmas` input only.

**Settings never stop a run (since 0.3.0).** A conflict between settings degrades to the nearest value the engine can run
and a warning in the report (`setting_adjusted`, `override_outside_bounds`, `forward_budget_exceeded`,
`preset_pin_mismatch`, `lora_not_recorded`, `phase_plan_ignored`, ...). A node stops only for what is impossible: a missing
file, a wrong model, latent, VAE or conditioning, a LoRA that cannot be mapped or has broken factors, a non-finite output,
a broken report or save input, an unknown preset or a broken preset file, a `profile_in` that does not belong to the MODEL
(`tests/unit/test_raise_whitelist.py` pins the list).

## 2. Example workflows (`workflows/`)

All six use the native loaders `UNETLoader`, `CLIPLoader` (type `moss_image`), `VAELoader`, `VAEDecode`, and end in
Diagnostics and Save. Each has a Note with versions, weights, preset status and the experimental features.

| File | Chain | Preset (status) | Needs |
|---|---|---|---|
| `sft_reference.json` | loaders, Prompt Layout, Sampler | `reference` (stable; bit-exact on 10 cases) | SFT, text encoder, VAE |
| `character_lora.json` | + one ordinary loader (`LoraLoaderModelOnly`, character 0.8) | `identity_safe` (candidate) | + `your_character_lora` (placeholder name) |
| `multi_lora.json` | + two chained ordinary loaders (0.8, 0.5) | `identity_safe` (candidate) | + `your_character_lora`, `your_second_lora` (placeholder names) |
| `turbo_8step.json` | + one ordinary loader (distill, 1.0) | `turbo_8step` (candidate; quality gate G5 not run) | + `lodestones-8-steps-dmd2-lora` (public) |
| `turbo_identity.json` | + distill loader, then character loader | `turbo_identity` (candidate; quality gate G5 not run) | + the distill LoRA and `your_character_lora` |
| `turbo_darkroom_portrait.json` | distill loader, then LoRA Phase (character 0.8, phase `identity`), the main Sampler on the darkroom preset | `turbo_darkroom_portrait` (candidate, experimental; uncalibrated except `w_commit`) | + the distill LoRA and `your_character_lora` |

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

### 3.3 LoRAs: ordinary loaders and Moss Photon LoRA Phase (`MossPhotonLoRAPhase`, 1 Setup)

**An ordinary ComfyUI LoRA loader (`LoraLoader`, `LoraLoaderModelOnly`) needs nothing else.** Connect its MODEL to the
sampler: the LoRA acts in every phase of the run (phase `all`) at the strength you set, the sampler accepts the patched MODEL
silently (an info `foreign_patches` in the report says there are patches no profile explains; it is not a warning and never an
error). No profile, no role, no `identity_profile` is needed. Turbo or not is decided by the **preset**, never by whether a
LoRA is on the model: a merged turbo checkpoint works with `turbo_8step` and no loader at all.

**Moss Photon LoRA Phase** chooses WHEN a LoRA acts. One node per LoRA, chained by the MODEL socket; the plan entries are
applied in the order of the nodes among themselves. **The plan entries are applied on top of every patch the MODEL already
carries (every ordinary loader), whatever the node order**: a graph `LoRA Phase(A) -> LoraLoader(D) -> sampler` runs D, then A
(the reverse of the graph; the sum is the same, but the rounding of the result differs bit for bit). To get the graph
order, put the ordinary loaders BEFORE the LoRA Phase nodes. When the MODEL got patches after a plan entry was
recorded, the report says so with the info `lora_plan_after_patches`. Inputs: `model`, `lora_name` (`none` passes the model through), `strength` (-20 to 20, like an ordinary loader;
0 passes the model through), `phase` (default `identity`). The node **patches nothing**: it validates the file (it exists, it
maps to MOSS, its factors fit and are finite, it is not the unsupported data-space DMD file, no trainer flag ComfyUI cannot
apply) and returns a clone of the MODEL that carries the entry `{lora_name, strength, phase, file name, size, sha256}` in its
options; the input MODEL is never modified. A MossPhoton sampler reads the plan and applies it on a **direct clone** of the
incoming MODEL per segment (never a clone of a clone; two neighbouring segments with the same LoRA set share one clone; at
most one segment clone is alive). Another sampler sees the MODEL without the LoRA. A run with no plan is the run of a model
without the node, bit for bit.

| `phase` | Darkroom phases it acts in (preset `turbo_darkroom_*`) | Meaning |
|---|---|---|
| `all` | compose, commit, hold, finish | exactly what an ordinary loader does; **never boosted** |
| `composition` | compose | layout and pose (before the face locks) |
| `identity` | compose, commit, hold, finish | **the character phase**: the whole run from the first step; takes the commit weight boost |
| `texture` | finish | the last phase: skin, fabric, fine detail |
| `composition+identity` | compose, commit, hold | everything but the finish; **never boosted**; the place of style and slider LoRAs (absent in the finish) |

**`identity` starts at the first step on MOSS (a conservative choice, not a measured one).** The character is present from the
first step because the maintainers' earlier tests found that a character LoRA weak early lost identity even at .9 late (a
50-step run without a distill LoRA), and that relaxing the character late costs identity. **No turbo A/B isolates the
character's presence in the compose segment:** in a later maintainers' test (10 generations, directional) the arms with the
character only from the commit on kept layout and hair, and they carry the commit boost as a confound. What that test does
show is that whatever LoRAs act in the compose segment shape layout and hair: the arm that left the extras out of compose
changed the hair, so moving an extra out of compose changes the picture. The windows are the preset's
own and are not calibrated for LoRA use.

The phase boundaries are the preset's own sigma-progress values `r_c`, `r_k`, `r_f` (and the Advanced knobs of the same
names): compose is `r < r_c`, commit `r_c <= r < r_k`, hold `r_k <= r < r_f`, finish `r >= r_f`. They are **not
calibrated** for LoRA use. A segment ends where the set of acting LoRAs changes (and where the guidance changes), so a run
has a handful of segments, each one stock `CFGGuider` + euler with the latent carried over exactly.

- **The commit boost.** On a darkroom preset the character weight moves `w_compose -> w_commit` over the commit window (the
  preset's `darkroom.commit`, 1.0 -> 1.35 for the portrait preset). It is added to the strength of the entries with phase
  **`identity`** only. The entry runs at `strength + (weight - w_compose)` in the commit, hold
  and finish segments (`weight == w_compose` in the compose segment, so no boost there). **The boost is additive**, which
  suits a character LoRA near 1.0 and is wrong for a slider (-2 would become -1.65, weaker) or a style LoRA (0.65 would become
  1.0): so `all` (exactly an ordinary loader), `composition+identity`, `composition` and `texture` entries never take it.
  Even a small positive strength gets the whole additive step: a strength of .3 at `w_commit` 1.35 peaks at .65 (+117 %).
  **A negative strength in `identity` takes no boost either** (the node's default phase is `identity`, so a slider left on
  it at -2 would become -1.65 and -0.2 would flip to +0.15): it keeps its value in every segment and the run gives the warning
  `darkroom_boost_skipped` (`detail.reason = negative_strength`, `detail.basename`, `detail.strength`; one per entry); put such
  a LoRA in `composition+identity` or `all`. **A known distill LoRA (its
  sha256 is in the pack's distill table) never takes it, in any phase.** With no
  boostable entry (and no character adapter of the deprecated profile) the boost is skipped and the report says so (warning
  `darkroom_boost_skipped`, `darkroom.character.boost_skipped = no_identity_lora`); the phases still run. Set `w_commit`
  equal to `w_compose` to drop the warning. A CONSTANT character LoRA of the deprecated profile next to a boosted entry does
  not take the boost (the entries do): the same warning names that LoRA (`detail.reason = identity_plan_takes_boost`). A
  PHASED character adapter and boosted entries are both boosted: `darkroom.character.mode =
  phased_adapter+lora_plan_extra_patch`. The report names the boosted entries in `darkroom.character.boosted_entries` (file
  names; present only when an entry took the boost).
- **Recommended stack.** Character LoRA: a LoRA Phase node with phase `identity` (the default). Style and slider LoRAs: a
  LoRA Phase node with phase `composition+identity` (they act in compose, commit and hold and are absent in the finish, and
  keep their strength: no boost). Turbo / distill LoRA: a **stock loader** (`LoraLoaderModelOnly`) placed BEFORE the LoRA
  Phase nodes (graph order, no `lora_plan_after_patches` info); a known distill file is never boosted even in a LoRA Phase
  node. `all` is exactly an ordinary loader (the same thing, in the report's plan) and stays that way. An ordinary loader is
  never in the plan and is never boosted. A style LoRA put through a LoRA Phase node with phase `identity` IS boosted: the
  pack cannot tell a style LoRA from a character LoRA by the file (only the known distill files are recognised), and there
  is no per-entry switch.
  **Evidence, honestly** (the maintainers' test, DIRECTIONAL: 8 generations, 2 seeds, ONE full-body prompt, preset
  `turbo_darkroom_fullbody`, 136 px face): removing the extras (a style LoRA, sliders) from the finish segment moved the
  mean ArcFace similarity .426 -> .461 (a control with the distill and the character only: .467) while the composition,
  pose and body of the extras stayed (mean abs RGB difference over the frame: the arm without extras in the finish differs
  from the stock-loader arm by 3.9 / 3.5 for seeds 42 / 7, the control from the stock-loader arm by 27.2 / 19.1).
  Seed 42 gained only +.013, inside the seed spread seen in the maintainers' earlier test (.02 to .03); seed 7 gained +.057. The skin texture
  (DoG detail) did not change (7.22 stock loaders, 7.07 absent in finish): this is NOT a cure for plastic skin, which the
  A/B could not judge at that face size. Cost: the face saturation rose by 7.8 points (70.8 -> 78.6). In every arm the
  character went through phase `all` at 1.0, so **no arm measured `identity` (the character in the compose segment)
  together with the commit boost**: the ArcFace gain is that of removing the extras from the finish, not of the boost.
  Starting point, not a verified setting.
- **Any other preset has no phases.** Every entry then runs for the whole run at its constant strength, exactly as an ordinary
  loader would, and an entry whose phase is not `all` or `identity` (the two that act in every phase) is named in the warning `phase_plan_ignored`.
- **The report** has `lora.plan` (basename, size, sha256, strength, phase, and `applied_in`: `["run"]` for a constant run, else
  the darkroom phases of ITS OWN model it ran in: the finish segments of a switched run belong to the `finish_model`, whose
  entries are its own; an entry that acts in none gives the warning `lora_plan_entry_inactive`) and, on a darkroom run, `darkroom.character.mode = lora_plan_extra_patch` when the
  boost was applied. The recipe hash covers the plan. Warnings of a file (`partial_lora_coverage` and the like) are kept
  on the entry and appear in the report with their plan index.
- **What stops** (typed errors, all before the first forward): a missing file (ComfyUI's own `FileNotFoundError`), a file
  that maps less than half of its modules to MOSS, factors that do not fit the weights in 5 % of the found modules or more
  or are NaN/Inf, the unsupported data-space DMD file, a trainer flag (rsLoRA / DoRA) ComfyUI does not apply, a model that is not
  MOSS (`incompatible_lora`, `unsupported_model`). A file that changed between the node and the sampler is
  `incompatible_lora`.
- A MODEL with its own plan can be the sampler's `finish_model`: its entries act in the finish phase of that model.
- Character LoRA -> `identity` (the default; it acts in every segment); style LoRA, slider ->
  `composition+identity` (absent in the finish; none of the other phases but `identity` is boosted either); a distill / turbo
  LoRA -> an ordinary loader placed before the LoRA Phase nodes (a known distill file is never boosted, whatever its phase).
  These are starting points, not verified settings.

### 3.3.0 Moss Photon LoRA Profile (`MossPhotonLoRAProfile`, 1 Setup) - DEPRECATED

**Deprecated since 0.3.0; kept so old graphs keep working** (the node says "(deprecated)" in its name, `DEPRECATED = True`).
Use an ordinary loader or Moss Photon LoRA Phase instead. Its profile is still read by the samplers, nothing about it changed,
and the shipped workflows no longer use it.

One adapter per node. Inputs: `model`, `lora_name` (`none` passes the model through), `strength_model` (constant),
`role` (`character`, `style`, `wardrobe`, `scene`, `other`, `distill`), `profile` (`constant`, or the experimental `phased`, see 3.3.1), optional `trigger`
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

#### 3.3.1 Phased LoRA weight (experimental, since 0.2.0)

`profile = phased` makes the LoRA weight follow the sampling progress instead of staying constant. It is **experimental,
outside every verification**, and works only with `MossPhotonSamplerAdvanced`; the simple sampler runs it **constant at its
nominal `strength_model`** with the warning `phased_lora_demoted` (it was `unsupported_feature`), except on a darkroom preset
(3.5), where the character adapter may be phased and the preset's own weights replace the profile's plan (a phased adapter
that is not the character runs constant there, with the same warning). Presets never enable it.

- Widgets of the LoRA Profile (used only with `profile = phased`): `w0` (weight before the window, default 0.3), `w1`
  (weight after it, 0.9), `a` and `b` (the window, as sampling progress `r` in 0..1, defaults 0.2 and 0.6, `a < b`).
  Between `a` and `b` the weight moves from `w0` to `w1` by a smoothstep; `w0 = w1` is exactly that constant weight.
  `strength_model` is not used (the report records `w1` in its place). Weights are limited to +-10.
- `r` is the share of the sigma path already walked at the start of a step (cumulative |d sigma| over the total of the
  run's own sigma array, native or the connected `sigmas`), never a step number.
- Rules: one phased adapter per chain; it stays on the **first character LoRA** that asks for it, and it is applied
  **after** all constant adapters of the chain, wherever it sits in the chain. Every other `profile = phased` request
  (a second phased node, `apply_text_encoder` together with `phased`, or a window with `a >= b`) runs **constant** at its
  own `strength_model` and the stack entry carries the warning `phased_lora_demoted` (it no longer stops the run). A
  character node placed after a non-character one that already took the phased slot re-applies that LoRA constant on its own
  MODEL. `apply_text_encoder` with no `clip` connected is ignored with the warning `lora_setting_ignored` (it is no longer
  `missing_required_input`). Transformer only for the phased weight.
- Result label: `not_reference` with the reason `lora_phased` (also on turbo presets), and the warning
  `phase_lora_experimental`. The report carries `lora.phase` (the plan, the applied weights and their hash, the segments).
- Mechanism: the weight is quantised to 6 equally spaced levels between `w0` and `w1`; the run is cut into at most 6
  segments, each sampled on its own clone of the model patched at that level, and the latent is handed from segment to
  segment unchanged. At most one level clone is alive at a time. `DEVIATIONS.md`, B11 gives the reasons.
- A phased adapter with `w0 = w1 = 0.8` produced the same latent and image, bit for bit, as the stock LoRA loader at 0.8
  (one case, GPU smoke, see below).

**Observation, one case and one seed; not a calibration and not a recommendation backed by a sample.** With the
native shift 5, `r` is very non-linear in the step number: step 6 is `r` = 0.026, step 10 is 0.047, step 20 is 0.117,
step 30 is 0.229, step 40 is 0.438 of 50. On the parity case C19 (1024x1024, seed 42, a character LoRA at
constant 0.8) the predicted clean image shows hair and silhouette already set by steps 6 to 10 (`r` about 0.03 to 0.05)
and the face by step 20 (`r` about 0.12). A weak-early profile (0.3 rising to 0.9, a = 0.2, b = 0.6) lost the
character's identity entirely in that case. A strong-early profile (1.0 falling to 0.6, window a = 0.25, b = 0.5, a different case: a full-body prompt at 832x1216,
one seed) kept the character's face and body build like constant 0.8 and changed late detail only. That is a visual
inspection by the maintainers of three images, one seed (`fullbody_A`, `fullbody_B`, `fullbody_C` in the development
repository); the run files make no likeness claim; not scored, not calibrated. If you want the face and the figure
of a character LoRA, a starting point to try is to keep the weight high for `r` below about 0.25 (roughly the first 30 of
50 steps) and ease it after that. Check the result yourself; nothing here says the profile is better than a constant
weight.

### 3.4 Moss Photon Schedule (`MossPhotonSchedule`, 2 Sampling)

Builds and validates a sigma schedule: inputs `model`, `latent`, `steps`, `shift`, `mode`. Mode `reference` equals the
native KSampler schedule bit for bit (`sgm_uniform`, the model's live shift); `shifted` recomputes it for `shift` and marks
the run not reference. The `scheduler` widget (`sgm_uniform`, the default and the base presets' scheduler, or `simple`, the scheduler of the
8-step turbo presets) picks the native KSampler spacing that `reference` reproduces; `simple` with 8 steps equals the
native turbo schedule. Outputs `sigmas` and `schedule_info`. Wire `sigmas` into the **Advanced** sampler only.

### 3.5 Moss Photon Sampler (`MossPhotonSampler`, 2 Sampling)

Inputs: `model`, `positive`, `negative`, `latent`, `seed`, `preset`, `cfg`, optional `identity_profile`, `constraints`,
`vae`, `preview`, `preset_pin`, `steps`. Outputs: `latent` and `report` (PHOTON_REPORT). `steps` is the last input
(0 = the preset's steps, 1 to 1000 overrides them; the Advanced sampler has the same widget in its old position).

- `preset` selects a versioned recipe (section 4). Everything the recipe fixes is shown in the web panel before the
  run and in the report after it.
- **`cfg`: 0.0 means "use the preset's CFG".** A value above 0 is an explicit override of the preset: the run is marked
  `customized` and the report lists the diff against the preset. A value equal to the preset's own is not customized.
  CFG 1.0 runs one forward per step (no negative branch); a value below 1 is moved to 1.0 with the warning
  `setting_adjusted`. The turbo presets declare bounds (steps 8 to 16, CFG 1 to 2); an override outside them runs as given
  with the warning `override_outside_bounds` (it was `invalid_override`). On a turbo preset a different `cfg` (or steps)
  keeps the parity class `native_turbo` (the run still equals the native turbo graph with the same settings) but sets
  `label.customized` and lists `cfg_differs` / `steps_differ` (and `turbo_setting_outside_verified` beyond 8 steps or
  CFG 1.1) in `not_reference_reasons`; there these reasons describe the run, they do not demote it.
- This sampler has no SLG widgets; an SLG override that reaches it on a turbo preset (an API caller) is dropped with the
  warning `setting_adjusted`. An SLG override on a turbo preset exists only in the Advanced sampler (3.6).
- **LoRAs.** Ordinary loaders and Moss Photon LoRA Phase work as they are (3.3); a preset that expects a LoRA role
  (`identity_safe`, the turbo presets) gives the note `lora_not_recorded` when none is recorded and nothing when the MODEL
  carries patches or a plan.
- **Darkroom presets** (`turbo_darkroom_portrait`, `turbo_darkroom_fullbody`; experimental, new in 0.3.0). The preset carries a `darkroom` block and the run goes through the phase engine: *compose* (guidance 1.0, one
  forward per step) until the face locks, *commit* (the character LoRA weight moves from `w_compose` to `w_commit`), then
  *finish* with a higher guidance and the negative prompt (two forwards per step). **No LoRA is required**: ordinary
  loaders act in every phase, Moss Photon LoRA Phase entries in the phases they name, and the commit boost goes to their
  entries with phase `identity` (3.3); a constant character profile of the deprecated node gets it as an extra patch of the same LoRA
  file, as before. **On
  these presets the `cfg` widget sets the finish guidance** (`darkroom.finish.cfg`, the phase where the negative acts),
  0 = the preset's value; the compose guidance is the preset's `sampling.cfg`. The node summary and the report say so
  (the `darkroom_experimental` warning carries `finish_cfg_source`). The report has a `darkroom` section; the label is
  always `not_reference` with the reason `darkroom_experimental`. There is no separate darkroom node any more.
- The preset also lists the checkpoint families it was written for. A checkpoint of another identified family runs with
  the warning `model_family_unverified` and the label `reference_unverified` (a turbo preset: `not_reference`).
- `vae` is a compatibility check only (128 channels, ratio 16); previews use ComfyUI's latent previewer.
- The node's text summarises: preset id and version, family, steps, sampler, scheduler, shift, CFG, SLG, forwards
  (expected / observed), LoRA stack, seed and noise policy, seconds, warning count and the **parity class**.
- Only what is impossible stops the run, before the first forward (typed codes such as `unsupported_model`,
  `invalid_schedule`, `incompatible_lora`, section 8); every settings conflict is a warning in the report and never hidden.
- Seeds: the default noise policy `native` is the stock KSampler noise. The Advanced sampler's
  `noise_policy = seed_plus_i` is reported and marks the run `not_reference`.

### 3.6 Moss Photon Sampler (Advanced) (`MossPhotonSamplerAdvanced`, 2 Sampling)

Everything of the simple sampler plus: `sigmas` (optional; a schedule that is not bit-equal to the native one gives
`custom_sigmas` and `not_reference`), `steps` (0 = preset), `slg_mode` / `slg_layers` / `slg_start` / `slg_end`
(sentinels: `from_preset`, empty, -1), `solver` (`euler` only), `cfg_profile` (`constant` only), `noise_policy`,
`max_forwards` (0 = unlimited; a plan above the limit is the warning `forward_budget_exceeded` and the run goes on), `fingerprint`, `report_step_times`. A
widget set to a value that **differs** from the preset makes the run `customized`; a value equal to the preset's own, or
a sentinel, does not. Experimental.

This is the only place where an SLG override may stand on a turbo preset (`turbo_8step`, `turbo_identity`). It is
**experimental**: the run warns `turbo_slg_experimental` and is labelled `not_reference` (reason `slg_differs`); the DMD2
recipe has SLG off, and at CFG 1.0 a forced SLG has no effect (info `slg_ineffective_cfg1`). The simple sampler has no such widgets. An SLG layer the model does not have is dropped (an `slg_mode` left without a layer becomes `off`), `slg_start` above `slg_end` is swapped, each with the warning `setting_adjusted`.

**Darkroom knobs** (optional inputs, last in the node so a saved workflow's widget values do not move): `schedule`
(`from_preset`, `native`, `sa2_8`; `sa2_8` keeps its eight measured knots at 8 steps and resamples them for another count with the warning `setting_adjusted`; a connected `sigmas` replaces it), `compose_cfg`,
`compose_slg`, `r_c`, `r_k`, `w_compose`, `w_commit`, `reexpose`, `sigma_r`, `r_f`, `finish_cfg`, `eta`, `switch_model`,
and the optional input `finish_model` (the second model of the finish phase, with `switch_model` on). Sentinels: `from_preset`,
0 for the two guidance knobs, a negative number for the rest; a sentinel means the preset's value. `r_c`, `r_k`, `r_f` are
sigma progress `r = (sigma_0 - sigma) / (sigma_0 - sigma_last)`; a step belongs to the phase its start `r` falls in. `steps`
is the shared widget. `finish_cfg` wins over the `cfg` widget (a warning says so when they differ). A darkroom value that differs from the
preset's (the `cfg` widget included) marks the run `customized`: the report lists it as a `darkroom.*` entry of the diff
and adds the reason `darkroom_setting_differs`; a value equal to the preset's does not. The `slg_*` widgets are not applied on a
darkroom preset (warning `darkroom_override_ignored`): use `compose_slg`. Bad orders and bounds degrade (they were
`invalid_override`): a value outside its range (guidance 1 to 30, fractions 0 to 1, weights 0 to 10, `sigma_r` at least 0.01) is moved to the
nearest bound, phases out of order (`0 <= r_c <= r_k <= r_f <= 1`) are sorted, a weight change over an empty commit window
keeps `w_compose`, a finish phase that holds no step, a re-expose point outside the run and `switch_model` without a
`finish_model` are left out, each with a warning (`setting_adjusted` / `darkroom_override_ignored`). **On a preset
without a darkroom block** a knob that is set is ignored with the warning `darkroom_override_ignored` (field names only) and a
`finish_model` with `finish_model_ignored`; the run is the stock run. `finish_model` with `switch_model` off is ignored the same way.

Reserved and absent: guard parameters, restart, ancestral noise, second order. A LoRA weight that changes over the run exists as the experimental `phased` profile since 0.2.0 (3.3.1); it runs only here.

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
| `identity_safe` | candidate | 50 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 100 | a character LoRA (expected, not required) |
| `wardrobe_safe` | candidate | 50 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 100 | wardrobe block |
| `scene_safe` | candidate | 50 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 100 | scene block |
| `night_photo` | experimental | 50 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 100 | - |
| `draft_fast` | preview_only | 25 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 50 | - |
| `advanced` | experimental | 50 | sgm_uniform | 3.0 | uncond_skip, layer 22 | 100 | - |
| `turbo_8step` | candidate (own gate) | 8 | simple | 1.0 | off | 8 | the distill LoRA (a note if not recorded; a merged turbo checkpoint needs none) |
| `turbo_identity` | candidate (own gate) | 8 | simple | 1.0 | off | 8 | distill + character LoRAs (expected) |
| `turbo_darkroom_portrait` | candidate (experimental) | 12 | simple | 1.0 compose / 2.0 finish | off | 17 on the 12-step simple schedule (the report has the exact count) | nothing (LoRAs optional; LoRA Phase `identity` takes the boost) |
| `turbo_darkroom_fullbody` | candidate (experimental) | 12 | simple | 1.0 compose / 2.0 finish | off | as above | nothing (as above) |

All presets use the Euler sampler and the model's native shift 5. The `*_safe` presets and `night_photo` are numerically
equal to reference; a difference between them is the input contract and the report, not the image maths.

## 5. `preset_pin` and the web panel

A saved ComfyUI workflow stores widget values only, so the pack adds the string widget `preset_pin`
(`<id>@<version>#<sha8>`, e.g. `reference@1.0.0#1936562e`). The web extension (`web/mossphoton.js`) fills it when the
node is created and when the preset changes. On the server an empty pin runs the installed version with the warning
`preset_unpinned`; a pin that names another id or version, or is malformed, runs the installed version with the warning
`preset_pin_mismatch` (it was the error `preset_version_mismatch`; a pin is advisory and never stops a run), and a pin that
differs only in its sha8 gives `preset_changed_same_version`. The report always names the installed version that ran. A pin
you type yourself is never overwritten. The pack works without the extension.

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

### Turbo presets at a glance

| Preset | Use it when | Forwards (steps) | Status | The `cfg` widget |
|---|---|---|---|---|
| `turbo_8step` | the fastest run: the distill LoRA only, no character | 8 (8) | candidate | the single guidance (preset 1.0) |
| `turbo_identity` | the same 8 steps with a character LoRA after the distill LoRA | 8 (8) | candidate | the single guidance (preset 1.0) |
| `turbo_darkroom_portrait` | a portrait / head-and-shoulders with a character LoRA, when you want the identity-oriented phases and accept about twice the forwards of `turbo_identity` | 17 on the simple schedule (12) | candidate, experimental, uncalibrated | the **finish** guidance (preset 2.0); the compose guidance is the preset's 1.0 |
| `turbo_darkroom_fullbody` | the same for a full-body frame | 17 (12) | candidate, experimental, uncalibrated | the finish guidance, as above |

The two darkroom presets need no LoRA (3.3: an ordinary loader for the distill LoRA, LoRA Phase `composition+identity` for style and slider LoRAs, LoRA Phase `identity` for the
character; the commit boost goes to its `identity` entries and is skipped with a warning when there is none). Their measurements
(the maintainers' calibration: 57 generations, constant profile, 3 seeds) are summarised in `DEVIATIONS.md`, B12: on the portrait preset only
`w_commit` was calibrated (1.2 to 1.35, a small identity gain); on the full-body preset the values were compared and
no robust gain was found, so they are unchanged. Nothing here is a quality guarantee. Example workflow:
`workflows/turbo_darkroom_portrait.json` (the main sampler).

Phase knobs of `MossPhotonSamplerAdvanced` (every one at its sentinel means "from the preset"; on a preset without a
`darkroom` block a set knob is ignored with `darkroom_override_ignored`; a value that differs from the preset marks the
run `customized`):

- Where the phases are (sigma progress `r`, 0 to 1): `r_c` ends compose and starts commit, `r_k` ends commit, `r_f`
  starts finish (`r_f >= r_k`); `schedule` picks the step spacing (`native` or the 8-step `sa2_8`; a connected `sigmas`
  replaces it).
- Guidance: `compose_cfg` / `compose_slg` for compose and commit, `finish_cfg` for the finish phase (it wins over the
  `cfg` widget).
- Character weight: `w_compose` before commit, `w_commit` after it (smoothstep over `r_c..r_k`).
- Finish extras: `eta` (noise injection in the finish phase, 0 = off), `reexpose` with `sigma_r` (re-noise and descend
  again at `r_k`; off in the presets, it smoothed the result), `switch_model` with the `finish_model` input (a second model
  for the finish phase).

`turbo_8step.json` and `turbo_identity.json` follow the 8-step DMD2 recipe of the LoRA's card: Euler, scheduler `simple`,
native shift 5, **CFG 1.0**, strength 1.0, no SLG. At CFG 1.0 the model runs once per step (8 forwards) and the negative
prompt is not used (info `negative_unused_cfg1`).

- Both are `candidate`. Their own gate ran at reduced size: bit-exact against the native turbo graph on 10 cases plus 3
  corner cases, determinism and stability passed; the quality sheets (G5) were **not run**. `recommended_final` is false.
- Expected LoRAs: the distill LoRA (and the character LoRA for `turbo_identity`, distill first), as ordinary loaders, LoRA
  Phase nodes or the deprecated profile chain. None is required: a role nobody recorded is the note `lora_not_recorded`
  (none when the MODEL carries patches), and a merged turbo checkpoint runs with no LoRA at all. The data-space DMD file
  `MOSS_Image_SFT_8_steps_dmd_v0.1` is still refused (`incompatible_lora`); an unknown distill file runs with
  `unverified_distill_lora`; a second distill LoRA gives `multiple_distill_loras`; a distill LoRA in a non-turbo preset gives
  `distill_lora_non_turbo`.
- The DMD2 LoRA was trained on Base by its author; on SFT it is unproven (info `lora_base_mismatch`).
- Experimental extras: steps up to 16 (Advanced sampler) and CFG up to 2 are allowed by the bounds but have no source;
  anything beyond 8 steps or CFG 1.1 gives `turbo_setting_outside_verified`. Raising CFG above 1.0 brings the negative
  branch back (2 forwards per step: up to 32).
- SLG is off in the turbo presets. An SLG override outside the Advanced sampler is dropped with a warning; the Advanced
  sampler lets it stand as an **experimental** override (warning `turbo_slg_experimental`, label `not_reference`), and at
  CFG 1.0 a forced SLG has no effect (`slg_ineffective_cfg1`). With the preset `advanced` a forced SLG at CFG 1.0 is
  reported with the info `slg_ineffective_cfg1`.
- The turbo run's quality and its effect on identity are unmeasured.

## 8. Typical stops and notes

Only the impossible stops a run. Since 0.3.0 the conflicts that used to stop it are warnings in the report (the run goes on
with the nearest usable value): a pin of another version (`preset_pin_mismatch`), a value outside a preset's bounds
(`override_outside_bounds`), a value the engine cannot run (`setting_adjusted`), a plan above `max_forwards`
(`forward_budget_exceeded`), a LoRA role nobody recorded (`lora_not_recorded`; an info for the distill role, which says a merged distill checkpoint is
assumed: turbo comes from the preset), a distill LoRA in a non-turbo preset
(`distill_lora_non_turbo`), a second distill LoRA (`multiple_distill_loras`), a phased profile on the simple sampler
(`phased_lora_demoted`), a LoRA Phase entry on a preset without phases (`phase_plan_ignored`), a LoRA Phase entry that acts in
no phase its MODEL runs (`lora_plan_entry_inactive`: e.g. a `texture` entry on the main MODEL while a switched `finish_model`
samples the finish phase; the entry changes nothing and `lora.plan[].applied_in` is empty); the info `lora_plan_after_patches` says the MODEL was patched
(an ordinary loader, a compute-dtype or hook node) after a LoRA Phase node (the entry is applied on top of that patch).

| You see | Meaning | Fix |
|---|---|---|
| `unsupported_model` | the MODEL is not a native MOSS Image model, or ComfyUI lacks MOSS support | load `moss_image_*.safetensors` with `UNETLoader`; update ComfyUI to commit `e087d8f6` or later |
| `missing_required_input` | `wardrobe_safe` / `scene_safe` without its block | fill the block in Prompt Layout and connect `constraints` |
| `incompatible_lora` | the file maps nothing (or less than half) onto MOSS, has NaN factors or factors that do not fit, is the refused DMD file, or a `profile_in` / `identity_profile` that does not belong to the MODEL | use a MOSS LoRA; connect the MODEL and profile of the same chain |
| `FileNotFoundError` (a LoRA) | the file is not in the loras folder (ComfyUI's own error) | pick an existing file |
| `invalid_dimensions` | width or height not a multiple of 16 | use multiples of 16 |
| `non_finite_output` | NaN/Inf in the result | the run stops with this error; read the message (precision, LoRA, model file) |

## 9. Out of scope in v1 (summary)

Base verification; calibration stages B to D; phased LoRA weight as a verified feature (it exists as the experimental `phased` profile since 0.2.0, 3.3.1); the nodes `MossPhotonSeedDrafts`,
`MossPhotonPreserveUpscale`, `MossPhotonIdentityDetailer`, `MossPhotonProtectedRefine`; identity and clothing scoring;
quantised-checkpoint matrix; the `guard` socket. See `DEVIATIONS.md` for the reasons.
