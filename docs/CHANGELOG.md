# Changelog

## 0.4.0 — 2026-11-06

### Added

- **`MossPhotonHiresFix`** (5 Finishing, experimental; USAGE 3.9): the second pass of a hires fix in one node, in place of
  `LatentUpscaleBy` + `KSampler`. Inputs: the `model`, `positive`, `negative`, `vae` and `latent` of the first pass, `seed`, `scale`
  (1.0 to 2.0, default 1.5), `steps`, `denoise`, `cfg`, `sampler_name`, `scheduler` (defaults 6, 0.32, 1.0, `res_multistep`,
  `linear_quadratic`: the recipe of the stock graph), `face_protect`, `face_sigma`, `identity_boost` and, optional, `character_lora`
  and `upscale_method` (`pixel_lanczos`, the default, or `latent_bicubic`). Outputs: `latent` and a text `summary` (no report). With
  `latent_bicubic`, `face_protect` off and `identity_boost` 0 it is that pair of nodes, bit for bit. The bicubic latent upscale
  leaves a semi-transparent doubled image on brows and eyes (visible in a plain decode; the stock graph hides it by redrawing from a
  high sigma), which a held face would keep: so the default upscale is `pixel_lanczos` (decode, Lanczos in pixels, encode), and a
  face is never held on a bicubic latent: with `latent_bicubic` and a face found the run uses `pixel_lanczos` and warns
  (`hires_bicubic_not_held`); with no face or protection off `latent_bicubic` stays the stock recipe. The first line of the summary
  names the method that ran. `face_sigma` defaults to 0.35 (on two seeds and one prompt, the lowest value tried kept the face closest
  to the first pass, with slightly less new detail); the summary line for a boosted `character_lora` says "on top of the loader's strength". With `face_protect` on, the faces found by the YOLO detector are
  held at the source until the schedule reaches `face_sigma` (a `noise_mask` and a `denoise_mask_function` on a clone of the MODEL;
  the pixels outside the faces are not touched). `identity_boost` is added to the strength of the character LoRA for this pass: the
  LoRA Phase entries with phase `identity` (the pass counts as the finish phase: `composition` and `composition+identity` entries do not
  act, `texture` and `all` do), and the file in `character_lora` for a MODEL patched by ordinary loaders; a known distill LoRA is
  never boosted, and a character LoRA that is also a boosted plan entry is boosted once. `face_sigma` and `identity_boost` are not
  calibrated (`identity_boost` is unmeasured on the pixel path). Every setting that cannot be honoured is a warning in the summary (`hires_no_detector`,
  `hires_detector_failed`, `hires_no_face`, `hires_face_sigma_above_schedule`, `hires_face_sigma_below_schedule`,
  `hires_boost_no_target`, `hires_boost_distill_skipped`, `hires_mask_not_combined`, `hires_bicubic_not_held`; the infos are `hires_boost_already_in_plan`
  and `hires_lora_phase_skipped`); the node stops only for a wrong MODEL or latent, a missing or unmappable LoRA file and a non-finite
  result. A result above 2 MP is noted, above 4 MP warned (`megapixels_out_of_documented_range`).
- `consts.NODE_IDS_HELPERS` lists the node (`NODE_IDS` stays the eight of the first queue); `finishing/backends.py` exposes the
  detector as the function `detect_faces`, which the detailer's backend now calls.

## 0.3.0 — 2026-11-04

Changes relative to 0.2.0. Scope unchanged: SFT only, Base not verified, presets are candidate or experimental (only `reference`
is stable). The numbers quoted below come from the maintainers' own tests; they are not part of this package.

### Added

- **`MossPhotonLoRAPhase`** (`lora_name`, `strength` -20..20, `phase` all / composition / identity / texture /
  composition+identity, default identity). It patches nothing: it validates the file (missing file, unmappable, NaN or
  misfit factors, the data-space DMD file, trainer flags) and returns a clone whose `model_options` carry the plan
  `{lora_name, strength, phase, basename, size, sha256}` (a clone deep-copies the options: nothing leaks into the input
  MODEL). The samplers apply the plan on DIRECT clones of the incoming MODEL: a Darkroom preset applies each entry in the
  phases of the maintainers' table (composition = compose; identity = whole run, see Changed; texture = finish;
  composition+identity = compose + commit + hold; all = every phase; the boundaries stay the preset's `r_c` / `r_k` / `r_f`,
  not calibrated), neighbouring segments with the same LoRA set share one clone; any other preset applies every entry
  constant (warning `phase_plan_ignored` for a phase other than `all`). Report: `lora.plan` (main entries, then the
  finish_model's, each marked `model`), `darkroom.character.mode = lora_plan_extra_patch`. A run with no plan is bit-equal to the
  previous build. Plan entries are applied on top of every patch the MODEL carries (an ordinary loader placed after the node
  runs BEFORE the entry): documented in USAGE 3.3, and the info `lora_plan_after_patches` says so when it happens (any later
  MODEL patch, not only a loader). An entry that acts in no phase its MODEL runs (a main `texture` entry next to a switched
  finish_model) has an empty `applied_in` and the warning `lora_plan_entry_inactive`.
- **`MossPhotonPromptSplit`**: the marker-delimited answer of a prompt enhancer into the nine fields of Prompt & Layout.
- The basic sampler has an optional `steps` input (appended last; Advanced keeps its widget position); stale or malformed
  `preset_pin` values warn.
- Darkroom phases in the main samplers (experimental): presets `turbo_darkroom_portrait` and `turbo_darkroom_fullbody`
  (candidate, uncalibrated) carry an optional `darkroom` block in the preset schema (`schedule`, `compose`, `commit`,
  `reexpose`, `finish`); the block is the single source of the phase values. They run in `MossPhotonSampler` and
  `MossPhotonSamplerAdvanced` (the simple sampler takes a phased or a constant character profile on them). On a darkroom
  preset the `cfg` widget sets the finish guidance. `MossPhotonSamplerAdvanced` gets the phase knobs (`schedule`,
  `compose_cfg`, `compose_slg`, `r_c`, `r_k`, `w_compose`, `w_commit`, `reexpose`, `sigma_r`, `r_f`, `finish_cfg`, `eta`,
  `switch_model`) and the optional `finish_model` input, appended after the existing inputs. Report: the `darkroom` section
  stays, its `recipe` is `<preset id>@<preset_version>`; label `not_reference`, reason `darkroom_experimental`. The web
  panel shows the block. `docs/presets/` has a page for each new preset. New example workflow
  `workflows/turbo_darkroom_portrait.json` (main sampler, distill + character LoRA, constant profiles).
- Warnings `finish_model_ignored` (a `finish_model` while the run has no model switch: `switch_model` off, or a preset
  without a darkroom block) and `darkroom_override_ignored` (a darkroom knob on a preset without a block, or a connected
  `sigmas` input that replaces `schedule`, or the `cfg` widget overridden by `finish_cfg`), `darkroom_boost_skipped` and
  `darkroom_experimental`.
- Contracts: `consts.NODE_IDS_HELPERS` (the two helper nodes), `consts.LORA_PHASES` (the one phase list: plan table, node
  combo, report schema, schema fixture), report `lora.plan`; `identity_safe.json` / `turbo_*.json` constraint severities
  changed (their sha256 and pins changed: workflows regenerated), `docs/presets/` regenerated.
- Tests: `tests/run_tests.py` has `--durations N` (slowest tests and files), cleans up models between integration tests, and
  `--jobs N` splits the integration suite over N worker processes.

### Changed

- **Ordinary LoRA loaders work as they are** (phase `all`): the MODEL they patch is accepted silently, the report records the
  patches as an info `foreign_patches` (it was a warning with advice to remove them). No preset requires a LoRA role
  any more; a turbo preset is turbo because of the preset, never because of a LoRA on the model (merged turbo checkpoints
  exist). The Darkroom presets dropped the distill and character requirements (`lora_policy`
  `use_explicit_user_profile_or_none`); `turbo_8step`, `turbo_identity` and `identity_safe` keep their `requires_lora_role`
  entry as a warning (reason `none`). The shipped workflows now use `LoraLoaderModelOnly` (and `MossPhotonLoRAPhase` on the
  Darkroom example).
- **`MossPhotonLoRAProfile` is deprecated** (`DEPRECATED = True`, display name "(deprecated)"); nothing about the node itself
  changed, old graphs keep working (see the no-raise changes below for what happens on a setting conflict).
- **No-raise policy (behaviour change for saved graphs):** every settings conflict degrades to a warning instead of an
  exception. New codes: `lora_not_recorded` (was `missing_lora`), `phase_plan_ignored`, `setting_adjusted`,
  `override_outside_bounds` (was `invalid_override` outside the turbo bounds), `forward_budget_exceeded` (was
  `budget_exceeded`), `preset_pin_mismatch` (was `preset_version_mismatch`), `multiple_distill_loras` (was an
  `incompatible_lora` error). Also warnings now: `sa2_8` with a step count other than 8 (the knots are resampled), a commit
  boost without an identity LoRA, a finish phase with no step, phases out of order (sorted), a value outside its range (cfg
  below 1, fractions, weights, `sigma_r`, SLG window and layers: moved to the nearest usable one), `switch_model` without a
  `finish_model`, a re-expose point outside the run, a phased profile on the simple sampler (`phased_lora_demoted`, runs
  constant), a distill LoRA in a non-turbo preset (`distill_lora_non_turbo`). What still stops: a missing file, a wrong model /
  latent / VAE / conditioning, an unmappable or broken LoRA, a non-finite output, a broken report or save input, an unknown
  preset or a broken preset file, a `profile_in` that does not belong to the MODEL. `tests/unit/test_raise_whitelist.py` pins
  the list by AST per call site, `(module, function, code)`, scanning typed errors, plain `RuntimeError` / `ValueError` /
  `TypeError` / `AssertionError` / `Exception` raises and the root `__init__.py` (proven by planting a violation) and
  `tests/integration/test_widget_fuzz_cpu.py` drives the sampler, LoRA Phase, the deprecated LoRA Profile and Prompt Split
  widgets with a seeded generator. A merged turbo checkpoint is legitimate: the `distill` note `lora_not_recorded` is an info
  that says a merged distill checkpoint is assumed (the other roles keep the warning).
- A LoRA Profile chain no longer raises on a setting conflict; it degrades with a warning on the stack entry. Of several
  `profile = phased` nodes the first character LoRA keeps the phased weight, every other phased request runs constant at its
  own `strength_model` with the new warning `phased_lora_demoted` (also for `apply_text_encoder` together with `phased`, and
  for a window with `a >= b`). `apply_text_encoder` without a `clip` is ignored with the new warning `lora_setting_ignored`
  (it was `missing_required_input`). What stays an error: a missing file, a wrong model or architecture, a corrupt LoRA, a
  profile that does not belong to the MODEL; those messages name the UI field to change.
- **`identity` covers the whole run; only `identity` takes the commit boost.** The commit boost (`w_commit - w_compose`, +0.35 for
  the portrait preset) is ADDITIVE: right for a character LoRA near 1.0, wrong for a slider (-2 would become -1.65, -0.2 would
  flip to +0.15) or a style LoRA (.65 would become 1.0).
  - An existing `identity` entry now acts in the compose segment too (was commit, hold, finish): on MOSS it starts at the first
    step. This is a conservative choice, not a measured one: in earlier maintainers' tests
    a character LoRA weak early lost identity even at .9 late, and relaxing the character late cost identity, but no turbo A/B
    isolates the character in compose (a later run kept layout and hair without it, with the commit boost as a confound).
  - A known distill LoRA (sha256 in `consts.KNOWN_DISTILL_LORAS`) under `identity` no longer takes the boost (it would change the
    distill recipe); it keeps its strength.
  - A negative strength under `identity` takes no boost (the node's default phase is `identity`) and the run says so
    (`darkroom_boost_skipped`, `detail.reason = negative_strength`, one warning per entry).
  - `composition+identity` takes no boost, `all` is exactly an ordinary loader; on a preset without Darkroom phases
    `phase_plan_ignored` no longer names `identity` (`covers_whole_run` derives from `PHASE_SEGMENTS`).
  - `lora_plan.BOOSTED_PHASES = ("identity",)` sits next to `PHASE_SEGMENTS` (a unit test cross-checks that it acts in the
    commit); `takes_boost(entry)` replaces `IDENTITY_PHASE`. An ordinary loader is not in the plan and is unaffected; a graph
    with no plan is bit-equal.
  - Report (additive, optional): `darkroom.character.boosted_entries` (file names of the boosted entries; present only when
    an entry took the boost). No new issue code; `darkroom_boost_skipped` gets the reason `negative_strength` next to
    `no_identity_lora` and `identity_plan_takes_boost`, and its messages name `identity`.
  - Recommended stack: character LoRA = `identity`; style and slider LoRAs = `composition+identity` (absent in the finish);
    turbo / distill LoRA = a stock loader placed before the LoRA Phase nodes. Docs: docs/USAGE.md 3.3 (phase table,
    recommended stack), the node tooltips, the preset doc. The shipped `turbo_darkroom_portrait.json` already has this shape.
  - Evidence (the maintainers' test, DIRECTIONAL, n=2 seeds, one full-body prompt, `turbo_darkroom_fullbody`): style / slider
    LoRAs absent in the finish segment: mean ArcFace .426 -> .461 (control, distill + character only: .467) with the composition
    kept (mean abs RGB difference to the stock-loader arm 3.9 / 3.5 for seeds 42 / 7, against 27.2 / 19.1 for the control).
    Seed 42 gained +.013, inside the seed spread (.02 to .03); seed 7 +.057. Skin texture (DoG) unchanged (7.22 -> 7.07), face
    saturation +7.8. The character went through `all` in every arm, so no arm measured `identity` with the boost. This is not
    evidence that it cures plastic skin: a 136 px face cannot show it, the question stays open.
- `turbo_darkroom_portrait`: `w_commit` 1.2 -> 1.35 (the maintainers' calibration, 57 generations, constant character profile;
  the only robust gain of that calibration: ArcFace mean +0.022 [+0.012, +0.032], same sign in all 3 seeds, texture flat).
  Other portrait values and all `turbo_darkroom_fullbody` values were kept (no arm beat the base robustly).
- On a darkroom preset the `slg_*` widgets are ignored with `darkroom_override_ignored` (they never reached the run); a darkroom
  value that differs from the preset marks the run `customized` (`darkroom.*` diff entries, reason
  `darkroom_setting_differs`); the web panel sends the Advanced darkroom knobs to the route.

### Fixed

- A `preset_pin` that differs from the installed preset only in sha8 (same id and same version, e.g. the source strings of
  a preset changed without a version bump) no longer stops the run with `preset_version_mismatch`: the installed
  preset runs and the report carries the warning `preset_changed_same_version` (names both pins; refresh `preset_pin`).
  A different id or version, a malformed pin stay errors; an empty pin stays `preset_unpinned`. The web panel refreshes a
  pin that differs only in sha8; it never rewrites a different version.
- Two paths sampled from a clone of a throwaway level clone: the SLG segments of a Darkroom run and every segment of a phased
  `MossPhotonLoRAProfile` with SLG on `MossPhotonSamplerAdvanced`. No change in sampled values. Both need a manual override
  (the `turbo_darkroom_*` presets ship SLG off). The maintainers saw "memory leak with model MOSSImage" warnings in their own
  ComfyUI; whether these two paths caused them is NOT shown, and there was no live server run. They were found through an
  emulation of the server's reference behaviour in the test process: when the cyclic gc frees such a patcher together with
  its throwaway parent, the `LoadedModel` stays dead. `guidance.apply_slg(..., in_place=True)` puts the hook on the level
  clone itself; `darkroom.run_plan` takes it off again after the segment, because neighbouring segments share a level clone.
  The warning count in a log is no evidence either way: it may not reach the server log file.
- Tests: the 89 dead entries the integration suite showed were test artefacts: a `Mock`'s `call_args` / stored exception kept a
  test-local MODEL chain alive until one gc pass freed it with the clones, and the count itself depended on when the automatic
  gc ran (47, 42, 7, 3 on one tree). Counting now forces a gc; `tests/integration/graph_cache.py` emulates the server (every
  sampling patcher in a reference cycle, every MODEL given to `solver.execute` kept alive), because otherwise a planted clone of
  a clone left 0 dead entries. `SamplingPatcherTest` asserts on the patcher that samples and on `is_dead`, also after an
  interrupt. `tests/run_tests.py` fails the integration suite when a test leaves a dead `LoadedModel`, and when the leak check
  itself cannot run (a comfy API change); set `MOSS_PHOTON_STRICT_LEAKS=0` to turn it off for a run.
- Not covered, not fixed: the Detailer (it samples a direct clone of its input through comfy's own sampler) and the `gpu` and
  `local` suites are outside the emulation. ComfyUI's own delegate clone in `comfy/samplers.py` when conds carry hooks on a
  dynamic patcher is ComfyUI's (the pack only warns, `adapter.py`). A throwaway parent of a cached output of the deprecated
  `MossPhotonLoRAProfile` (`nodes/lora.py`) is benign without a reference cycle and is left as is.

### Removed

- The separate `MossPhotonDarkroomSampler` node was never released (0.2.0 does not contain it); its engine is the
  main samplers' darkroom path above.

## 0.2.0 — 2026-10-25

### Added

- Experimental phased LoRA weight: `MossPhotonLoRAProfile` `profile = phased` with `w0`, `w1`, `a`, `b`; the weight
  follows the sigma progress by a smoothstep. Runs only in `MossPhotonSamplerAdvanced` (the simple sampler refuses it);
  label `not_reference` / `lora_phased`, warning `phase_lora_experimental`; one phased adapter per chain, applied after
  the constant ones, transformer only. Clone per level (6 levels), not native hooks (`DEVIATIONS.md`, B11). Outside every
  verification; see `USAGE.md`, 3.3.1 for usage and for one observation (one case, one seed).

### Fixed

- `missing_lora` now names a known distill LoRA loaded with a non-distill role and tells which node to fix; new warning
  `distill_lora_role_mismatch`.

## 0.1.0 (first version)

Scope: SFT only. Base: not verified in v1. Nothing is calibrated (`CALIBRATION.md`). Licence: MIT (`LICENSE`).

### Added

- Eight nodes in four categories: `MossPhoton/1 Setup` (`MossPhotonEmptyLatent`, `MossPhotonPromptLayout`,
  `MossPhotonLoRAProfile`), `MossPhoton/2 Sampling` (`MossPhotonSchedule`, `MossPhotonSampler`,
  `MossPhotonSamplerAdvanced`), `MossPhoton/3 Diagnostics` (`MossPhotonDiagnostics`) and `MossPhoton/4 Output`
  (`MossPhotonSaveImage`).
- Ten versioned presets with a JSON schema: `reference` (stable), `photo_safe`, `identity_safe`, `wardrobe_safe`,
  `scene_safe` (candidate, numerically equal to reference), `night_photo` (experimental), `draft_fast` (preview only),
  `advanced` (experimental), `turbo_8step` and `turbo_identity` (candidate, own gate pending, not recommended as final).
  The turbo presets are a deliberate deviation from SPEC 1 (`DEVIATIONS.md`, B1): 8 Euler steps, scheduler `simple`,
  CFG 1.0, SLG off, with the public lodestones 8-step DMD2 LoRA (role `distill`, recognised by sha256).
- LoRA profile with clone-based isolation, file coverage check, stack record (with each file's size) and the roles
  `character`, `style`, `wardrobe`, `scene`, `other`, `distill`.
- Typed errors and warnings with a code registry; a run report (PHOTON_REPORT) with a parity label (`native_turbo` for
  the turbo presets), fingerprints, forward counts and a recipe hash.
- `cfg` 0.0 means "use the preset's CFG" (B3 in `DEVIATIONS.md`).
- `MossPhotonSchedule` has a `scheduler` widget (`sgm_uniform`, `simple`); `simple` with 8 steps equals the native turbo
  sigmas.
- Save node with a PNG text chunk and a JSON sidecar, never overwriting.
- Web extension: `preset_pin` auto-fill and a read-only effective-settings panel fed by
  `GET /mossphoton/presets/<id>/resolved`.
- Example workflows `sft_reference`, `character_lora`, `multi_lora`, `turbo_8step`, `turbo_identity` with neutral file
  names (the official text encoder `qwen3.5_9b_bf16.safetensors`, the public lodestones DMD2 LoRA, and the placeholders
  `your_character_lora` / `your_second_lora`); a recorded `/object_info` and `tests/quality/test_workflows.py` that
  validates them offline; scripts in the maintainers' development repository (not shipped) that regenerate the workflows, the snapshot and the preset docs.
- The quality suite is part of `tests/run_tests.py` (`--suite quality`, and `--suite all`).
- Documentation: `INSTALL.md`, `USAGE.md`, `CALIBRATION.md`, `DEVIATIONS.md`, one page per preset in `presets/`.
- Release metadata: MIT `LICENSE`, `pyproject.toml` with licence, `requires-python >=3.10` and the Comfy Registry
  block (the `PublisherId` is still to be filled in by the maintainers).

### Fixed after the production review

- Turbo labels: a customized steps or CFG value keeps `native_turbo` with `label.customized` and descriptive reasons
  instead of demoting the run (`steps_differ`, `cfg_differs`, `turbo_setting_outside_verified`).
- SLG override on a turbo preset: refused by `MossPhotonSampler`, allowed only in `MossPhotonSamplerAdvanced` as an
  experimental override (warning `turbo_slg_experimental`, label `not_reference`).
- A checkpoint family the preset does not list gives the warning `model_family_unverified` (`reference_unverified`; for
  turbo `not_reference`). New info `reference_photos_recommended`.
- Reports: `lora.stack[].size` is required; per-LoRA warnings carry `detail.stack_index`; an unknown attention backend
  is `unknown:<name>`; `sampling.scheduler` is `custom` for input sigmas that are not the native schedule; conditioning
  area, mask and hooks are recorded as the modifier `cond_extras`.
- One broken preset file disables only that preset (warning in the report) instead of the catalog; cheap preset,
  profile and guard checks run before the checkpoint hash; an unknown noise policy is a typed `unsupported_feature`.
- Privacy: the `attribute_conflict` warning names the attribute and the field names only, never the colour words; tests
  prove that no prompt word reaches logs, report, PNG chunk or sidecar of a run.
- LoRA node: the last loaded state is kept and mapped once; the hash cache writes through a unique temporary file and
  keys by real path; the Prompt node has tooltips for every input and the display name "Moss Photon Prompt & Layout".
- Web panel: route cache, override bounds clamp; the quality suite compares the recorded `/object_info` with the live
  node declarations; a GPU run that finds no tests fails instead of passing empty.
- Documentation: the quality suite is in `run_tests.py`; `preset_version_mismatch` is an error; the node categories are
  the four `MossPhoton/<n> ...` groups; a widget set to the preset's own value does not make a run customized.

### Verification

- GPU gates of 2026-10-25 (ComfyUI 0.42.0 `8f04fd6ec`, RTX 5090, pack `5b33c0e`): `reference` and the turbo presets are
  bit-exact against the native graph on 10 fixed cases (+3 turbo corner cases), determinism and 30-job stability
  passed, at sizes reduced by the maintainers' decision (`DEVIATIONS.md` B10); no quality claim is made. Device-level VRAM
  growth of the process over 30 jobs: 0.00 % (`reference`), 2.63 % (turbo), limit 5 % (post-unload residual, Windows
  counter only).

### Known limits

- GPU gates ran at reduced size (`DEVIATIONS.md` B10); the turbo quality sheets (G5) and D2 were not run. No identity or
  clothing score, no calibration manifest.
- Not included: the second-queue nodes of SPEC 4.2, Base verification, phased LoRA weight, a Base workflow, a
  masked-detailer workflow.
- The turbo path on SFT, steps above 8 and CFG above 1.1 are unproven (the DMD2 LoRA was trained on Base).
