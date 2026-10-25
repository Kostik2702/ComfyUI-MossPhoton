# Changelog

## 0.2.0 — 2026-10-25

### Added

- Experimental phased LoRA weight (MP-16): `MossPhotonLoRAProfile` `profile = phased` with `w0`, `w1`, `a`, `b`; the weight
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
  validates them offline; scripts in the development repository (`research/`) that regenerate the workflows, the snapshot and the preset docs.
- The quality suite is part of `tests/run_tests.py` (`--suite quality`, and `--suite all`).
- Documentation: `INSTALL.md`, `USAGE.md`, `CALIBRATION.md`, `DEVIATIONS.md`, one page per preset in `presets/`.
- Release metadata: MIT `LICENSE`, `pyproject.toml` with licence, `requires-python >=3.10` and the Comfy Registry
  block (the `PublisherId` is still to be filled in by the owner).

### Fixed after the production review (MP-30)

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
  passed, at sizes reduced by the owner's decision (`DEVIATIONS.md` B10); no quality claim is made. Device-level VRAM
  growth of the process over 30 jobs: 0.00 % (`reference`), 2.63 % (turbo), limit 5 % (post-unload residual, Windows
  counter only).

### Known limits

- GPU gates ran at reduced size (`DEVIATIONS.md` B10); the turbo quality sheets (G5) and D2 were not run. No identity or
  clothing score, no calibration manifest.
- Not included: the second-queue nodes of SPEC 4.2, Base verification, phased LoRA weight, a Base workflow, a
  masked-detailer workflow.
- The turbo path on SFT, steps above 8 and CFG above 1.1 are unproven (the DMD2 LoRA was trained on Base).
