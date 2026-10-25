# Install

Version 0.1.0. Scope: SFT only. Base: not verified in v1.

## 1. What you need

| Item | Requirement | Source |
|---|---|---|
| ComfyUI | a build with native MOSS Image support: commit `e087d8f6` or later. The adapter checks for `comfy.model_base.MOSSImage` and refuses with `unsupported_model` ("ComfyUI must contain commit e087d8f6 or newer") when it is missing | SPEC 3; `core/adapter.py` |
| Tested on | ComfyUI 0.42.0, commit `8f04fd6ec`, web frontend 1.55.18, Python 3.13, torch 2.9 / CUDA 13.0 (embedded Windows build). Nothing else has been tried. The recorded `/object_info` used by the workflow tests is from this build | `tests/quality/object_info_snapshot.json`; docs/01-investigation.md |
| Python | 3.10 or newer (`requires-python` in `pyproject.toml`, ComfyUI's own floor). The code uses no syntax newer than 3.9 grammar (checked by parsing every file), but it has only been run on 3.13 | `pyproject.toml` |
| Python packages | **none** beyond ComfyUI's own environment. The `reference` path and every node of v1 use the standard library, torch and numpy that ComfyUI already needs. `pyproject.toml` lists no dependencies | SPEC 8; `pyproject.toml` |
| Optional packages | none are used in v1. Face detectors, ArcFace and similar extras belong to the second-queue nodes, which are not in this release (see `USAGE.md`, section 9) | SPEC 4.2, 8 |
| Network | none while generating. The pack loads local files only and runs no remote code | SPEC 8 |

## 2. Put the pack into ComfyUI

The pack is the folder `ComfyUI-MossPhoton` (the one that holds `__init__.py`, `moss_photon/`, `presets/`, `web/`,
`workflows/`). It must end up as `ComfyUI/custom_nodes/ComfyUI-MossPhoton`.

**Junction (a repository checkout that you keep updating).** Windows, in a normal command prompt:

```text
mklink /J "<ComfyUI>\custom_nodes\ComfyUI-MossPhoton" "<repository>\ComfyUI-MossPhoton"
```

Linux and macOS: `ln -s <repository>/ComfyUI-MossPhoton <ComfyUI>/custom_nodes/ComfyUI-MossPhoton`. A junction or link
shows every `git pull` immediately; to remove it, delete the link itself (not its target).

**Copy (a fixed snapshot).** Copy the folder into `custom_nodes/`. Later changes of the repository do not reach the
copy; delete the old folder before copying a new version so that no stale file stays behind.

Restart ComfyUI after either way.

## 3. Weights

The workflows name these files. The pack and the workflows download nothing: get the files yourself and put them where
the table says (or pick yours in the loader combos).

| Role | Location under the ComfyUI `models` folder | Notes |
|---|---|---|
| Diffusion model | `diffusion_models/moss_image_sft_bf16.safetensors` | the official MOSS-Image-SFT bf16 file; load it with `UNETLoader` (Load Diffusion Model). Its sha256 is recorded in `core/consts.py` (`SFT_BF16_SHA256`) |
| Text encoder | `text_encoders/qwen3.5_9b_bf16.safetensors` | the official Qwen3.5 9B bf16 text encoder of MOSS-Image. Load with `CLIPLoader`, **type `moss_image`**. Another text encoder file works if you pick it in the combo, but the parity statements are made for the file recorded in the report (DV20) |
| VAE | `vae/flux2-vae.safetensors` | the Flux 2 VAE (128 channels, 16x downscale) |
| Distill LoRA (turbo workflows only) | `loras/lodestones-8-steps-dmd2-lora.safetensors` | the public lodestones 8-step DMD2 LoRA; the pack recognises it by sha256 (see `DEVIATIONS.md`, B1). Keep this file name or pick yours in the combo |
| Character LoRA (character and turbo_identity workflows) | `loras/your_character_lora.safetensors` | a **placeholder name**: copy or rename your own character LoRA (trained for MOSS-Image SFT) to it, or pick it in the `lora_name` combo |
| Second LoRA (`multi_lora` only) | `loras/your_second_lora.safetensors` | a placeholder name for the chaining demo; any MOSS LoRA |

A freshly opened workflow shows "value not in list" on every file you do not have under exactly that name; that is
expected for the placeholders. Open the combo and pick your file. ComfyUI lists a file in a subfolder with a backslash
on Windows (`sub\file.safetensors`) and a forward slash on Linux; the shipped workflows use flat names, so the
spelling does not matter for them.

Quantised weights (int8, fp8) load but are labelled `reference_modified` and are outside the verified matrix.

## 4. First start

- The ComfyUI console must not contain `[MossPhoton] module ... disabled`. A line like that means one node module failed
  to import; the other nodes still load. The message carries the cause.
- Add Node, category **MossPhoton**: four groups `1 Setup`, `2 Sampling`, `3 Diagnostics`, `4 Output`, eight nodes.
- `GET /mossphoton/presets/reference/resolved` (in a browser, same host and port as ComfyUI) returns the effective
  settings of the preset as JSON.
- Sampler nodes show an "effective settings" panel and fill `preset_pin` themselves when the web extension is loaded. The
  pack works without it (the server then warns `preset_unpinned`).
- The first run hashes the checkpoint and every LoRA file (sha256, `fingerprint = cached_or_compute`), which takes a
  while for multi-gigabyte files; the hashes are cached in `hash_cache.json` under ComfyUI's user directory
  (`moss_photon` folder) and keyed by path, size and modification time. The `fingerprint` widget of the LoRA Profile and
  of the Advanced sampler accepts `cached_only` or `off` to skip hashing; then a distill LoRA can not be verified by
  hash.

## 5. Check the install

From the pack folder, with the Python of the ComfyUI you installed into (embedded build shown):

```text
<ComfyUI>\python_embeded\python.exe tests\run_tests.py --suite unit
<ComfyUI>\python_embeded\python.exe tests\run_tests.py --suite integration
<ComfyUI>\python_embeded\python.exe tests\run_tests.py --suite quality
<ComfyUI>\python_embeded\python.exe tests\run_tests.py --suite all
```

- `unit` imports no ComfyUI. `integration` runs the sampler, LoRA and save paths on a tiny random MOSS model on the CPU
  (no weights needed). `quality` is part of `run_tests.py`: it checks the example workflows, the preset docs and the
  release metadata offline. `all` runs the three. The `gpu` suite is opt-in (`MOSS_PHOTON_GPU=1`) and needs a GPU and
  real weights; it is not part of a normal install check.
- `quality` reads a recording of `/object_info` (`tests/quality/object_info_snapshot.json`), so it needs no server. The
  recording holds the file lists of the machine it was made on; the example workflows use neutral file names, and the
  few that are not in the recording are listed as placeholders inside `tests/quality/test_workflows.py`. After a
  ComfyUI update refresh the recording with the script in the development repository (`research/refresh_object_info_snapshot.py`; not shipped in this package; the
  server must be running; GET requests only).

These tests show that the code and the workflows are consistent. They do not show image quality.

## 6. Load an example workflow

Drag a file from `workflows/` onto the ComfyUI canvas, or use Workflow > Open. Each workflow carries a Note with the
ComfyUI version, the weights, the preset status and the experimental features. See `USAGE.md`, section 2.

## 7. Update and remove

- Update: pull the repository (junction) or replace the copied folder, restart ComfyUI. A workflow saved with an older
  preset file keeps its `preset_pin`; when the installed preset differs from the pin the sampler stops with the
  **error** `preset_version_mismatch` before the first forward. Select the preset again so the web extension rewrites
  the pin, or clear the `preset_pin` field to run the installed version (an empty pin only warns `preset_unpinned`).
- Remove: delete the junction or the folder and restart. Nothing else is written outside the folder except the hash
  cache and the images and sidecars the Save node writes into the ComfyUI output folder.
