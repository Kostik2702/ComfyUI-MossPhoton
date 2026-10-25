# ComfyUI-MossPhoton

ComfyUI nodes for text-to-image with MOSS-Image: a sampler that is bit-equal to the native graph on the 10 fixed cases measured on one GPU (see Status), with versioned presets, a LoRA profile
that stacks adapters without leaking them between jobs, a prompt node that never rewrites your text, and diagnostics and
a save node that record what really ran.

Version 0.1.0, MIT licence. Tested only on ComfyUI 0.42.0 (commit `8f04fd6ec`), frontend 1.55.18, Python 3.13.

**Scope: SFT only. Base: not verified in v1.** Calibration stages B to D are not done. The
presets carry the vendor's reference recipe; the pack makes no claim about image quality, identity or realism.

## Status

| Part | Status |
|---|---|
| 8 nodes (Sampler, Sampler Advanced, Schedule, LoRA Profile, Prompt & Layout, Empty Latent, Diagnostics, Save Image) | implemented; unit and CPU integration tests on a tiny random model |
| Preset `reference` | `stable` label = the vendor recipe (50 Euler steps, CFG 3, shift 5, SLG layer 22). Measured: bit-exact vs the native graph on 10 fixed cases (ComfyUI 0.42.0 `8f04fd6ec`, RTX 5090, pack `5b33c0e`): equal latents and images, equal sigma and forward counts, F = 0 |
| Determinism and stability of `reference` | measured: 10 cases x 3 repeats give identical latents; 30 stability jobs (interleaved LoRA stacks and sizes) with no failed job and no allocator growth. The sizes are smaller than SPEC section 10 by the owner's decision, see [DEVIATIONS](docs/DEVIATIONS.md), B10; device-level VRAM growth over 30 jobs was 0.00 % (`reference`) and 2.63 % (turbo), limit 5 %; measured after unload with the Windows per-process counter, so it is a residual, not a peak |
| Presets `photo_safe`, `identity_safe`, `wardrobe_safe`, `scene_safe` | `candidate`; numerically equal to reference, they differ in input contract and report |
| Presets `night_photo`, `advanced` | `experimental` |
| Preset `draft_fast` | `preview_only` |
| Presets `turbo_8step`, `turbo_identity` | `candidate`. Measured: bit-exact vs the native turbo graph on 10 cases plus 3 corner cases (16 steps at CFG 2, 12 at 1.5), determinism on 12 cases x 3, 30 stability jobs, mechanics checks. The quality sheets (G5) and the second identity check (D2) were **not run**, so nothing is known about the quality of 8-step DMD2 on SFT; not recommended as final |
| Base checkpoints | recognised by hash, not verified (MP-19) |
| Calibration stages B to D | not done |
| Identity / clothing scoring, masked detailer, upscale, seed drafts | not in v1 |
| Layout JSON, text-encoder LoRA, Advanced sampler | experimental |
| Phased LoRA weight (`profile = phased`, Advanced sampler only) | experimental, outside every verification; clone per level, label `not_reference`; with a constant weight it is bit-exact to the stock loader in one GPU smoke case; see [USAGE](docs/USAGE.md), 3.3.1 |

## Quick start

1. Install (no extra pip packages): `cd <ComfyUI>/custom_nodes` and
   `git clone https://github.com/Kostik2702/ComfyUI-MossPhoton`, or see [docs/INSTALL.md](docs/INSTALL.md) for a
   junction or a copy.
2. Drag `workflows/sft_reference.json` onto the ComfyUI canvas. Download the weights named in its Note (and in
   INSTALL, section 3) or pick your own in the loaders; the LoRA workflows use placeholder names you replace.
3. Read [docs/USAGE.md](docs/USAGE.md): nodes one by one, the rule "cfg 0 = take the preset's value", `preset_pin`, the
   effective-settings panel and the turbo workflow.

## Documentation

- [Install](docs/INSTALL.md)
- [Usage](docs/USAGE.md)
- [Calibration status](docs/CALIBRATION.md)
- [Deviations from SPEC](docs/DEVIATIONS.md)
- [Changelog](docs/CHANGELOG.md)
- Presets: [reference](docs/presets/reference.md), [photo_safe](docs/presets/photo_safe.md),
  [identity_safe](docs/presets/identity_safe.md), [wardrobe_safe](docs/presets/wardrobe_safe.md),
  [scene_safe](docs/presets/scene_safe.md), [night_photo](docs/presets/night_photo.md),
  [draft_fast](docs/presets/draft_fast.md), [advanced](docs/presets/advanced.md),
  [turbo_8step](docs/presets/turbo_8step.md), [turbo_identity](docs/presets/turbo_identity.md)

## Licence

MIT, see [LICENSE](LICENSE).
