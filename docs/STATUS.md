# Status

Version 0.4.0. What is implemented, what was measured and what was not, part by part. Scope: SFT only. Base: not verified in v1.

| Part | Status |
|---|---|
| 12 nodes: 8 (Sampler, Sampler Advanced, Schedule, LoRA Profile (deprecated), Prompt & Layout, Empty Latent, Diagnostics, Save Image), plus LoRA Phase, Prompt Split, Hires Fix (experimental) and Identity Detailer (experimental) | implemented; unit and CPU integration tests on a tiny random model |
| Identity Detailer (`MossPhotonIdentityDetailer`) | experimental; verified on the turbo stack only. One GPU smoke, 3 full-body seeds at 1 MP: the ArcFace gate accepted 1 (0.467 to 0.509) and rolled back 2, leaving 0 pixels changed; see [USAGE](USAGE.md), 3.10 |
| Preset `reference` | `stable` label = the vendor recipe (50 Euler steps, CFG 3, shift 5, SLG layer 22). Measured: bit-exact vs the native graph on 10 fixed cases (ComfyUI 0.42.0 `8f04fd6ec`, RTX 5090, pack `5b33c0e`): equal latents and images, equal sigma and forward counts, F = 0 |
| Determinism and stability of `reference` | measured: 10 cases x 3 repeats give identical latents; 30 stability jobs (interleaved LoRA stacks and sizes) with no failed job (the torch allocator counter stays flat under dynamic VRAM and proves nothing about leaks, so device memory was measured instead). The sizes are smaller than SPEC section 10 by the maintainers' decision, see [DEVIATIONS](DEVIATIONS.md), B10; device-level VRAM growth over 30 jobs was 0.00 % (`reference`) and 2.63 % (turbo), limit 5 %; measured after unload with the Windows per-process counter, so it is a residual, not a peak |
| Presets `photo_safe`, `identity_safe`, `wardrobe_safe`, `scene_safe` | `candidate`; numerically equal to reference, they differ in input contract and report |
| Presets `night_photo`, `advanced` | `experimental` |
| Preset `draft_fast` | `preview_only` |
| Presets `turbo_8step`, `turbo_identity` | `candidate`. Measured: bit-exact vs the native turbo graph on 10 cases plus 3 corner cases (16 steps at CFG 2, 12 at 1.5), determinism on 12 cases x 3, 30 stability jobs, mechanics checks. The quality sheets (G5) and the second identity check (D2) were **not run**, so nothing is known about the quality of 8-step DMD2 on SFT; not recommended as final |
| Presets `turbo_darkroom_portrait`, `turbo_darkroom_fullbody` (darkroom phases: compose, commit, finish) | `candidate`, uncalibrated, experimental (label `not_reference`, reason `darkroom_experimental`). They run in the two samplers (the phase knobs are in the Advanced sampler); no separate darkroom node. Nothing is known about their quality beyond the maintainers' own look at a few images; see [USAGE](USAGE.md), 3.5 |
| Base checkpoints | recognised by hash, not verified |
| Calibration stages B to D | not done |
| Identity / clothing scoring, masked detailer, seed drafts | not in v1 (no scoring node; the Identity Detailer has its own ArcFace accept gate, see its row) |
| Upscale | no upscale node and no pixel upscaler in v1. The one exception is the second pass of a hires fix, `MossPhotonHiresFix` (its default upscale is Lanczos in pixels, with a latent-bicubic option): **experimental**, its defaults measured on two seeds and one prompt only; see [USAGE](USAGE.md), 3.9 |
| Layout JSON, text-encoder LoRA, Advanced sampler | experimental |
| Phased LoRA weight (`profile = phased`, Advanced sampler only) | experimental, outside every verification; clone per level, label `not_reference`; with a constant weight it is bit-exact to the stock loader in one GPU smoke case; see [USAGE](USAGE.md), 3.3.1 |
