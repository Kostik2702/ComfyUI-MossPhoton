# Calibration status

Version 0.1.0. Short version: **stage A (parity) is done for SFT within reduced sizes; nothing else is calibrated.** The
presets carry the recipe of the vendor's reference graph (or, for turbo, of the LoRA card). No calibration manifest
exists. The GPU parity, determinism and stability checks were run on 2026-10-25 at sizes smaller than SPEC 10 (owner
decision, `DEVIATIONS.md` B10); the quality gates of SPEC 10 were not run. Scope: SFT only. Base: not verified in v1.

## 1. What exists today

| Item | State | Evidence |
|---|---|---|
| Calibration manifests (`moss_photon/calibration/*.json`) | none; the folder holds no manifest. Every preset reports `calibration.state` `uncalibrated` (`reference`: `official`, meaning the recipe is the vendor's) and `manifest: null`; the resolver then runs the preset numbers and adds the info `manifest_missing` | `presets/*.json`; `moss_photon/calibration/` |
| Sigma schedule of the reference recipe | the 50-step `sgm_uniform` array at shift 5 was computed on the CPU from ComfyUI's own code; its sha256 (`9b2c66b3fb335d998fc03e73ddbafe997fe2d3cf3b187f51629bdf0332f5a536`) is a constant of the pack and a unit test compares the Schedule node with the native function. The array with a CUDA-resident model sampling is unverified | `core/consts.py`; docs/01-investigation.md 1.3 |
| Turbo schedule | 8-step `simple` array, sha256 `c4fe20a8f589897444d75ce8b7a6bb594b7c5c991c8e8bc7c1d18b3cdc20f693`, computed on the CPU | docs/03-turbo.md 3.2 |
| Tests | unit and integration tests run on a tiny random 24-layer MOSS on the CPU (no weights). They check code paths (schedule, noise, SLG, LoRA isolation, save) against the native ComfyUI functions on that fixture | `tests/` |
| GPU parity of `reference` with the native graph | **measured, reduced size:** 10 fixed cases (C01 C02 C03 C04 C07 C09 C13 C17 C19 C20) bit-exact (mode exact, F = 0): equal latent, image, sigma array and forward counts. ComfyUI 0.42.0 `8f04fd6ec`, torch 2.9.0+cu130, CUDA 13.0, RTX 5090, pack commit `5b33c0e`, clean tree | in the development repository (`research/parity/results/2026-10-25/reference-153349/`: `parity.md`, `env.json`) |
| Turbo gate (own gate of `turbo_8step` / `turbo_identity`) | **measured, reduced size:** G1 10/10 and 3/3 corner cases (16@CFG 2, 12@1.5) bit-exact against the native turbo graph; G2 determinism 12 cases x 3 identical; G3 30 stability jobs without a failed job (live patcher count went from 3 to 4 and stayed 4, the gate's pass rule accepted it); G4 mechanics 5/5. **G5 (observational quality sheets) and D2 were not run** (owner time decision) | in the development repository (`research/parity/results/2026-10-25/turbo-152509/summary.json`) |
| Real-weight behaviour (bf16 attention, fused kernels, weights streaming) | not measured | design 8.3 |

Bit-exact parity with the native graph is measured on the cases above only; it says nothing about image quality or
identity. Nothing in this repository states a measured quality or a measured identity effect.

## 2. Stages of SPEC 9 and their state

| Stage | Content | State |
|---|---|---|
| A, reference | record versions, checkpoint hashes, CUDA/PyTorch, sigma array and the control graph; parity of `reference` on 1024x1024, a portrait and a wide frame | **done for SFT at reduced size** (10 parity cases instead of 20, determinism 10 x 3, stability 30 jobs at 10 steps instead of 100 at 50; protocol in design 6.7; owner decision, `DEVIATIONS.md` B10) |
| B, trajectory measurement | decode intermediate `denoised` tensors, find when layout, face, clothes and background settle; write phase limits into manifests | **not done**, outside v1 |
| C, one-factor A/B | CFG profile, SLG on/off and layers, constant vs phased LoRA weight, strength, LoRA order, other solvers, noise, a second pass, upscale and detailer; one factor per experiment | **not done**, outside v1 |
| D, preset release | a value enters `photo_safe` or `identity_safe` only if the improvement repeats in a blind comparison and breaks no hard gate; each preset gets a manifest | **not done**, outside v1 |
| Turbo gate | own gate for `turbo_8step` / `turbo_identity`: parity with the native turbo graph, determinism, stability, mechanics, observational quality sheets | **G0 to G4 done at reduced size; G5 and D2 not run** (docs/03-turbo.md 5.6) |
| Base | same gates on a Base checkpoint | **not started** (MP-19) |

## 3. Gates of SPEC 10 (parity, determinism and stability run at reduced size; the quality gates not run)

| Gate | Criterion (SPEC 10) | State |
|---|---|---|
| Reference parity | `reference` equals the native graph on 20 fixed cases, same sigma and forward count | passed on 10 cases (reduced by the owner), bit-exact |
| Determinism | 3 repeats of 10 cases give equal hashes or stay inside the approved tolerance | passed on 10 cases x 3 repeats, identical latents (the same size as SPEC 10) |
| Face | `identity_safe` not worse than the constant LoRA in at least 95 % of pairs by human vote | not run; no identity score in v1 |
| Clothes / scene | annotated violations not above `reference` plus 2 points; critical substitutions 0 | not run |
| Realism | blind pairwise preference of `photo_safe` over `reference` at least 55 % with a 95 % bootstrap interval above 50 % | not run |
| Masks | exact source pixels outside M=0 | not applicable in v1 (no masked node) |
| Stability | 100 jobs interleaving Base/SFT and LoRA with no crash, GPU leak above 5 % or leftover patch | 30 jobs at 10 steps, no failed job, no allocator growth; device-level VRAM of our process after unload (Windows counter only, residual not peak): `reference` 0.00 % and turbo G3 2.63 % growth over 30 jobs, limit 5 % (`DEVIATIONS.md` B10); isolation also tested on the CPU fixture |
| Compatibility | Base/SFT bf16 on a fixed ComfyUI; quantised separately | SFT bf16 run on the one machine above; quantised not claimed |

The thresholds of SPEC 10 are project requirements, not properties proven for MOSS. They may be changed only before the
closed test set is looked at.

## 4. What a calibration run must record

SPEC 6.4: versions of ComfyUI, the pack, Torch/CUDA; model, VAE and LoRA hashes; dataset and seed list; metrics with
confidence intervals; date. A change of solver, sigma, CFG/SLG, LoRA timing or postprocess bumps `preset_version` and
repeats the evaluation. Per preset: effective parameters (see `presets/<id>.md`), A/B report, examples of successes and
failures, worst cases. The A/B sections of the preset docs say "pending MP-12 / calibration" (generated text); parity is measured as above, the A/B calibration is not done.

## 5. Where the protocol lives

- SPEC section 9 (stages A to D) and section 10 (evaluation set and gates): `SPEC.md` in the repository root.
- GPU harness, case list C01 to C20, tolerance rule and stability sequence: `docs/02-design.md` section 6.7.
- Turbo gate G0 to G5 and D2: `docs/03-turbo.md` section 5.6.
- Open risks that only a GPU run can close: `docs/02-design.md` section 8.3 and `docs/03-turbo.md` section 7.

## 6. Known unverified points

DMD2 (trained on Base) on SFT; the bf16 merge of the fp32 distill delta; steps above 8 and CFG above 1.1 in turbo;
identity under the distill LoRA; a clean `CoreModelPatcher` on the real model reporting no foreign modifiers; the
`lora_compute_dtype` of the GPU; Each is a question for a later ticket, not an answered one. Bit parity of `reference` and
the turbo presets on the real GPU is answered for the 10 + 3 cases above on one machine (RTX 5090) and one ComfyUI
build; it has to be re-checked after any ComfyUI upgrade.
