"""Stability and patch-leak gates on the GPU: reference (SPEC 10: 100 jobs, design 6.7) and turbo G3
(docs/03-turbo.md 5.6). Opt-in, see ``_env``. The sequence engine lives in ``research/parity/extras.py``.

Reference: one base MODEL, a cycle of 10 jobs [no LoRA, A, no LoRA, B, no LoRA, A+B, no LoRA, A, no LoRA, B], seeds and
sizes cycling; jobs 7, 19, 41, 73 are cancelled at step 10, jobs 13 and 57 get a NaN output. Turbo G3: a cycle of 10 over
the turbo and reference presets (two jobs must raise ``missing_lora`` / ``incompatible_lora`` before sampling), cancel at
step 3. After EVERY job: base MODEL untouched, every output equals its own first run (exact in deterministic mode, else
within 2F), no LoRA output equals its no-LoRA counterpart, every 5th job 5 sampled base weights equal the snapshot, at
jobs 10, 20, ... allocated memory grows <= 5% against job 1, DEVICE-level VRAM of our process (NVML / nvidia-smi / Windows
GPU Process Memory counter, after unload_all_models + soft_empty_cache + gc; torch's allocated counter is flat under
dynamic VRAM) grows <= 5% against job 1, and live ModelPatcher objects stay within baseline + 2.

Size: ``smoke`` runs ``MOSS_PHOTON_GPU_JOBS`` jobs (default 10), ``full`` the 100 of ``cases.json``.
"""
import copy
import json
import os
import unittest

from . import _env

S = None
JOBS_ENV = "MOSS_PHOTON_GPU_JOBS"


def setUpModule():
    global S
    S = _env.session()


def total_jobs(limit):
    value = int(os.environ.get(JOBS_ENV, 10 if _env.size() == "smoke" else limit))
    return min(value, limit)


class ReferenceStabilityTest(unittest.TestCase):
    def test_reference_sequence(self):
        import extras
        cfg = S.cases["stability"]
        jobs = extras.stability_jobs_reference(S.cases, total_jobs(cfg["jobs"]))
        result = extras.run_sequence(S.assets, jobs, cfg, tol=0.0, live=S.gallery, label="[stability]", group="reference")
        S.harness.write_json(os.path.join(S.out_dir, "stability.json"), dict(result, size=_env.size()))
        self.assertEqual([(r["job"], r["problems"]) for r in result["failures"]], [], json.dumps(result["failures"])[:2000])


class TurboStabilityTest(unittest.TestCase):
    def test_turbo_g3_sequence(self):
        import extras
        g3 = S.cases["turbo"]["g3_stability"]
        cfg = copy.deepcopy(S.cases["stability"])
        cfg["cancel_step"] = g3["cancel_step"]
        jobs = extras.stability_jobs_turbo(S.cases, total_jobs(g3["jobs"]))
        result = extras.run_sequence(S.assets, jobs, cfg, tol=0.0, live=S.gallery, label="[g3]", group="turbo G3")
        S.harness.write_json(os.path.join(S.out_dir, "stability_turbo.json"), dict(result, size=_env.size()))
        self.assertEqual([(r["job"], r["problems"]) for r in result["failures"]], [], json.dumps(result["failures"])[:2000])


if __name__ == "__main__":
    unittest.main()
