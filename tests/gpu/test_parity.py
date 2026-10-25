"""Parity gates on the GPU: reference (design 6.7) and turbo (docs/03-turbo.md 5.6 G1/G1c). Opt-in, see ``_env``.

``smoke`` size runs C01 only; ``full`` runs the 20 fixed cases (the turbo G1c corner set after them). The verdict of
every case is the OD3 rule of ``research/parity/verdict.py``; a failure keeps its artefacts in
``research/parity/results/<date>/tests``.
"""
import json
import os
import unittest

from . import _env

S = None


def setUpModule():
    global S
    S = _env.session()


def case_ids(cases):
    return ["C01"] if _env.size() == "smoke" else [c["id"] for c in cases["cases"]]


def dump(name, doc):
    S.harness.write_json(os.path.join(S.out_dir, name), doc)


class EnvGateTest(unittest.TestCase):
    """G0: the environment gate is recorded and its conditions hold (no --fast, no attention override, SFT sha)."""

    def test_reference_env_gate(self):
        env = S.harness.collect_env(S.assets, S.cases, gate="reference", machine=S.machine)
        dump("env.json", env)
        self.assertTrue(env["gate_conditions"]["sft_sha_matches_official"], env["files"]["checkpoint"])
        self.assertTrue(env["gate_conditions"]["no_fast"], env["fast_flags"])
        self.assertTrue(env["gate_conditions"]["no_attention_override"])
        self.assertTrue(env["gate_conditions"]["te_is_not_official"])
        self.assertFalse(env["text_encoder"]["official"])

    def test_turbo_env_gate(self):
        env = S.harness.collect_env(S.assets, S.cases, gate="turbo", machine=S.machine)
        dump("env_turbo.json", env)
        self.assertTrue(env["turbo"]["distill_matches"], env["turbo"])
        self.assertIn(env["turbo"]["lora_compute_dtype"], ("torch.float16", "torch.float32"))


class ReferenceParityTest(unittest.TestCase):
    def test_reference_parity(self):
        import gate
        specs = gate.specs_for(S.cases, "reference", case_ids(S.cases))
        doc = gate.run_reference_like_gate(S.assets, S.cases, "reference", specs, S.out_dir, label="[ref]")
        dump("parity.json", doc)
        failed = [(r["id"], r["verdict"]["failed"]) for r in doc["cases"] if not r["verdict"]["pass"]]
        self.assertEqual(failed, [], json.dumps(failed))


class TurboParityTest(unittest.TestCase):
    def test_turbo_parity_default(self):
        import gate
        specs = gate.specs_for(S.cases, "turbo", case_ids(S.cases))
        doc = gate.run_reference_like_gate(S.assets, S.cases, "turbo", specs, S.out_dir, label="[turbo]")
        dump("parity_turbo.json", doc)
        failed = [(r["id"], r["verdict"]["failed"]) for r in doc["cases"] if not r["verdict"]["pass"]]
        self.assertEqual(failed, [], json.dumps(failed))

    @unittest.skipIf(_env.size() == "smoke", "corner set (G1c) is part of the full size")
    def test_turbo_parity_corner(self):
        import gate
        corner = S.cases["turbo"]["g1c_corner"]
        specs = []
        for item in corner["cases"]:
            by_id = {c["id"]: c for c in S.cases["cases"]}
            case = by_id[item["id"]]
            spec = S.harness.build_spec(S.cases, "turbo", case, variant="corner", overrides={
                "steps": corner["steps"], "cfg": corner["cfg"], "sigmas_sha256": corner["sigmas_sha256"]})
            self.assertEqual(spec["preset"], item["preset"])
            specs.append(spec)
        doc = gate.run_reference_like_gate(S.assets, S.cases, "turbo", specs, S.out_dir, probe_ids=("C01", "C19"),
                                           label="[turbo-corner]")
        dump("parity_turbo_corner.json", doc)
        failed = [(r["id"], r["verdict"]["failed"]) for r in doc["cases"] if not r["verdict"]["pass"]]
        self.assertEqual(failed, [], json.dumps(failed))


if __name__ == "__main__":
    unittest.main()
