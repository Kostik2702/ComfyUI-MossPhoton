"""Determinism gates (SPEC 10: 10 cases x 3 repeats; docs/03-turbo.md 5.6 G2). Opt-in, see ``_env``.

Three Photon repeats per case. Equal ``output.latent_sha256`` always passes. A difference is judged by the OD3 rule:
the case's own ``F`` is measured with two extra stock runs; ``F == 0`` means the backend is exact on this case, so any
Photon difference fails; ``F > 0`` allows ``max pair difference <= 2F`` (``verdict.repeats_ok``).
"""
import json
import os
import unittest

from . import _env

S = None


def setUpModule():
    global S
    S = _env.session()


def pick(ids):
    return ids[:1] if _env.size() == "smoke" else ids


class _Base(unittest.TestCase):
    def run_cases(self, specs, name):
        import extras
        S.gallery.plan(cases=len(specs), runs=3 * len(specs))
        rows = []
        for spec in specs:
            row = extras.determinism_row(S.assets, spec, 3, None, label="[det %s %s]" % (spec["gate"], spec["id"]))
            rows.append(row)
            S.gallery.add_determinism(row)
        S.harness.write_json(os.path.join(S.out_dir, name), {"schema_version": 1, "size": _env.size(), "rows": rows})
        bad = [r for r in rows if r["verdict"] == "FAIL"]
        self.assertEqual(bad, [], json.dumps(bad))


class ReferenceDeterminismTest(_Base):
    def test_reference_three_repeats(self):
        import gate
        ids = pick(S.cases["determinism_gate"]["cases"])
        self.run_cases(gate.specs_for(S.cases, "reference", ids), "determinism.json")


class TurboDeterminismTest(_Base):
    def test_turbo_default_three_repeats(self):
        import gate
        ids = pick(S.cases["turbo"]["g2_determinism"]["default_cases"])
        self.run_cases(gate.specs_for(S.cases, "turbo", ids), "determinism_turbo.json")

    @unittest.skipIf(_env.size() == "smoke", "corner determinism (G2) is part of the full size")
    def test_turbo_corner_three_repeats(self):
        import gate
        g1c = S.cases["turbo"]["g1c_corner"]
        specs = []
        for item in S.cases["turbo"]["g2_determinism"]["corner_cases"]:
            by_id = {c["id"]: c for c in S.cases["cases"]}
            specs.append(S.harness.build_spec(S.cases, "turbo", by_id[item["id"]], variant="corner", overrides={
                "steps": item["steps"], "cfg": item["cfg"], "sigmas_sha256": g1c["sigmas_sha256"]}))
        self.run_cases(specs, "determinism_turbo_corner.json")


if __name__ == "__main__":
    unittest.main()
