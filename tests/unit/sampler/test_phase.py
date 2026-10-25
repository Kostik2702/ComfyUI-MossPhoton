"""phase.py is ONLY the extension point (design 1.5, LD10, 6.4)."""
import ast
import os
import unittest

from moss_photon.core import errors, phase


class RejectPhasePlanTest(unittest.TestCase):
    def test_no_profile_and_no_plan_pass(self):
        self.assertIsNone(phase.reject_phase_plan(None, node="N"))
        self.assertIsNone(phase.reject_phase_plan({"phase_plan": None}, node="N"))
        self.assertIsNone(phase.reject_phase_plan({}, node="N"))

    def test_a_plan_raises_unsupported_feature_naming_mp16(self):
        with self.assertRaises(errors.UnsupportedFeature) as cm:
            phase.reject_phase_plan({"phase_plan": {"kind": "phased"}}, node="MossPhotonSampler")
        self.assertIn("MP-16", str(cm.exception))
        self.assertEqual(cm.exception.code, "unsupported_feature")

    def test_the_module_holds_nothing_but_the_extension_point(self):
        path = os.path.join(os.path.dirname(phase.__file__), "phase.py")
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        functions = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        self.assertEqual(functions, ["reject_phase_plan"])


if __name__ == "__main__":
    unittest.main()
