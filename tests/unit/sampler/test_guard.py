"""guard.check_finite / guard_items / required_guards (design 1.5, 6.4)."""
import sys
import unittest

from moss_photon.core import errors, guard, presets


class FakeTensor:
    """Has ``isfinite().all()`` like a torch tensor; torch is not needed."""

    def __init__(self, finite):
        self._finite = finite

    def isfinite(self):
        outer = self

        class Mask:
            def all(self):
                return outer._finite
        return Mask()


class CheckFiniteTest(unittest.TestCase):
    def test_finite_passes(self):
        self.assertIsNone(guard.check_finite(FakeTensor(True), node="N", where="step 0"))

    def test_non_finite_raises_with_the_place(self):
        with self.assertRaises(errors.NonFiniteOutput) as cm:
            guard.check_finite(FakeTensor(False), node="MossPhotonSampler", where="step 3")
        self.assertIn("step 3", str(cm.exception))
        self.assertEqual(cm.exception.issue.detail, {"where": "step 3"})

    def test_torch_is_not_imported(self):
        import subprocess
        code = ("import sys\nsys.path.insert(0, sys.argv[1])\nfrom moss_photon.core import guard\n"
                "assert 'torch' not in sys.modules and 'comfy' not in sys.modules\nprint('OK')\n")
        import os
        pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        proc = subprocess.run([sys.executable, "-I", "-c", code, pkg], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-400:])


class GuardItemsTest(unittest.TestCase):
    def test_every_guard_of_every_shipped_preset_is_not_run_with_a_reason(self):
        seen = 0
        for pid, preset in presets.catalog().items():
            items = guard.guard_items(preset)
            self.assertEqual([i["id"] for i in items], [g["id"] for g in preset["guards"]], pid)
            for item in items:
                seen += 1
                self.assertEqual(item["status"], "not_run")
                self.assertIn(item["reason"], ("missing_guard_dependency", "not_implemented_v1"))
        self.assertGreater(seen, 0, "control: the catalog has guards to look at")

    def test_reason_follows_the_declared_dependency(self):
        preset = {"guards": [
            {"id": "a", "required": False, "requires_dependency": "insightface"},
            {"id": "b", "required": False, "requires_dependency": "none"},
            {"id": "c", "required": True}]}
        self.assertEqual(guard.guard_items(preset), [
            {"id": "a", "status": "not_run", "reason": "missing_guard_dependency"},
            {"id": "b", "status": "not_run", "reason": "not_implemented_v1"},
            {"id": "c", "status": "not_run", "reason": "not_implemented_v1"}])
        self.assertEqual(guard.required_guards(preset), ["c"])

    def test_no_guards(self):
        self.assertEqual(guard.guard_items({"guards": []}), [])
        self.assertEqual(guard.guard_items({}), [])
        self.assertEqual(guard.required_guards({}), [])

    def test_the_preset_is_not_mutated(self):
        preset = presets.catalog()["identity_safe"]
        before = repr(preset)
        guard.guard_items(preset)
        self.assertEqual(repr(preset), before)


if __name__ == "__main__":
    unittest.main()
