"""MP-88 review: the dead-LoadedModel check of ``tests/run_tests.py`` must not fail open.

If a comfy version lacks an API the check calls, strict mode (the default for the integration suite) has to fail the run: a
check that switches itself off and exits 0 attests to nothing. Without strict mode the old degrade stays (report once, go on).
No comfy is imported: a stand-in ``comfy.model_management`` is put into ``sys.modules`` for one test and taken out again.
"""
import io
import sys
import types
import unittest

from tests import run_tests


class Entry:
    def is_dead(self):
        return False


class LeakCheckFailsOpenTest(unittest.TestCase):
    def setUp(self):
        mm = types.ModuleType("comfy.model_management")
        mm.current_loaded_models = [Entry()]          # no ``cleanup_models``: the API the check calls is missing
        mm.unload_all_models = lambda: None
        self.assertNotIn("comfy.model_management", sys.modules)
        sys.modules["comfy.model_management"] = mm
        self.addCleanup(sys.modules.pop, "comfy.model_management", None)
        cls = run_tests._TimedResult
        saved = {name: getattr(cls, name, False) for name in ("model_cleanup", "track_leaks", "strict")}
        self.addCleanup(lambda: [setattr(cls, name, value) for name, value in saved.items()])

    def result(self, strict):
        cls = run_tests._TimedResult
        cls.model_cleanup, cls.track_leaks, cls.strict = True, True, strict
        return cls(io.StringIO(), False, 1)

    def release(self, result):
        stderr, sys.stderr = sys.stderr, io.StringIO()
        try:
            result._release_models("tests.x.Case.test_y")
        finally:
            sys.stderr = stderr

    def test_strict_mode_fails_the_run_when_the_check_cannot_run(self):
        result = self.result(strict=True)
        self.assertTrue(result.wasSuccessful(), "control: nothing went wrong before the check ran")
        self.release(result)
        self.assertIsNotNone(result.leak_check_error)
        self.assertFalse(result.wasSuccessful())

    def test_without_strict_mode_the_check_degrades_and_the_run_goes_on(self):
        result = self.result(strict=False)
        self.release(result)
        self.assertIsNone(result.leak_check_error)
        self.assertTrue(result.wasSuccessful())
        self.assertFalse(run_tests._TimedResult.track_leaks, "the check switched itself off, as before")


if __name__ == "__main__":
    unittest.main()
