"""Forward counts and the budget (INV 4.3, design 6.4): cfg 3 -> 2 per step, cfg 1 -> 1 per step, SLG adds none."""
import types
import unittest

from moss_photon.core import errors, presets, sigma, solver

LATENT = types.SimpleNamespace(batch=1)


def sigmas(steps):
    return [1.0 - i / steps for i in range(steps)] + [0.0]


class ForwardsTableTest(unittest.TestCase):
    def test_table(self):
        for steps, cfg, expected in ((50, 3.0, 100), (50, 1.0, 50), (8, 1.0, 8), (16, 2.0, 32), (12, 1.5, 24),
                                     (8, 1.01, 16), (25, 3.0, 50)):
            with self.subTest(steps=steps, cfg=cfg):
                self.assertEqual(sigma.expected_forwards(steps, cfg), expected)

    def test_slg_does_not_add_forwards(self):
        for pid in ("reference", "draft_fast", "advanced"):
            r = presets.resolve(pid)
            self.assertEqual(r.effective["sampling"]["slg_mode"], "uncond_skip")
            plan = solver.plan_run(r, LATENT, sigmas(r.effective["sampling"]["steps"]), max_forwards=0)
            self.assertEqual(plan.expected_branch_forwards, r.effective["sampling"]["steps"] * 2)


class PlanRunTest(unittest.TestCase):
    def plan(self, preset="reference", overrides=None, batch=1, n=None, max_forwards=0):
        r = presets.resolve(preset, overrides=overrides)
        steps = n if n is not None else r.effective["sampling"]["steps"]
        return solver.plan_run(r, types.SimpleNamespace(batch=batch), sigmas(steps), max_forwards=max_forwards)

    def test_reference(self):
        plan = self.plan()
        self.assertEqual((plan.steps, plan.cfg, plan.branches, plan.extra), (50, 3.0, 2, 0))
        self.assertEqual((plan.expected_branch_forwards, plan.expected_sample_forwards), (100, 100))
        self.assertIsNone(plan.limit)

    def test_batch_multiplies_the_sample_count_only(self):
        plan = self.plan(batch=3)
        self.assertEqual((plan.expected_branch_forwards, plan.expected_sample_forwards), (100, 300))

    def test_cfg_one_is_a_single_branch(self):
        plan = self.plan(overrides={"cfg": 1.0})
        self.assertEqual((plan.branches, plan.expected_branch_forwards), (1, 50))

    def test_custom_sigmas_replace_the_preset_steps(self):
        self.assertEqual(self.plan(n=10).expected_branch_forwards, 20)

    def test_extra_forwards_are_added(self):
        r = presets.resolve("reference")
        r.effective["budget"]["extra_forwards"] = 2
        plan = solver.plan_run(r, LATENT, sigmas(25), max_forwards=0)
        self.assertEqual(plan.expected_branch_forwards, 52)

    def test_advanced_limit_exceeded(self):
        with self.assertRaises(errors.BudgetExceeded) as cm:
            self.plan(max_forwards=99)
        self.assertEqual(cm.exception.issue.detail["planned"], 100)
        self.assertEqual(cm.exception.issue.detail["limit"], 99)
        self.assertIn("lower steps or raise max_forwards", str(cm.exception))

    def test_advanced_limit_exactly_met_passes(self):
        self.assertEqual(self.plan(max_forwards=100).limit, 100)

    def test_zero_means_unlimited(self):
        self.assertIsNone(self.plan(max_forwards=0).limit)

    def test_preset_budget_limit(self):
        r = presets.resolve("reference")
        r.effective["budget"]["max_forwards"] = 40
        with self.assertRaises(errors.BudgetExceeded):
            solver.plan_run(r, LATENT, sigmas(50), max_forwards=0)
        self.assertEqual(solver.plan_run(r, LATENT, sigmas(20), max_forwards=0).limit, 40)

    def test_the_smaller_limit_is_reported(self):
        r = presets.resolve("reference")
        r.effective["budget"]["max_forwards"] = 500
        self.assertEqual(solver.plan_run(r, LATENT, sigmas(50), max_forwards=150).limit, 150)


if __name__ == "__main__":
    unittest.main()
