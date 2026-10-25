"""MP-16: the phase math (SPEC 5): progress by cumulative |d sigma|, smoothstep, the weight, the plan and its schedule.

Every expectation below is computed by a second, independent route (a plain loop, the textbook polynomial), so a change in
``core/phase.py`` that bends the math cannot also bend its test.
"""
import unittest

from moss_photon.core import errors, phase

UNIFORM = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
# the shape of the native shifted schedule: dense near 1, sparse near 0
SHIFTED = [5 * t / (1 + 4 * t) for t in [1 - i / 20 for i in range(21)]]


def independent_progress(sigmas):
    """r at the start of every step, written as 'distance walked so far over the whole distance'."""
    total = sum(abs(sigmas[i + 1] - sigmas[i]) for i in range(len(sigmas) - 1))
    return [sum(abs(sigmas[j + 1] - sigmas[j]) for j in range(k)) / total for k in range(len(sigmas) - 1)]


def textbook_weight(r, w0, w1, a, b):
    t = min(1.0, max(0.0, (r - a) / (b - a)))
    return w0 + (w1 - w0) * (3 * t ** 2 - 2 * t ** 3)


class SmoothstepTest(unittest.TestCase):
    def test_polynomial_and_fixed_points(self):
        for i in range(0, 1001):
            t = i / 1000.0
            self.assertAlmostEqual(phase.smoothstep(t), 3 * t ** 2 - 2 * t ** 3, places=14)
        self.assertEqual((phase.smoothstep(0.0), phase.smoothstep(0.5), phase.smoothstep(1.0)), (0.0, 0.5, 1.0))

    def test_clamped_outside_the_unit_interval(self):
        self.assertEqual((phase.smoothstep(-3.0), phase.smoothstep(7.5)), (0.0, 1.0))

    def test_monotone(self):
        values = [phase.smoothstep(i / 500.0) for i in range(501)]
        self.assertEqual(values, sorted(values))


class ProgressTest(unittest.TestCase):
    def test_uniform_steps(self):
        got = phase.progress_fractions(UNIFORM)
        for g, w in zip(got, [0.0, 0.2, 0.4, 0.6, 0.8]):
            self.assertAlmostEqual(g, w, places=12)

    def test_matches_an_independent_computation_on_a_shifted_schedule(self):
        for sigmas in (SHIFTED, UNIFORM):
            for g, w in zip(phase.progress_fractions(sigmas), independent_progress(sigmas)):
                self.assertAlmostEqual(g, w, places=12)

    def test_it_is_the_walked_distance_not_the_step_number(self):
        steps = len(SHIFTED) - 1
        by_index = [k / steps for k in range(steps)]
        got = phase.progress_fractions(SHIFTED)
        self.assertGreater(max(abs(g - i) for g, i in zip(got, by_index)), 0.05)

    def test_starts_at_zero_and_stays_below_one_and_increases(self):
        got = phase.progress_fractions(SHIFTED)
        self.assertEqual(got[0], 0.0)
        self.assertLess(got[-1], 1.0)
        self.assertEqual(list(got), sorted(got))
        self.assertEqual(len(got), len(SHIFTED) - 1)

    def test_a_rising_array_gives_the_same_progress(self):
        up = list(reversed(UNIFORM))
        for g, w in zip(phase.progress_fractions(up), phase.progress_fractions(UNIFORM)):
            self.assertAlmostEqual(g, w, places=12)

    def test_degenerate_arrays_are_typed_errors(self):
        for bad in ([1.0], [], [0.5, 0.5, 0.5], [1.0, float("nan"), 0.0], [1.0, float("inf")]):
            with self.subTest(bad=bad):
                with self.assertRaises(errors.UnsupportedFeature):
                    phase.progress_fractions(bad)


class WeightTest(unittest.TestCase):
    def test_equal_weights_are_exactly_the_constant_for_every_r(self):
        for w in (0.0, 0.3, 0.8, 1.0, -0.5):
            for i in range(0, 101):
                self.assertEqual(phase.lora_weight(i / 100.0, w, w, 0.2, 0.6), w)

    def test_formula_edges_and_middle(self):
        w0, w1, a, b = 0.3, 0.9, 0.2, 0.6
        self.assertEqual(phase.lora_weight(0.0, w0, w1, a, b), w0)
        self.assertEqual(phase.lora_weight(a, w0, w1, a, b), w0)
        self.assertAlmostEqual(phase.lora_weight(b, w0, w1, a, b), w1, places=14)
        self.assertAlmostEqual(phase.lora_weight(1.0, w0, w1, a, b), w1, places=14)
        self.assertAlmostEqual(phase.lora_weight((a + b) / 2, w0, w1, a, b), (w0 + w1) / 2, places=14)
        for i in range(101):
            r = i / 100.0
            self.assertAlmostEqual(phase.lora_weight(r, w0, w1, a, b), textbook_weight(r, w0, w1, a, b), places=14)

    def test_reversed_weights_mirror(self):
        for i in range(101):
            r = i / 100.0
            up = phase.lora_weight(r, 0.3, 0.9, 0.2, 0.6)
            down = phase.lora_weight(r, 0.9, 0.3, 0.2, 0.6)
            self.assertAlmostEqual(up + down, 1.2, places=14)


class PlanValidationTest(unittest.TestCase):
    def test_valid(self):
        plan = phase.make_plan(1, 0.3, 0.9, 0.2, 0.6)
        self.assertEqual((plan.stack_index, plan.w0, plan.w1, plan.a, plan.b, plan.levels),
                         (1, 0.3, 0.9, 0.2, 0.6, phase.PHASE_LEVELS))

    def test_each_bad_value_is_refused_by_name(self):
        good = dict(stack_index=0, w0=0.3, w1=0.9, a=0.2, b=0.6)
        bad = [("w0", float("nan")), ("w0", 10.5), ("w1", -11.0), ("w0", True), ("w1", "0.5"), ("a", -0.01),
               ("b", 1.01), ("a", 0.6), ("b", 0.2), ("levels", 1), ("levels", 99), ("levels", 2.5),
               ("stack_index", -1), ("stack_index", True)]
        for name, value in bad:
            with self.subTest(name=name, value=value):
                with self.assertRaises(errors.UnsupportedFeature) as ctx:
                    phase.make_plan(**dict(good, **{name: value}))
                self.assertEqual(ctx.exception.issue.detail["feature"], "phase_plan")

    def test_a_equal_to_b_is_refused(self):
        with self.assertRaises(errors.UnsupportedFeature):
            phase.make_plan(0, 0.3, 0.9, 0.4, 0.4)

    def test_dict_round_trip_and_strict_parse(self):
        plan = phase.make_plan(2, 0.9, 0.3, 0.1, 0.7, levels=4)
        obj = phase.plan_to_dict(plan)
        self.assertEqual(phase.plan_from_dict(obj, node="N"), plan)
        for key in obj:
            broken = dict(obj)
            del broken[key]
            with self.assertRaises(errors.UnsupportedFeature):
                phase.plan_from_dict(broken, node="N")
        for key, value in (("mechanism", "hooks"), ("kind", "x"), ("version", 2), ("progress", "step"),
                           ("apply_order", "first")):
            with self.assertRaises(errors.UnsupportedFeature):
                phase.plan_from_dict(dict(obj, **{key: value}), node="N")
        with self.assertRaises(errors.UnsupportedFeature):
            phase.plan_from_dict(dict(obj, extra=1), node="N")


class ScheduleTest(unittest.TestCase):
    def schedule(self, sigmas=SHIFTED, **kw):
        args = dict(stack_index=0, w0=0.3, w1=0.9, a=0.2, b=0.6)
        args.update(kw)
        return phase.build_schedule(sigmas, phase.make_plan(**args))

    def test_planned_weights_follow_smoothstep_of_the_walked_distance(self):
        for sigmas in (SHIFTED, UNIFORM):
            sch = self.schedule(sigmas)
            want = [textbook_weight(r, 0.3, 0.9, 0.2, 0.6) for r in independent_progress(sigmas)]
            self.assertEqual(len(sch.planned), len(sigmas) - 1)
            for g, w in zip(sch.planned, want):
                self.assertAlmostEqual(g, w, places=12)
            self.assertEqual(sch.planned, phase.plan_step_weights(sigmas, sch.plan))

    def test_applied_weights_are_levels_within_half_a_level_of_the_plan(self):
        for kw in ({}, {"w0": 0.9, "w1": 0.3}, {"levels": 3}, {"levels": 16}):
            sch = self.schedule(**kw)
            levels = sch.plan.levels
            allowed = {phase._level_weight(sch.plan, j) for j in range(levels)}
            half = abs(sch.plan.w1 - sch.plan.w0) / (2 * (levels - 1)) + 1e-12
            for planned, applied in zip(sch.planned, sch.applied):
                self.assertIn(applied, allowed)
                self.assertLessEqual(abs(applied - planned), half)

    def test_plateaus_are_exactly_w0_and_w1_and_the_ramp_is_monotone(self):
        sch = self.schedule()
        for k, rk in enumerate(independent_progress(SHIFTED)):
            if rk <= 0.2:
                self.assertEqual(sch.applied[k], 0.3)
            if rk >= 0.6:
                self.assertEqual(sch.applied[k], 0.9)
        self.assertEqual(list(sch.applied), sorted(sch.applied))
        down = self.schedule(w0=0.9, w1=0.3)
        self.assertEqual(list(down.applied), sorted(down.applied, reverse=True))
        self.assertEqual((down.applied[0], down.applied[-1]), (0.9, 0.3))

    def test_segments_tile_the_steps_and_are_bounded_by_the_levels(self):
        sch = self.schedule()
        self.assertEqual(sch.segments[0].first, 0)
        self.assertEqual(sch.segments[-1].stop, len(sch.applied))
        for left, right in zip(sch.segments, sch.segments[1:]):
            self.assertEqual(left.stop, right.first)
            self.assertNotEqual(left.weight, right.weight)
        for seg in sch.segments:
            self.assertTrue(all(w == seg.weight for w in sch.applied[seg.first:seg.stop]))
        self.assertLessEqual(len(sch.segments), sch.plan.levels)
        self.assertGreater(len(sch.segments), 2)

    def test_equal_weights_are_one_segment_of_exactly_w0(self):
        for w in (0.8, 0.0, 1.0):
            sch = self.schedule(w0=w, w1=w)
            self.assertEqual(len(sch.segments), 1)
            self.assertEqual(sch.segments[0], phase.Segment(0, len(SHIFTED) - 1, w))
            self.assertEqual(set(sch.applied), {w})
            self.assertEqual(set(sch.planned), {w})

    def test_describe_is_deterministic_and_sensitive(self):
        a = phase.describe_schedule(self.schedule(), clones_built=4)
        b = phase.describe_schedule(self.schedule(), clones_built=4)
        self.assertEqual(a, b)
        for other in (self.schedule(w1=0.8), self.schedule(a=0.25), self.schedule(sigmas=UNIFORM)):
            self.assertNotEqual(phase.describe_schedule(other, clones_built=4)["weights_sha256"], a["weights_sha256"])
        self.assertEqual(sum(s["last_step"] - s["first_step"] + 1 for s in a["segments"]), a["steps"])
        self.assertEqual((a["mechanism"], a["clones_built"]), ("clone_per_level", 4))


class GateTest(unittest.TestCase):
    def profile(self, plan=None, kinds=("constant", "phased")):
        stack = []
        for i, kind in enumerate(kinds):
            if kind == "constant":
                wp = {"kind": "constant", "w": 1.0}
            elif plan is not None:
                wp = phase.weight_profile_of(plan)
            else:
                wp = {"kind": "phased", "w0": 0.3, "w1": 0.9, "a": 0.2, "b": 0.6}
            stack.append({"index": i, "weight_profile": wp})
        return {"stack": stack, "phase_plan": phase.plan_to_dict(plan) if plan is not None else None}

    def test_no_profile_and_a_constant_profile_pass_everywhere(self):
        for allowed in (False, True):
            self.assertIsNone(phase.check_phase_plan(None, node="N", allowed=allowed))
            self.assertIsNone(phase.check_phase_plan(self.profile(kinds=("constant",)), node="N", allowed=allowed))
            self.assertIsNone(phase.check_phase_plan({"stack": [], "phase_plan": None}, node="N", allowed=allowed))
            self.assertIsNone(phase.check_phase_plan({}, node="N", allowed=allowed))

    def test_a_phased_profile_is_refused_outside_the_advanced_node_and_names_it(self):
        plan = phase.make_plan(1, 0.3, 0.9, 0.2, 0.6)
        with self.assertRaises(errors.UnsupportedFeature) as ctx:
            phase.check_phase_plan(self.profile(plan), node="MossPhotonSampler", allowed=False)
        self.assertIn("MossPhotonSamplerAdvanced", str(ctx.exception))
        self.assertEqual(ctx.exception.issue.node, "MossPhotonSampler")

    def test_advanced_gets_the_plan(self):
        plan = phase.make_plan(1, 0.3, 0.9, 0.2, 0.6)
        self.assertEqual(phase.check_phase_plan(self.profile(plan), node="N", allowed=True), plan)

    def test_half_phased_profiles_are_refused(self):
        plan = phase.make_plan(1, 0.3, 0.9, 0.2, 0.6)
        wrong_index = phase.plan_to_dict(phase.make_plan(0, 0.3, 0.9, 0.2, 0.6))
        wrong_w0 = phase.plan_to_dict(phase.make_plan(1, 0.4, 0.9, 0.2, 0.6))
        cases = {
            "plan, no phased entry": dict(self.profile(plan), stack=self.profile(kinds=("constant", "constant"))["stack"]),
            "phased entry, no plan": dict(self.profile(plan), phase_plan=None),
            "plan points at the wrong entry": dict(self.profile(plan), phase_plan=wrong_index),
            "entry differs from the plan": dict(self.profile(plan), phase_plan=wrong_w0),
            "two phased entries": self.profile(plan, kinds=("phased", "phased")),
            "unknown kind": dict(self.profile(plan), stack=[{"index": 0, "weight_profile": {"kind": "ramp"}}]),
        }
        for name, profile in cases.items():
            with self.subTest(name):
                with self.assertRaises(errors.UnsupportedFeature):
                    phase.check_phase_plan(profile, node="N", allowed=True)


if __name__ == "__main__":
    unittest.main()
