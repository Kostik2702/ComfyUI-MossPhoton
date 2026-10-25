"""Sigma math (design 1.2, 2.6, 6.1): the reference array is a literal, validation runs on lists only."""
import json
import math
import os
import struct
import subprocess
import sys
import unittest

from moss_photon.core import consts, errors, sigma

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# The 51 float32 values of the native 50-step sgm_uniform schedule at shift 5 / multiplier 1.0, as repr() of the
# python floats (research/spike_mp2/sigmas_native_sgm_uniform_50.json, INV 1.3). A literal on purpose: the test
# must not recompute what it checks.
REFERENCE_50 = [
    1.0, 0.9959554672241211, 0.9917780756950378,
    0.9874609112739563, 0.9829968810081482, 0.9783784747123718,
    0.973597526550293, 0.9686450958251953, 0.9635122418403625,
    0.9581883549690247, 0.9526629447937012, 0.9469242691993713,
    0.9409597516059875, 0.9347557425498962, 0.9282975196838379,
    0.9215691089630127, 0.9145533442497253, 0.9072311520576477,
    0.8995822668075562, 0.8915842175483704, 0.8832125663757324,
    0.8744403719902039, 0.8652384281158447, 0.8555740714073181,
    0.8454118371009827, 0.8347120881080627, 0.8234310746192932,
    0.811519980430603, 0.7989246249198914, 0.7855842709541321,
    0.7714307308197021, 0.75638747215271, 0.7403677701950073,
    0.7232733368873596, 0.704992413520813, 0.6853967905044556,
    0.6643396019935608, 0.6416508555412292, 0.6171333193778992,
    0.5905565619468689, 0.5616498589515686, 0.530092716217041,
    0.49550309777259827, 0.45742183923721313, 0.4152921140193939,
    0.3684321343898773, 0.3159978687763214, 0.25693169236183167,
    0.1898905336856842, 0.11314225941896439, 0.0,
]
LITERAL_HASH = "9b2c66b3fb335d998fc03e73ddbafe997fe2d3cf3b187f51629bdf0332f5a536"


def codes(issues):
    return [i.code for i in issues]


class ReferenceLiteralTest(unittest.TestCase):
    def test_reference_literal_50_equals_native_array(self):
        self.assertEqual(len(REFERENCE_50), 51)
        self.assertEqual(sigma.validate_sigmas(REFERENCE_50, node="T", expect_steps=50), [])
        # the hash is pinned twice: against a literal in this file and against consts
        self.assertEqual(sigma.sigma_hash(REFERENCE_50), LITERAL_HASH)
        self.assertEqual(consts.REFERENCE_SIGMAS_SHA256, LITERAL_HASH)
        self.assertEqual((REFERENCE_50[0], REFERENCE_50[1]), (1.0, 0.9959554672241211))
        self.assertEqual((REFERENCE_50[-2], REFERENCE_50[-1]), (0.11314225941896439, 0.0))

    def test_hash_is_float32_little_endian(self):
        self.assertEqual(len(struct.pack("<51f", *REFERENCE_50)), 204)
        self.assertNotEqual(sigma.sigma_hash(REFERENCE_50[:-1] + [1e-30]), LITERAL_HASH)

    def test_native_step_ratio_margin(self):
        observed = sigma.observed_jump_ratio(REFERENCE_50)
        self.assertAlmostEqual(observed, 1.1447933744323253, places=12)
        self.assertLess(observed, sigma.JUMP_RATIO_MAX / 2)

    def test_hash_accepts_tuple_and_tensor_like(self):
        class Tensorish:
            def tolist(self):
                return list(REFERENCE_50)
        self.assertEqual(sigma.sigma_hash(tuple(REFERENCE_50)), LITERAL_HASH)
        self.assertEqual(sigma.sigma_hash(Tensorish()), LITERAL_HASH)


class RejectionTest(unittest.TestCase):
    """One failure each, reason invalid_schedule, message naming the index."""

    def check(self, values, needle, **kw):
        issues = sigma.validate_sigmas(values, node="MossPhotonSchedule", **kw)
        self.assertEqual(codes(issues), ["invalid_schedule"], issues)
        self.assertEqual(issues[0].severity, "error")
        self.assertIn(needle, issues[0].message)
        return issues[0]

    def test_validate_rejects_nan(self):
        v = list(REFERENCE_50)
        v[7] = float("nan")
        self.check(v, "sigmas[7]")

    def test_validate_rejects_inf(self):
        v = list(REFERENCE_50)
        v[3] = float("inf")
        self.check(v, "sigmas[3]")

    def test_validate_rejects_non_monotonic(self):
        v = list(REFERENCE_50)
        v[10], v[11] = v[11], v[10]
        self.check(v, "not strictly decreasing at index 11")

    def test_validate_rejects_duplicate(self):
        v = list(REFERENCE_50)
        v[12] = v[11]
        self.check(v, "duplicate value %r at index 12" % v[11])

    def test_validate_rejects_not_ending_zero(self):
        v = list(REFERENCE_50)
        v[-1] = 1e-6
        self.check(v, "sigmas[50]")

    def test_validate_rejects_first_above_max(self):
        v = [1.5] + REFERENCE_50[1:]
        self.check(v, "sigmas[0]", sigma_max=1.0)

    def test_validate_rejects_first_zero_and_negative(self):
        self.check([0.0, 0.0], "sigmas[0]")
        self.check([1.0, -0.5, 0.0], "sigmas[1]")

    def test_validate_rejects_two_dim(self):
        self.check([[1.0, 0.5, 0.0]], "one-dimensional")

    def test_validate_rejects_two_dim_tensor_like(self):
        class Tensorish:
            def tolist(self):
                return [[1.0], [0.0]]
        self.check(Tensorish(), "one-dimensional")

    def test_validate_rejects_too_short_and_too_long(self):
        self.check([0.0], "1 values")
        self.check([], "0 values")
        long = [1.0 - i / 1002.0 for i in range(1001)] + [0.0]
        self.check(long, "1002 values")

    def test_validate_rejects_wrong_length(self):
        self.check(REFERENCE_50, "expected steps + 1 = 31", expect_steps=30)

    def test_validate_rejects_non_numbers(self):
        self.check([1.0, "0.5", 0.0], "sigmas[1]")
        self.check([1.0, True, 0.0], "sigmas[1]")
        self.check(5, "not an array")

    def test_validate_rejects_step_ratio(self):
        v = [1.0, 0.9, 0.8, 0.7, 0.69, 0.2, 0.0]  # deltas .1 .1 .1 .01 ... ratio 0.1 between steps 2 and 3
        issue = self.check(v, "step size changes")
        self.assertEqual(issue.detail["check"], "step_ratio")

    def test_step_ratio_is_a_warning_outside_reference_mode(self):
        v = [1.0, 0.9, 0.8, 0.7, 0.69, 0.2, 0.0]
        issues = sigma.validate_sigmas(v, node="MossPhotonSchedule", ratio_is_error=False)
        self.assertEqual(codes(issues), ["sigma_step_ratio"])
        self.assertEqual(issues[0].severity, "warning")

    def test_last_step_to_zero_is_excluded_from_the_ratio(self):
        # the final delta (0.97) is ~100x the one before; that must not trigger
        self.assertEqual(sigma.validate_sigmas([1.0, 0.99, 0.98, 0.97, 0.0], node="T", sigma_max=1.0), [])

    def test_validation_never_imports_torch(self):
        code = ("import sys\nsys.path.insert(0, sys.argv[1])\nfrom moss_photon.core import sigma\n"
                "sigma.validate_sigmas([1.0, 0.5, 0.0], node='n')\nsigma.build_schedule_info("
                "[1.0, 0.5, 0.0], mode='reference', scheduler='sgm_uniform', steps=2, shift_model=5.0, "
                "shift_used=5.0, multiplier=1.0, native_reference=[1.0, 0.5, 0.0])\n"
                "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'comfy', 'numpy'))\n"
                "assert not bad, bad[:5]\nprint('PURE')\n")
        proc = subprocess.run([sys.executable, "-I", "-c", code, PKG_ROOT], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])


class ForwardsTest(unittest.TestCase):
    def test_forwards_table(self):
        self.assertEqual(sigma.expected_forwards(50, 3.0), 100)
        self.assertEqual(sigma.expected_forwards(50, 1.0), 50)
        self.assertEqual(sigma.expected_forwards(25, 3.0, extra=2), 52)

    def test_branches_use_isclose(self):
        self.assertEqual(sigma.branches_per_step(1.0), 1)
        self.assertEqual(sigma.branches_per_step(1.0 + 1e-12), 1)
        self.assertEqual(sigma.branches_per_step(1.01), 2)
        self.assertEqual(sigma.branches_per_step(0.0), 2)


class CompareTest(unittest.TestCase):
    def test_equal_arrays(self):
        r = sigma.compare_sigmas(REFERENCE_50, list(REFERENCE_50))
        self.assertEqual(r, {"equal": True, "bit_equal": True, "max_abs_diff": 0.0, "n_differ": 0, "same_length": True})

    def test_compare_sigmas_detects_1e7_difference(self):
        # the SD3-style multiplier-1000 array differs from native by up to 1.19e-7 (INV 1.3): one float32 ulp
        b = list(REFERENCE_50)
        b[25] = struct.unpack("<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", b[25]))[0] + 1))[0]
        r = sigma.compare_sigmas(REFERENCE_50, b)
        self.assertFalse(r["equal"])
        self.assertFalse(r["bit_equal"])
        self.assertEqual(r["n_differ"], 1)
        self.assertGreater(r["max_abs_diff"], 0.0)
        self.assertLess(r["max_abs_diff"], 2e-7)

    def test_length_mismatch(self):
        r = sigma.compare_sigmas(REFERENCE_50, REFERENCE_50[:-2])
        self.assertFalse(r["same_length"])
        self.assertFalse(r["equal"])
        self.assertEqual(r["n_differ"], 2)

    def test_negative_zero_is_equal_but_not_bit_equal(self):
        r = sigma.compare_sigmas([1.0, 0.0], [1.0, -0.0])
        self.assertTrue(r["equal"])
        self.assertFalse(r["bit_equal"])


class ScheduleInfoTest(unittest.TestCase):
    def build(self, **kw):
        args = dict(mode="reference", scheduler="sgm_uniform", steps=50, shift_model=5.0, shift_used=5.0,
                    multiplier=1.0, native_reference=list(REFERENCE_50))
        args.update(kw)
        return sigma.build_schedule_info(REFERENCE_50, **args)

    def test_info_equals_the_contract_fixture(self):
        path = os.path.join(PKG_ROOT, "tests", "fixtures", "contracts", "schedule_info.json")
        with open(path, "r", encoding="utf-8") as f:
            fixture = json.load(f)
        self.assertEqual(self.build(), fixture)

    def test_shifted_mode_is_never_native_reference(self):
        info = self.build(mode="shifted")
        self.assertFalse(info["is_native_reference"])
        self.assertTrue(info["native_reference"]["bit_equal"])  # the arrays do match; the mode decides the label

    def test_differing_native_reference(self):
        other = [x * 0.99 for x in REFERENCE_50]
        info = self.build(native_reference=other)
        self.assertFalse(info["is_native_reference"])
        self.assertGreater(info["native_reference"]["n_differ"], 0)

    def test_native_reference_is_required(self):
        with self.assertRaises(ValueError):
            sigma.build_schedule_info(REFERENCE_50, mode="reference", scheduler="sgm_uniform", steps=50,
                                      shift_model=5.0, shift_used=5.0, multiplier=1.0)

    def test_warnings_are_stored_as_dicts(self):
        w = errors.make_issue("shift_input_ignored", "MossPhotonSchedule", "shift", "m", "f")
        info = self.build(warnings=[w])
        self.assertEqual(info["warnings"], [w.to_dict()])
        self.assertTrue(all(math.isfinite(info[k]) for k in ("sigma_first", "sigma_last")))


if __name__ == "__main__":
    unittest.main()
