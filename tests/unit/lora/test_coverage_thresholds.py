"""Coverage thresholds (design 1.6, 1.9, A11): 0.95 ok, 0.50 warn, below error; allow_low never rescues 0."""
import unittest

from moss_photon.core import lora


class CoverageThresholdsTest(unittest.TestCase):
    def check(self, total, mapped, verdict, code, **kw):
        got = lora.coverage_verdict(total, mapped, **kw)
        self.assertEqual((got["verdict"], got["code"]), (verdict, code), (total, mapped, kw))
        return got

    def test_table(self):
        self.check(100, 100, "ok", None)
        self.check(100, 95, "ok", None)
        self.check(100, 94, "warn", "partial_lora_coverage")
        self.check(100, 50, "warn", "partial_lora_coverage")
        self.check(100, 49, "error", "incompatible_lora")
        self.check(100, 0, "error", "incompatible_lora")

    def test_allow_low_turns_49_into_a_forced_warning(self):
        got = self.check(100, 49, "warn", "lora_low_coverage_forced", allow_low=True)
        self.assertTrue(got["forced"])
        self.assertFalse(self.check(100, 94, "warn", "partial_lora_coverage", allow_low=True)["forced"])

    def test_allow_low_never_rescues_zero_or_an_empty_file(self):
        self.check(100, 0, "error", "incompatible_lora", allow_low=True)
        self.check(0, 0, "error", "incompatible_lora", allow_low=True)
        self.check(0, 5, "error", "incompatible_lora", allow_low=True)

    def test_ratio_value_and_cap(self):
        self.assertEqual(lora.coverage_verdict(252, 252)["ratio"], 1.0)
        self.assertAlmostEqual(lora.coverage_verdict(100, 95)["ratio"], 0.95)
        # a module may map to more than one patch entry (bias diff): the ratio stays within the schema's [0, 1]
        self.assertEqual(lora.coverage_verdict(10, 12)["ratio"], 1.0)

    def test_constants_are_the_design_values(self):
        self.assertEqual((lora.COVERAGE_OK, lora.COVERAGE_MIN), (0.95, 0.50))

    def test_real_gap_between_the_measured_files(self):
        # the six compatible files = 1.0, the two incompatible = 0.0 (design 1.9): both sit far from the thresholds
        for total, mapped, verdict in ((252, 252, "ok"), (164, 164, "ok"), (246, 246, "ok"),
                                       (240, 0, "error"), (264, 0, "error")):
            self.assertEqual(lora.coverage_verdict(total, mapped)["verdict"], verdict)


if __name__ == "__main__":
    unittest.main()
