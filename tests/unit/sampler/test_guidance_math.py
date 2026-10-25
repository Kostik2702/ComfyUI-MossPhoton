"""guidance.parse_layers / validate_slg / slg_sigma_window / describe_slg (design 1.5, 6.4)."""
import types
import unittest

from moss_photon.core import errors, guidance

NODE = "MossPhotonSamplerAdvanced"


class FlowSampling:
    """Pure copy of ModelSamplingDiscreteFlow.percent_to_sigma (CUI/comfy/model_sampling.py:331-336)."""

    def __init__(self, shift):
        self.shift = shift

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        t = 1.0 - percent
        return t if self.shift == 1.0 else self.shift * t / (1 + (self.shift - 1) * t)


class ParseLayersTest(unittest.TestCase):
    def test_strings_like_the_native_regex(self):
        for text, expected in (("22", [22]), ("22, 24", [22, 24]), ("[22,24]", [22, 24]), ("24 22", [22, 24]),
                               ("", []), ("  ", []), ("22 22", [22]), ("7, 8, 9", [7, 8, 9]), ("a22b", [22])):
            with self.subTest(text=text):
                self.assertEqual(guidance.parse_layers(text), expected)

    def test_sequences_are_sorted_and_unique(self):
        self.assertEqual(guidance.parse_layers([24, 22, 22]), [22, 24])
        self.assertEqual(guidance.parse_layers(()), [])
        self.assertEqual(guidance.parse_layers(None), [])

    def test_bad_sequence_items_are_a_programmer_error(self):
        for bad in ([True], [1.5], ["22"]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    guidance.parse_layers(bad)


class ValidateSlgTest(unittest.TestCase):
    def codes(self, *args, **kw):
        return [(i.code, i.param) for i in guidance.validate_slg(*args, node=NODE, **kw)]

    def test_off_needs_nothing(self):
        self.assertEqual(self.codes("off", [], -5, 7, num_layers=24), [])

    def test_valid_reference_setting(self):
        self.assertEqual(self.codes("uncond_skip", [22], 0.0, 1.0, num_layers=24), [])
        self.assertEqual(self.codes("uncond_skip", [22], 0.0, 1.0, num_layers=None), [])

    def test_unknown_mode(self):
        self.assertEqual(self.codes("none", [22], 0.0, 1.0, num_layers=24), [("invalid_override", "slg_mode")])

    def test_empty_layers(self):
        self.assertEqual(self.codes("uncond_skip", [], 0.0, 1.0, num_layers=24), [("invalid_override", "slg_layers")])

    def test_layer_must_exist_so_slg_cannot_be_a_silent_no_op(self):
        issues = guidance.validate_slg("uncond_skip", [22], 0.0, 1.0, num_layers=4, node=NODE)
        self.assertEqual([i.code for i in issues], ["invalid_override"])
        self.assertIn("22", issues[0].message)
        self.assertEqual(issues[0].detail["num_layers"], 4)
        self.assertEqual(self.codes("uncond_skip", [23], 0.0, 1.0, num_layers=24), [])
        self.assertEqual(self.codes("uncond_skip", [24], 0.0, 1.0, num_layers=24), [("invalid_override", "slg_layers")])

    def test_negative_or_non_integer_layer(self):
        self.assertEqual(self.codes("uncond_skip", [-1], 0.0, 1.0, num_layers=24), [("invalid_override", "slg_layers")])
        self.assertEqual(self.codes("uncond_skip", [True], 0.0, 1.0, num_layers=24),
                         [("invalid_override", "slg_layers")])

    def test_window(self):
        self.assertEqual(self.codes("uncond_skip", [22], 0.5, 0.25, num_layers=24), [("invalid_override", "slg_start")])
        self.assertEqual(self.codes("uncond_skip", [22], -0.1, 1.0, num_layers=24), [("invalid_override", "slg_start")])
        self.assertEqual(self.codes("uncond_skip", [22], 0.0, 1.5, num_layers=24), [("invalid_override", "slg_end")])
        self.assertEqual(self.codes("uncond_skip", [22], 0.25, 0.75, num_layers=24), [])

    def test_issues_raise_as_invalid_override(self):
        with self.assertRaises(errors.InvalidOverride):
            errors.raise_first_error(guidance.validate_slg("uncond_skip", [], 0.0, 1.0, num_layers=24, node=NODE))


class SigmaWindowTest(unittest.TestCase):
    def test_shift5_half(self):
        self.assertEqual(guidance.slg_sigma_window(FlowSampling(5.0), 0.0, 0.5), (0.8333333333333334, 1.0))
        self.assertEqual(guidance.slg_sigma_window(FlowSampling(5.0), 0.5, 1.0), (0.0, 0.8333333333333334))

    def test_ends(self):
        self.assertEqual(guidance.slg_sigma_window(FlowSampling(5.0), 0.0, 1.0), (0.0, 1.0))

    def test_shift_changes_the_window(self):
        self.assertNotEqual(guidance.slg_sigma_window(FlowSampling(3.0), 0.0, 0.5),
                            guidance.slg_sigma_window(FlowSampling(5.0), 0.0, 0.5))

    def test_the_live_model_sampling_is_used(self):
        calls = []

        class Spy(FlowSampling):
            def percent_to_sigma(self, percent):
                calls.append(percent)
                return super().percent_to_sigma(percent)

        guidance.slg_sigma_window(Spy(5.0), 0.25, 0.75)
        self.assertEqual(sorted(calls), [0.25, 0.75])


class MiscTest(unittest.TestCase):
    def test_has_external_cond_hook(self):
        clean = types.SimpleNamespace(model_options={"transformer_options": {}})
        hooked = types.SimpleNamespace(model_options={"sampler_calc_cond_batch_function": lambda a: a})
        self.assertFalse(guidance.has_external_cond_hook(clean))
        self.assertTrue(guidance.has_external_cond_hook(hooked))
        self.assertFalse(guidance.has_external_cond_hook(types.SimpleNamespace()))

    def test_apply_slg_without_layers_returns_the_same_object_and_needs_no_comfy(self):
        sentinel = object()
        self.assertIs(guidance.apply_slg(sentinel, layers=[], start=0.0, end=1.0), sentinel)
        self.assertIs(guidance.apply_slg(sentinel, layers="", start=0.0, end=1.0), sentinel)

    def test_describe_slg(self):
        self.assertEqual(guidance.describe_slg("uncond_skip", "22", 0, 1, (0.0, 1.0)),
                         {"mode": "uncond_skip", "layers": [22], "start": 0.0, "end": 1.0, "sigma_window": [0.0, 1.0]})


if __name__ == "__main__":
    unittest.main()
