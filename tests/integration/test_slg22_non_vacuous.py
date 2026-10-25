"""M3 (REQUIRED): SLG layer 22 is not a no-op on the 24-layer fixture, for the native node AND for our closure, and ours
equals native; on a 4-layer model it IS a no-op, which is why ``validate_slg`` rejects a layer the model lacks."""
import unittest

from . import _comfy_env, tiny_moss
from . import sampler_support as ss


class Slg22NonVacuousTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        from moss_photon.core import adapter, errors, guidance
        cls.adapter, cls.errors, cls.guidance = adapter, errors, guidance

    def sha(self, model):
        return ss.tensor_sha(tiny_moss.sample_euler(self.tiny, model))

    def test_native_and_ours_both_change_the_result_and_agree(self):
        none = self.sha(self.tiny.patcher)
        native = self.sha(ss.native_slg(self.tiny.patcher, 22))
        ours = self.sha(self.guidance.apply_slg(self.tiny.patcher, layers=[22], start=0.0, end=1.0))
        self.assertNotEqual(native, none)
        self.assertNotEqual(ours, none)
        self.assertEqual(ours, native)

    def test_photon_run_with_the_preset_slg_differs_from_slg_off(self):
        on = ss.run(self.tiny, overrides={"steps": 6})
        off = ss.run(self.tiny, overrides={"steps": 6, "slg_mode": "off"})
        self.assertNotEqual(ss.tensor_sha(on.latent["samples"]), ss.tensor_sha(off.latent["samples"]))
        self.assertEqual(on.report["forwards"]["slg_skip_calls"], 6)
        self.assertEqual(off.report["forwards"]["slg_skip_calls"], 0)

    def test_on_four_layers_slg22_is_vacuous_and_validate_slg_refuses_it(self):
        small = tiny_moss.build_tiny_moss(layers=4)
        base = ss.tensor_sha(tiny_moss.sample_euler(small))
        ours = ss.tensor_sha(tiny_moss.sample_euler(
            small, self.guidance.apply_slg(small.patcher, layers=[22], start=0.0, end=1.0)))
        self.assertEqual(ours, base, "control: the vacuous case is real, otherwise the refusal below is pointless")
        facts = self.adapter.probe_model(small.patcher, reference_dtypes=ss.F32)
        issues = self.guidance.validate_slg("uncond_skip", [22], 0.0, 1.0, num_layers=facts.num_layers,
                                            node="MossPhotonSampler")
        self.assertEqual([i.code for i in issues], ["invalid_override"])
        with self.assertRaises(self.errors.InvalidOverride):
            ss.run(small, overrides={"steps": 6})  # the reference preset skips layer 22 of a 4-layer model


if __name__ == "__main__":
    unittest.main()
