"""MP-50: the LoRA profile node warns at load time when a known distill file is loaded with another role, and then the
sampler's missing_lora error names that file. The file is synthetic; its sha256 is overridden to the real lodestones
hash (the node's own hashing is not the subject)."""
import unittest
from unittest import mock

from . import lora_synth, tiny_moss
from . import sampler_support as ss

DMD2 = "7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2"
NAME = "lodestones-8-steps-dmd2-lora.safetensors"


class DistillRoleWarningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        from moss_photon.core import errors, lora
        cls.errors, cls.lora = errors, lora

    def setUp(self):
        self.env = lora_synth.install(self)
        self.patcher = self.tiny.patcher
        self.env.write(NAME, lora_synth.make_lora_sd(self.patcher, seed=11, modules=("qkv", "o")),
                       {"ss_base_model_version": "moss_image"})
        real = self.lora.describe_file

        def known_hash(path, **kw):
            return dict(real(path, **kw), sha256=DMD2)
        patch = mock.patch.object(self.lora, "describe_file", known_hash)
        patch.start()
        self.addCleanup(patch.stop)

    def codes(self, profile):
        return [w["code"] for w in profile["stack"][0]["warnings"]]

    def test_a_known_distill_file_under_role_other_warns_at_load_time(self):
        model, profile, _ = lora_synth.run_node(self.patcher, NAME, role="other")
        self.assertIn("distill_lora_role_mismatch", self.codes(profile))
        warning = next(w for w in profile["stack"][0]["warnings"] if w["code"] == "distill_lora_role_mismatch")
        self.assertIn(NAME, warning["message"])
        self.assertIn("role 'other'", warning["message"])
        self.assertEqual(profile["stack"][0]["role"], "other")  # never changed for the user

    def test_the_distill_role_has_no_such_warning(self):
        _, profile, _ = lora_synth.run_node(self.patcher, NAME, role="distill")
        self.assertNotIn("distill_lora_role_mismatch", self.codes(profile))

    def test_the_sampler_error_then_names_the_file(self):
        model, profile, _ = lora_synth.run_node(self.patcher, NAME, role="other")
        with ss.known_sft(), self.assertRaises(self.errors.MissingLora) as cm:
            ss.run(self.tiny, model=model, identity_profile=profile, preset_id="turbo_8step", cfg=1.0)
        text = str(cm.exception)
        for needle in (NAME, "'other'", "role=distill"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
