"""MP-60 / audit F03 on the real MossPhotonLoRAProfile node and the real sampler: a file whose sha256 is the known 8-step
distill LoRA is a distill whatever role the user gave it.

Audit repro B: with ``role=character`` the load only warned; ``identity_safe`` then took it as the character LoRA and a
50-step preset did not see a distill. The file is synthetic; its sha256 is overridden to the real lodestones hash (the
node's own hashing is not the subject), as in ``test_distill_role_warning_cpu.py``."""
import unittest
from unittest import mock

from . import lora_synth, tiny_moss
from . import sampler_support as ss

DMD2 = "7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2"
NAME = "lodestones-8-steps-dmd2-lora.safetensors"
OTHER = "someone-elses-lora.safetensors"


class KnownDistillBeatsRoleTest(unittest.TestCase):
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
        self.env.write(OTHER, lora_synth.make_lora_sd(self.patcher, seed=12, modules=("qkv", "o")),
                       {"ss_base_model_version": "moss_image"})
        real = self.lora.describe_file

        def known_hash(path, **kw):
            info = real(path, **kw)
            return dict(info, sha256=DMD2) if info["basename"] == NAME else info
        patch = mock.patch.object(self.lora, "describe_file", known_hash)
        patch.start()
        self.addCleanup(patch.stop)

    def profile(self, name, role):
        model, profile, _ = lora_synth.run_node(self.patcher, name, role=role)
        return model, profile

    def sample(self, model, profile, preset_id, **kw):
        with ss.known_sft(), ss.NoSample():
            return ss.run(self.tiny, model=model, identity_profile=profile, preset_id=preset_id, **kw)

    # ---------------------------------------------------------------- repro B
    def test_repro_b_identity_safe_does_not_take_the_known_distill_as_the_character(self):
        model, profile = self.profile(NAME, "character")
        self.assertEqual(profile["stack"][0]["role"], "character")  # never changed silently
        self.assertEqual(profile["summary"]["has_character"], False)
        self.assertEqual(profile["summary"]["has_distill"], True)
        # MP-84: the run is not stopped; the report names the file and the roles in the note
        res = self.sample(model, profile, "identity_safe")
        note = next(w for w in res.report["warnings"] if w["code"] == "lora_not_recorded")
        text = note["message"] + " " + note["fix"]
        for needle in (NAME, "'character'", "known distill", "role=distill"):
            self.assertIn(needle, text)

    def test_repro_b_a_normal_preset_warns_with_the_file_and_its_role_named(self):
        for role in ("character", "style", "other"):
            for preset_id in ("reference", "photo_safe"):
                with self.subTest(role=role, preset=preset_id):
                    model, profile = self.profile(NAME, role)
                    res = self.sample(model, profile, preset_id)
                    note = next(w for w in res.report["warnings"] if w["code"] == "distill_lora_non_turbo")
                    self.assertEqual(note["severity"], "warning")
                    text = note["message"] + " " + note["fix"]
                    for needle in (NAME, "role '%s'" % role, "role=distill", "turbo"):
                        self.assertIn(needle, text)

    def test_advanced_warns_and_names_the_file(self):
        model, profile = self.profile(NAME, "character")
        res = self.sample(model, profile, "advanced", overrides={"steps": 4})
        warning = [w for w in res.report["warnings"] if w["code"] == "distill_lora_non_turbo"]
        self.assertEqual(len(warning), 1)
        self.assertIn(NAME, warning[0]["message"])

    # ---------------------------------------------------------------- what must keep working
    def test_an_unknown_hash_keeps_its_user_role(self):
        model, profile = self.profile(OTHER, "character")
        self.assertEqual(profile["summary"]["has_character"], True)
        res = self.sample(model, profile, "identity_safe", overrides={"steps": 4})  # satisfied, and no distill error
        self.assertEqual(res.report["lora"]["stack"][0]["role"], "character")

    def test_the_known_distill_stays_a_distill_next_to_a_real_character(self):
        m1, p1, _ = lora_synth.run_node(self.patcher, NAME, role="style")
        m2, p2, _ = lora_synth.run_node(m1, OTHER, role="character", profile_in=p1)
        res = self.sample(m2, p2, "identity_safe")  # the distill is still in the stack: a warning, not an error (MP-84)
        self.assertIn("distill_lora_non_turbo", [w["code"] for w in res.report["warnings"]])
        self.assertNotIn("lora_not_recorded", [w["code"] for w in res.report["warnings"]], "the real character is there")

    def test_the_distill_role_in_a_turbo_preset_is_untouched(self):
        model, profile = self.profile(NAME, "distill")
        res = self.sample(model, profile, "turbo_8step", cfg=1.0)
        self.assertEqual(res.report["lora"]["stack"][0]["role"], "distill")

    def test_turbo_keeps_noting_the_explicit_distill_role(self):
        """MP-50: no silent re-roling, the note names the file (MP-84: a note, no longer an error)."""
        model, profile = self.profile(NAME, "character")
        res = self.sample(model, profile, "turbo_8step", cfg=1.0)
        note = next(w for w in res.report["warnings"] if w["code"] == "lora_not_recorded")
        self.assertIn(NAME, note["message"])

    def test_two_known_distills_under_any_roles_warn_at_the_second_node(self):
        m1, p1, _ = lora_synth.run_node(self.patcher, NAME, role="distill")
        m2, p2, _ = lora_synth.run_node(m1, NAME + "", role="character", profile_in=p1)  # MP-84: a warning, not an error
        second = [w for w in p2["stack"][1]["warnings"] if w["code"] == "multiple_distill_loras"]
        self.assertEqual(len(second), 1)
        self.assertIn("second distill", second[0]["message"])


if __name__ == "__main__":
    unittest.main()
