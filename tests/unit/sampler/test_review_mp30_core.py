"""MP-32 (MP-30 review) at the solver preflight: m1 order, M2 SLG flag, M7 family gate, n1 typed noise-policy error.

Same harness as ``test_solver_plan`` (fake adapter, spies on everything that touches a GPU). The full-run effects of
these fixes (label, report, warnings) are in ``tests/integration/test_review_mp30_core_cpu.py``.
"""
import copy
import unittest
from unittest import mock

from moss_photon.core import canon, errors, presets, solver

from .test_solver_plan import FakeSigmas, Scenario, Stop, constraints_fixture, good_facts, profile_fixture

IDENT = {"identity": {"sha256": None, "size": 0, "basename": "unknown", "hash_source": "unavailable",
                      "classified_by": "none", "official": None, "variant": None, "verified_matrix": None},
         "family": "unknown-moss", "warnings": []}


class SpyScenario(Scenario):
    """``Scenario`` whose checkpoint identification is a spy the test can inspect."""

    def __init__(self, test, **kw):
        super().__init__(test, **kw)
        self.identify = mock.Mock(name="identify_checkpoint", return_value=copy.deepcopy(IDENT))

    def run(self, **kw):
        with mock.patch.object(solver, "adapter", self.fake), \
                mock.patch.object(solver.rng, "make_noise", self.spies["make_noise"]), \
                mock.patch.object(solver, "build_guider", self.spies["build_guider"]), \
                mock.patch.object(solver, "install_counter", self.spies["install_counter"]), \
                mock.patch.object(solver.guidance, "apply_slg", self.spies["apply_slg"]), \
                mock.patch.object(solver.fingerprint, "identify_checkpoint", self.identify):
            return solver.execute(self.request(**kw))


def distill_profile(role="distill"):
    profile = profile_fixture()
    profile["stack"][0]["role"] = role
    return profile, good_facts(lora_stack_attachment=canon.canonical_json(profile["attachment"]),
                               patches_uuid=profile["attachment"]["patches_uuid"],
                               patch_entries=profile["attachment"]["patch_entries_total"])


class CheapChecksBeforeTheHashTest(unittest.TestCase):
    """MP-30 m1: the checkpoint sha256 costs seconds to minutes on a cache miss; a run that a cheap check refuses must
    not pay it first."""

    def refused(self, code, **kw):
        s = SpyScenario(self, **kw.pop("scenario", {}))
        s.expect(code, **kw)
        self.assertEqual(s.identify.call_count, 0, "the checkpoint was hashed before the %s check" % code)

    def test_missing_character_lora(self):
        self.refused("missing_lora", preset_id="identity_safe")

    def test_missing_distill_lora(self):
        self.refused("missing_lora", preset_id="turbo_8step", cfg=1.0)

    def test_missing_required_block(self):
        self.refused("missing_required_input", preset_id="wardrobe_safe")

    def test_source_latent_for_a_scene_preset(self):
        from .test_solver_plan import constraints_fixture, good_latent
        self.refused("unsupported_source_input", preset_id="scene_safe", constraints=constraints_fixture(),
                     scenario={"latent": good_latent(is_empty=False)})

    def test_a_profile_that_does_not_belong_to_the_model(self):
        self.refused("incompatible_lora", identity_profile=profile_fixture())

    def test_a_phase_plan(self):
        profile, facts = distill_profile("character")
        profile["phase_plan"] = {"kind": "phased"}
        self.refused("unsupported_feature", identity_profile=profile, scenario={"facts": facts})

    # MP-36 m1 rest (MP-38): SLG, layout, sigma and budget checks are cheap too
    def test_an_slg_layer_the_model_does_not_have(self):
        self.refused("invalid_override", scenario={"facts": good_facts(num_layers=4)})

    def test_a_layout_outside_the_canvas(self):
        c = constraints_fixture()
        c["layout"] = {"items": 1, "extent": [-5000, 0, 0, 0]}
        self.refused("invalid_layout", constraints=c)

    def test_invalid_native_sigmas(self):
        self.refused("invalid_schedule", scenario={"sigma_values": [1.0, float("nan"), 0.5, 0.0]})

    def test_invalid_input_sigmas(self):
        self.refused("invalid_schedule", sigmas=FakeSigmas([1.0, 0.5, 0.6, 0.0]))

    def test_the_forward_budget(self):
        self.refused("budget_exceeded", max_forwards=10)

    def test_the_preset_budget_of_a_turbo_run(self):
        profile, facts = distill_profile("distill")
        self.refused("budget_exceeded", preset_id="turbo_8step", cfg=1.0, identity_profile=profile, max_forwards=7,
                     scenario={"facts": facts})

    def test_a_manifest_changed_budget_is_still_planned_after_the_family_is_known(self):
        """The early plan uses the first resolve; if the family-aware resolve differs, the plan is made again."""
        s = SpyScenario(self)
        s.identify.return_value = dict(copy.deepcopy(IDENT), family="moss-image-sft")
        real = presets.resolve
        calls = []

        def spy(*a, **kw):
            r = real(*a, **kw)
            calls.append(kw.get("model_family"))
            if kw.get("model_family"):
                eff = copy.deepcopy(r.effective)
                eff["budget"]["max_forwards"] = 5
                r = type(r)(**dict(r.__dict__, effective=eff))
            return r
        with mock.patch.object(solver.presets, "resolve", spy), self.assertRaises(errors.BudgetExceeded):
            s.run()
        self.assertEqual(s.identify.call_count, 1)
        self.assertEqual(calls[-1], "moss-image-sft")

    def test_a_required_guard(self):
        s = SpyScenario(self)
        preset = copy.deepcopy(presets.resolve("reference").preset)
        preset["guards"] = [{"id": "needs_detector", "required": True, "requires_dependency": "x"}]
        real = presets.resolve

        def patched(*a, **kw):
            r = real(*a, **kw)
            return type(r)(**dict(r.__dict__, preset=preset))
        with mock.patch.object(solver.presets, "resolve", patched):
            s.expect("missing_guard_dependency")
        self.assertEqual(s.identify.call_count, 0)

    def test_control_a_good_request_hashes_exactly_once_and_then_goes_on(self):
        s = SpyScenario(self)
        with self.assertRaises(Stop):
            s.run()
        self.assertEqual(s.identify.call_count, 1)

    def test_the_family_still_reaches_the_final_resolve(self):
        """The manifest layer is keyed by family, so the final resolve must come after the identification."""
        s = SpyScenario(self)
        s.identify.return_value = dict(copy.deepcopy(IDENT), family="moss-image-sft")
        seen = []
        real = presets.resolve

        def spy(*a, **kw):
            seen.append(kw.get("model_family"))
            return real(*a, **kw)
        with mock.patch.object(solver.presets, "resolve", spy), self.assertRaises(Stop):
            s.run()
        self.assertEqual(seen[-1], "moss-image-sft")


class KnownDistillWrongRoleTest(unittest.TestCase):
    """MP-50: the owner's graph loaded lodestones-8-steps-dmd2-lora with role=other; the run must say which file."""

    DMD2 = "7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2"
    NAME = "lodestones-8-steps-dmd2-lora.safetensors"

    def test_the_run_stops_with_an_error_that_names_the_file_and_the_roles(self):
        profile, facts = distill_profile("other")
        profile["stack"][0]["lora"].update(basename=self.NAME, sha256=self.DMD2)
        s = SpyScenario(self, facts=facts)
        exc = s.expect("missing_lora", preset_id="turbo_8step", cfg=1.0, identity_profile=profile)
        text = str(exc)
        for needle in (self.NAME, "'other'", "role=distill", "known distill LoRA"):
            self.assertIn(needle, text)
        self.assertEqual(s.identify.call_count, 0)

    def test_no_role_is_changed_the_profile_is_untouched(self):
        profile, facts = distill_profile("other")
        profile["stack"][0]["lora"].update(basename=self.NAME, sha256=self.DMD2)
        before = copy.deepcopy(profile)
        SpyScenario(self, facts=facts).expect("missing_lora", preset_id="turbo_8step", cfg=1.0, identity_profile=profile)
        self.assertEqual(profile, before)

    def test_an_unknown_file_gets_the_pairs_listing(self):
        profile, facts = distill_profile("other")
        profile["stack"][0]["lora"].update(basename="mine.safetensors", sha256="a" * 64)
        exc = SpyScenario(self, facts=facts).expect("missing_lora", preset_id="turbo_8step", cfg=1.0,
                                                   identity_profile=profile)
        self.assertIn("mine.safetensors:other", str(exc))
        self.assertNotIn("known distill", str(exc))


class SlgOverrideOnTurboTest(unittest.TestCase):
    """MP-30 M2: the SLG override on a turbo preset is allowed for the Advanced node only."""

    SLG = {"slg_mode": "uncond_skip", "slg_layers": [22]}

    def turbo_scenario(self):
        profile, facts = distill_profile("distill")
        return SpyScenario(self, facts=facts), profile

    def test_the_default_request_still_refuses_it(self):
        s, profile = self.turbo_scenario()
        s.expect("invalid_override", preset_id="turbo_8step", cfg=1.5, identity_profile=profile, overrides=dict(self.SLG))
        self.assertEqual(s.identify.call_count, 0)

    def test_the_advanced_flag_lets_it_through_to_the_slg_clone(self):
        s, profile = self.turbo_scenario()
        with self.assertRaises(Stop):
            s.run(preset_id="turbo_8step", cfg=1.5, identity_profile=profile, overrides=dict(self.SLG),
                  allow_slg_override=True)
        self.assertTrue(s.spies["apply_slg"].called, "SLG must be applied at cfg 1.5")

    def test_the_flag_changes_nothing_for_other_overrides(self):
        s, profile = self.turbo_scenario()
        s.expect("invalid_override", preset_id="turbo_8step", cfg=3.0, identity_profile=profile,
                 overrides=dict(self.SLG), allow_slg_override=True)

    def test_the_request_flag_defaults_to_off(self):
        self.assertFalse(solver.SampleRequest.__dataclass_fields__["allow_slg_override"].default)


class FamilyGateTest(unittest.TestCase):
    """MP-30 M7: ``model_families`` of the preset is read; checkpoint_policy has nothing to enforce today."""

    def issues(self, preset_id, family):
        return solver.family_issues(presets.catalog()[preset_id], family, "MossPhotonSampler")

    def test_a_base_checkpoint_on_a_turbo_preset_warns(self):
        got = self.issues("turbo_8step", "moss-image-base")
        self.assertEqual([(i.code, i.severity, i.param) for i in got], [("model_family_unverified", "warning", "model")])
        self.assertIn("moss-image-base", got[0].message)
        self.assertIn("moss-image-sft", got[0].message)
        self.assertEqual(got[0].detail, {"family": "moss-image-base", "preset_families": ["moss-image-sft"]})
        self.assertEqual([i.code for i in self.issues("turbo_identity", "moss-image-base")], ["model_family_unverified"])

    def test_listed_and_unidentified_families_do_not_warn(self):
        self.assertEqual(self.issues("turbo_8step", "moss-image-sft"), [])
        self.assertEqual(self.issues("reference", "moss-image-base"), [])
        # an unidentified checkpoint is already covered by unverified_checkpoint / fingerprint_unavailable
        self.assertEqual(self.issues("turbo_8step", "unknown-moss"), [])

    def test_a_synthetic_sft_only_non_turbo_preset_warns_too(self):
        preset = copy.deepcopy(presets.catalog()["reference"])
        preset["model_families"] = ["moss-image-sft"]
        self.assertEqual([i.code for i in solver.family_issues(preset, "moss-image-base", "n")],
                         ["model_family_unverified"])

    def test_the_checkpoint_policy_values_have_no_second_behaviour(self):
        """Recorded no-op: the only policy any preset uses is native-moss-only (any MOSS checkpoint, an unknown hash is
        the warning unverified_checkpoint); native-moss-verified-only is reserved by the design and unused."""
        self.assertEqual({p["checkpoint_policy"] for p in presets.catalog().values()}, {"native-moss-only"})


class NoisePolicyTest(unittest.TestCase):
    def test_an_unknown_noise_policy_is_a_typed_error_not_a_value_error(self):
        """MP-30 n1 (solver.py:560): R3 says typed errors only; the node combo hides it, the API does not."""
        s = SpyScenario(self)
        exc = s.expect("unsupported_feature", noise_policy="bogus")
        self.assertIsInstance(exc, errors.UnsupportedFeature)
        self.assertEqual(exc.issue.param, "noise_policy")
        self.assertIn("bogus", exc.issue.message)


if __name__ == "__main__":
    unittest.main()
