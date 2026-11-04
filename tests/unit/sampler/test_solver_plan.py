"""solver.execute preflight with a fake adapter whose GPU-touching functions are spies (design 6.4, MP-9 acceptance 1).

Every typed failure happens before ``model.clone()``, the noise, the guider and ``guider.sample``: the spies for those
four were NEVER called. A positive control proves the spies are wired: a request that passes the whole preflight
reaches ``model.clone()`` (which then stops the run on purpose).
"""
import copy
import json
import os
import types
import unittest
from unittest import mock

from moss_photon.core import adapter, canon, consts, constraints_contract, errors, fingerprint, solver

CONTRACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "fixtures", "contracts")


class Stop(Exception):
    """Raised by the clone / SLG spies: the preflight was passed."""


class FakeSigmas:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class FakeModel:
    def __init__(self):
        self.clone_calls = 0
        self.model_options = {"transformer_options": {}}
        self.patches = {}

    def clone(self):
        self.clone_calls += 1
        raise Stop()

    def get_model_object(self, name):
        assert name == "model_sampling", name
        return types.SimpleNamespace(percent_to_sigma=lambda p: 1.0 - p)


def good_facts(**kw):
    base = dict(is_moss=True, class_name="MOSSImage", num_layers=24, dim=128, text_dim=64,
                model_sampling_class="ModelSamplingDiscreteFlow+CONST", is_flow=True, is_const=True, shift=5.0,
                multiplier=1.0, sigma_max=1.0, latent_channels=128, spacial_downscale_ratio=16,
                dtype="torch.float32", load_device="cpu", is_dynamic=False, weight_dtype="torch.float32",
                manual_cast_dtype=None, foreign_modifiers=(), patches_uuid="u0", patch_entries=0,
                checkpoint_path=None, lora_stack_attachment=None)
    base.update(kw)
    return adapter.ModelFacts(**base)


def good_latent(**kw):
    base = dict(batch=1, channels=128, height=8, width=8, dtype="torch.float32", device="cpu", is_nested=False,
                is_empty=True, has_noise_mask=False, has_batch_index=False, raw_shape=(1, 128, 8, 8),
                raw_downscale_ratio_spacial=None, rescaled_by_native_fix=False, pixel_width=128, pixel_height=128)
    base.update(kw)
    return adapter.LatentFacts(**base)


def good_cond(dim=64, **kw):
    base = dict(entries=1, token_counts=(23,), dim=dim, dtype="torch.float32", sha256="0" * 64,
                has_area_or_hooks=False)
    base.update(kw)
    return adapter.CondFacts(**base)


class Scenario:
    """One preflight experiment with every spy installed."""

    def __init__(self, test, facts=None, latent=None, cond=None, vae=None, sigma_values=None):
        self.test = test
        self.model = FakeModel()
        self.facts = facts or good_facts()
        self.latent = latent or good_latent()
        self.cond = cond or good_cond()
        self.vae = vae
        self.sigma_values = sigma_values
        self.spies = {"make_noise": mock.Mock(name="make_noise"), "build_guider": mock.Mock(name="build_guider"),
                      "install_counter": mock.Mock(name="install_counter"),
                      "apply_slg": mock.Mock(name="apply_slg", side_effect=Stop())}
        fake = types.SimpleNamespace(**{n: getattr(adapter, n) for n in adapter.__all__})
        fake.probe_model = lambda model, reference_dtypes=None: self.facts
        fake.fix_empty_latent = lambda model, latent, node=None: object()
        fake.probe_latent = lambda latent, fixed: self.latent
        fake.probe_conditioning = lambda cond, name: self.cond
        fake.probe_vae = lambda vae: self.vae
        fake.env_info = lambda model=None: {}
        fake.native_sigmas = lambda model, scheduler, steps: FakeSigmas(
            self.sigma_values or [1.0 - i / steps for i in range(steps)] + [0.0])
        self.fake = fake

    def request(self, **kw):
        fields = dict(node_id="MossPhotonSampler", model=self.model, positive=[], negative=[],
                      latent={"samples": object()}, seed=1, preset_id="reference", preset_pin="", cfg=3.0,
                      preview=False)
        fields.update(kw)
        return solver.SampleRequest(**fields)

    def run(self, **kw):
        ident = {"identity": {"sha256": None, "size": 0, "basename": "unknown", "hash_source": "unavailable",
                              "classified_by": "none", "official": None, "variant": None, "verified_matrix": None},
                 "family": "unknown-moss", "warnings": []}
        with mock.patch.object(solver, "adapter", self.fake), \
                mock.patch.object(solver.rng, "make_noise", self.spies["make_noise"]), \
                mock.patch.object(solver, "build_guider", self.spies["build_guider"]), \
                mock.patch.object(solver, "install_counter", self.spies["install_counter"]),                 mock.patch.object(solver.guidance, "apply_slg", self.spies["apply_slg"]), \
                mock.patch.object(solver.fingerprint, "identify_checkpoint", return_value=ident):
            return solver.execute(self.request(**kw))

    def expect(self, code, **kw):
        with self.test.assertRaises(errors.MossPhotonError) as cm:
            self.run(**kw)
        self.test.assertEqual(cm.exception.code, code, str(cm.exception))
        self.assert_untouched()
        return cm.exception

    def reach(self, **kw):
        """The request passes the whole preflight (the clone / SLG spy stops the run on purpose: ``Stop``); returns the
        codes of the warnings the run noted on the way. MP-84: a setting conflict degrades to a warning, so the run goes on."""
        seen = []
        self.noted = []
        real = solver.report.ReportBuilder.warn

        def warn(builder, issue, scope=None):
            item = issue.to_dict() if hasattr(issue, "to_dict") else dict(issue)
            self.noted.append(item)
            seen.append(item["code"])
            return real(builder, issue, scope)

        with mock.patch.object(solver.report.ReportBuilder, "warn", warn):
            with self.test.assertRaises(Stop):
                self.run(**kw)
        return seen

    def assert_untouched(self):
        self.test.assertEqual(self.model.clone_calls, 0, "model.clone() was called")
        for name, spy in self.spies.items():
            self.test.assertFalse(spy.called, "%s was called" % name)


def profile_fixture():
    with open(os.path.join(CONTRACTS, "identity_profile.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def constraints_fixture(bind=True, **blocks):
    """The constraints fixture with the given block flags (and a field text that agrees with each flag), bound to
    the empty conditioning the Scenario passes (``bind=False``: the fixture's own, unbound ``conditioning: null``).
    Since MP-60 the sampler validates the contract and runs its checks only for constraints that describe the
    conditioning it received."""
    with open(os.path.join(CONTRACTS, "constraints.json"), "r", encoding="utf-8") as f:
        c = json.load(f)
    for name, present in blocks.items():
        c["blocks_present"][name] = present
        c["fields"][name] = (c["fields"][name] or "x") if present else ""
    return constraints_contract.bind_conditioning(c, [], []) if bind else c


def layout_with_extent(extent):
    return {"items": 1, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx", "bboxes": [list(extent)],
            "extent": list(extent), "experimental": True}


class PreflightErrorsTest(unittest.TestCase):
    def test_positive_control_a_good_request_reaches_clone(self):
        s = Scenario(self)
        with self.assertRaises(Stop):
            s.run()  # reference preset: SLG on, so the SLG clone is the first heavy step
        self.assertTrue(s.spies["apply_slg"].called)
        s = Scenario(self)
        with self.assertRaises(Stop):
            s.run(overrides={"slg_mode": "off"})
        self.assertEqual(s.model.clone_calls, 1)

    def test_unsupported_model(self):
        Scenario(self, facts=good_facts(is_moss=False, class_name="Flux")).expect("unsupported_model")

    def test_multiplier_1000(self):
        e = Scenario(self, facts=good_facts(multiplier=1000.0)).expect("invalid_model_sampling")
        self.assertIn("ModelSamplingAuraFlow", str(e))

    def test_foreign_cond_hook_with_slg_requested(self):
        Scenario(self, facts=good_facts(foreign_modifiers=("cond_batch_function",))).expect("conflicting_patch")

    def test_foreign_cond_hook_without_slg_is_only_a_warning(self):
        s = Scenario(self, facts=good_facts(foreign_modifiers=("cond_batch_function",)))
        with self.assertRaises(Stop):
            s.run(overrides={"slg_mode": "off"})

    def test_four_channel_non_empty_latent(self):
        Scenario(self, latent=good_latent(channels=4, is_empty=False, raw_shape=(1, 4, 8, 8))).expect("invalid_latent")

    def test_noise_mask(self):
        Scenario(self, latent=good_latent(has_noise_mask=True)).expect("unsupported_source_input")

    def test_wrong_text_dim(self):
        Scenario(self, cond=good_cond(dim=4096)).expect("invalid_conditioning")

    def test_empty_conditioning(self):
        Scenario(self, cond=good_cond(entries=0, token_counts=(), dim=None)).expect("invalid_conditioning")

    def test_bad_vae(self):
        Scenario(self, vae=adapter.VaeFacts(4, 8)).expect("invalid_vae", vae=object())

    def test_unknown_preset(self):
        Scenario(self).expect("invalid_preset", preset_id="nope")

    def test_pin_mismatch_and_malformed_pin_are_warnings_and_the_run_goes_on(self):
        for pin in ("reference@9.9.9", "reference@1.0", "other@1.0.0", "not a pin"):
            with self.subTest(pin=pin):
                self.assertIn("preset_pin_mismatch", Scenario(self).reach(preset_pin=pin))

    def test_a_value_of_the_wrong_type_is_still_an_invalid_override(self):
        Scenario(self).expect("invalid_override", overrides={"steps": "five"})

    def test_steps_above_the_limit_are_moved_to_it_with_a_warning(self):
        self.assertIn("setting_adjusted", Scenario(self).reach(overrides={"steps": 5000}))

    def test_slg_layer_that_does_not_exist_on_this_model(self):
        """The preset's OWN layer (reference: 22) on a model without it: the wrong model, still an error."""
        Scenario(self, facts=good_facts(num_layers=4)).expect("invalid_override")

    def test_slg_layers_the_model_does_not_have_are_dropped_with_a_warning(self):
        codes = Scenario(self, facts=good_facts(num_layers=30)).reach(
            overrides={"slg_mode": "uncond_skip", "slg_layers": [22, 99]})
        self.assertIn("setting_adjusted", codes)

    def test_cfg_out_of_range_is_moved_into_range_with_a_warning(self):
        for cfg in (31.0, 0.5):
            with self.subTest(cfg=cfg):
                self.assertIn("setting_adjusted", Scenario(self).reach(cfg=cfg))

    def test_identity_safe_without_a_lora_profile_notes_the_role_and_goes_on(self):
        self.assertIn("lora_not_recorded", Scenario(self).reach(preset_id="identity_safe"))

    def test_identity_safe_with_a_profile_without_a_character_notes_the_role_and_goes_on(self):
        profile = profile_fixture()
        profile["stack"][0]["role"] = "style"
        facts = good_facts(lora_stack_attachment=canon.canonical_json(profile["attachment"]),
                           patches_uuid=profile["attachment"]["patches_uuid"],
                           patch_entries=profile["attachment"]["patch_entries_total"])
        self.assertIn("lora_not_recorded", Scenario(self, facts=facts).reach(preset_id="identity_safe",
                                                                             identity_profile=profile))

    def test_ordinary_loader_patches_are_not_a_missing_role(self):
        """A MODEL with patches no profile explains (a stock LoraLoader) is phase all: no role note, an info at most."""
        codes = Scenario(self, facts=good_facts(patch_entries=252)).reach(preset_id="identity_safe")
        self.assertNotIn("lora_not_recorded", codes)
        self.assertIn("foreign_patches", codes)

    def test_wardrobe_safe_with_an_empty_wardrobe_block(self):
        Scenario(self).expect("missing_required_input", preset_id="wardrobe_safe",
                              constraints=constraints_fixture(wardrobe=False))

    def test_wardrobe_safe_without_constraints_at_all(self):
        Scenario(self).expect("missing_required_input", preset_id="wardrobe_safe")

    def test_scene_safe_with_an_empty_scene_block(self):
        Scenario(self).expect("missing_required_input", preset_id="scene_safe",
                              constraints=constraints_fixture(scene=False))

    def test_scene_safe_with_a_source_latent(self):
        Scenario(self, latent=good_latent(is_empty=False)).expect(
            "unsupported_source_input", preset_id="scene_safe", constraints=constraints_fixture())

    def test_wardrobe_safe_with_its_block_passes_the_preflight(self):
        with self.assertRaises(Stop):
            Scenario(self).run(preset_id="wardrobe_safe", constraints=constraints_fixture())

    def test_a_profile_that_does_not_belong_to_the_model(self):
        Scenario(self).expect("incompatible_lora", identity_profile=profile_fixture())

    def test_a_phase_plan_that_does_not_match_the_stack_is_a_broken_profile(self):
        """A profile that does not describe itself consistently is refused; a CONSISTENT phased profile on the basic sampler
        degrades to a constant weight (integration test: ``test_lora_plan_cpu``)."""
        profile = profile_fixture()
        profile["phase_plan"] = {"kind": "phased"}
        facts = good_facts(lora_stack_attachment=canon.canonical_json(profile["attachment"]),
                           patches_uuid=profile["attachment"]["patches_uuid"],
                           patch_entries=profile["attachment"]["patch_entries_total"])
        Scenario(self, facts=facts).expect("unsupported_feature", identity_profile=profile)

    def test_layout_outside_the_canvas(self):
        c = constraints_fixture()
        c["layout"] = layout_with_extent([-5000, 0, 0, 0])
        Scenario(self).expect("invalid_layout", constraints=c)

    def test_invalid_schedule_nan(self):
        Scenario(self, sigma_values=[1.0, float("nan"), 0.5, 0.0]).expect("invalid_schedule")

    def test_invalid_schedule_not_ending_at_zero(self):
        Scenario(self, sigma_values=[1.0, 0.7, 0.4, 0.1]).expect("invalid_schedule")

    def test_input_sigmas_are_validated_before_any_forward(self):
        Scenario(self).expect("invalid_schedule", sigmas=FakeSigmas([1.0, 0.5, 0.6, 0.0]))

    def test_a_plan_over_the_forward_limit_is_a_warning_and_the_run_goes_on(self):
        self.assertIn("forward_budget_exceeded", Scenario(self).reach(max_forwards=10))
        self.assertNotIn("forward_budget_exceeded", Scenario(self).reach(max_forwards=1000))

    def test_seed_plus_i_with_batch_index(self):
        s = Scenario(self, latent=good_latent(has_batch_index=True))
        s.expect("unsupported_feature", noise_policy="seed_plus_i")

    def test_a_required_guard_cannot_run_in_v1(self):
        s = Scenario(self)
        preset = copy.deepcopy(solver.presets.resolve("reference").preset)
        preset["guards"] = [{"id": "needs_detector", "required": True, "requires_dependency": "x"}]
        real = solver.presets.resolve

        def patched(*a, **kw):
            r = real(*a, **kw)
            # the constraints and guards are read from the first resolve now (MP-30 m1: before the hash), so every call
            return type(r)(**dict(r.__dict__, preset=preset))
        with mock.patch.object(solver.presets, "resolve", patched):
            s.expect("missing_guard_dependency")


if __name__ == "__main__":
    unittest.main()
