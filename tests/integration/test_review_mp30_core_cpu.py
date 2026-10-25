"""MP-32 (MP-30 review) end to end on the tiny CPU MOSS: M2 Advanced SLG on turbo, M3 LoRA size, M6/m7 registered
codes, M7 family gate, m4 conditioning extras, m8 per-LoRA warnings, m15 unknown attention backend, m22 broken preset
file, n3 scheduler with input sigmas. The sampling steps are skipped (``NoSample``) where the subject is the label
or the report; the model, its patches and the conditioning are real."""
import copy
import os
import shutil
import tempfile
import unittest
from unittest import mock

from . import _comfy_env, tiny_moss
from . import sampler_support as ss


def known_family(family, sha, *, verified, variant="bf16"):
    """Patch ``fingerprint.identify_checkpoint`` so the fixture counts as a known checkpoint of ``family``."""
    from moss_photon.core import fingerprint
    ident = {"identity": {"sha256": sha, "size": 10164655808, "basename": "%s_%s.safetensors" % (family, variant),
                          "hash_source": "cache", "classified_by": "known_table", "official": True, "variant": variant,
                          "verified_matrix": verified}, "family": family, "warnings": []}
    return mock.patch.object(fingerprint, "identify_checkpoint", return_value=ident)


BASE_SHA = "cacd158728a1fb9e57c408a922b54752ec71fceb566b3821269927964e62b41f"


class CoreReviewFixesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.core import adapter, errors, presets, report, solver
        from moss_photon.nodes import sampler
        cls.torch, cls.adapter, cls.errors, cls.presets, cls.report = torch, adapter, errors, presets, report
        cls.solver, cls.sampler = solver, sampler
        base = cls.tiny.patcher
        cls.distill = ss.apply_lora(base, ss.make_lora_dict(base, seed=21), 1.0)
        cls.distill_profile = ss.make_profile(cls.distill, base_model=base, role="distill",
                                              name="lodestones-8-steps-dmd2-lora.safetensors")

    def assert_valid_and_registered(self, rep):
        """Every code in ``report.warnings`` is a key of REASONS and the whole report validates (MP-30 M6)."""
        self.assertEqual(self.report.validate_report(rep), [])
        for w in rep["warnings"]:
            self.assertIn(w["code"], self.errors.REASONS, w)
        return rep

    def turbo(self, profile=None, model=None, **kw):
        kw.setdefault("cfg", 1.0)
        kw.setdefault("preset_id", "turbo_8step")
        with ss.known_sft(), ss.NoSample():
            return self.assert_valid_and_registered(
                ss.run(self.tiny, model=model or self.distill, identity_profile=profile or self.distill_profile,
                       **kw).report)

    # ------------------------------------------------------------------ M1 (label through the real run)
    def test_a_customized_turbo_run_keeps_native_turbo_with_reasons(self):
        rep = self.turbo(cfg=1.5, overrides={"steps": 12})
        label = rep["label"]
        self.assertEqual(label["parity_class"], "native_turbo")
        self.assertTrue(label["customized"])
        self.assertEqual(label["not_reference_reasons"],
                         ["steps_differ", "cfg_differs", "turbo_setting_outside_verified"])
        self.assertEqual(self.turbo()["label"]["not_reference_reasons"], [])

    # ------------------------------------------------------------------ M2
    def advanced(self, **kw):
        t = self.tiny
        args = dict(model=self.distill, positive=t.make_cond(23), negative=t.make_cond(9),
                    latent={"samples": t.latent.clone()}, seed=tiny_moss.SEED, preset="turbo_8step", cfg=1.5, steps=0,
                    slg_mode="uncond_skip", slg_layers="22", slg_start=-1.0, slg_end=-1.0, solver="euler",
                    cfg_profile="constant", noise_policy="native", max_forwards=0, fingerprint="cached_or_compute",
                    identity_profile=self.distill_profile, preview=False)
        args.update(kw)
        with ss.known_sft(), ss.NoSample():
            return self.sampler.MossPhotonSamplerAdvanced().sample(**args)["result"][1]

    def test_the_advanced_node_runs_slg_on_a_turbo_preset_as_experimental_not_reference(self):
        rep = self.assert_valid_and_registered(self.advanced())
        codes = ss.codes(rep)
        self.assertIn("turbo_slg_experimental", codes)
        warning = next(w for w in rep["warnings"] if w["code"] == "turbo_slg_experimental")
        self.assertEqual(warning["severity"], "warning")
        self.assertEqual(rep["sampling"]["slg"]["mode"], "uncond_skip")
        self.assertEqual(rep["sampling"]["slg"]["layers"], [22])
        self.assertEqual(rep["label"]["parity_class"], "not_reference")
        self.assertIn("slg_differs", rep["label"]["not_reference_reasons"])
        self.assertTrue(rep["label"]["customized"])

    def test_the_advanced_node_slg_at_cfg1_is_inert_and_stays_native_turbo(self):
        rep = self.advanced(cfg=1.0)
        self.assertIn("slg_ineffective_cfg1", ss.codes(rep))
        # the nodes use the real bf16 reference set, so the float32 fixture reads as a modifier; SLG changes nothing
        self.assertEqual(rep["label"]["modifiers"], ["weight_dtype:torch.float32"])
        self.assertEqual(rep["label"]["not_reference_reasons"], [])

    def test_the_simple_sampler_path_still_refuses_an_slg_override_on_turbo(self):
        with ss.known_sft(), self.assertRaises(self.errors.InvalidOverride) as cm:
            ss.run(self.tiny, model=self.distill, identity_profile=self.distill_profile, preset_id="turbo_8step",
                   cfg=1.5, overrides={"slg_mode": "uncond_skip", "slg_layers": [22]})
        self.assertIn("MossPhotonSamplerAdvanced", str(cm.exception))

    def test_the_advanced_node_without_slg_widgets_is_unchanged(self):
        rep = self.advanced(slg_mode="from_preset", slg_layers="", cfg=1.0)
        self.assertNotIn("turbo_slg_experimental", ss.codes(rep))
        self.assertEqual(rep["label"]["not_reference_reasons"], [])
        self.assertEqual(rep["sampling"]["slg"]["mode"], "off")

    # ------------------------------------------------------------------ M3
    def test_the_real_producer_emits_the_lora_size_and_it_enters_the_recipe_without_a_hash(self):
        rep = self.turbo()
        item = rep["lora"]["stack"][0]
        self.assertEqual(item["size"], 1)  # make_profile writes size 1
        same = copy.deepcopy(self.distill_profile)
        same["stack"][0]["lora"]["sha256"] = None
        other = copy.deepcopy(same)
        other["stack"][0]["lora"]["size"] = 2
        a, b = self.turbo(profile=same), self.turbo(profile=other)
        self.assertEqual(a["lora"]["stack"][0]["sha256"], None)
        self.assertNotEqual(a["recipe_hash"], b["recipe_hash"])

    # ------------------------------------------------------------------ M6 / m7
    def test_identity_safe_reports_a_registered_recommendation_not_the_code_none(self):
        base = self.tiny.patcher
        patched = ss.apply_lora(base, ss.make_lora_dict(base, seed=31), 1.0)
        profile = ss.make_profile(patched, base_model=base, role="character")
        with ss.known_sft(), ss.NoSample():
            rep = self.assert_valid_and_registered(
                ss.run(self.tiny, model=patched, identity_profile=profile, preset_id="identity_safe").report)
        self.assertIn("reference_photos_recommended", ss.codes(rep))
        self.assertNotIn("none", ss.codes(rep))
        warning = next(w for w in rep["warnings"] if w["code"] == "reference_photos_recommended")
        self.assertIn("MossPhotonDiagnostics", warning["fix"])

    def test_every_preset_run_reports_registered_codes_only(self):
        with ss.known_sft(), ss.NoSample():
            for pid in ("reference", "photo_safe", "draft_fast", "night_photo", "advanced"):
                with self.subTest(preset=pid):
                    self.assert_valid_and_registered(ss.run(self.tiny, preset_id=pid).report)

    # ------------------------------------------------------------------ M7
    def test_a_base_checkpoint_on_a_turbo_preset_warns_and_is_not_reference(self):
        with known_family("moss-image-base", BASE_SHA, verified=False), ss.NoSample():
            rep = self.assert_valid_and_registered(
                ss.run(self.tiny, model=self.distill, identity_profile=self.distill_profile, preset_id="turbo_8step",
                       cfg=1.0).report)
        self.assertIn("model_family_unverified", ss.codes(rep))
        self.assertEqual(rep["model"]["family"], "moss-image-base")
        self.assertEqual(rep["label"]["parity_class"], "not_reference")
        self.assertIn("model_family_unverified", rep["label"]["not_reference_reasons"])

    def test_the_sft_checkpoint_on_a_turbo_preset_is_clean(self):
        rep = self.turbo()
        self.assertNotIn("model_family_unverified", ss.codes(rep))

    def test_a_base_checkpoint_on_a_preset_that_lists_it_gets_no_family_warning(self):
        with known_family("moss-image-base", BASE_SHA, verified=False), ss.NoSample():
            rep = self.assert_valid_and_registered(ss.run(self.tiny, preset_id="reference").report)
        self.assertNotIn("model_family_unverified", ss.codes(rep))
        self.assertEqual(rep["label"]["parity_class"], "reference_unverified")

    def test_an_unidentified_turbo_checkpoint_keeps_the_existing_path(self):
        with ss.NoSample():
            rep = self.assert_valid_and_registered(
                ss.run(self.tiny, model=self.distill, identity_profile=self.distill_profile, preset_id="turbo_8step",
                       cfg=1.0).report)
        self.assertNotIn("model_family_unverified", ss.codes(rep))
        self.assertIn("fingerprint_unavailable", ss.codes(rep))
        self.assertEqual(rep["label"]["parity_class"], "reference_unverified")

    # ------------------------------------------------------------------ m1 through a real model
    def test_a_wrong_graph_is_refused_before_the_checkpoint_is_hashed(self):
        from moss_photon.core import fingerprint
        with mock.patch.object(fingerprint, "identify_checkpoint") as identify, \
                self.assertRaises(self.errors.MissingLora):
            ss.run(self.tiny, preset_id="turbo_8step", cfg=1.0)
        identify.assert_not_called()

    # ------------------------------------------------------------------ m4
    def test_a_conditioning_with_an_area_is_a_modifier(self):
        positive = self.tiny.make_cond(23)
        positive[0][1]["area"] = ("percentage", 0.5, 0.5, 0.0, 0.0)
        with ss.known_sft(), ss.NoSample():
            rep = self.assert_valid_and_registered(ss.run(self.tiny, positive=positive).report)
        self.assertEqual(rep["label"]["parity_class"], "reference_modified")
        self.assertEqual(rep["label"]["modifiers"], ["cond_extras"])
        warning = next(w for w in rep["warnings"] if w["code"] == "foreign_modifiers")
        self.assertEqual(warning["param"], "positive")

    def test_a_plain_conditioning_is_not(self):
        with ss.known_sft(), ss.NoSample():
            rep = ss.run(self.tiny).report
        self.assertEqual((rep["label"]["parity_class"], rep["label"]["modifiers"]), ("reference_exact", []))

    def test_conditioning_extras_merge_with_model_modifiers_sorted_and_unique(self):
        from comfy_extras.nodes_cfg import CFGNorm
        model = CFGNorm.execute(self.tiny.patcher, 1.0).args[0]
        negative = self.tiny.make_cond(9)
        negative[0][1]["mask"] = self.torch.ones(1, 8, 8)
        with ss.known_sft(), ss.NoSample():
            rep = ss.run(self.tiny, model=model, negative=negative).report
        self.assertEqual(rep["label"]["modifiers"], ["cond_extras", "post_cfg_function"])

    # ------------------------------------------------------------------ m8
    def test_the_warnings_of_two_loras_are_both_kept(self):
        profile = copy.deepcopy(self.distill_profile)
        second = copy.deepcopy(profile["stack"][0])
        second["lora"]["basename"] = "second.safetensors"
        profile["stack"].append(second)
        for i, entry in enumerate(profile["stack"]):
            # check_stack puts the entry position under detail.index; the report keeps one key, stack_index (NEW-5)
            entry["warnings"] = [{"code": "lora_layers_missing", "severity": "info", "node": "MossPhotonLoRAProfile",
                                  "param": "lora_name", "message": "layer 22 is not in the file", "fix": "none needed",
                                  "detail": {"index": i, "layers": [22]}}]
        rep = self.turbo(profile=profile)
        got = [w for w in rep["warnings"] if w["code"] == "lora_layers_missing"]
        self.assertEqual(len(got), 2)
        self.assertEqual([w["detail"]["stack_index"] for w in got], [0, 1])
        self.assertEqual([sorted(w["detail"]) for w in got], [["layers", "stack_index"]] * 2)
        self.assertEqual(profile["stack"][1]["warnings"][0]["detail"]["index"], 1)  # the profile itself is untouched

    # ------------------------------------------------------------------ m15
    def test_an_attention_backend_outside_the_table_is_reported_as_unknown(self):
        import comfy.ldm.modules.attention as att

        def attention_future_backend(*a, **k):
            return None
        saved, saved_base = att.optimized_attention, self.adapter._env_base
        att.optimized_attention, self.adapter._env_base = attention_future_backend, None
        try:
            info = self.adapter.env_info()
        finally:
            att.optimized_attention, self.adapter._env_base = saved, saved_base
        self.assertEqual(info["attention"]["global"], "unknown:attention_future_backend")
        rep = ss.read_contract("report.json")
        rep["env"]["attention"]["global"] = info["attention"]["global"]
        self.assertEqual(self.report.validate_report(rep), [])

    def test_a_known_backend_is_still_its_table_name(self):
        import comfy.ldm.modules.attention as att
        saved, saved_base = att.optimized_attention, self.adapter._env_base
        att.optimized_attention, self.adapter._env_base = att.attention_pytorch, None
        try:
            self.assertEqual(self.adapter.env_info()["attention"]["global"], "pytorch")
        finally:
            att.optimized_attention, self.adapter._env_base = saved, saved_base

    # ------------------------------------------------------------------ m22
    def test_a_broken_preset_file_is_a_warning_in_the_report_and_the_others_run(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        for name in os.listdir(self.presets.preset_dir()):
            shutil.copy(os.path.join(self.presets.preset_dir(), name), tmp)
        with open(os.path.join(tmp, "photo_safe.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        with mock.patch.object(self.presets, "preset_dir", lambda: tmp), ss.known_sft(), ss.NoSample():
            rep = self.assert_valid_and_registered(ss.run(self.tiny, preset_id="reference").report)
            with self.assertRaises(self.errors.InvalidPresetFile):
                ss.run(self.tiny, preset_id="photo_safe")
        got = [w for w in rep["warnings"] if w["code"] == "invalid_preset_file"]
        self.assertEqual([(w["severity"], w["param"]) for w in got], [("warning", "presets/photo_safe.json")])
        self.assertEqual(rep["label"]["parity_class"], "reference_exact")

    # ------------------------------------------------------------------ n3
    def test_the_scheduler_is_not_claimed_for_custom_input_sigmas(self):
        with ss.known_sft(), ss.NoSample():
            native = ss.run(self.tiny).report
            custom = ss.run(self.tiny, sigmas=[1.0, 0.7, 0.3, 0.0]).report
            same = ss.run(self.tiny, sigmas=self.adapter.native_sigmas(self.tiny.patcher, "sgm_uniform", 50)).report
        self.assertEqual((native["sampling"]["scheduler"], native["sampling"]["sigmas"]["source"]),
                         ("sgm_uniform", "native_scheduler"))
        self.assertEqual((custom["sampling"]["scheduler"], custom["sampling"]["sigmas"]["source"]),
                         ("custom", "input"))
        # input sigmas that equal the native schedule ARE that scheduler's schedule
        self.assertEqual((same["sampling"]["scheduler"], same["sampling"]["sigmas"]["source"]),
                         ("sgm_uniform", "input"))

    # ------------------------------------------------------------------ n1
    def test_an_unknown_noise_policy_is_typed(self):
        with self.assertRaises(self.errors.UnsupportedFeature):
            ss.run(self.tiny, noise_policy="bogus")


if __name__ == "__main__":
    unittest.main()
