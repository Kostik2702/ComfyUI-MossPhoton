"""INPUT_TYPES, RETURN_TYPES, descriptions and request building of the two sampler nodes (design 4.5, 4.6, 6.4)."""
import sys
import unittest
from unittest import mock

from moss_photon.core import consts, presets, solver
from moss_photon.nodes import sampler

BASIC_REQUIRED = ["model", "positive", "negative", "latent", "seed", "preset", "cfg"]
BASIC_OPTIONAL = ["identity_profile", "constraints", "vae", "preview", "preset_pin"]
ADVANCED_REQUIRED = BASIC_REQUIRED + ["steps", "slg_mode", "slg_layers", "slg_start", "slg_end", "solver",
                                      "cfg_profile", "noise_policy", "max_forwards", "fingerprint"]
ADVANCED_OPTIONAL = BASIC_OPTIONAL + ["sigmas", "report_step_times"]


class BasicSchemaTest(unittest.TestCase):
    def setUp(self):
        self.spec = sampler.MossPhotonSampler.INPUT_TYPES()

    def test_names_and_order(self):
        self.assertEqual(list(self.spec["required"]), BASIC_REQUIRED)
        self.assertEqual(list(self.spec["optional"]), BASIC_OPTIONAL)

    def test_types(self):
        req, opt = self.spec["required"], self.spec["optional"]
        self.assertEqual([req[k][0] for k in ("model", "positive", "negative", "latent")],
                         ["MODEL", "CONDITIONING", "CONDITIONING", "LATENT"])
        self.assertEqual((opt["identity_profile"][0], opt["constraints"][0], opt["vae"][0]),
                         ("PHOTON_IDENTITY_PROFILE", "PHOTON_CONSTRAINTS", "VAE"))

    def test_seed_cfg_preview_pin(self):
        req, opt = self.spec["required"], self.spec["optional"]
        self.assertEqual((req["seed"][0], req["seed"][1]["default"], req["seed"][1]["max"]), ("INT", 0, 2 ** 64 - 1))
        cfg = req["cfg"][1]
        self.assertEqual((req["cfg"][0], cfg["default"], cfg["min"], cfg["max"], cfg["step"], cfg["round"]),
                         ("FLOAT", 0.0, 0.0, 30.0, 0.1, 0.01))
        self.assertEqual((opt["preview"][0], opt["preview"][1]["default"]), ("BOOLEAN", True))
        self.assertEqual((opt["preset_pin"][0], opt["preset_pin"][1]["default"]), ("STRING", ""))

    def test_cfg_tooltip_says_zero_is_the_preset_value(self):
        self.assertIn("0 = preset value", self.spec["required"]["cfg"][1]["tooltip"])

    def test_preset_combo_is_the_catalog_with_reference_as_default(self):
        values, options = self.spec["required"]["preset"]
        self.assertEqual(values, presets.catalog_ids())
        self.assertEqual(options["default"], "reference")
        for row in presets.describe_catalog():
            self.assertIn("%s: %s" % (row["id"], row["status"]), options["tooltip"])

    def test_contract_lines(self):
        cls = sampler.MossPhotonSampler
        self.assertEqual(cls.RETURN_TYPES, ("LATENT", "PHOTON_REPORT"))
        self.assertEqual(cls.RETURN_NAMES, ("latent", "report"))
        self.assertEqual(cls.FUNCTION, "sample")
        self.assertEqual(cls.CATEGORY, consts.CAT_SAMPLING)
        self.assertEqual(len(cls.OUTPUT_TOOLTIPS), 2)

    def test_description_lists_every_preset_with_its_status(self):
        for row in presets.describe_catalog():
            self.assertIn("%s (%s)" % (row["id"], row["status"]), sampler.MossPhotonSampler.DESCRIPTION)
        for pid in consts.PRESET_IDS:
            self.assertIn(pid, sampler.MossPhotonSampler.DESCRIPTION)

    def test_is_changed_follows_the_catalog_hash(self):
        self.assertEqual(sampler.MossPhotonSampler.IS_CHANGED(), presets.catalog_sha256())
        self.assertEqual(sampler.MossPhotonSampler.IS_CHANGED(seed=1, preset="x"), presets.catalog_sha256())

    def test_registered_ids_and_display_names(self):
        self.assertEqual(sorted(sampler.NODE_CLASS_MAPPINGS), ["MossPhotonSampler", "MossPhotonSamplerAdvanced"])
        self.assertEqual(sampler.NODE_DISPLAY_NAME_MAPPINGS["MossPhotonSampler"], "Moss Photon Sampler")
        self.assertEqual(sampler.NODE_DISPLAY_NAME_MAPPINGS["MossPhotonSamplerAdvanced"],
                         "Moss Photon Sampler (Advanced)")
        self.assertTrue(set(sampler.NODE_CLASS_MAPPINGS) <= set(consts.NODE_IDS))

    def test_the_module_pulls_in_neither_comfy_nor_torch(self):
        self.assertNotIn("comfy", sys.modules)


class AdvancedSchemaTest(unittest.TestCase):
    def setUp(self):
        self.spec = sampler.MossPhotonSamplerAdvanced.INPUT_TYPES()

    def test_names_and_order(self):
        self.assertEqual(list(self.spec["required"]), ADVANCED_REQUIRED)
        self.assertEqual(list(self.spec["optional"]), ADVANCED_OPTIONAL)

    def test_basic_spec_is_untouched_by_building_the_advanced_one(self):
        self.assertEqual(list(sampler.MossPhotonSampler.INPUT_TYPES()["required"]), BASIC_REQUIRED)
        self.assertEqual(list(sampler.MossPhotonSampler.INPUT_TYPES()["optional"]), BASIC_OPTIONAL)

    def test_sentinels_reproduce_the_preset(self):
        req = self.spec["required"]
        self.assertEqual(req["steps"][1]["default"], 0)
        self.assertEqual(req["slg_mode"][0], ["from_preset", "off", "uncond_skip"])
        self.assertEqual(req["slg_mode"][1]["default"], "from_preset")
        self.assertEqual(req["slg_layers"][1]["default"], "")
        for name in ("slg_start", "slg_end"):
            self.assertEqual((req[name][1]["default"], req[name][1]["min"], req[name][1]["max"]), (-1.0, -1.0, 1.0))
        self.assertEqual(req["max_forwards"][1]["default"], 0)
        self.assertEqual(req["noise_policy"][0], ["native", "seed_plus_i"])
        self.assertEqual(req["noise_policy"][1]["default"], "native")
        self.assertEqual(req["fingerprint"][0], ["cached_or_compute", "cached_only", "off"])
        self.assertEqual((req["solver"][0], req["cfg_profile"][0]), (["euler"], ["constant"]))
        self.assertEqual(self.spec["optional"]["sigmas"][0], "SIGMAS")
        self.assertEqual(self.spec["optional"]["report_step_times"][1]["default"], False)

    def test_all_sentinels_resolve_to_a_zero_diff_for_every_preset(self):
        req = self.spec["required"]
        overrides = sampler.build_overrides(req["steps"][1]["default"], req["slg_mode"][1]["default"],
                                            req["slg_layers"][1]["default"], req["slg_start"][1]["default"],
                                            req["slg_end"][1]["default"], has_sigmas=False)
        self.assertEqual(overrides, {})
        for pid in presets.catalog_ids():
            with self.subTest(pid=pid):
                resolved = presets.resolve(pid, overrides=overrides or None)
                self.assertEqual((resolved.customized, resolved.diff), (False, ()))

    def test_same_contract_as_the_basic_node(self):
        cls = sampler.MossPhotonSamplerAdvanced
        self.assertEqual((cls.RETURN_TYPES, cls.RETURN_NAMES, cls.FUNCTION, cls.CATEGORY),
                         (sampler.MossPhotonSampler.RETURN_TYPES, ("latent", "report"), "sample", consts.CAT_SAMPLING))


class BuildOverridesTest(unittest.TestCase):
    def test_each_widget_alone(self):
        b = sampler.build_overrides
        self.assertEqual(b(30, "from_preset", "", -1.0, -1.0, has_sigmas=False), {"steps": 30})
        self.assertEqual(b(0, "off", "", -1.0, -1.0, has_sigmas=False), {"slg_mode": "off"})
        self.assertEqual(b(0, "from_preset", "22, 24", -1.0, -1.0, has_sigmas=False), {"slg_layers": [22, 24]})
        self.assertEqual(b(0, "from_preset", "", 0.25, -1.0, has_sigmas=False), {"slg_start": 0.25})
        self.assertEqual(b(0, "from_preset", "", -1.0, 0.0, has_sigmas=False), {"slg_end": 0.0})

    def test_steps_are_ignored_when_sigmas_are_connected(self):
        self.assertEqual(sampler.build_overrides(30, "from_preset", "", -1.0, -1.0, has_sigmas=True), {})

    def test_overrides_flow_into_resolve(self):
        overrides = sampler.build_overrides(30, "uncond_skip", "21", 0.0, 0.5, has_sigmas=False)
        resolved = presets.resolve("reference", overrides=overrides)
        self.assertTrue(resolved.customized)
        self.assertEqual([d["path"] for d in resolved.diff],
                         ["budget.nominal_steps", "sampling.slg_end", "sampling.slg_layers", "sampling.steps"])


class RequestBuildingTest(unittest.TestCase):
    def capture(self, node, **kw):
        seen = []

        def fake_execute(req):
            seen.append(req)
            return solver.SampleResult(latent={"samples": 1}, report={"r": 1}, summary="S")
        with mock.patch.object(sampler.core_solver, "execute", fake_execute):
            out = node.sample(**kw)
        return seen[0], out

    BASE = dict(model="M", positive="P", negative="N", latent={"samples": 0}, seed=7, preset="reference", cfg=3.0)

    def test_cfg_zero_takes_the_preset_value_and_says_so(self):
        for preset, expected in (("reference", 3.0), ("turbo_8step", 1.0), ("draft_fast", 3.0)):
            with self.subTest(preset=preset):
                req, out = self.capture(sampler.MossPhotonSampler(), **dict(self.BASE, preset=preset, cfg=0.0))
                self.assertEqual(req.cfg, expected)
                self.assertIn("cfg %g from the preset" % expected, out["ui"]["text"][0])

    def test_cfg_above_zero_is_the_users_explicit_value(self):
        req, out = self.capture(sampler.MossPhotonSampler(), **dict(self.BASE, cfg=3.0))
        self.assertEqual(req.cfg, 3.0)
        self.assertIn("cfg 3 from the user (widget)", out["ui"]["text"][0])
        req, _ = self.capture(sampler.MossPhotonSampler(), **dict(self.BASE, preset="turbo_8step", cfg=3.0))
        self.assertEqual(req.cfg, 3.0)  # explicit: the turbo bounds reject it later, in the resolver

    def test_cfg_zero_with_an_unknown_preset_or_bad_pin_raises_the_resolver_error(self):
        from moss_photon.core import errors
        with self.assertRaises(errors.UnknownPreset):
            self.capture(sampler.MossPhotonSampler(), **dict(self.BASE, preset="nope", cfg=0.0))
        with self.assertRaises(errors.PresetVersionMismatch):
            self.capture(sampler.MossPhotonSampler(), preset_pin="reference@9.9.9", **dict(self.BASE, cfg=0.0))

    def test_advanced_uses_the_same_rule(self):
        req, out = self.capture(
            sampler.MossPhotonSamplerAdvanced(), steps=0, slg_mode="from_preset", slg_layers="", slg_start=-1.0,
            slg_end=-1.0, solver="euler", cfg_profile="constant", noise_policy="native", max_forwards=0,
            fingerprint="off", **dict(self.BASE, preset="turbo_8step", cfg=0.0))
        self.assertEqual(req.cfg, 1.0)
        self.assertIn("from the preset", out["ui"]["text"][0])

    def test_basic_node(self):
        req, out = self.capture(sampler.MossPhotonSampler(), preset_pin=" reference@1.0.0\n", **self.BASE)
        self.assertEqual((req.node_id, req.preset_id, req.preset_pin, req.cfg, req.seed),
                         ("MossPhotonSampler", "reference", "reference@1.0.0", 3.0, 7))
        self.assertEqual((req.overrides, req.noise_policy, req.fingerprint_mode, req.max_forwards, req.sigmas),
                         ({}, "native", "cached_or_compute", 0, None))
        self.assertTrue(req.preview)
        self.assertEqual(out["result"], ({"samples": 1}, {"r": 1}))
        self.assertEqual(out["ui"]["text"], ["S\ncfg 3 from the user (widget)"])

    def test_advanced_node(self):
        req, out = self.capture(
            sampler.MossPhotonSamplerAdvanced(), steps=30, slg_mode="off", slg_layers="", slg_start=-1.0,
            slg_end=-1.0, solver="euler", cfg_profile="constant", noise_policy="seed_plus_i", max_forwards=77,
            fingerprint="off", sigmas=None, report_step_times=True, **self.BASE)
        self.assertEqual(req.node_id, "MossPhotonSamplerAdvanced")
        self.assertEqual(req.overrides, {"steps": 30, "slg_mode": "off"})
        self.assertEqual((req.noise_policy, req.max_forwards, req.fingerprint_mode, req.report_step_times),
                         ("seed_plus_i", 77, "off", True))

    def test_advanced_drops_steps_for_connected_sigmas(self):
        req, _ = self.capture(
            sampler.MossPhotonSamplerAdvanced(), steps=30, slg_mode="from_preset", slg_layers="", slg_start=-1.0,
            slg_end=-1.0, solver="euler", cfg_profile="constant", noise_policy="native", max_forwards=0,
            fingerprint="cached_or_compute", sigmas="SIG", **self.BASE)
        self.assertEqual((req.overrides, req.sigmas), ({}, "SIG"))

    def test_only_euler_exists(self):
        from moss_photon.core import errors
        with self.assertRaises(errors.UnsupportedFeature):
            self.capture(sampler.MossPhotonSamplerAdvanced(), steps=0, slg_mode="from_preset", slg_layers="",
                         slg_start=-1.0, slg_end=-1.0, solver="heun", cfg_profile="constant", noise_policy="native",
                         max_forwards=0, fingerprint="off", **self.BASE)


if __name__ == "__main__":
    unittest.main()
