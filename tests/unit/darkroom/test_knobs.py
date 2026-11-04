"""MP-71: the darkroom surface of the two sampler nodes. The simple node gets no new widget; the Advanced node gets the phase
knobs (each at a sentinel that means "from the preset") and the optional ``finish_model``, appended AFTER every existing input so
the positional widget values of a saved workflow do not move; a darkroom preset reaches the solver as a ``DarkroomRequest`` with the
cfg widget as the finish guidance; a stock preset reaches it exactly as before (no request at all) unless a knob is set.
"""
import inspect
import sys
import unittest
from unittest import mock

from moss_photon.core import darkroom, lora, solver
from moss_photon.nodes import sampler

from .support import DARKROOM_IDS, PORTRAIT, preset_of, settings_of

KNOBS = ("schedule", "compose_cfg", "compose_slg", "r_c", "r_k", "w_compose", "w_commit", "reexpose", "sigma_r", "r_f",
         "finish_cfg", "eta", "switch_model")
BASIC_REQUIRED = ["model", "positive", "negative", "latent", "seed", "preset", "cfg"]
BASIC_OPTIONAL = ["identity_profile", "constraints", "vae", "preview", "preset_pin"]
V020_ADVANCED_OPTIONAL = BASIC_OPTIONAL + ["sigmas", "report_step_times"]  # what a saved v0.2.0 workflow was made with


class SchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = sampler.MossPhotonSamplerAdvanced.INPUT_TYPES()

    def test_the_simple_node_has_no_darkroom_widget(self):
        spec = sampler.MossPhotonSampler.INPUT_TYPES()
        self.assertEqual(list(spec["required"]), BASIC_REQUIRED)
        self.assertEqual(list(spec["optional"]), BASIC_OPTIONAL + ["steps"])  # MP-84: appended, nothing before it moves

    def test_the_knobs_are_appended_after_every_widget_of_v020(self):
        self.assertEqual(list(self.spec["optional"]), V020_ADVANCED_OPTIONAL + ["finish_model"] + list(KNOBS))
        self.assertEqual(self.spec["optional"]["finish_model"][0], "MODEL")

    def test_the_darkroom_presets_are_in_the_preset_combo_of_both_nodes(self):
        for cls in (sampler.MossPhotonSampler, sampler.MossPhotonSamplerAdvanced):
            values = cls.INPUT_TYPES()["required"]["preset"][0]
            for pid in DARKROOM_IDS:
                self.assertIn(pid, values)

    def test_every_knob_defaults_to_its_sentinel(self):
        opt = self.spec["optional"]
        for name in KNOBS:
            with self.subTest(knob=name):
                widget = opt[name]
                default = widget[1]["default"]
                if isinstance(widget[0], list):
                    self.assertEqual((default, widget[0][0]), (sampler.FROM_PRESET, sampler.FROM_PRESET))
                elif name in ("compose_cfg", "finish_cfg"):
                    self.assertEqual(default, 0.0)
                else:
                    self.assertEqual((default, widget[1]["min"]), (-1.0, -1.0))

    def test_the_schedule_combo_is_from_preset_plus_the_engine_schedules(self):
        self.assertEqual(self.spec["optional"]["schedule"][0], [sampler.FROM_PRESET, "native", "sa2_8"])
        self.assertEqual(self.spec["optional"]["schedule"][0][1:], list(darkroom.SCHEDULES))

    def test_the_cfg_tooltip_says_what_the_widget_does_on_a_darkroom_preset(self):
        for cls in (sampler.MossPhotonSampler, sampler.MossPhotonSamplerAdvanced):
            tip = cls.INPUT_TYPES()["required"]["cfg"][1]["tooltip"]
            self.assertIn("0 = preset value", tip)
            self.assertIn("FINISH guidance", tip)
            self.assertIn("negative prompt", tip)

    def test_the_sample_signature_matches_the_inputs(self):
        params = inspect.signature(sampler.MossPhotonSamplerAdvanced.sample).parameters
        declared = list(self.spec["required"]) + list(self.spec["optional"])
        self.assertEqual(sorted(list(params)[1:]), sorted(declared))
        for name, p in list(params.items())[1:]:
            self.assertEqual(p.default is inspect.Parameter.empty, name in self.spec["required"], name)

    def test_the_defaults_of_the_signature_are_the_widget_sentinels(self):
        params = inspect.signature(sampler.MossPhotonSamplerAdvanced.sample).parameters
        for name in KNOBS:
            self.assertEqual(params[name].default, self.spec["optional"][name][1]["default"], name)

    def test_the_node_module_imports_no_comfy_or_torch(self):
        src = inspect.getsource(sampler)
        self.assertNotIn("import torch", src)
        self.assertNotIn("import comfy", src)
        self.assertNotIn("comfy", sys.modules)


class SentinelsTest(unittest.TestCase):
    SENTINELS = dict(schedule="from_preset", compose_cfg=0.0, compose_slg="from_preset", r_c=-1.0, r_k=-1.0, w_compose=-1.0,
                     w_commit=-1.0, reexpose="from_preset", sigma_r=-1.0, r_f=-1.0, finish_cfg=0.0, eta=-1.0,
                     switch_model="from_preset")

    def test_every_sentinel_means_the_preset(self):
        overrides = sampler.build_darkroom_overrides(**self.SENTINELS)
        self.assertEqual(sorted(overrides), sorted(darkroom.OVERRIDE_NAMES))
        self.assertEqual(set(overrides.values()), {None})
        for pid in DARKROOM_IDS:
            self.assertEqual(darkroom.resolve_settings(preset_of(pid), overrides), settings_of(pid))

    def test_the_widget_defaults_are_the_sentinels(self):
        opt = sampler.MossPhotonSamplerAdvanced.INPUT_TYPES()["optional"]
        defaults = {name: opt[name][1]["default"] for name in KNOBS}
        self.assertEqual(defaults, self.SENTINELS)

    def test_a_real_value_overrides_and_is_named(self):
        for widget, name, value in (("compose_cfg", "compose_cfg", 1.5), ("r_c", "r_c", 0.0), ("eta", "eta", 0.0),
                                    ("w_commit", "w_commit", 0.0), ("finish_cfg", "finish_cfg", 3.0),
                                    ("schedule", "schedule", "native"), ("switch_model", "switch", "off")):
            with self.subTest(widget=widget):
                ov = sampler.build_darkroom_overrides(**dict(self.SENTINELS, **{widget: value}))
                self.assertIsNotNone(ov[name])
                self.assertEqual([k for k, v in ov.items() if v is not None], [name])

    def test_the_combos_map_to_booleans(self):
        ov = sampler.build_darkroom_overrides(**dict(self.SENTINELS, compose_slg="on", reexpose="off", switch_model="on"))
        self.assertEqual((ov["compose_slg"], ov["reexpose"], ov["switch"]), (True, False, True))

    def test_zero_is_a_real_value_for_the_fractions_and_noise(self):
        ov = sampler.build_darkroom_overrides(**dict(self.SENTINELS, r_c=0.0, eta=0.0, w_commit=0.0))
        self.assertEqual((ov["r_c"], ov["eta"], ov["w_commit"]), (0.0, 0.0, 0.0))


class RequestTest(unittest.TestCase):
    """What the nodes hand the solver; ``execute`` is replaced by a recorder."""

    BASE = dict(model="M", positive="P", negative="N", latent={"samples": 0}, seed=7)
    ADV = dict(steps=0, slg_mode="from_preset", slg_layers="", slg_start=-1.0, slg_end=-1.0, solver="euler",
               cfg_profile="constant", noise_policy="native", max_forwards=0, fingerprint="off")

    def capture(self, node, **kw):
        seen = []
        report = {"r": 1, "darkroom": {"settings": {"finish": {"cfg": kw.pop("report_finish", 2.0)}}}}

        def fake_execute(req):
            seen.append(req)
            return solver.SampleResult(latent={"samples": 1}, report=report, summary="S")
        with mock.patch.object(sampler.core_solver, "execute", fake_execute):
            out = node.sample(**kw)
        return seen[0], out

    def test_a_darkroom_preset_in_the_simple_node(self):
        req, out = self.capture(sampler.MossPhotonSampler(), preset=PORTRAIT, cfg=0.0, **self.BASE)
        self.assertEqual((req.preset_id, req.cfg, req.node_id), (PORTRAIT, 1.0, "MossPhotonSampler"))
        self.assertTrue(req.allow_phased_lora, "the character adapter may be a phased profile on a darkroom preset")
        self.assertIsInstance(req.darkroom, darkroom.DarkroomRequest)
        self.assertEqual((req.darkroom.overrides, req.darkroom.cfg_widget, req.darkroom.finish_model), ({}, None, None))
        self.assertIs(req.darkroom.boost_loader, lora.boost_patches)
        self.assertIn("compose cfg 1 from the preset", out["ui"]["text"][0])
        self.assertIn("finish cfg 2 from the preset", out["ui"]["text"][0])

    def test_the_cfg_widget_is_the_finish_guidance_and_the_summary_says_so(self):
        req, out = self.capture(sampler.MossPhotonSampler(), preset=PORTRAIT, cfg=3.0, report_finish=3.0, **self.BASE)
        self.assertEqual(req.cfg, 1.0, "sampling.cfg stays the compose guidance")
        self.assertEqual(req.darkroom.cfg_widget, 3.0)
        text = out["ui"]["text"][0]
        self.assertIn("finish cfg 3 from the cfg widget", text)
        self.assertIn("the cfg widget sets the finish guidance", text)

    def test_the_advanced_knobs_reach_the_request_and_compose_cfg_is_the_sampling_cfg(self):
        req, out = self.capture(
            sampler.MossPhotonSamplerAdvanced(), preset=PORTRAIT, cfg=0.0, compose_cfg=1.1, finish_cfg=2.5, r_f=0.3,
            switch_model="on", finish_model="FM", report_finish=2.5, **self.ADV, **self.BASE)
        self.assertEqual(req.cfg, 1.1)
        self.assertEqual(req.darkroom.overrides, {"finish_cfg": 2.5, "r_f": 0.3, "switch": True})
        self.assertEqual(req.darkroom.finish_model, "FM")
        self.assertEqual(req.node_id, "MossPhotonSamplerAdvanced")
        self.assertIn("compose cfg 1.1 from the user (compose_cfg widget)", out["ui"]["text"][0])
        self.assertIn("finish cfg 2.5 from the finish_cfg widget", out["ui"]["text"][0])

    def test_a_stock_preset_gets_the_request_it_always_got(self):
        for node, extra in ((sampler.MossPhotonSampler(), {}), (sampler.MossPhotonSamplerAdvanced(), self.ADV)):
            for pid, cfg, want in (("reference", 0.0, 3.0), ("turbo_identity", 0.0, 1.0), ("turbo_8step", 1.5, 1.5)):
                with self.subTest(node=type(node).__name__, preset=pid):
                    req, out = self.capture(node, preset=pid, cfg=cfg, **extra, **self.BASE)
                    self.assertIsNone(req.darkroom)
                    self.assertEqual(req.cfg, want)
                    self.assertEqual(req.allow_phased_lora, isinstance(node, sampler.MossPhotonSamplerAdvanced))
                    self.assertNotIn("finish cfg", out["ui"]["text"][0])

    def test_a_knob_or_a_finish_model_on_a_stock_preset_still_reaches_the_solver_which_warns(self):
        req, _ = self.capture(sampler.MossPhotonSamplerAdvanced(), preset="reference", cfg=0.0, r_f=0.3, eta=0.1,
                              **self.ADV, **self.BASE)
        self.assertEqual(req.darkroom.overrides, {"r_f": 0.3, "eta": 0.1})
        self.assertIsNone(req.darkroom.cfg_widget)
        self.assertEqual(req.cfg, 3.0, "the cfg widget keeps its meaning on a preset without a block")
        req, _ = self.capture(sampler.MossPhotonSamplerAdvanced(), preset="reference", cfg=0.0, finish_model="FM",
                              **self.ADV, **self.BASE)
        self.assertEqual((req.darkroom.overrides, req.darkroom.finish_model), ({}, "FM"))

    def test_an_unknown_preset_raises_the_resolver_error_and_a_bad_pin_does_not(self):
        """MP-84: a pin is advisory: the request goes on (the resolver warns ``preset_pin_mismatch``)."""
        from moss_photon.core import errors
        with self.assertRaises(errors.UnknownPreset):
            self.capture(sampler.MossPhotonSampler(), preset="nope", cfg=0.0, **self.BASE)
        req, _ = self.capture(sampler.MossPhotonSampler(), preset=PORTRAIT, cfg=0.0, preset_pin="%s@9.9.9" % PORTRAIT,
                              **self.BASE)
        self.assertEqual(req.preset_id, PORTRAIT)


if __name__ == "__main__":
    unittest.main()
