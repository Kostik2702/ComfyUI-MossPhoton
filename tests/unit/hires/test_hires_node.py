"""MossPhotonHiresFix, the declared contract of the node (no ComfyUI): inputs, defaults, bounds, outputs, tooltips, where it is
registered, and that it adds no raise site of its own (it stops only through helpers that are already on the no-raise list).
The run of the node on a tiny MODEL is tests/integration/test_hires_cpu.py."""
import sys
import unittest
from unittest import mock

from moss_photon.core import consts
from moss_photon.finishing import deps
from moss_photon.nodes import hires as node

from tests.unit import test_raise_whitelist as guard

REQUIRED = ("model", "positive", "negative", "vae", "latent", "seed", "scale", "steps", "denoise", "cfg", "sampler_name",
            "scheduler", "face_protect", "face_sigma", "identity_boost")
OPTIONAL = ("character_lora", "upscale_method")


class DeclaredContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # the embedded interpreter can import ComfyUI's folder_paths, which drags comfy into the unit run: block it, the combo
        # then holds just "none" (the contract of _lora_choices)
        with mock.patch.dict(sys.modules, {"folder_paths": None}):
            cls.spec = node.MossPhotonHiresFix.INPUT_TYPES()

    def widget(self, name):
        section = self.spec["required"] if name in self.spec["required"] else self.spec["optional"]
        return section[name]

    def test_the_inputs_in_the_order_of_the_ticket(self):
        self.assertEqual(tuple(self.spec["required"]), REQUIRED)
        self.assertEqual(tuple(self.spec["optional"]), OPTIONAL)

    def test_the_socket_types(self):
        for name, kind in (("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("vae", "VAE"),
                           ("latent", "LATENT")):
            self.assertEqual(self.widget(name)[0], kind, name)

    def test_defaults_and_bounds_are_the_recipe(self):
        for name, kind, default, lo, hi in (("scale", "FLOAT", 1.5, 1.0, 2.0), ("steps", "INT", 6, 1, 50),
                                            ("denoise", "FLOAT", 0.32, 0.05, 1.0), ("cfg", "FLOAT", 1.0, 0.0, 30.0),
                                            ("face_sigma", "FLOAT", 0.35, 0.0, 1.0), ("identity_boost", "FLOAT", 0.2, 0.0, 2.0)):
            kind_got, opts = self.widget(name)
            self.assertEqual(kind_got, kind, name)
            self.assertEqual((opts["default"], opts["min"], opts["max"]), (default, lo, hi), name)
        self.assertEqual(self.widget("scale")[1]["step"], 0.05)
        self.assertEqual(self.widget("face_sigma")[1]["step"], 0.01)
        self.assertEqual(self.widget("identity_boost")[1]["step"], 0.05)
        self.assertEqual(self.widget("face_protect")[0], "BOOLEAN")
        self.assertIs(self.widget("face_protect")[1]["default"], True)

    def test_the_seed_is_a_stock_seed(self):
        kind, opts = self.widget("seed")
        self.assertEqual((kind, opts["default"], opts["min"], opts["max"]), ("INT", 0, 0, 0xFFFFFFFFFFFFFFFF))
        self.assertIs(opts["control_after_generate"], True)

    def test_the_combos(self):
        samplers, opts = self.widget("sampler_name")
        self.assertEqual(opts["default"], "res_multistep")
        self.assertIn("res_multistep", samplers)
        self.assertIn("euler", samplers)
        schedulers, opts = self.widget("scheduler")
        self.assertEqual(opts["default"], "linear_quadratic")
        self.assertIn("linear_quadratic", schedulers)
        lora, opts = self.widget("character_lora")
        self.assertEqual((lora[0], opts["default"]), ("none", "none"))
        methods, opts = self.widget("upscale_method")
        self.assertEqual((list(methods), opts["default"]), (["latent_bicubic", "pixel_lanczos"], "pixel_lanczos"))

    def test_every_widget_has_a_tooltip_and_the_uncalibrated_ones_say_so(self):
        for name in REQUIRED + OPTIONAL:
            self.assertTrue(self.widget(name)[1].get("tooltip", "").strip(), name)
        for name in ("denoise", "face_sigma"):
            self.assertIn("calibrated", self.widget(name)[1]["tooltip"], name)
        self.assertIn("2 seeds and one prompt", self.widget("face_sigma")[1]["tooltip"])
        self.assertNotIn("first guess", self.widget("face_sigma")[1]["tooltip"])
        self.assertIn("not measured yet on the pixel_lanczos path", " ".join(self.widget("identity_boost")[1]["tooltip"].split()))

    def test_the_default_upscale_is_the_pixel_one_and_the_tooltip_says_why(self):
        text = " ".join(self.widget("upscale_method")[1]["tooltip"].split())
        self.assertIn("doubled image", text)
        self.assertIn("the stock recipe", text)
        import inspect
        self.assertEqual(inspect.signature(node.MossPhotonHiresFix.hires).parameters["upscale_method"].default, "pixel_lanczos")

    def test_the_denoise_tooltip_warns_that_it_is_not_the_fraction_of_noise(self):
        text = " ".join(self.widget("denoise")[1]["tooltip"].split())
        self.assertIn("not the fraction of noise", text)
        self.assertIn("linear_quadratic", text)
        self.assertIn("0.32", text)
        self.assertIn("0.86", text)

    def test_outputs_have_no_report(self):
        cls = node.MossPhotonHiresFix
        self.assertEqual(cls.RETURN_TYPES, ("LATENT", "STRING"))
        self.assertEqual(cls.RETURN_NAMES, ("latent", "summary"))
        self.assertEqual(len(cls.OUTPUT_TOOLTIPS), 2)
        self.assertNotIn("PHOTON_REPORT", cls.RETURN_TYPES)
        self.assertIn("not calibrated", cls.DESCRIPTION)

    def test_category_names_and_registration(self):
        cls = node.MossPhotonHiresFix
        self.assertEqual(cls.CATEGORY, consts.CAT_FINISHING)
        self.assertEqual(node.NODE_CLASS_MAPPINGS, {"MossPhotonHiresFix": cls})
        self.assertEqual(node.NODE_DISPLAY_NAME_MAPPINGS, {"MossPhotonHiresFix": "Moss Photon Hires Fix"})
        self.assertIn("MossPhotonHiresFix", consts.NODE_IDS_HELPERS)
        self.assertNotIn("MossPhotonHiresFix", consts.NODE_IDS)
        self.assertEqual(len(consts.NODE_IDS), 8)

    def test_the_probe_is_the_shared_dependency_probe(self):
        with mock.patch.object(deps, "probe", return_value="sentinel"):
            self.assertEqual(node._probe(), "sentinel")


class NoRaiseSiteOfItsOwnTest(unittest.TestCase):
    """The owner's rule: a node raises only for what is impossible, and every raise site is on the whitelist. This node and its
    pure module have no raise site at all: the wrong MODEL / latent, a LoRA that cannot be mapped and a non-finite output come
    out of helpers that are already listed (adapter.validate_*, lora.plan_patches, guard.check_finite)."""

    def test_no_raise_site_in_the_node_or_its_pure_module(self):
        found = guard.raise_sites(guard.package_sources())
        own = sorted(site for site in found if site[0] in ("nodes/hires.py", "finishing/hires.py"))
        self.assertEqual(own, [])

    def test_control_the_scan_sees_the_helpers_it_relies_on(self):
        found = guard.raise_sites(guard.package_sources())
        for site in (("core/adapter.py", "validate_model_facts", "unsupported_model"),
                     ("core/lora.py", "plan_patches", "incompatible_lora"),
                     ("core/guard.py", "check_finite", "non_finite_output"),
                     ("core/adapter.py", "validate_latent_facts", "invalid_latent")):
            self.assertIn(site, found)
            self.assertIn(site, guard.ALLOWED)

    def test_control_a_planted_raise_in_the_node_would_be_found(self):
        text = guard.package_sources()["nodes/hires.py"] + (
            "\n\ndef planted(node):\n    raise errors.OutputError(errors.make_issue('output_error', node, 'p', 'm', 'f'))\n")
        found = guard.raise_sites({"nodes/hires.py": text})
        self.assertIn(("nodes/hires.py", "planted", "output_error"), found)


if __name__ == "__main__":
    unittest.main()
