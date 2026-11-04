"""(e) A missing detector disables MossPhotonIdentityDetailer with a clear message and the pack still loads (MP-58).

Also the node's declared contract (inputs, defaults, outputs), the dependency probe, and the profile helpers that need
no ComfyUI. The run of the node on a real (tiny) MODEL is tests/integration/test_detailer_cpu.py."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from moss_photon.core import consts, errors
from moss_photon.finishing import deps
from moss_photon.nodes import detailer

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Loads the pack the way ComfyUI does, with the given modules made un-importable, and reports what happened.
PROBE = (
    "import sys, importlib.util, json\n"
    "for name in sys.argv[2].split(','):\n"
    "    sys.modules[name] = None\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "spec = importlib.util.spec_from_file_location('mp_pkg', sys.argv[1] + '/__init__.py', "
    "submodule_search_locations=[sys.argv[1]])\n"
    "pkg = importlib.util.module_from_spec(spec)\n"
    "sys.modules['mp_pkg'] = pkg\n"
    "spec.loader.exec_module(pkg)\n"
    "from moss_photon.finishing import deps\n"
    "from moss_photon.nodes import detailer\n"
    "av = deps.probe()\n"
    "out = {'ids': sorted(pkg.NODE_CLASS_MAPPINGS), 'detector_ok': av.detector_ok, 'detector_reason': av.detector_reason,\n"
    "       'arcface_ok': av.arcface_ok, 'heavy': sorted(m for m in sys.modules if sys.modules[m] is not None and "
    "m.split('.')[0] in ('torch', 'comfy', 'numpy', 'PIL', 'folder_paths', 'ultralytics', 'insightface'))}\n"
    "try:\n"
    "    detailer.MossPhotonIdentityDetailer().detail(None, None, None, None, {}, {}, 0, 110, 0.25)\n"
    "    out['error'] = None\n"
    "except Exception as exc:\n"
    "    out['error'] = [type(exc).__name__, str(exc), getattr(exc, 'code', None)]\n"
    "print(json.dumps(out))\n"
)


def run_probe(blocked):
    proc = subprocess.run([sys.executable, "-I", "-c", PROBE, PKG_ROOT, ",".join(blocked)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr[-1500:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


class PackLoadsWithoutTheDetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # folder_paths blocked as well: the embedded python can import ComfyUI's own module, which pulls comfy in
        cls.without = run_probe(["ultralytics", "insightface", "folder_paths"])

    def test_the_pack_loads_and_every_node_is_registered(self):
        expected = sorted(consts.NODE_IDS + consts.NODE_IDS_SECOND_QUEUE + consts.NODE_IDS_HELPERS)
        self.assertEqual(self.without["ids"], expected)
        self.assertIn("MossPhotonIdentityDetailer", self.without["ids"])

    def test_the_probe_reports_what_is_missing(self):
        self.assertFalse(self.without["detector_ok"])
        self.assertIn("ultralytics", self.without["detector_reason"])
        self.assertFalse(self.without["arcface_ok"])

    def test_the_node_is_disabled_with_a_clear_message_not_a_crash(self):
        name, message, code = self.without["error"]
        self.assertEqual((name, code), ("MissingGuardDependency", "missing_guard_dependency"))
        self.assertIn("MossPhotonIdentityDetailer is disabled", message)
        self.assertIn("ultralytics", message)
        self.assertIn("pip install ultralytics", message)
        self.assertIn("face_yolov8m.pt", message)
        self.assertLessEqual(len(message), errors.MAX_MESSAGE_CHARS)

    def test_loading_imports_no_heavy_module(self):
        self.assertEqual(self.without["heavy"], [])

    def test_the_other_eight_nodes_are_unaffected(self):
        for node_id in consts.NODE_IDS:
            self.assertIn(node_id, self.without["ids"])

    def test_the_same_pack_without_blocking_anything_loads_the_same_nodes(self):
        with_everything = run_probe(["folder_paths"])
        self.assertEqual(with_everything["ids"], self.without["ids"])

    def test_a_node_module_that_fails_to_import_is_skipped_by_the_loader(self):
        """The generic safety net behind it all: __init__ isolates every node module."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mp_pkg_loader_probe", os.path.join(PKG_ROOT, "__init__.py"), submodule_search_locations=[PKG_ROOT])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mp_pkg_loader_probe"] = mod
        self.addCleanup(sys.modules.pop, "mp_pkg_loader_probe", None)
        spec.loader.exec_module(mod)
        classes, _ = mod._load_node_modules(["moss_photon.nodes.detailer", "moss_photon.nodes.not_there"],
                                            package="mp_pkg_loader_probe")
        self.assertIn("MossPhotonIdentityDetailer", classes)


class NodeRefusesWithoutADetectorTest(unittest.TestCase):
    def call(self, availability):
        with mock.patch.object(detailer, "_probe", lambda: availability):
            return detailer.MossPhotonIdentityDetailer().detail(None, None, None, None, {}, {}, 0, 110, 0.25)

    def test_a_missing_package(self):
        av = deps.Availability("python package ultralytics is not installed", None, None, "/x")
        with self.assertRaises(errors.MissingGuardDependency) as ctx:
            self.call(av)
        self.assertEqual(ctx.exception.code, "missing_guard_dependency")
        self.assertIn("is disabled: python package ultralytics is not installed", str(ctx.exception))

    def test_a_missing_model_file(self):
        av = deps.Availability("face model face_yolov8m.pt not found in models/ultralytics/bbox", None, None, "/x")
        with self.assertRaises(errors.MissingGuardDependency) as ctx:
            self.call(av)
        self.assertIn("not found in models/ultralytics/bbox", str(ctx.exception))
        self.assertIn("huggingface.co/Bingsu/adetailer", str(ctx.exception))

    def test_the_check_is_the_first_thing_even_the_report_is_not_looked_at(self):
        av = deps.Availability("python package ultralytics is not installed", None, None, "/x")
        with mock.patch.object(detailer, "_require_valid_report", side_effect=AssertionError("looked at the report")):
            with self.assertRaises(errors.MissingGuardDependency):
                self.call(av)


class ProbeTest(unittest.TestCase):
    def tree(self, files=(), dirs=()):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for f in files:
            path = os.path.join(tmp.name, *f.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").close()
        for d in dirs:
            os.makedirs(os.path.join(tmp.name, *d.split("/")), exist_ok=True)
        return tmp.name

    @staticmethod
    def installed(*names):
        return lambda name: object() if name in names else None

    def test_everything_present(self):
        models = self.tree(files=["ultralytics/bbox/face_yolov8m.pt"], dirs=["insightface/models/buffalo_l"])
        av = deps.probe(models, self.installed("ultralytics", "insightface", "onnxruntime"), [])
        self.assertTrue(av.detector_ok and av.arcface_ok)
        self.assertEqual(av.detector_path, os.path.join(models, "ultralytics", "bbox", "face_yolov8m.pt"))
        self.assertEqual(av.arcface_root, os.path.join(models, "insightface"))
        self.assertIn("ready", av.describe())

    def test_package_missing(self):
        models = self.tree(files=["ultralytics/bbox/face_yolov8m.pt"], dirs=["insightface/models/buffalo_l"])
        av = deps.probe(models, self.installed("insightface", "onnxruntime"), [])
        self.assertFalse(av.detector_ok)
        self.assertIn("ultralytics", av.detector_reason)
        self.assertTrue(av.arcface_ok)
        self.assertIn("MISSING", av.describe())

    def test_model_file_missing(self):
        models = self.tree(dirs=["ultralytics/bbox"])
        av = deps.probe(models, self.installed("ultralytics"), [])
        self.assertIn("face_yolov8m.pt", av.detector_reason)

    def test_the_impact_subpack_folder_and_the_plain_ultralytics_folder_are_found(self):
        models = self.tree(files=["ultralytics/face_yolov8m.pt"])
        self.assertTrue(deps.probe(models, self.installed("ultralytics"), []).detector_ok)
        elsewhere = self.tree(files=["somewhere/face_yolov8m.pt"])
        registered = [os.path.join(elsewhere, "somewhere")]
        self.assertTrue(deps.probe(self.tree(), self.installed("ultralytics"), registered).detector_ok)

    def test_arcface_needs_the_package_onnxruntime_and_the_pack(self):
        models = self.tree(files=["ultralytics/bbox/face_yolov8m.pt"], dirs=["insightface/models/buffalo_l"])
        self.assertIn("onnxruntime", deps.probe(models, self.installed("ultralytics", "insightface"), []).arcface_reason)
        self.assertIn("insightface", deps.probe(models, self.installed("ultralytics"), []).arcface_reason)
        bare = self.tree(files=["ultralytics/bbox/face_yolov8m.pt"])
        reason = deps.probe(bare, self.installed("ultralytics", "insightface", "onnxruntime"), []).arcface_reason
        self.assertIn("buffalo_l", reason)

    def test_unknown_models_folder_is_a_reason_not_a_crash(self):
        with mock.patch.object(deps, "_models_dir", lambda: None):
            av = deps.probe(None, self.installed("ultralytics", "insightface", "onnxruntime"), [])
        self.assertFalse(av.detector_ok)
        self.assertFalse(av.arcface_ok)
        self.assertIn("models folder", av.detector_reason)


class NodeContractTest(unittest.TestCase):
    def setUp(self):
        self.spec = detailer.MossPhotonIdentityDetailer.INPUT_TYPES()

    def test_registration(self):
        self.assertEqual(detailer.NODE_CLASS_MAPPINGS, {"MossPhotonIdentityDetailer": detailer.MossPhotonIdentityDetailer})
        self.assertIn("MossPhotonIdentityDetailer", consts.NODE_IDS_SECOND_QUEUE)
        self.assertNotIn("MossPhotonIdentityDetailer", consts.NODE_IDS)
        self.assertEqual(detailer.NODE_DISPLAY_NAME_MAPPINGS["MossPhotonIdentityDetailer"], "Moss Photon Identity Detailer")
        self.assertEqual(detailer.MossPhotonIdentityDetailer.CATEGORY, consts.CAT_FINISHING)

    def test_outputs(self):
        cls = detailer.MossPhotonIdentityDetailer
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE", "PHOTON_REPORT", "STRING"))
        self.assertEqual(len(cls.RETURN_NAMES), 3)
        self.assertEqual(len(cls.OUTPUT_TOOLTIPS), 3)
        self.assertTrue(hasattr(cls, cls.FUNCTION))

    def test_inputs_and_the_decided_defaults(self):
        req, opt = self.spec["required"], self.spec["optional"]
        self.assertEqual({k: v[0] for k, v in req.items() if k not in ("seed", "size_gate_px", "denoise")},
                         {"image": "IMAGE", "model": "MODEL", "clip": "CLIP", "vae": "VAE", "report": "PHOTON_REPORT",
                          "identity_profile": "PHOTON_IDENTITY_PROFILE"})
        self.assertEqual(req["size_gate_px"][0], "INT")
        self.assertEqual(req["size_gate_px"][1]["default"], 110)    # the MP-56 decision
        self.assertEqual(req["denoise"][1]["default"], 0.25)        # the MP-56 decision
        self.assertEqual(opt["consent_reference_photos"][1]["default"], False)
        self.assertEqual(opt["max_faces"][1]["default"], 8)
        self.assertEqual(opt["accept_without_references"], ("BOOLEAN", opt["accept_without_references"][1]))
        self.assertFalse(opt["accept_without_references"][1]["default"])   # N04: pass-through is the default
        for name in ("reference_images", "other_reference_images_1", "other_reference_images_2"):
            self.assertEqual(opt[name][0], "IMAGE")

    def test_every_input_has_a_tooltip(self):
        for section in self.spec.values():
            for name, (kind, cfg) in section.items():
                self.assertTrue(cfg.get("tooltip"), name)

    def test_the_widget_defaults_equal_the_settings_defaults(self):
        from moss_photon.finishing import face_detailer as fd
        d = fd.Settings()
        self.assertEqual(self.spec["required"]["size_gate_px"][1]["default"], d.size_gate_px)
        self.assertEqual(self.spec["required"]["denoise"][1]["default"], d.denoise)
        self.assertEqual(self.spec["optional"]["max_faces"][1]["default"], d.max_faces)
        self.assertEqual(self.spec["optional"]["accept_without_references"][1]["default"], d.accept_without_references)
        self.assertEqual(self.spec["optional"]["character_match_min"][1]["default"], d.match_min)

    def test_the_mp56_recipe_is_the_default_settings(self):
        from moss_photon.finishing import face_detailer as fd
        d = fd.Settings()
        self.assertEqual((d.crop_factor, d.work_px, d.steps, d.cfg, d.sampler, d.scheduler, d.denoise, d.feather),
                         (1.8, 1024, 8, 1.0, "euler", "simple", 0.25, 0.08))
        self.assertTrue(d.color_match)


class ProfileHelpersTest(unittest.TestCase):
    def test_trigger_comes_from_the_character_adapter_only(self):
        profile = {"stack": [{"role": "distill", "trigger_hints": ["wrong"]},
                             {"role": "character", "trigger_hints": ["", " MyTrigger ", "second"]}]}
        self.assertEqual(detailer.trigger_of(profile), "MyTrigger")

    def test_no_trigger(self):
        self.assertIsNone(detailer.trigger_of({"stack": []}))
        self.assertIsNone(detailer.trigger_of({"stack": [{"role": "character", "trigger_hints": []}]}))
        self.assertIsNone(detailer.trigger_of({}))

    def test_the_prompt_is_the_face_and_the_trigger_only(self):
        self.assertEqual(detailer.PROMPT_WITH_TRIGGER % "MyTrigger", "close-up photo of MyTrigger's face, natural skin texture")
        self.assertNotIn("%s", detailer.PROMPT_NO_TRIGGER)

    def test_a_profile_without_a_distill_adapter_is_refused(self):
        with self.assertRaises(errors.UnsupportedFeature) as ctx:
            detailer._require_turbo_profile({"stack": [{"role": "character"}]})
        self.assertIn("distill", str(ctx.exception))
        detailer._require_turbo_profile({"stack": [{"role": "distill"}, {"role": "character"}]})  # no raise


if __name__ == "__main__":
    unittest.main()
