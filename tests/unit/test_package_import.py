import importlib.util
import os
import sys
import unittest

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NAME = "mossphoton_pkg_under_test"


def _load_package():
    spec = importlib.util.spec_from_file_location(
        _NAME, os.path.join(PKG_ROOT, "__init__.py"),
        submodule_search_locations=[PKG_ROOT])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


class PackageImportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pkg = _load_package()

    def test_no_comfy_import(self):
        self.assertNotIn("comfy", sys.modules)

    def test_mappings_are_dicts(self):
        self.assertIsInstance(self.pkg.NODE_CLASS_MAPPINGS, dict)
        self.assertIsInstance(self.pkg.NODE_DISPLAY_NAME_MAPPINGS, dict)

    def test_web_directory(self):
        self.assertEqual(self.pkg.WEB_DIRECTORY, "./web")

    def test_failing_module_is_skipped(self):
        with self.assertLogs("MossPhoton", level="WARNING") as cm:
            classes, names = self.pkg._load_node_modules(
                ["moss_photon.nodes.does_not_exist"], package=_NAME)
        self.assertEqual(classes, {})
        self.assertEqual(names, {})
        self.assertIn("[MossPhoton] module moss_photon.nodes.does_not_exist disabled:", cm.output[0])

    def test_good_module_still_loads_next_to_failing_one(self):
        classes, _ = self.pkg._load_node_modules(
            ["moss_photon.nodes.does_not_exist", "moss_photon.core"], package=_NAME)
        self.assertEqual(classes, {})


if __name__ == "__main__":
    unittest.main()
