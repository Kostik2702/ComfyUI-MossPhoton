"""The full v1 registration list and the guarded routes call (design 1.0, 6.1, 7.1).

Wave 1 runs in parallel, so a module owned by a ticket that is not merged yet is reported as PENDING and the
"all 8 IDs" assertion is skipped with a message listing them; once every module exists the skip turns into a hard
assertion (the skipped count of the unit suite must drop). The two modules of MP-6 are always asserted.
"""
import importlib
import importlib.util
import os
import re
import subprocess
import sys
import unittest
from unittest.mock import patch

from moss_photon.core import consts

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

EXPECTED_MODULES = [
    "moss_photon.nodes.latent", "moss_photon.nodes.prompt", "moss_photon.nodes.lora",
    "moss_photon.nodes.schedule", "moss_photon.nodes.sampler",
    "moss_photon.nodes.diagnostics", "moss_photon.nodes.save",
]
MY_MODULES = {"moss_photon.nodes.latent": {"MossPhotonEmptyLatent"}, "moss_photon.nodes.schedule": {"MossPhotonSchedule"}}

# One interpreter per probe: the unit process must keep comfy / torch out of sys.modules.
PROBE = (
    "import sys, importlib.util, json\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "spec = importlib.util.spec_from_file_location('mp_pkg', sys.argv[1] + '/__init__.py', "
    "submodule_search_locations=[sys.argv[1]])\n"
    "pkg = importlib.util.module_from_spec(spec)\n"
    "sys.modules['mp_pkg'] = pkg\n"
    "spec.loader.exec_module(pkg)\n"
    "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'comfy', 'numpy', 'PIL', 'folder_paths'))\n"
    "print(json.dumps({'ids': sorted(pkg.NODE_CLASS_MAPPINGS), 'names': pkg.NODE_DISPLAY_NAME_MAPPINGS, "
    "'modules': pkg._NODE_MODULES, 'bad': bad[:5]}))\n"
)


def module_exists(name):
    return importlib.util.find_spec(name) is not None


def run_probe():
    proc = subprocess.run([sys.executable, "-I", "-c", PROBE, PKG_ROOT], capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr[-800:])
    import json
    return json.loads(proc.stdout.strip().splitlines()[-1])


class RegistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = run_probe()

    def test_module_list_is_the_full_v1_list(self):
        self.assertEqual(self.probe["modules"], EXPECTED_MODULES)
        with open(os.path.join(PKG_ROOT, "__init__.py"), "r", encoding="utf-8") as f:
            src = f.read()
        for name in EXPECTED_MODULES:
            self.assertEqual(len(re.findall(re.escape('"%s"' % name), src)), 1, name)

    def test_loading_the_package_imports_no_comfy_or_torch(self):
        self.assertEqual(self.probe["bad"], [])

    def test_my_two_nodes_are_registered_and_named(self):
        for module, ids in MY_MODULES.items():
            for node_id in ids:
                self.assertIn(node_id, self.probe["ids"])
                self.assertEqual(self.probe["names"][node_id], "Moss Photon " + re.sub(
                    r"(?<!^)(?=[A-Z])", " ", node_id[len("MossPhoton"):]))

    def test_registered_ids_are_a_subset_of_the_design_ids(self):
        self.assertTrue(set(self.probe["ids"]) <= set(consts.NODE_IDS), self.probe["ids"])

    def test_all_eight_ids_exist_once_every_ticket_is_merged(self):
        pending = [m for m in EXPECTED_MODULES if not module_exists(m)]
        if pending:
            self.skipTest("PENDING modules of other tickets: %s" % ", ".join(pending))
        self.assertEqual(sorted(self.probe["ids"]), sorted(consts.NODE_IDS))

    def test_modules_present_import_without_comfy_or_torch(self):
        present = [m for m in EXPECTED_MODULES if module_exists(m)]
        self.assertTrue(set(MY_MODULES) <= set(present))  # control: my modules are found by the same lookup
        code = ("import sys, importlib\nsys.path.insert(0, sys.argv[1])\n"
                "for name in sys.argv[2:]:\n    importlib.import_module(name)\n"
                "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'comfy', 'numpy', 'PIL', "
                "'folder_paths'))\nassert not bad, bad[:5]\nprint('OK')\n")
        proc = subprocess.run([sys.executable, "-I", "-c", code, PKG_ROOT] + present, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])

    def test_pack_version_equals_pyproject(self):
        with open(os.path.join(PKG_ROOT, "pyproject.toml"), "r", encoding="utf-8") as f:
            m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
        self.assertEqual(m.group(1), consts.PACK_VERSION)


class RoutesCallTest(unittest.TestCase):
    def load(self):
        spec = importlib.util.spec_from_file_location(
            "mp_pkg_routes_probe", os.path.join(PKG_ROOT, "__init__.py"), submodule_search_locations=[PKG_ROOT])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mp_pkg_routes_probe"] = mod
        self.addCleanup(sys.modules.pop, "mp_pkg_routes_probe", None)
        spec.loader.exec_module(mod)
        return mod

    def test_missing_routes_module_is_tolerated(self):
        # A None entry in sys.modules makes the import raise ModuleNotFoundError(name=<routes>) regardless of
        # whether routes.py exists on disk, so the missing-module branch runs in every environment.
        baseline = self.load()
        routes_name = "mp_pkg_routes_probe.moss_photon.core.routes"
        with patch.dict(sys.modules, {routes_name: None}):
            with self.assertLogs("MossPhoton", level="DEBUG") as logs:
                pkg = self.load()
        self.assertTrue(any("routes module not present yet" in line for line in logs.output), logs.output)
        self.assertTrue(pkg.NODE_CLASS_MAPPINGS)
        self.assertEqual(sorted(pkg.NODE_CLASS_MAPPINGS), sorted(baseline.NODE_CLASS_MAPPINGS))
        self.assertEqual(pkg.NODE_DISPLAY_NAME_MAPPINGS, baseline.NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(pkg.WEB_DIRECTORY, "./web")

    def test_a_broken_routes_module_is_logged_not_raised(self):
        pkg = self.load()
        import types
        fake = types.ModuleType("mp_pkg_routes_probe.moss_photon.core.routes")

        def boom():
            raise RuntimeError("no server")
        fake.register_routes = boom
        saved = sys.modules.get(fake.__name__)
        sys.modules[fake.__name__] = fake
        try:
            with self.assertLogs("MossPhoton", level="WARNING") as logs:
                self.assertFalse(pkg._register_routes("mp_pkg_routes_probe"))
        finally:
            if saved is None:
                sys.modules.pop(fake.__name__, None)
            else:
                sys.modules[fake.__name__] = saved
        self.assertIn("no server", logs.output[0])

    def test_a_working_routes_module_is_called_once_and_its_result_returned(self):
        pkg = self.load()
        import types
        calls = []
        fake = types.ModuleType("mp_pkg_routes_probe.moss_photon.core.routes")
        fake.register_routes = lambda: calls.append(1) or True
        saved = sys.modules.get(fake.__name__)
        sys.modules[fake.__name__] = fake
        try:
            self.assertTrue(pkg._register_routes("mp_pkg_routes_probe"))
        finally:
            if saved is None:
                sys.modules.pop(fake.__name__, None)
            else:
                sys.modules[fake.__name__] = saved
        self.assertEqual(calls, [1])

    def test_package_init_calls_register_routes_at_import(self):
        with open(os.path.join(PKG_ROOT, "__init__.py"), "r", encoding="utf-8") as f:
            src = f.read()
        self.assertRegex(src, r"(?m)^_register_routes\(\)$")
        self.assertIn("register_routes()", src)


if __name__ == "__main__":
    unittest.main()
