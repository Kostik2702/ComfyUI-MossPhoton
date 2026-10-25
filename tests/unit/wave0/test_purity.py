"""Wave 0 leaves import in a fresh interpreter without pulling comfy, torch or numpy (design R1, 7.3).

A subprocess is used because the unit process may legitimately hold other modules; the AST guard in
tests/unit/test_import_graph.py covers the source, this covers the actual import.
"""
import os
import subprocess
import sys
import unittest

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CODE = (
    "import sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from moss_photon.core import errors, canon, consts\n"
    "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('comfy', 'comfy_extras', 'torch', 'numpy', 'PIL'))\n"
    "assert not bad, 'heavy modules imported, first: %s of %d' % (bad[:5], len(bad))\n"
    "assert errors.REASONS and canon.canonical_json({}) == '{}' and consts.PACK_NAME\n"
    "print('PURE')\n"
)


class LeafPurityTest(unittest.TestCase):
    def test_leaves_import_without_comfy_or_torch(self):
        proc = subprocess.run([sys.executable, "-I", "-c", CODE, PKG_ROOT], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-600:])
        self.assertEqual(proc.stdout.strip(), "PURE")

    def test_control_the_probe_detects_a_heavy_import(self):
        """Same probe, with a stdlib stand-in listed as heavy, must fail (the probe can fail)."""
        code = CODE.replace("('comfy', 'comfy_extras', 'torch', 'numpy', 'PIL')", "('json',)")
        proc = subprocess.run([sys.executable, "-I", "-c", code, PKG_ROOT], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
