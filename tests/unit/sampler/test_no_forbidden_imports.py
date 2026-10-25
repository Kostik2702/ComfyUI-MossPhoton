"""MP-9 consumes PHOTON_CONSTRAINTS and PHOTON_IDENTITY_PROFILE as dicts only (M4, LD5): by AST, none of its files
imports ``core.constraints`` or ``core.lora`` -- in any spelling, at any depth, also inside a function."""
import ast
import os
import unittest

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = os.path.join(PKG, "moss_photon")
MINE = ("core/solver.py", "core/guidance.py", "core/phase.py", "core/guard.py", "core/routes.py", "nodes/sampler.py")
FORBIDDEN = ("constraints", "lora")


def forbidden_imports(source):
    """Lines of ``source`` that import a forbidden module (relative or absolute spelling)."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".")
            if node.level and parts[-1] in FORBIDDEN or any(p in FORBIDDEN for p in parts):
                hits.append(node.lineno)
            elif any(alias.name in FORBIDDEN for alias in node.names):
                hits.append(node.lineno)
        elif isinstance(node, ast.Import):
            if any(p in FORBIDDEN for alias in node.names for p in alias.name.split(".")):
                hits.append(node.lineno)
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "import_module":
            hits.append(node.lineno)
    return hits


class ForbiddenImportsTest(unittest.TestCase):
    def test_my_files_are_clean(self):
        for rel in MINE:
            path = os.path.join(ROOT, *rel.split("/"))
            self.assertTrue(os.path.isfile(path), "control: %s must exist" % rel)
            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(forbidden_imports(f.read()), [], rel)

    def test_the_scanner_goes_red_on_every_spelling(self):
        for src in ("from . import constraints\n", "from .constraints import compile_prompt\n",
                    "from ..core import lora\n", "from ..core.lora import apply_lora\n",
                    "import moss_photon.core.lora\n", "def f():\n    from . import lora\n",
                    "import importlib\nimportlib.import_module('x')\n"):
            with self.subTest(src=src):
                self.assertTrue(forbidden_imports(src), src)

    def test_the_scanner_is_green_on_legal_imports(self):
        for src in ("from . import adapter, errors\n", "from ..core import presets\nimport math\n",
                    "from ..core import solver as core_solver\n"):
            with self.subTest(src=src):
                self.assertEqual(forbidden_imports(src), [])


if __name__ == "__main__":
    unittest.main()
