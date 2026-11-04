"""MP-9 consumes PHOTON_CONSTRAINTS and PHOTON_IDENTITY_PROFILE as dicts only (M4, LD5): by AST, none of its files
imports ``core.constraints`` or ``core.lora`` -- in any spelling, at any depth, also inside a function."""
import ast
import os
import unittest

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = os.path.join(PKG, "moss_photon")
MINE = ("core/solver.py", "core/guidance.py", "core/phase.py", "core/guard.py", "core/routes.py", "core/lora_plan.py")
# MP-71 amendment: nodes/sampler.py injects the darkroom boost loader (``lora.boost_patches``) into the solver request, the way
# the deleted nodes/darkroom.py did; it may import ``core.lora`` for that one callable and never ``constraints``. The solver
# (above) still imports neither, and ``test_the_sampler_node_uses_lora_for_the_loaders_only`` pins the uses. MP-84 adds the
# second callable: ``lora.plan_patches``, the loader of the LoRA Phase plan a MODEL carries.
NODE_FILE = "nodes/sampler.py"
FORBIDDEN = ("constraints", "lora")
ALLOWED_LORA_ATTRS = ("boost_patches", "plan_patches")


def forbidden_imports(source, forbidden=FORBIDDEN):
    """Lines of ``source`` that import a forbidden module (relative or absolute spelling)."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".")
            if node.level and parts[-1] in forbidden or any(p in forbidden for p in parts):
                hits.append(node.lineno)
            elif any(alias.name in forbidden for alias in node.names):
                hits.append(node.lineno)
        elif isinstance(node, ast.Import):
            if any(p in forbidden for alias in node.names for p in alias.name.split(".")):
                hits.append(node.lineno)
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "import_module":
            hits.append(node.lineno)
    return hits


def lora_pin_violations(source):
    """``[(line, why)]`` where ``source`` (nodes/sampler.py) touches ``core.lora`` in any way but the one allowed: the
    top-level ``from ..core import lora`` (no alias) and the attributes ``lora.boost_patches`` (MP-71 amendment; MP-75 N1) and
    ``lora.plan_patches`` (MP-84)."""
    tree = ast.parse(source)
    top = {id(n) for n in tree.body}
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            parts = (n.module or "").split(".")
            names = [a for a in n.names if a.name == "lora"]
            if "lora" in parts:
                hits.append((n.lineno, "import from the lora module"))
            elif names:
                if not (n.level == 2 and n.module == "core" and id(n) in top and all(a.asname is None for a in names)):
                    hits.append((n.lineno, "lora imported another way than a top-level 'from ..core import lora'"))
        elif isinstance(n, ast.Import):
            if any("lora" in a.name.split(".") for a in n.names):
                hits.append((n.lineno, "absolute import of lora"))
        elif isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "import_module":
            hits.append((n.lineno, "import_module"))
    bases = {id(n.value): n for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id == "lora":
            use = bases.get(id(n))
            if use is None:
                hits.append((n.lineno, "lora used as a value"))
            elif use.attr not in ALLOWED_LORA_ATTRS:
                hits.append((n.lineno, "lora.%s" % use.attr))
    return hits


class ForbiddenImportsTest(unittest.TestCase):
    def test_my_files_are_clean(self):
        for rel in MINE:
            path = os.path.join(ROOT, *rel.split("/"))
            self.assertTrue(os.path.isfile(path), "control: %s must exist" % rel)
            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(forbidden_imports(f.read()), [], rel)

    def test_the_sampler_node_never_imports_constraints(self):
        with open(os.path.join(ROOT, *NODE_FILE.split("/")), "r", encoding="utf-8") as f:
            self.assertEqual(forbidden_imports(f.read(), ("constraints",)), [])

    def test_the_sampler_node_uses_lora_for_the_loaders_only(self):
        with open(os.path.join(ROOT, *NODE_FILE.split("/")), "r", encoding="utf-8") as f:
            src = f.read()
        self.assertEqual(lora_pin_violations(src), [])
        for name in ALLOWED_LORA_ATTRS:
            self.assertIn("lora.%s" % name, src, "control: the use exists")

    def test_the_pin_goes_red_on_every_other_way_in(self):
        base = "from ..core import consts, lora\n"
        for name, src in (
                ("another name of the module", base + "from ..core.lora import apply_lora\n"),
                ("the module under an alias", base + "from ..core import lora as lora2\n"),
                ("another attribute", base + "x = lora.apply_lora\n"),
                ("the module as a value", base + "y = lora\n"),
                ("a nested import", "def f():\n    from ..core import lora\n    return lora.boost_patches\n"),
                ("another relative form", "from .. import lora\n"),
                ("an absolute import", base + "import moss_photon.core.lora\n")):
            with self.subTest(name):
                self.assertTrue(lora_pin_violations(src), name)
        self.assertEqual(lora_pin_violations(base + "z = lora.boost_patches\nw = lora.plan_patches\n"), [])

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
