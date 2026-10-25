"""Import graph guard (design 7.3 and rules R1/R2/R6), by AST: nothing is imported or executed.

Checked over every .py file under ``moss_photon/``:

* ``errors``, ``canon`` and ``consts`` are leaves: stdlib only, wherever the import sits (also inside a function),
  and no import of any other module of the package.
* Edges between package modules follow the allowed table of 7.3 (a new edge needs a design amendment). Notably
  ``solver`` never imports ``constraints``, ``lora``, ``saveio`` or ``imagestats`` (M4). ``nodes/*`` may import
  ``core/*`` and never each other.
* R1: no module-level import of comfy*, torch, numpy, PIL, folder_paths, nodes, execution, server in ``core/`` or
  ``nodes/`` (function-level imports are the lazy form and are allowed). R6: ``jsonschema`` is tests only.
* R2: inside the package only relative imports. No import cycles.

The checker itself is proven in this file by planting each violation in a scratch tree.
"""
import ast
import os
import sys
import tempfile
import unittest

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKAGE_DIR = os.path.join(PKG_ROOT, "moss_photon")
PACKAGE_NAME = "moss_photon"

LEAVES = ("errors", "canon", "consts")

# Design 7.3, allowed edges between core modules (left imports right).
CORE_ALLOWED = {
    "errors": (), "canon": (), "consts": (),
    "fingerprint": ("errors",),
    "sigma": ("errors", "canon"),
    "rng": ("errors",),
    "report": ("canon", "consts", "errors"),
    "adapter": ("errors", "canon", "fingerprint", "consts"),
    "schema_lite": (),
    "presets": ("canon", "errors", "schema_lite", "consts", "sigma"),
    "constraints": ("errors", "canon", "consts", "fingerprint"),
    "lora": ("errors", "canon", "consts", "fingerprint"),
    "guidance": ("errors",),
    "phase": ("errors",),
    "guard": ("errors",),
    "routes": ("presets", "errors"),
    "solver": ("errors", "canon", "consts", "fingerprint", "adapter", "sigma", "rng", "report", "presets",
               "guidance", "phase", "guard"),
    "saveio": ("errors", "canon", "consts"),
    "imagestats": ("errors",),
}
SOLVER_NEVER = ("constraints", "lora", "saveio", "imagestats")

# R1: imported lazily (inside functions) only.
LAZY_ONLY_ROOTS = ("torch", "numpy", "PIL", "folder_paths", "nodes", "execution", "server")
# R6
TESTS_ONLY_ROOTS = ("jsonschema",)


def _is_comfy_root(root):
    return root == "comfy" or root.startswith("comfy_")


class _Imports(ast.NodeVisitor):
    def __init__(self):
        self.items = []  # (kind, module, level, names, lazy, lineno)
        self._depth = 0

    def _scope(self, node):
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _scope

    def visit_Import(self, node):
        for alias in node.names:
            self.items.append(("import", alias.name, 0, [], self._depth > 0, node.lineno))

    def visit_ImportFrom(self, node):
        self.items.append(("from", node.module or "", node.level, [a.name for a in node.names],
                           self._depth > 0, node.lineno))


def _module_parts(rel):
    """``core/errors.py`` -> (("core", "errors"), False); ``core/__init__.py`` -> (("core",), True)."""
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        return tuple(parts[:-1]), True
    return tuple(parts), False


def _name(mod):
    return ".".join(mod) if mod else "(root)"


def collect_files(package_dir):
    if not os.path.isdir(package_dir):
        raise FileNotFoundError("import graph root does not exist: %s" % package_dir)
    files = {}
    for dirpath, dirnames, filenames in os.walk(package_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fname in filenames:
            if fname.endswith(".py"):
                path = os.path.join(dirpath, fname)
                rel = os.path.relpath(path, package_dir).replace(os.sep, "/")
                with open(path, "r", encoding="utf-8") as f:
                    files[rel] = f.read()
    return files


def check_tree(package_dir):
    """Return ``(module tuples scanned, violations)``; a violation is a readable string."""
    return check_sources(collect_files(package_dir))


def check_sources(files):
    """``files``: relative path (``core/errors.py``) -> source text."""
    violations = []
    mods = {rel: _module_parts(rel) for rel in sorted(files)}
    known = {m for m, _ in mods.values()}
    edges = {}

    for rel in sorted(files):
        src, is_init = mods[rel]
        try:
            tree = ast.parse(files[rel], filename=rel)
        except SyntaxError as exc:
            violations.append("%s: does not parse (%s)" % (rel, exc.msg))
            continue
        coll = _Imports()
        coll.visit(tree)
        in_core_or_nodes = bool(src) and src[0] in ("core", "nodes")
        is_leaf = len(src) == 2 and src[0] == "core" and src[1] in LEAVES

        for kind, module, level, names, lazy, lineno in coll.items:
            where = "%s:%d" % (rel, lineno)
            if level == 0:
                root = module.split(".")[0]
                if root == PACKAGE_NAME:
                    violations.append("%s: absolute import of the package (%s); use relative imports (R2)"
                                      % (where, module))
                elif root in TESTS_ONLY_ROOTS:
                    violations.append("%s: %s is for tests only (R6)" % (where, module))
                elif is_leaf:
                    if root not in sys.stdlib_module_names:
                        violations.append("%s: leaf module %s may import the stdlib only, found %s"
                                          % (where, _name(src), module))
                elif in_core_or_nodes and not lazy and (_is_comfy_root(root) or root in LAZY_ONLY_ROOTS):
                    violations.append("%s: module-level import of %s; import it inside the function (R1)"
                                      % (where, module))
                continue

            # relative import: resolve against the package tree
            pkg = list(src) if is_init else list(src[:-1])
            up = level - 1
            if up > len(pkg):
                violations.append("%s: relative import leaves the package" % where)
                continue
            target = tuple(pkg[:len(pkg) - up] + (module.split(".") if module else []))
            if module and target in known:
                dsts = [target]  # "from .mod import name": an edge to the module itself
            elif target in known:
                # "from . import a, b": each name is a submodule if one exists, else an attribute of the package
                dsts = [target + (n,) if target + (n,) in known else target for n in names]
            else:
                violations.append("%s: cannot resolve relative import of %s" % (where, "." * level + module))
                continue
            for dst in dsts:
                edges.setdefault(src, set()).add(dst)
                _check_edge(src, dst, is_leaf, where, violations)

    _check_cycles(edges, violations)
    return sorted({m for m, _ in mods.values()}), violations


def _check_edge(src, dst, is_leaf, where, violations):
    if is_leaf:
        violations.append("%s: leaf module %s imports %s" % (where, _name(src), _name(dst)))
    elif len(src) == 2 and src[0] == "core":
        allowed = CORE_ALLOWED.get(src[1])
        if allowed is None:
            violations.append("%s: unknown core module %s: a new module needs a design amendment (7.3)"
                              % (where, _name(src)))
        elif not (len(dst) == 2 and dst[0] == "core" and dst[1] in allowed):
            extra = ""
            if src[1] == "solver" and len(dst) == 2 and dst[1] in SOLVER_NEVER:
                extra = " (M4: solver never imports %s)" % dst[1]
            violations.append("%s: edge %s -> %s is not allowed by design 7.3%s"
                              % (where, _name(src), _name(dst), extra))
    elif src and src[0] == "core":
        violations.append("%s: %s is a package __init__ or a nested module; it must not import (7.3)"
                          % (where, _name(src)))
    elif src and src[0] == "nodes":
        if not (dst and dst[0] == "core"):
            violations.append("%s: node module %s may import core/* only, found %s"
                              % (where, _name(src), _name(dst)))
    # other packages (calibration, finishing, root): only cycles and the import rules above apply


def _check_cycles(edges, violations):
    state = {}

    def visit(node, stack):
        state[node] = 1
        for nxt in sorted(edges.get(node, ())):
            if state.get(nxt) == 1:
                cycle = stack[stack.index(nxt):] + [nxt] if nxt in stack else [node, nxt]
                violations.append("import cycle: " + " -> ".join(_name(c) for c in cycle))
            elif nxt not in state:
                visit(nxt, stack + [nxt])
        state[node] = 2

    for node in sorted(edges):
        if node not in state:
            visit(node, [node])


class RealTreeTest(unittest.TestCase):
    def test_package_import_graph_is_clean(self):
        modules, violations = check_tree(PACKAGE_DIR)
        self.assertEqual(violations, [])
        # control specimen: the scan really saw the Wave 0 leaves
        for leaf in LEAVES:
            self.assertIn(("core", leaf), modules)

    def test_allowed_table_keeps_leaves_edge_free(self):
        for leaf in LEAVES:
            self.assertEqual(CORE_ALLOWED[leaf], ())

    def test_missing_package_dir_fails(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                check_tree(os.path.join(root, "nope"))


class PlantedViolationTest(unittest.TestCase):
    """Every rule goes red on a planted violation and stays green on the legal form."""

    BASE = {
        "__init__.py": "",
        "core/__init__.py": "",
        "core/errors.py": "import dataclasses\n",
        "core/canon.py": "import json\n",
        "core/consts.py": "X = 1\n",
        "nodes/__init__.py": "",
    }

    def run_with(self, extra):
        files = dict(self.BASE)
        files.update(extra)
        return check_sources(files)[1]

    def assertClean(self, extra=None):
        self.assertEqual(self.run_with(extra or {}), [])

    def assertRed(self, needle, extra):
        v = self.run_with(extra)
        self.assertTrue(any(needle in line for line in v), "expected %r in %r" % (needle, v))

    def test_baseline_is_clean(self):
        self.assertClean()

    def test_leaf_with_forbidden_import_goes_red(self):
        self.assertRed("canon", {"core/canon.py": "import json\nimport torch\n"})
        self.assertRed("errors", {"core/errors.py": "from comfy import model_base\n"})

    def test_leaf_forbids_lazy_import_too(self):
        self.assertRed("canon", {"core/canon.py": "def f():\n    import torch\n"})
        self.assertRed("consts", {"core/consts.py": "def f():\n    import numpy\n"})

    def test_leaf_forbids_package_edges(self):
        self.assertRed("leaf module core.errors imports core.canon",
                       {"core/errors.py": "from .canon import sha256_hex\n"})
        self.assertRed("leaf module core.canon imports core.errors", {"core/canon.py": "from . import errors\n"})

    def test_leaf_allows_stdlib(self):
        self.assertClean({"core/canon.py": "import hashlib, json\nfrom typing import NamedTuple\nimport os.path\n"})

    def test_module_level_heavy_import_goes_red_but_lazy_is_fine(self):
        for src in ("import torch\n", "import numpy as np\n", "from PIL import Image\n", "import folder_paths\n",
                    "import comfy.model_base\n", "from comfy_extras import nodes_slg\n",
                    "try:\n    import torch\nexcept ImportError:\n    torch = None\n",
                    "class A:\n    import torch\n"):
            with self.subTest(src=src):
                self.assertRed("R1", {"core/sigma.py": src})
        self.assertClean({"core/sigma.py": "def f():\n    import torch\n    import comfy.samplers\n    return torch\n"})
        self.assertClean({"core/sigma.py": "f = lambda: 1\n"})

    def test_node_module_level_comfy_goes_red(self):
        for src in ("from comfy import samplers\n", "import execution\n", "import server\n"):
            with self.subTest(src=src):
                self.assertRed("R1", {"nodes/schedule.py": src})
        self.assertClean({"nodes/schedule.py": "def run():\n    from comfy import samplers\n"})

    def test_jsonschema_is_tests_only_even_lazily(self):
        self.assertRed("R6", {"core/sigma.py": "def f():\n    import jsonschema\n"})

    def test_absolute_package_import_goes_red(self):
        self.assertRed("R2", {"core/sigma.py": "from moss_photon.core import errors\n"})
        self.assertRed("R2", {"nodes/schedule.py": "import moss_photon.core.sigma\n"})

    def test_allowed_edges_are_clean(self):
        self.assertClean({
            "core/sigma.py": "from .errors import Issue\nfrom .canon import short\n",
            "core/rng.py": "from . import errors\n",
        })
        self.assertClean({
            "core/sigma.py": "",
            "nodes/schedule.py": "from ..core import sigma\nfrom ..core.errors import Issue\n",
        })

    def test_disallowed_core_edge_goes_red(self):
        self.assertRed("edge core.sigma -> core.adapter",
                       {"core/sigma.py": "from .adapter import x\n", "core/adapter.py": ""})
        self.assertRed("edge core.sigma -> core.consts", {"core/sigma.py": "from .consts import X\n"})

    def test_solver_never_imports_constraints_lora_saveio_imagestats(self):
        for mod in SOLVER_NEVER:
            with self.subTest(mod=mod):
                self.assertRed("M4", {"core/solver.py": "from .%s import x\n" % mod, "core/%s.py" % mod: ""})

    def test_unknown_core_module_goes_red(self):
        self.assertRed("unknown core module core.helpers", {"core/helpers.py": "from .errors import Issue\n"})

    def test_core_init_must_not_import(self):
        self.assertRed("must not import", {"core/__init__.py": "from . import errors\n"})

    def test_nodes_never_import_each_other(self):
        self.assertRed("core/* only", {"nodes/a.py": "from . import b\n", "nodes/b.py": ""})
        self.assertRed("core/* only", {"nodes/a.py": "from .b import X\n", "nodes/b.py": ""})

    def test_nodes_may_not_import_other_packages(self):
        self.assertRed("core/* only", {"nodes/a.py": "from ..calibration import m\n",
                                       "calibration/__init__.py": "", "calibration/m.py": ""})

    def test_cycle_outside_the_table_goes_red(self):
        v = self.run_with({"calibration/__init__.py": "", "calibration/a.py": "from . import b\n",
                           "calibration/b.py": "from . import a\n"})
        self.assertTrue(any(line.startswith("import cycle:") for line in v), v)

    def test_unresolvable_and_escaping_relative_imports_go_red(self):
        self.assertRed("cannot resolve", {"core/sigma.py": "from .nope import x\n"})
        self.assertRed("leaves the package", {"core/sigma.py": "from .... import x\n"})

    def test_syntax_error_is_reported_not_swallowed(self):
        self.assertRed("does not parse", {"core/sigma.py": "def (:\n"})


if __name__ == "__main__":
    unittest.main()
