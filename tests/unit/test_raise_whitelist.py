"""MP-84 guard (owner 2026-11-02, "no-raise" policy): the exceptions the package can raise are a short, pinned list.

The owner's rule: every settings conflict degrades with a warning in the report. A node raises only for what is impossible:
a missing file, a wrong model / latent / VAE / conditioning, an unmappable or broken LoRA, a non-finite output, a broken report or
save input, an unknown preset or a broken preset file, a ``profile_in`` that does not belong to the MODEL. This test reads the
SOURCE (by AST, nothing is imported or run) and lists every place that can build a raisable error:

* a call of an exception class of ``core/errors.py``, and a call that carries a string equal to a code that has an exception
  class unless the same call says ``severity`` is a warning or an info (the code is the key), and
* a ``raise`` of a plain ``RuntimeError`` / ``ValueError`` / ``TypeError`` / ``AssertionError`` / ``Exception`` (the class name
  is the key): these are programmer-contract guards, never a way to refuse a setting.

Every ``(module, function, code)`` it finds must be in ``WHITELIST`` below, with the reason: the unit is the CALL SITE's function,
so a new raise of an already listed code in a listed module is a red test unless the function that holds it is listed too (a new
function is a new decision). A whitelist row nobody uses is red as well (the list may only shrink on purpose). The package root
``__init__.py`` is scanned like the modules.

The scanner is proven here by planting each violation in a scratch tree and in the real source text.
``tests/integration/test_widget_fuzz_cpu.py`` is the other half: it drives the sampler, LoRA Phase, LoRA Profile (deprecated) and
Prompt Split widgets with a seeded generator and asserts that nothing outside the list is raised at run time.
"""
import ast
import os
import unittest

from moss_photon.core import errors

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKAGE_DIR = os.path.join(PKG_ROOT, "moss_photon")
ROOT_INIT = "../__init__.py"   # the pack's root __init__.py, relative to PACKAGE_DIR

PLAIN_EXCEPTIONS = ("RuntimeError", "ValueError", "TypeError", "AssertionError", "Exception")
INTERNAL = "internal programmer-contract guard: "

# module -> function -> {code: why it may raise}. "wrong input" = the graph is wired to something the node cannot run at all.
WHITELIST = {
    "core/adapter.py": {
        "fix_empty_latent": {"invalid_latent": "wrong latent"},
        "validate_cond_facts": {"invalid_conditioning": "wrong conditioning"},
        "validate_dimensions": {"invalid_dimensions": "wrong latent (size)"},
        "validate_latent_facts": {"invalid_latent": "wrong latent",
                                  "unsupported_source_input": "wrong latent (a noise mask)"},
        "validate_model_facts": {
            "unsupported_model": "wrong model (not MOSS)",
            "invalid_model_sampling": "wrong model (a sampling the route cannot run)",
            "conflicting_patch": "wrong model (a foreign cond hook where SLG needs the slot)"},
        "validate_vae_facts": {"invalid_vae": "wrong VAE"},
    },
    "core/canon.py": {
        "sha256_hex": {"TypeError": INTERNAL + "hashing something that is neither bytes nor text"},
        "short": {"ValueError": INTERNAL + "a prefix length below 1"},
    },
    "core/constraints.py": {
        "_layout_error": {"invalid_layout": "a layout JSON the compiler cannot read (broken prompt input)"},
        "_reject_constant": {"ValueError": "the json.loads hook for NaN / Infinity; the layout parser turns it into invalid_layout"},
    },
    "core/darkroom.py": {
        "_bad": {"invalid_override": "a value of the wrong TYPE or an unwalkable schedule (API caller / broken sigmas)"},
        "run_plan": {"RuntimeError": INTERNAL + "the solver always passes the character adapter the plan's boost mode needs"},
    },
    "core/fingerprint.py": {
        "JsonCache._load": {"ValueError": "a hash-cache file of another layout; caught on the next line, the cache starts empty"},
        "_hash_with_source": {"ValueError": INTERNAL + "a hash mode outside HASH_MODES (the node's combo hides it)"},
    },
    "core/guard.py": {"check_finite": {"non_finite_output": "non-finite output"}},
    "core/guidance.py": {
        "parse_layers": {"ValueError": INTERNAL + "a layer list with a non-integer (the widget is text, the presets are integers)"},
        "validate_slg.bad": {"invalid_override": "an SLG layer the preset's OWN settings name and the model does not have"},
    },
    "core/imagestats.py": {"_bad": {"output_error": "broken save input"}},
    "core/lora.py": {
        "StateCache.__init__": {"ValueError": INTERNAL + "a cache with no slot"},
        "_check_verified_strength": {"ValueError": "a broken sidecar value; read_sidecar's caller warns and ignores the sidecar"},
        "_parse_sidecar": {"ValueError": "a broken sidecar; read_sidecar's caller warns and ignores the sidecar"},
        "read_sidecar": {"ValueError": "a broken sidecar; MossPhotonLoRAProfile warns and ignores the sidecar"},
        "build_attachment": {"ValueError": INTERNAL + "an attachment entry the profile code itself built without a field"},
        "check_profile_in": {
            "incompatible_lora": "profile_in of another MODEL",
            "unsupported_feature": "profile_in newer than this pack knows / of an unknown kind (broken profile_in)"},
        "distill_issues": {"incompatible_lora": "the unsupported data-space DMD file"},
        "plan_patches": {"incompatible_lora": "a LoRA Phase file that is missing, unmappable or changed since the node ran"},
        "strip_trainer_flags": {"incompatible_lora": "a trainer flag ComfyUI does not apply"},
        "validate_mapping": {"incompatible_lora": "unmappable / broken LoRA (maps less than half, misfit or non-finite factors)"},
    },
    "core/lora_phase.py": {
        "check_adapter.bad": {"incompatible_lora": "a phased profile whose adapter is not on the MODEL (the profile does not belong to it)"},
        "run_segments": {"unsupported_feature": INTERNAL + "weight segments exceed the levels"},
    },
    "core/lora_plan.py": {
        "make_entry": {"invalid_override": "a phase outside the list or a strength outside +-20 (a combo and a bounded widget "
                                           "cannot produce either: API caller)"},
    },
    "core/phase.py": {
        "_issue": {"unsupported_feature": "a profile whose phase plan does not describe itself consistently (broken profile)"},
    },
    "core/presets.py": {
        "_mismatch": {"preset_version_mismatch": "parse_pin of a malformed pin; resolve() catches it and warns preset_pin_mismatch"},
        "_override_error": {"invalid_override": "an override of the wrong type or an unknown name (API caller)"},
        "_read_json": {"invalid_preset_file": "broken preset file"},
        "_scan_catalog": {"invalid_preset_file": "broken preset file / schema"},
        "load_schema": {"invalid_preset_file": "broken preset schema file"},
        "resolve": {
            "calibration_required": "a preset that names a calibration manifest which is not installed (no shipped preset does)",
            "invalid_preset": "unknown preset",
            "invalid_preset_file": "broken preset file"},
    },
    "core/report.py": {
        "<module>": {"missing_guard_dependency": "a table of the report schema (the reason a guard did not run), not a raise"},
        "ReportBuilder.finalize": {"ValueError": INTERNAL + "the builder's own report must validate (the producer is wrong, "
                                                            "not the user)"},
        "ReportBuilder.mark_phase": {"ValueError": INTERNAL + "a phase name outside the two the solver uses"},
        "ReportBuilder.section": {"ValueError": INTERNAL + "a section the builder owns, or a private object copied into the report"},
        "ReportBuilder.warn": {"ValueError": INTERNAL + "an error issue is raised, never recorded; the solver filters them first"},
        "_check_known_version": {"unsupported_feature": "a report newer than this pack knows"},
        "_type_ok": {"AssertionError": INTERNAL + "a type name the report schema table does not define"},
        "with_detailer": {"output_error": "broken report input"},
    },
    "core/rng.py": {
        "_check_policy": {"ValueError": INTERNAL + "a noise policy outside NOISE_POLICIES (the solver checks it first)"},
        "make_noise": {"unsupported_feature": "an unknown noise policy (the combo hides it)"},
    },
    "core/routes.py": {
        "_bad_override": {"invalid_override": "an unparseable query value of the preset panel (HTTP 400)"},
        "_parse_query": {"ValueError": "an unparseable query value; caught in the same function and answered as HTTP 400"},
    },
    "core/saveio.py": {
        "_output_error": {"output_error": "broken save input"},
        "next_free_path": {"ValueError": INTERNAL + "no extension asked for"},
    },
    "core/sigma.py": {
        "build_schedule_info": {"ValueError": INTERNAL + "the schedule node always passes the fresh native reference"},
        "validate_sigmas.err": {"invalid_schedule": "connected sigmas that cannot be walked (broken input)"},
    },
    "core/solver.py": {
        "_finish_model_issues": {"unsupported_model": "a finish_model of another architecture"},
        "_profile_shape_issue": {
            "incompatible_lora": "an identity_profile that is no profile / does not belong to the MODEL",
            "unsupported_feature": "profile newer than this pack"},
        "_resolve_plan": {"RuntimeError": INTERNAL + "the sampler nodes inject the plan loader with every request"},
        "check_constraints_input": {
            "invalid_conditioning": "constraints that are no valid contract (broken input)",
            "unsupported_feature": "constraints newer than this pack"},
        "check_layout_extent": {"invalid_layout": "layout boxes outside the canvas (the PromptLayout and the latent disagree)"},
        "check_layout_extent.bad": {"invalid_layout": "layout boxes outside the canvas (the PromptLayout and the latent disagree)"},
        "check_profile": {"incompatible_lora": "an identity_profile that does not belong to the MODEL"},
        "execute": {
            "invalid_override": "the cfg widget of a darkroom preset is not a number (API caller)",
            "missing_guard_dependency": "a preset that REQUIRES a guard v1 cannot run (no shipped preset does)",
            "unsupported_feature": "an unknown noise policy or a noise policy the latent cannot use"},
        "get_sampler": {"unsupported_feature": "a solver other than euler (the combo hides it)"},
    },
    "finishing/colormatch.py": {"lab_match": {"ValueError": INTERNAL + "a skin mask with no weight (the detailer checks it first)"}},
    "finishing/face_detailer.py": {
        "_check_image": {"output_error": "broken image input"},
        "best_assignment": {"ValueError": INTERNAL + "more characters than the matcher supports"},
        "detail_image": {"missing_guard_dependency": "no face detector installed", "output_error": "broken image input"},
    },
    "finishing/geometry.py": {"assert_outside_mask_equal": {"output_error": "broken image input"}},
    # MossPhotonIdentityDetailer (second queue, outside MP-84): it redraws faces with the CHARACTER adapter of the profile, so a
    # missing profile, no distill role and a missing detector stay refusals. Reported to the lead.
    "nodes/detailer.py": {
        "MossPhotonIdentityDetailer.detail": {"missing_guard_dependency": "no face detector installed",
                                              "missing_required_input": "the detailer needs its inputs"},
        "_check_model_and_profile": {"missing_lora": "the detailer needs the character adapter"},
        "_images_as_numpy": {"output_error": "broken image input"},
        "_require_detector": {"missing_guard_dependency": "no face detector installed"},
        "_require_turbo_profile": {"unsupported_feature": "the face pass is verified on the turbo stack only"},
        "_require_valid_report": {"output_error": "broken report input"},
    },
    "nodes/diagnostics.py": {"_require_valid_report": {"output_error": "broken report / image input"}},
    "nodes/lora.py": {
        "MossPhotonLoRAProfile.apply": {
            "incompatible_lora": "MossPhotonLoRAProfile (deprecated): unmappable / broken LoRA",
            "unsupported_feature": "MossPhotonLoRAProfile: a profile / role outside the lists, a ComfyUI that patches differently"},
    },
    "nodes/prompt.py": {"MossPhotonPromptLayout.encode": {"missing_required_input": "an empty positive prompt"}},
    "nodes/sampler.py": {"MossPhotonSamplerAdvanced.sample": {"unsupported_feature": "a solver other than euler (the combo hides it)"}},
    "nodes/save.py": {"_fail": {"output_error": "broken save input"}},
    "nodes/schedule.py": {
        "MossPhotonSchedule.build": {"invalid_model_sampling": "wrong model",
                                     "unsupported_feature": "a schedule mode the node cannot build"},
    },
}
ALLOWED = {(module, function, code)
           for module, functions in WHITELIST.items() for function, codes in functions.items() for code in codes}

_CLASS_CODES = {cls.__name__: cls.CODE for cls in errors._ERROR_CLASSES}
_RAISABLE = {code for code, info in errors.REASONS.items() if info.exc is not None}
# the typed codes the whitelist allows anywhere (the widget fuzz asserts a raised MossPhotonError is one of them)
ALLOWED_CODES = {code for (_module, _function, code) in ALLOWED if code in _RAISABLE}


def _name_of(node):
    return node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)


class _Collector(ast.NodeVisitor):
    """Every raise site of one module, keyed by the function (qualified by its classes and enclosing functions) that holds it."""

    def __init__(self, rel, found):
        self.rel, self.found, self.scope = rel, found, []

    def _in_scope(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _in_scope

    def _add(self, code, line):
        self.found.setdefault((self.rel, ".".join(self.scope) or "<module>", code), []).append(line)

    def visit_Raise(self, node):
        exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if _name_of(exc) in PLAIN_EXCEPTIONS:
            self._add(_name_of(exc), node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node):
        name = _name_of(node.func)
        if name in _CLASS_CODES:
            self._add(_CLASS_CODES[name], node.lineno)
        else:
            severity = None
            for kw in node.keywords:
                if kw.arg == "severity":
                    severity = kw.value.value if isinstance(kw.value, ast.Constant) else "<computed>"
            if severity not in ("warning", "info"):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value in _RAISABLE:
                        self._add(arg.value, node.lineno)
        self.generic_visit(node)


def raise_sites(sources):
    """``{(module, function, code): [line, ...]}`` for ``sources`` (``{relative path: text}``): every call of an exception class,
    every call with a string argument equal to a raisable code unless it says ``severity`` is a warning or an info, and every
    ``raise`` of a plain exception class (the class name is the code)."""
    found = {}
    for rel, text in sources.items():
        if rel == "core/errors.py":
            continue
        _Collector(rel, found).visit(ast.parse(text, filename=rel))
    return found


def package_sources(package_dir=PACKAGE_DIR):
    files = {}
    for dirpath, dirnames, filenames in os.walk(package_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fname in filenames:
            if fname.endswith(".py"):
                path = os.path.join(dirpath, fname)
                with open(path, "r", encoding="utf-8") as f:
                    files[os.path.relpath(path, package_dir).replace(os.sep, "/")] = f.read()
    root = os.path.join(os.path.dirname(package_dir), "__init__.py")
    if os.path.isfile(root):
        with open(root, "r", encoding="utf-8") as f:
            files[ROOT_INIT] = f.read()
    return files


class WhitelistTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = package_sources()
        cls.found = raise_sites(cls.sources)

    def test_every_raise_site_is_whitelisted(self):
        stray = sorted((module, function, code, lines) for (module, function, code), lines in self.found.items()
                       if (module, function, code) not in ALLOWED)
        self.assertEqual(stray, [], "a raise site outside the no-raise whitelist: degrade it to a warning (MP-84) "
                                    "or, if it is impossible to run on, add (module, function, code) to WHITELIST with the reason")

    def test_every_whitelist_row_is_used(self):
        dead = sorted(ALLOWED - set(self.found))
        self.assertEqual(dead, [], "a whitelist row nobody raises any more: delete it (the list may only shrink on purpose)")

    def test_the_whitelist_names_real_modules_functions_and_raisable_codes(self):
        for module, functions in WHITELIST.items():
            self.assertIn(module, self.sources, module)
            for function, codes in functions.items():
                for code, why in codes.items():
                    self.assertTrue(code in _RAISABLE or code in PLAIN_EXCEPTIONS,
                                    "%s %s: %s has no exception class" % (module, function, code))
                    self.assertTrue(why.strip(), "%s %s %s has no reason" % (module, function, code))

    def test_the_scan_covers_the_package_root_and_every_module(self):
        self.assertIn(ROOT_INIT, self.sources)
        for rel in ("core/solver.py", "core/darkroom.py", "core/report.py", "nodes/lora_plan.py", "nodes/prompt_split.py"):
            self.assertIn(rel, self.sources, rel)
        self.assertIn(("core/solver.py", "_resolve_plan", "RuntimeError"), self.found, "control: a plain raise is found")

    def test_the_conflicts_the_owner_listed_are_not_raisable_any_more(self):
        """The audit list of the ticket: each of these codes was raised for a settings conflict and must be gone from the
        modules that raised it (they may remain for the other, impossible, reasons in the whitelist)."""
        pairs = {(module, code) for (module, _function, code) in self.found}
        gone = {("core/solver.py", "missing_lora"), ("core/solver.py", "budget_exceeded"),
                ("core/darkroom.py", "incompatible_lora"), ("core/presets.py", "incompatible_lora"),
                ("core/lora.py", "missing_lora")}
        self.assertEqual(sorted(gone & pairs), [])
        self.assertNotIn("preset_version_mismatch", {code for (module, code) in pairs if module != "core/presets.py"})

    def test_the_whitelist_is_not_a_blanket(self):
        """Control: the codes the audit degraded are not whitelisted anywhere (they never were impossible to run on)."""
        for code in ("missing_lora", "budget_exceeded"):
            for module, functions in WHITELIST.items():
                if module != "nodes/detailer.py":
                    for function, codes in functions.items():
                        self.assertNotIn(code, codes, "%s %s" % (module, function))

    def test_a_new_call_site_of_a_listed_code_in_a_listed_module_is_red(self):
        """The case the per-module list let through: ``invalid_override`` is whitelisted in core/solver.py (``execute``), a new
        function that raises it there is a new site, outside the list."""
        text = self.sources["core/solver.py"] + (
            "\n\ndef planted_new_site(node):\n"
            "    errors.raise_first_error([errors.make_issue('invalid_override', node, 'p', 'm', 'f', severity='error')])\n")
        found = raise_sites(dict(self.sources, **{"core/solver.py": text}))
        stray = sorted(site for site in found if site not in ALLOWED)
        self.assertEqual(stray, [("core/solver.py", "planted_new_site", "invalid_override")])
        self.assertIn(("core/solver.py", "execute", "invalid_override"), ALLOWED, "control: the code IS listed in the module")


class ScannerGoesRedTest(unittest.TestCase):
    """An invariant is proven by breaking it: each spelling of a raise, planted in a scratch tree, is found."""

    def sites(self, text, rel="core/solver.py"):
        return raise_sites({rel: text})

    def codes(self, text, rel="core/solver.py"):
        return {code for (_module, _function, code) in self.sites(text, rel)}

    def test_a_planted_error_issue_is_found(self):
        text = ("from . import errors\n"
                "def f():\n"
                "    errors.raise_first_error([errors.make_issue('budget_exceeded', 'n', 'p', 'm', 'f', severity='error')])\n")
        self.assertEqual(set(self.sites(text)), {("core/solver.py", "f", "budget_exceeded")})

    def test_a_planted_default_severity_issue_is_found(self):
        """No ``severity`` at all: a code with an exception class defaults to an error."""
        self.assertEqual(set(self.sites("x = make_issue('missing_lora', 'n', 'p', 'm', 'f')\n")),
                         {("core/solver.py", "<module>", "missing_lora")})

    def test_a_planted_exception_class_is_found(self):
        text = "from . import errors\nraise errors.BudgetExceeded(issue)\n"
        self.assertEqual(self.codes(text), {"budget_exceeded"})
        self.assertEqual(self.codes("raise MissingLora(issue)\n"), {"missing_lora"})

    def test_a_computed_severity_is_still_found(self):
        self.assertEqual(self.codes("make_issue('incompatible_lora', 'n', 'p', 'm', 'f', severity=level)\n"),
                         {"incompatible_lora"})

    def test_a_warning_or_an_info_is_not_a_raise(self):
        for severity in ("warning", "info"):
            text = "make_issue('budget_exceeded', 'n', 'p', 'm', 'f', severity='%s')\n" % severity
            self.assertEqual(self.sites(text), {}, severity)

    def test_a_status_code_is_not_raisable(self):
        self.assertEqual(self.sites("make_issue('setting_adjusted', 'n', 'p', 'm', 'f')\n"), {})

    def test_a_plain_exception_is_found_in_every_spelling(self):
        for text in ("def f():\n    raise RuntimeError('x')\n", "def f():\n    raise RuntimeError\n",
                     "def f(a):\n    raise builtins.RuntimeError('x') from a\n"):
            self.assertEqual(set(self.sites(text)), {("core/solver.py", "f", "RuntimeError")}, text)
        for name in PLAIN_EXCEPTIONS:
            self.assertEqual(self.codes("def f():\n    raise %s('x')\n" % name), {name})

    def test_an_exception_that_is_not_in_the_plain_list_or_a_bare_re_raise_is_not_a_site(self):
        self.assertEqual(self.sites("def f():\n    try:\n        pass\n    except OSError:\n        raise\n"), {})
        self.assertEqual(self.sites("def f():\n    raise KeyError('x')\n"), {})

    def test_the_site_is_keyed_by_the_function_qualified_by_its_classes_and_enclosing_functions(self):
        text = ("class A:\n"
                "    def m(self):\n"
                "        def inner():\n"
                "            raise ValueError('x')\n"
                "def top():\n"
                "    raise ValueError('y')\n"
                "raise ValueError('z')\n")
        self.assertEqual(set(self.sites(text)), {("core/solver.py", "A.m.inner", "ValueError"),
                                                 ("core/solver.py", "top", "ValueError"),
                                                 ("core/solver.py", "<module>", "ValueError")})

    def test_the_root_init_is_scanned_like_a_module(self):
        found = raise_sites({ROOT_INIT: "def register():\n    raise RuntimeError('planted')\n"})
        self.assertEqual(set(found), {(ROOT_INIT, "register", "RuntimeError")})
        self.assertTrue([site for site in found if site not in ALLOWED])

    def test_the_planted_site_is_outside_the_whitelist_and_so_the_guard_would_fail(self):
        found = self.sites("make_issue('missing_lora', 'n', 'p', 'm', 'f')\n")
        self.assertTrue([site for site in found if site not in ALLOWED])
        found = self.sites("def execute():\n    make_issue('incompatible_lora', 'n', 'p', 'm', 'f')\n", rel="core/lora.py")
        self.assertTrue([site for site in found if site not in ALLOWED], "a listed code in a listed module, new function: red")
        found = self.sites("def check_profile():\n    make_issue('incompatible_lora', 'n', 'p', 'm', 'f')\n", rel="core/solver.py")
        self.assertFalse([site for site in found if site not in ALLOWED], "control: a whitelisted site is accepted")


if __name__ == "__main__":
    unittest.main()
