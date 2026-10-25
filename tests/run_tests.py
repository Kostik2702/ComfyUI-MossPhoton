# -*- coding: utf-8 -*-
"""Run the MossPhoton unittest suites (no pytest in the ComfyUI embedded interpreter).

    <embedded python> tests/run_tests.py [--suite unit|integration|gpu|quality|all]      (default: unit)

Every suite runs in its OWN subprocess (design A14): the unit suite asserts that ``comfy`` stays out of
``sys.modules`` and the integration suite imports it. Exit code is 0 only if every selected suite passed.

* unit        no comfy; after the run the child fails if ``comfy`` got imported anyway.
* integration comfy on CPU with the tiny random MOSS (no weights); tests skip with a message when ComfyUI is missing.
* gpu         opt-in (env MOSS_PHOTON_GPU=1, checked by the tests themselves); an empty suite is allowed, except with
              MOSS_PHOTON_GPU=1, where discovering no test is a failure.
* quality     workflow-level checks (tests/quality); an empty suite is a failure.
* local       opt-in (env MOSS_PHOTON_LOCAL=1): tests that read the owner's real model files; names come from the
              untracked tests/local/local_models.json. Without the flag it is not run (reported as skipped by design);
              with the flag, discovering no test is a failure. ``all`` includes it only when the flag is set.
"""
import argparse
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)

SUITES = ("unit", "integration", "gpu", "quality", "local")
LOCAL_FLAG = "MOSS_PHOTON_LOCAL"
# A suite that must contain tests; gpu is added to it when MOSS_PHOTON_GPU=1 (an opt-in run that finds nothing is red).
MUST_HAVE_TESTS = ("unit", "integration", "quality")


def _run_suite_in_this_process(suite):
    # The embedded python ignores the script dir, so put the roots on sys.path.
    for p in (PKG_ROOT, HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
    start = os.path.join(HERE, suite)
    discovered = unittest.defaultTestLoader.discover(start, pattern="test_*.py", top_level_dir=PKG_ROOT)
    result = unittest.TextTestRunner(verbosity=2).run(discovered)
    ok = result.wasSuccessful()
    problems = []
    gpu_opted_in = suite == "gpu" and os.environ.get("MOSS_PHOTON_GPU") == "1"
    local_opted_in = suite == "local" and os.environ.get(LOCAL_FLAG) == "1"
    if (suite in MUST_HAVE_TESTS or gpu_opted_in or local_opted_in) and result.testsRun == 0:
        ok = False
        problems.append("no tests discovered in tests/%s" % suite)
    if suite == "unit" and "comfy" in sys.modules:
        ok = False
        problems.append("comfy was imported by the unit suite")
    print("run_tests[%s]: %s (%d tests, %d skipped)%s" % (
        suite, "OK" if ok else "FAILED", result.testsRun, len(result.skipped),
        "".join("; " + p for p in problems)))
    return 0 if ok else 1


def _run_suites_in_subprocesses(suites):
    codes = {}
    for suite in suites:
        if suite == "local" and os.environ.get(LOCAL_FLAG) != "1":
            print("run_tests[local]: SKIPPED by design (opt-in, set %s=1)" % LOCAL_FLAG)
            codes[suite] = None
            continue
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--suite", suite, "--child"], cwd=PKG_ROOT)
        codes[suite] = proc.returncode
    print("run_tests: " + ", ".join(
        "%s=%s" % (s, "SKIPPED" if c is None else "OK" if c == 0 else "FAILED(%d)" % c) for s, c in codes.items()))
    return 0 if all(c in (0, None) for c in codes.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run MossPhoton test suites, one subprocess per suite.")
    parser.add_argument("--suite", choices=SUITES + ("all",), default="unit")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        if args.suite == "all":
            parser.error("--child needs a single suite")
        return _run_suite_in_this_process(args.suite)
    if args.suite == "all":
        # local is part of "all" only when opted in (its own skip-by-design line is printed for single runs).
        selected = tuple(s for s in SUITES if s != "local" or os.environ.get(LOCAL_FLAG) == "1")
    else:
        selected = (args.suite,)
    return _run_suites_in_subprocesses(selected)


if __name__ == "__main__":
    sys.exit(main())
