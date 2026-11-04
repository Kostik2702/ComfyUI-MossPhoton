# -*- coding: utf-8 -*-
"""Run the MossPhoton unittest suites (no pytest in the ComfyUI embedded interpreter).

    <embedded python> tests/run_tests.py [--suite unit|integration|gpu|quality|all]      (default: unit)
                                         [--durations N] [--jobs N] [--leaks] [--no-model-cleanup]

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

Speed and diagnostics (all of them leave the test code alone; they live in this runner only):

--durations N     after each suite print the N slowest tests and the N slowest files (wall time per test, measured
                  by a TestResult subclass; the cleanup below is not counted in a test's time).
--jobs N          integration only: split the test CLASSES (one file can hold most of the time) over N worker
                  subprocesses (default 1 = one process, the old behaviour; capped at 4). Each worker discovers the
                  whole suite and runs only its classes, with OMP/MKL/OPENBLAS_NUM_THREADS=4 and
                  torch.set_num_threads(4) (the integration helper tests/integration/_comfy_env.py hardcodes 8
                  threads, so the worker overrides its THREADS before ``setup()`` runs). Classes are balanced by the
                  per-class seconds of the last serial run, kept as a hint in the system temp dir (env
                  MOSS_PHOTON_TEST_DURATIONS overrides the path); without the hint, by test count. One class
                  (FULL_THREAD_CLASSES: it asserts THREADS == 8) runs afterwards in its own process without the cap.
                  The merged summary line has the usual format. Tests must not share fixed paths; the current ones use mkdtemp.
--no-model-cleanup  turn off the model cleanup that otherwise runs between tests in a process that imported comfy:
                  one gc.collect(), drop the dead ``LoadedModel`` entries (patcher gone, module still referenced: a
                  leak that makes comfy do a full gc.collect() and log "memory leak" on EVERY later model load, so one
                  leaking test slows every test after it), then ``comfy.model_management.unload_all_models()``.
                  The dead entries still alive after the gc are counted per file and reported (diagnostic, not a
                  failure; they are the leaks of the tests, the cleanup only stops them from slowing the rest).
--leaks           also list the dead entries per test (and count them even with --no-model-cleanup). While leaks are
                  tracked a full gc runs after every test, so the count does not depend on when the automatic gc happened
                  to run (without it an entry is only seen once something else has triggered a collection).
env MOSS_PHOTON_STRICT_LEAKS     a dead ``LoadedModel`` entry left by a test (after gc) fails the run, with the per-test list
                  printed. ON by default for the integration suite (MP-88: the tests there run with a stand-in for the
                  server, tests/integration/graph_cache.py, so a clone of a clone shows up); ``=0`` turns it off for a
                  run. Other suites: only with ``=1``. It costs one gc per test (about 0.1s).
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)

SUITES = ("unit", "integration", "gpu", "quality", "local")
LOCAL_FLAG = "MOSS_PHOTON_LOCAL"
STRICT_LEAKS_FLAG = "MOSS_PHOTON_STRICT_LEAKS"
DURATIONS_CACHE_ENV = "MOSS_PHOTON_TEST_DURATIONS"
# A suite that must contain tests; gpu is added to it when MOSS_PHOTON_GPU=1 (an opt-in run that finds nothing is red).
MUST_HAVE_TESTS = ("unit", "integration", "quality")
MAX_JOBS = 4
WORKER_THREADS = 4
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
# Classes that assert the helper's own 8 threads (test_wave0_tiny_moss: THREADS == 8): with --jobs they run after the
# workers, alone, without the thread cap.
FULL_THREAD_CLASSES = ("tests.integration.test_wave0_tiny_moss.EnvTest",)


class _TimedResult(unittest.TextTestResult):
    """TextTestResult with the wall time of every test and, when comfy is loaded, a model cleanup between tests."""

    model_cleanup = False  # set by the child before the run
    track_leaks = False
    strict = False         # a leak check that cannot run fails the run (set with track_leaks)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timings = []  # (seconds, test id, module name)
        self.leaks_by_test = {}  # test id -> dead LoadedModel entries it left behind (still dead after a gc)
        self.leak_check_error = None  # strict mode: why the check could not run (a comfy API it needs is gone)
        self.cleanup_seconds = 0.0
        self._t0 = 0.0
        self._previous = None  # (test id, module) of the test that ran last
        self._seen_dead = []  # kept alive on purpose: a freed object's id() could be reused

    def startTest(self, test):
        # Between tests, not in stopTest: by now the unittest loop has dropped the previous test instance, so what
        # it only referenced through ``self`` is collectable.
        if self._previous is not None:
            self._release_models(self._previous[0])
        self._previous = (test.id(), type(test).__module__)
        super().startTest(test)
        self._t0 = time.perf_counter()

    def stopTest(self, test):
        self.timings.append((time.perf_counter() - self._t0, test.id(), type(test).__module__))
        super().stopTest(test)

    def stopTestRun(self):
        if self._previous is not None:
            self._release_models(self._previous[0])
        super().stopTestRun()

    def wasSuccessful(self):
        return super().wasSuccessful() and self.leak_check_error is None

    @property
    def leaks_by_module(self):
        by_module = {}
        for tid, n in self.leaks_by_test.items():
            module = tid.rsplit(".", 2)[0]
            by_module[module] = by_module.get(module, 0) + n
        return by_module

    def _release_models(self, owner):
        """Count the dead LoadedModel entries ``owner`` left, drop them and unload the rest (model_cleanup)."""
        # Never imports comfy: a process that did not import it (unit suite) has nothing to clean.
        mm = sys.modules.get("comfy.model_management")
        if mm is None or not (self.model_cleanup or self.track_leaks):
            return
        t0 = time.perf_counter()
        try:
            # A full gc of a process with torch loaded is ~0.1s. An entry dies only when the cyclic collector frees its
            # patcher together with a throwaway parent, so what a test leaves is visible only after a collection: when
            # leaks are tracked the gc is forced (otherwise the count depends on when the automatic gc happened to run),
            # when they are not, only an entry that is already dead (the case comfy itself answers with a gc) triggers it.
            dead = [m for m in mm.current_loaded_models if m.is_dead()]
            if dead or self.track_leaks:
                gc.collect()
                mm.cleanup_models()
                dead = [m for m in mm.current_loaded_models if m.is_dead()]
            new = [m for m in dead if not any(m is s for s in self._seen_dead)]
            self._seen_dead.extend(new)
            if new:
                self.leaks_by_test[owner] = len(new)
            if self.model_cleanup:
                mm.current_loaded_models[:] = [m for m in mm.current_loaded_models if not m.is_dead()]
                mm.unload_all_models()
        except AttributeError as exc:  # a comfy version without this API: report once, stop touching it
            print("run_tests: model cleanup disabled, comfy.model_management lacks %s" % exc, file=sys.stderr)
            if type(self).strict and self.leak_check_error is None:
                # a check that switches itself off and exits 0 attests to nothing: the run fails
                self.leak_check_error = "comfy.model_management lacks %s" % exc
            type(self).model_cleanup = type(self).track_leaks = False
        self.cleanup_seconds += time.perf_counter() - t0


def _file_key(module, suite):
    parts = module.split(".")
    if parts[:2] == ["tests", suite]:
        return "/".join(parts[2:]) + ".py"
    return module


def _print_durations(timings, top, suite):
    by_file = {}
    for secs, _tid, module in timings:
        by_file[_file_key(module, suite)] = by_file.get(_file_key(module, suite), 0.0) + secs
    print("\nrun_tests[%s]: slowest %d tests (of %d, %.1fs in tests)" % (
        suite, min(top, len(timings)), len(timings), sum(t[0] for t in timings)))
    for secs, tid, _m in sorted(timings, reverse=True)[:top]:
        print("  %8.2fs  %s" % (secs, tid))
    print("run_tests[%s]: slowest %d files (of %d)" % (suite, min(top, len(by_file)), len(by_file)))
    for key, secs in sorted(by_file.items(), key=lambda kv: -kv[1])[:top]:
        print("  %8.2fs  %s" % (secs, key))


def _durations_cache_path():
    return os.environ.get(DURATIONS_CACHE_ENV) or os.path.join(tempfile.gettempdir(), "moss_photon_test_durations.json")


def _load_durations_hint():
    try:
        with open(_durations_cache_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _class_key(test_id):
    """``tests.integration.test_x.SomeTest.test_y`` -> ``tests.integration.test_x.SomeTest``."""
    return test_id.rsplit(".", 1)[0]


def _store_durations_hint(timings):
    per_class = {}
    for secs, tid, _module in timings:
        key = _class_key(tid)
        per_class[key] = per_class.get(key, 0.0) + secs
    merged = _load_durations_hint()
    merged.update(per_class)
    path = _durations_cache_path()
    try:
        tmp = path + ".%d.tmp" % os.getpid()
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: round(v, 3) for k, v in sorted(merged.items())}, f, indent=1)
        os.replace(tmp, path)
    except OSError:
        pass  # only a balancing hint


def _iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


def _discover(suite, class_keys):
    """All tests of the suite, or only those whose class key is in ``class_keys`` (discovery order is kept)."""
    discovered = unittest.defaultTestLoader.discover(
        os.path.join(HERE, suite), pattern="test_*.py", top_level_dir=PKG_ROOT)
    if class_keys is None:
        return discovered
    selected = unittest.TestSuite()
    for test in _iter_tests(discovered):
        if _class_key(test.id()) in class_keys:
            selected.addTest(test)
    return selected


def _list_test_classes(suite):
    counts = {}
    for test in _iter_tests(_discover(suite, None)):
        key = _class_key(test.id())
        counts[key] = counts.get(key, 0) + 1
    return counts


def _apply_threads(n):
    """The integration helper calls torch.set_num_threads(THREADS) in setup(); lower THREADS before that runs."""
    for name, mod in list(sys.modules.items()):
        if name.endswith("_comfy_env") and hasattr(mod, "THREADS") and hasattr(mod, "setup"):
            mod.THREADS = n


def _run_suite_in_this_process(suite, args):
    if args.threads:
        for var in THREAD_ENV:
            os.environ[var] = str(args.threads)
    # The embedded python ignores the script dir, so put the roots on sys.path.
    for p in (PKG_ROOT, HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
    if args.list_classes:
        with open(args.result_json, "w", encoding="utf-8") as f:
            json.dump(_list_test_classes(suite), f)
        return 0
    class_keys = set(k for k in args.classes.split(",") if k) if args.classes else None
    if suite == "integration":
        # the process stands in for the server where a dead LoadedModel is decided (see the module); not for gpu / local,
        # which hold real checkpoints that must not be kept alive
        from tests.integration import graph_cache
        graph_cache.install()
    discovered = _discover(suite, class_keys)
    if args.threads:
        _apply_threads(args.threads)
    strict = os.environ.get(STRICT_LEAKS_FLAG, "1" if suite == "integration" else "0") == "1"
    _TimedResult.model_cleanup = not args.no_model_cleanup
    _TimedResult.track_leaks = args.leaks or strict
    _TimedResult.strict = strict
    result = unittest.TextTestRunner(verbosity=2, resultclass=_TimedResult).run(discovered)
    if args.durations:
        _print_durations(result.timings, args.durations, suite)
    ok = result.wasSuccessful()
    problems = []
    if result.leak_check_error:
        problems.append("the dead-LoadedModel check could not run (%s=1): %s" % (STRICT_LEAKS_FLAG, result.leak_check_error))
    if result.leaks_by_module:
        print("\nrun_tests[%s]: dead LoadedModel entries left by tests after gc (a leak, diagnostic): %d in %d files"
              % (suite, sum(result.leaks_by_module.values()), len(result.leaks_by_module)))
        for module, n in sorted(result.leaks_by_module.items(), key=lambda kv: -kv[1]):
            print("  %4d  %s" % (n, module))
        if args.leaks or strict:
            for tid, n in sorted(result.leaks_by_test.items(), key=lambda kv: -kv[1]):
                print("  %4d    %s" % (n, tid))
        if strict:
            ok = False
            problems.append("%d dead LoadedModel entries left by tests (%s=1)"
                            % (sum(result.leaks_by_module.values()), STRICT_LEAKS_FLAG))
    if result.cleanup_seconds:
        print("run_tests[%s]: model cleanup/leak accounting took %.1fs" % (suite, result.cleanup_seconds))
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
    if suite == "integration" and class_keys is None:
        _store_durations_hint(result.timings)
    if args.result_json:
        with open(args.result_json, "w", encoding="utf-8") as f:
            json.dump({"ok": ok, "tests": result.testsRun, "skipped": len(result.skipped), "problems": problems,
                       "timings": result.timings}, f)
    return 0 if ok else 1


def _partition(class_counts, hint, jobs):
    """Longest-processing-time-first balance of test classes. Weight = hinted seconds of the class; a class the
    hint does not know weighs its test count times the hint's mean seconds per test (1.0 without any hint)."""
    known = [(hint[k], n) for k, n in class_counts.items() if k in hint]
    per_test = (sum(s for s, _ in known) / max(1, sum(n for _, n in known))) if known else 1.0
    weight = {k: hint.get(k, n * per_test) for k, n in class_counts.items()}
    bins = [[] for _ in range(jobs)]
    load = [0.0] * jobs
    for k in sorted(class_counts, key=lambda k: (-weight[k], k)):
        i = load.index(min(load))
        bins[i].append(k)
        load[i] += weight[k]
    return [b for b in bins if b]


def _run_integration_in_workers(jobs, args):
    workdir = tempfile.mkdtemp(prefix="moss_photon_workers_")
    listing = os.path.join(workdir, "classes.json")
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--suite", "integration", "--child",
                           "--list-classes", "--result-json", listing], cwd=PKG_ROOT)
    try:
        with open(listing, "r", encoding="utf-8") as f:
            class_counts = json.load(f)
    except (OSError, ValueError):
        class_counts = {}
    if proc.returncode != 0 or not class_counts:
        shutil.rmtree(workdir, ignore_errors=True)
        print("run_tests[integration]: FAILED (0 tests, 0 skipped); could not list the test classes of tests/integration")
        return 1
    pinned = [k for k in FULL_THREAD_CLASSES if k in class_counts]
    pool = {k: n for k, n in class_counts.items() if k not in pinned}
    bins = _partition(pool, _load_durations_hint(), min(jobs, len(pool))) if pool else []
    capped_env = dict(os.environ)
    for var in THREAD_ENV:
        capped_env[var] = str(WORKER_THREADS)

    def launch(label, group, threads):
        base = os.path.join(workdir, label)
        cmd = [sys.executable, os.path.abspath(__file__), "--suite", "integration", "--child",
               "--classes", ",".join(group), "--result-json", base + ".json"]
        if threads:
            cmd += ["--threads", str(threads)]
        if args.leaks:
            cmd.append("--leaks")
        if args.no_model_cleanup:
            cmd.append("--no-model-cleanup")
        log = open(base + ".log", "wb")
        proc = subprocess.Popen(cmd, cwd=PKG_ROOT, env=capped_env if threads else dict(os.environ),
                                stdout=log, stderr=subprocess.STDOUT)
        return label, group, base, proc, log

    merged = {"ok": True, "tests": 0, "skipped": 0, "problems": [], "timings": []}

    def collect(worker):
        label, group, base, proc, log = worker
        code = proc.wait()
        log.close()
        print("\n==== integration %s: %d classes, %d tests, exit %d ====" % (
            label, len(group), sum(class_counts[c] for c in group), code))
        sys.stdout.flush()
        with open(base + ".log", "rb") as f:
            sys.stdout.buffer.write(f.read())
        sys.stdout.buffer.flush()
        try:
            with open(base + ".json", "r", encoding="utf-8") as f:
                part = json.load(f)
        except (OSError, ValueError):
            part = {"ok": False, "tests": 0, "skipped": 0, "timings": [],
                    "problems": ["%s left no result (crashed?)" % label]}
        merged["ok"] = merged["ok"] and part["ok"] and code == 0
        merged["tests"] += part["tests"]
        merged["skipped"] += part["skipped"]
        merged["problems"] += ["%s: %s" % (label, p) for p in part["problems"]]
        merged["timings"] += [tuple(t) for t in part["timings"]]

    started = time.perf_counter()
    workers = [launch("worker%d" % (k + 1), group, WORKER_THREADS) for k, group in enumerate(bins)]
    print("run_tests[integration]: %d tests in %d classes over %d workers x %d threads%s" % (
        sum(class_counts.values()), len(class_counts), len(workers), WORKER_THREADS,
        ", then %d classes alone with the default threads" % len(pinned) if pinned else ""))
    sys.stdout.flush()
    for worker in workers:
        collect(worker)
    if pinned:
        collect(launch("default-threads run", pinned, 0))
    wall = time.perf_counter() - started
    shutil.rmtree(workdir, ignore_errors=True)
    print("\nrun_tests[integration]: %d workers, wall %.1fs" % (len(workers), wall))
    if args.durations:
        _print_durations(merged["timings"], args.durations, "integration")
    _store_durations_hint(merged["timings"])
    # A class-level skip (setUpClass) runs none of its tests, so the totals only have to agree when nothing skipped.
    if merged["skipped"] == 0 and merged["tests"] != sum(class_counts.values()):
        merged["ok"] = False
        merged["problems"].append("workers ran %d tests, discovery found %d"
                                  % (merged["tests"], sum(class_counts.values())))
    print("run_tests[integration]: %s (%d tests, %d skipped)%s" % (
        "OK" if merged["ok"] else "FAILED", merged["tests"], merged["skipped"],
        "".join("; " + p for p in merged["problems"])))
    return 0 if merged["ok"] else 1


def _run_suites_in_subprocesses(suites, args):
    codes = {}
    for suite in suites:
        if suite == "local" and os.environ.get(LOCAL_FLAG) != "1":
            print("run_tests[local]: SKIPPED by design (opt-in, set %s=1)" % LOCAL_FLAG)
            codes[suite] = None
            continue
        if suite == "integration" and args.jobs > 1:
            codes[suite] = _run_integration_in_workers(args.jobs, args)
            continue
        cmd = [sys.executable, os.path.abspath(__file__), "--suite", suite, "--child"]
        if args.durations:
            cmd += ["--durations", str(args.durations)]
        if args.leaks:
            cmd.append("--leaks")
        if args.no_model_cleanup:
            cmd.append("--no-model-cleanup")
        proc = subprocess.run(cmd, cwd=PKG_ROOT)
        codes[suite] = proc.returncode
    print("run_tests: " + ", ".join(
        "%s=%s" % (s, "SKIPPED" if c is None else "OK" if c == 0 else "FAILED(%d)" % c) for s, c in codes.items()))
    return 0 if all(c in (0, None) for c in codes.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run MossPhoton test suites, one subprocess per suite.")
    parser.add_argument("--suite", choices=SUITES + ("all",), default="unit")
    parser.add_argument("--durations", type=int, default=0, metavar="N",
                        help="print the N slowest tests and the N slowest files per suite")
    parser.add_argument("--jobs", type=int, default=1, metavar="N",
                        help="integration: N worker subprocesses over the test classes (default 1, max %d)" % MAX_JOBS)
    parser.add_argument("--leaks", action="store_true", help="per-test attribution of dead LoadedModel entries")
    parser.add_argument("--no-model-cleanup", action="store_true",
                        help="do not unload comfy models / drop dead entries between tests")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--classes", default="", help=argparse.SUPPRESS)
    parser.add_argument("--list-classes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.jobs > MAX_JOBS:
        print("run_tests: --jobs %d capped at %d (shared desktop: %d workers x %d threads)"
              % (args.jobs, MAX_JOBS, MAX_JOBS, WORKER_THREADS))
        args.jobs = MAX_JOBS
    if args.child:
        if args.suite == "all":
            parser.error("--child needs a single suite")
        return _run_suite_in_this_process(args.suite, args)
    if args.suite == "all":
        # local is part of "all" only when opted in (its own skip-by-design line is printed for single runs).
        selected = tuple(s for s in SUITES if s != "local" or os.environ.get(LOCAL_FLAG) == "1")
    else:
        selected = (args.suite,)
    return _run_suites_in_subprocesses(selected, args)


if __name__ == "__main__":
    sys.exit(main())
