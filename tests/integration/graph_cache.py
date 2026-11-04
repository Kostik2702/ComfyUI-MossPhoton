"""The integration process stands in for the ComfyUI server in the two ways that decide whether a patcher leaves a dead
``LoadedModel`` behind (``comfy/model_management.py`` ``LoadedModel._set_model`` / ``_switch_parent``).

A ``LoadedModel`` keeps a weakref to its patcher and ONE weakref to the patcher's ``parent``; when the patcher dies it moves to
the parent. If the cyclic gc frees the patcher and that parent in one pass the parent's weakref is already cleared, nothing is
moved, and the entry stays dead for good: comfy then runs a full ``gc.collect()`` and logs "memory leak with model ..." on
every later load. That happens only when the parent is a THROWAWAY clone (a clone of a clone) and the patcher is collected by the
cyclic gc, not by refcount. In a bare test process neither holds, so a clone of a clone would pass unnoticed:

* the patcher a run samples from dies by refcount the moment the run returns (nothing in the tiny fixtures keeps it in a cycle),
  and its throwaway parent right after it: the chain unwinds through ``_switch_parent`` one step at a time. ``build_guider`` is
  wrapped to put every sampling patcher into a reference cycle, as the real server does (the maintainers' earlier probe of the same mechanism: two levels without a cycle are fine, two
  levels and a cycle leave the entry dead);
* the MODEL a node receives is held by the graph cache for as long as the workflow lives, a test's local variable dies with the
  test. ``solver.execute`` is wrapped to keep every MODEL (and ``finish_model``) that enters it alive for the process, so a test
  dropping its own input no longer produces a dead entry.

Limits (this is an emulation, not the server): the MODELs are pinned for the WHOLE process, while the server evicts old cache
entries (``execution.py``, ``comfy_execution/caching.py``). A patcher that is a direct clone, still waiting in a cycle, whose
incoming MODEL is evicted before the gc runs, gives a dead entry on a real ``LoadedModel`` that this process never sees: a
false green is possible in that window. So a green run means "no clone of a throwaway on a covered path", not "no warning on the
server". Covered: patchers handed to ``solver.build_guider``. Not covered: the Detailer, the ``gpu`` and ``local`` suites,
ComfyUI's own delegate clone for conds with hooks on a dynamic patcher (``comfy/samplers.py:1204``; ``adapter.py`` only warns),
and the throwaway parent of a cached output of the deprecated ``MossPhotonLoRAProfile`` (``nodes/lora.py``, benign without a
cycle).

``tests/run_tests.py`` calls ``install()`` for the integration suite only (a test file run on its own is the bare process). It
touches nothing the sampling computes: no value, no patch, no order.
"""
_pinned = {}
_installed = False


def install():
    global _installed
    if _installed:
        return
    from moss_photon.core import solver
    real_execute, real_build_guider = solver.execute, solver.build_guider

    def execute(request, *args, **kwargs):
        darkroom = getattr(request, "darkroom", None)
        for model in (getattr(request, "model", None), getattr(request, "finish_model", None),
                      getattr(darkroom, "finish_model", None)):
            if model is not None:
                _pinned[id(model)] = model
        return real_execute(request, *args, **kwargs)

    def build_guider(patcher, *args, **kwargs):
        patcher.emulated_server_cycle = patcher
        return real_build_guider(patcher, *args, **kwargs)

    solver.execute, solver.build_guider = execute, build_guider
    _installed = True
