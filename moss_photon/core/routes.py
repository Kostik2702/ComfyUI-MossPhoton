"""``GET /mossphoton/presets/{id}/resolved``: the effective settings of a preset before the run (design 4.9, LD7).

The handler is PURE: no model, no GPU, no latent, no file besides the preset catalog. ``handle_resolved`` maps a
preset id and a query to ``(status, payload)``; ``register_routes`` wires it to the running ComfyUI server and never
raises. Imports: ``presets`` and ``errors``; ``aiohttp`` only inside ``register_routes``.
"""
import logging
import re
import sys

from . import errors, presets

__all__ = ["ROUTE_PATH", "QUERY_PARAMS", "DARKROOM_PARAMS", "NODE_PARAM", "ADVANCED_NODE", "handle_resolved", "build_handler",
           "register_routes"]

_log = logging.getLogger("MossPhoton")

ROUTE_PATH = "/mossphoton/presets/{id}/resolved"
QUERY_PARAMS = ("cfg", "steps", "slg_mode", "slg_layers", "slg_start", "slg_end")
# The class of the node that asks (NEW-1): the solver lets an SLG override stand on a turbo preset in the Advanced
# node only, and the panel must show what that node's run will do. Anything else counts as the simple node.
NODE_PARAM = "node"
ADVANCED_NODE = "MossPhotonSamplerAdvanced"
_NODE = "MossPhotonSampler"
_FLOAT_PARAMS = ("cfg", "slg_start", "slg_end")
# The Advanced darkroom knobs (MP-71), by widget name; only a request that sets one carries it (sentinels are left out).
# ``switch_model`` is the widget of the engine's ``switch``; the three combos are ``on`` / ``off``.
DARKROOM_PARAMS = ("schedule", "compose_cfg", "compose_slg", "r_c", "r_k", "w_compose", "w_commit", "reexpose", "sigma_r",
                   "r_f", "finish_cfg", "eta", "switch_model")
_DARKROOM_ENGINE_NAME = {"switch_model": "switch"}
_DARKROOM_FLAGS = ("compose_slg", "reexpose", "switch_model")


def _bad_override(param, message):
    return errors.make_issue("invalid_override", _NODE, param, message,
                             "send %s as a number inside the preset bounds, or leave it out" % param)


def _parse_query(query):
    """Overrides dict from query parameters; ``ValueError``-free: returns ``(overrides, [Issue])``."""
    overrides, problems = {}, []
    for name in QUERY_PARAMS:
        raw = query.get(name)
        if raw is None or raw == "":
            continue
        try:
            if name == "steps":
                overrides[name] = int(raw)
            elif name in _FLOAT_PARAMS:
                value = float(raw)
                if value != value or value in (float("inf"), float("-inf")):
                    raise ValueError("not finite")
                overrides[name] = value
            elif name == "slg_layers":
                overrides[name] = sorted({int(token) for token in re.findall(r"\d+", raw)})
            else:
                overrides[name] = str(raw)
        except (TypeError, ValueError):
            problems.append(_bad_override(name, "%s = %r is not a valid value" % (name, raw)))
    return overrides, problems


def _parse_darkroom(query):
    """``(knobs by engine name, [Issue])`` from the darkroom query parameters."""
    knobs, problems = {}, []
    for name in DARKROOM_PARAMS:
        raw = query.get(name)
        if raw is None or raw == "":
            continue
        engine = _DARKROOM_ENGINE_NAME.get(name, name)
        if name in _DARKROOM_FLAGS and raw in ("on", "off"):
            knobs[engine] = raw == "on"
        elif name == "schedule" and isinstance(raw, str):
            knobs[engine] = raw
        elif name not in _DARKROOM_FLAGS and name != "schedule":
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = float("nan")
            if value != value or value in (float("inf"), float("-inf")):
                problems.append(_bad_override(name, "%s = %r is not a valid number" % (name, raw)))
            else:
                knobs[engine] = value
        else:
            problems.append(_bad_override(name, "%s = %r is not a valid value" % (name, raw)))
    return knobs, problems


def handle_resolved(preset_id, query=None):
    """``(http_status, payload)`` for one request; never raises for user input.

    200: the payload of ``presets.effective_view`` (design 4.9; a value outside what a run can use is moved or dropped
    with the warning ``setting_adjusted`` / ``override_outside_bounds`` in ``warnings``, as the run does, MP-84).
    400: ``{"error": Issue}`` for an unparseable query value (``invalid_override``). 404: ``invalid_preset`` for an unknown id.
    500: ``invalid_preset_file`` when the installed catalog or this preset's file is broken. ``query`` is any mapping
    ``name -> str``; ``node=MossPhotonSamplerAdvanced`` resolves with the SLG-on-turbo override allowed.
    """
    overrides, problems = _parse_query(query or {})
    knobs, knob_problems = _parse_darkroom(query or {})
    problems = problems + knob_problems
    if problems:
        return 400, {"error": problems[0].to_dict()}
    try:
        allow_slg = (query or {}).get(NODE_PARAM) == ADVANCED_NODE
        node = ADVANCED_NODE if allow_slg else _NODE
        return 200, presets.effective_view(preset_id, overrides or None, allow_slg_override=allow_slg,
                                           darkroom_knobs=knobs or None, node=node)
    except errors.UnknownPreset as exc:
        return 404, {"error": exc.to_dict()}
    except errors.InvalidOverride as exc:
        return 400, {"error": exc.to_dict()}
    except errors.MossPhotonError as exc:
        return 500, {"error": exc.to_dict()}


def build_handler(json_response):
    """The aiohttp request handler, with ``json_response(payload, status=...)`` injected (``aiohttp.web`` in
    production, a stub in tests)."""
    async def resolved(request):
        status, payload = handle_resolved(request.match_info.get("id", ""), dict(request.rel_url.query))
        return json_response(payload, status=status)
    return resolved


_registered = False


def register_routes():
    """Register the GET route on the running ``server.PromptServer``; ``True`` once registered.

    Returns ``False`` (never raises) when no ComfyUI server module is loaded, when ``PromptServer.instance`` does
    not exist yet, when aiohttp is missing or when anything else goes wrong; a second call is a no-op returning
    ``True``. ASSUMED: ``server`` is looked up in ``sys.modules`` instead of imported (ComfyUI imports it before it
    loads custom nodes), so a plain interpreter or a test process never pulls the whole server in.
    """
    global _registered
    try:
        if _registered:
            return True
        server = sys.modules.get("server")
        instance = getattr(getattr(server, "PromptServer", None), "instance", None)
        routes = getattr(instance, "routes", None)
        if routes is None:
            return False
        from aiohttp import web
        routes.get(ROUTE_PATH)(build_handler(web.json_response))
        _registered = True
        return True
    except Exception as exc:  # noqa: BLE001 - the pack works without the route
        _log.warning("[MossPhoton] route %s not registered: %s", ROUTE_PATH, exc)
        return False
