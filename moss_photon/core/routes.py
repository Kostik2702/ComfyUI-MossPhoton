"""``GET /mossphoton/presets/{id}/resolved``: the effective settings of a preset before the run (design 4.9, LD7).

The handler is PURE: no model, no GPU, no latent, no file besides the preset catalog. ``handle_resolved`` maps a
preset id and a query to ``(status, payload)``; ``register_routes`` wires it to the running ComfyUI server and never
raises. Imports: ``presets`` and ``errors``; ``aiohttp`` only inside ``register_routes``.
"""
import logging
import re
import sys

from . import errors, presets

__all__ = ["ROUTE_PATH", "QUERY_PARAMS", "NODE_PARAM", "ADVANCED_NODE", "handle_resolved", "build_handler",
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


def handle_resolved(preset_id, query=None):
    """``(http_status, payload)`` for one request; never raises for user input.

    200: the payload of ``presets.effective_view`` (design 4.9). 400: ``{"error": Issue}`` for an unparseable query
    value or an override outside the preset bounds (``invalid_override``). 404: ``invalid_preset`` for an unknown id.
    500: ``invalid_preset_file`` when the installed catalog or this preset's file is broken. ``query`` is any mapping
    ``name -> str``; ``node=MossPhotonSamplerAdvanced`` resolves with the SLG-on-turbo override allowed.
    """
    overrides, problems = _parse_query(query or {})
    if problems:
        return 400, {"error": problems[0].to_dict()}
    try:
        allow_slg = (query or {}).get(NODE_PARAM) == ADVANCED_NODE
        return 200, presets.effective_view(preset_id, overrides or None, allow_slg_override=allow_slg)
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
