"""
ComfyUI-MossPhoton
ComfyUI nodes for text-to-image with MOSS-Image (SFT): a sampler that targets parity with the native graph, driven by
versioned presets (including 8-step DMD2 turbo presets), a LoRA profile that stacks adapters without leaking them between
jobs, a prompt node that never rewrites the user's text, an empty-latent and a schedule node, and diagnostics and save
nodes that record what really ran. Standard library, torch and numpy only; MIT licence.
"""
import importlib
import logging

_log = logging.getLogger("MossPhoton")

# Node modules (relative to the moss_photon package) to load, one by one: the 7 modules of the 8 v1 nodes
# (design 1.0; moss_photon.nodes.sampler holds both samplers). A module that does not exist yet only logs
# the "module ... disabled" warning below.
_NODE_MODULES = [
    "moss_photon.nodes.latent", "moss_photon.nodes.prompt", "moss_photon.nodes.lora",
    "moss_photon.nodes.schedule", "moss_photon.nodes.sampler",
    "moss_photon.nodes.diagnostics", "moss_photon.nodes.save",
]


def _load_node_modules(module_names, package=__name__):
    """Import each module in isolation; a failing one is logged and skipped.

    Returns (class_mappings, display_name_mappings). Never raises for a
    broken module, so one bad node cannot take the whole package down.
    """
    classes = {}
    names = {}
    for mod_name in module_names:
        try:
            mod = importlib.import_module("." + mod_name, package)
            classes.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
            names.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            _log.warning("[MossPhoton] module %s disabled: %s", mod_name, exc)
            print("[MossPhoton] module %s disabled: %s" % (mod_name, exc))
    return classes, names


_NODES, _NAMES = _load_node_modules(_NODE_MODULES)


def _register_routes(package=__name__):
    """GET /mossphoton/presets/{id}/resolved (design 4.9); guarded, the pack works without it."""
    try:
        routes = importlib.import_module(".moss_photon.core.routes", package)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.endswith("core.routes"):
            _log.debug("[MossPhoton] routes module not present yet")
        else:
            _log.warning("[MossPhoton] routes disabled: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001 - routes are optional
        _log.warning("[MossPhoton] routes disabled: %s", exc)
        return False
    try:
        return bool(routes.register_routes())
    except Exception as exc:  # noqa: BLE001
        _log.warning("[MossPhoton] routes registration failed: %s", exc)
        return False


_register_routes()

NODE_CLASS_MAPPINGS = {**_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_NAMES}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
