"""Skip-layer guidance (SLG) for MOSS: parsing, validation and the uncond-only closure (design 1.5, D3).

``apply_slg`` re-implements the closure of ``comfy_extras.nodes_slg.SkipLayerGuidanceDiTSimple`` (CUI/comfy_extras/
nodes_slg.py:132-156): only the unconditional pass skips the chosen single blocks, inside the sigma window of the
LIVE ``model_sampling``. The incoming MODEL is never written to. Imports: stdlib and ``errors``; comfy lazily.
"""
import re

from . import errors

__all__ = [
    "SLG_MODES", "SKIP_BLOCK_NAME", "parse_layers", "validate_slg", "slg_sigma_window", "has_external_cond_hook",
    "apply_slg", "describe_slg",
]

SLG_MODES = ("off", "uncond_skip")
# MOSS has one kind of block; the native node addresses them as "single_block" (INV 4.2).
SKIP_BLOCK_NAME = "single_block"


def parse_layers(spec):
    """Layer indices from a widget string or a sequence: sorted, unique ``list[int]``.

    A string yields every run of digits in it, like ``re.findall(r'\\d+', text)`` of the native node ("22", "22, 24",
    "[22,24]", "22 24" -> the integers; "" -> ``[]``; a minus sign is ignored as in the native node). A sequence must
    hold ``int`` values (``bool`` and floats raise ``ValueError``: a programmer error, the widget is a string).
    """
    if spec is None:
        return []
    if isinstance(spec, str):
        return sorted({int(token) for token in re.findall(r"\d+", spec)})
    out = set()
    for value in spec:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("layer indices must be integers, got %r" % (value,))
        out.add(value)
    return sorted(out)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_slg(mode, layers, start, end, *, num_layers, node):
    """Pure checks of an SLG setting; returns ``list[Issue]`` (``invalid_override``).

    ``mode`` is ``off`` or ``uncond_skip`` (anything else is an error). ``off`` needs nothing else. ``uncond_skip``
    needs at least one layer, every layer an integer in ``0 .. num_layers - 1`` (``num_layers`` ``None`` skips the
    upper bound), and a window ``0 <= start <= end <= 1`` (fractions of the schedule). A layer the model does not
    have is an error because SLG would silently do nothing (MP-5 R5: SLG 22 on a 4-layer model equals no SLG).
    """
    def bad(param, message, fix, **detail):
        return errors.make_issue("invalid_override", node, param, message, fix, detail=detail or None)

    if mode not in SLG_MODES:
        return [bad("slg_mode", "slg_mode = %r, expected one of %s" % (mode, ", ".join(SLG_MODES)),
                    "choose off or uncond_skip", value=mode)]
    if mode == "off":
        return []
    out = []
    layers = list(layers)
    if not layers:
        out.append(bad("slg_layers", "slg_mode uncond_skip needs at least one layer, got none",
                       "enter the layer indices to skip, for example 22, or set slg_mode off"))
    for layer in layers:
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            out.append(bad("slg_layers", "layer %r is not a non-negative integer" % (layer,),
                           "use layer indices counted from 0", layer=layer))
        elif num_layers is not None and layer >= num_layers:
            out.append(bad("slg_layers",
                           "layer %d does not exist: the model has %d layers (0..%d), so SLG would do nothing"
                           % (layer, num_layers, num_layers - 1),
                           "use a layer index below %d" % num_layers, layer=layer, num_layers=num_layers))
    for name, value in (("slg_start", start), ("slg_end", end)):
        if not _is_number(value) or not 0.0 <= value <= 1.0:
            out.append(bad(name, "%s = %r is outside 0..1" % (name, value), "use a fraction between 0 and 1",
                           value=value))
    if _is_number(start) and _is_number(end) and start > end:
        out.append(bad("slg_start", "slg_start %r must be <= slg_end %r" % (start, end),
                       "lower slg_start or raise slg_end", start=start, end=end))
    return out


def slg_sigma_window(model_sampling, start, end):
    """``(sigma_end, sigma_start)``: the sigma interval in which SLG is active.

    ``model_sampling.percent_to_sigma(percent)`` of the LIVE model (INV 4.2), as the native node: ``start`` and
    ``end`` are fractions of the schedule, so ``percent_to_sigma(end) <= sigma <= percent_to_sigma(start)``.
    """
    return float(model_sampling.percent_to_sigma(end)), float(model_sampling.percent_to_sigma(start))


def has_external_cond_hook(model):
    """True when the MODEL already carries a non-empty ``sampler_calc_cond_batch_function``.

    The slot is single (CUI/comfy/model_patcher.py:654-655): a native SLG node upstream would be overwritten.
    """
    return bool((getattr(model, "model_options", None) or {}).get("sampler_calc_cond_batch_function"))


def apply_slg(model, *, layers, start, end):
    """A CLONE of ``model`` whose unconditional pass skips ``layers``; the original is untouched (D3).

    With no layers the input object itself is returned (nothing to install). Otherwise the clone gets the native
    uncond-only closure: inside ``sigma_end <= sigma <= sigma_start`` and when an unconditional branch exists
    (CFG != 1) the conditional pass runs normally and the unconditional pass runs with ``single_block`` ``layer``
    replaced by identity; everywhere else the stock ``calc_cond_batch``. Bit-equal to the native node
    (``tests/integration/test_slg_equals_native.py``).
    """
    layers = parse_layers(layers)
    if not layers:
        return model
    import comfy.model_patcher
    import comfy.samplers

    def skip(args, extra_args):
        return args

    sigma_end, sigma_start = slg_sigma_window(model.get_model_object("model_sampling"), start, end)

    def calc_cond_batch_function(args):
        x = args["input"]
        net = args["model"]
        conds = args["conds"]
        sigma = args["sigma"]
        model_options = args["model_options"]
        slg_model_options = model_options.copy()
        for layer in layers:
            slg_model_options = comfy.model_patcher.set_model_options_patch_replace(
                slg_model_options, skip, "dit", SKIP_BLOCK_NAME, layer)
        cond, uncond = conds
        sigma_value = sigma[0].item()
        if sigma_end <= sigma_value <= sigma_start and uncond is not None:
            cond_out, _ = comfy.samplers.calc_cond_batch(net, [cond, None], x, sigma, model_options)
            _, uncond_out = comfy.samplers.calc_cond_batch(net, [None, uncond], x, sigma, slg_model_options)
            return [cond_out, uncond_out]
        return comfy.samplers.calc_cond_batch(net, conds, x, sigma, model_options)

    clone = model.clone()
    clone.set_model_sampler_calc_cond_batch_function(calc_cond_batch_function)
    return clone


def describe_slg(mode, layers, start, end, window):
    """Report block ``sampling.slg``: ``{mode, layers, start, end, sigma_window: [sigma_end, sigma_start]}``."""
    return {"mode": mode, "layers": parse_layers(layers), "start": float(start), "end": float(end),
            "sigma_window": [float(window[0]), float(window[1])]}
