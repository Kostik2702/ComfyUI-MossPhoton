// MossPhoton sampler panel (MP-18; design 4.9, SPEC 6.2 / 6.3).
//
// For MossPhotonSampler and MossPhotonSamplerAdvanced:
//   1. fills the preset_pin widget ("<id>@<version>#<sha8>", the installed pin) when the node is created and
//      when the preset combo changes;
//   2. shows a read-only "effective settings" panel fed by GET /mossphoton/presets/{id}/resolved.
// Nothing is computed here. The route runs the same Python as the sampler (presets.effective_view), and the pack
// works with this file switched off: the server then warns preset_unpinned and still runs the installed version.
//
// Pin policy: an empty pin is filled. A pin that is exactly the installed pin of some preset is the value this
// script (or a previous session of it) wrote, so it follows the combo. Any other non-empty pin was typed by the
// user: it is never changed, a mismatch with the selected preset is shown in the panel instead.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASSES = ["MossPhotonSampler", "MossPhotonSamplerAdvanced"];
const PIN_RE = /^([a-z0-9_]+)@([0-9]+[.][0-9]+[.][0-9]+)(?:#([0-9a-f]{8}))?$/;
const DEBOUNCE_MS = 150;
const LINE_H = 15;

// Every payload field this file reads. tests/unit/web/test_route_fields.py parses this literal and checks it
// against the real route payload, so a renamed server field fails a test instead of blanking the panel.
// "[]" = every item of a list.
export const READ_FIELDS = [
    "preset.id",
    "preset.version",
    "preset.pin",
    "preset.label",
    "preset.status",
    "preset.summary",
    "preset.recommended_final",
    "effective.sampling.steps",
    "effective.sampling.sampler",
    "effective.sampling.scheduler",
    "effective.sampling.shift",
    "effective.sampling.cfg",
    "effective.sampling.slg_mode",
    "effective.sampling.slg_layers",
    "effective.sampling.slg_start",
    "effective.sampling.slg_end",
    "effective.budget.max_forwards",
    "forwards.branches_per_step",
    "forwards.expected_branch_forwards",
    "customized",
    "requirements[].kind",
    "requirements[].severity",
    "warnings[].message",
];
// Extra requirement fields by kind (read only for that kind).
export const REQUIREMENT_FIELDS = {
    requires_lora_role: "role",
    requires_block: "block",
    recommends_input: "input",
    forbids_source_latent: null,
    limits_lora_roles: "roles",
};
// Turbo presets only: payload.override_bounds[<key>] carries these (checked by the same test).
export const BOUNDS_KEYS = ["steps", "cfg"];
export const BOUNDS_FIELDS = ["min", "max", "verified_min", "verified_max"];
// Fields of an error body (HTTP 400 / 404 / 500).
export const ERROR_FIELDS = ["error.message", "error.fix"];

const STATUS_COLOR = {
    stable: "#199e70", candidate: "#3987e5", experimental: "#e0a33a", draft: "#8a9096", preview_only: "#8a9096",
};

// ---------------------------------------------------------------- pure helpers

export const parsePin = (pin) => {
    const m = PIN_RE.exec(String(pin ?? "").trim());
    return m ? { id: m[1], version: m[2], sha8: m[3] ?? null } : null;
};

const fmt = (v) => (typeof v === "number" ? String(Number(v.toPrecision(6))) : String(v));

// "reference@1.0.0#1936562e" vs the installed pin; a pin without sha8 only compares id and version.
export function pinMatches(pin, installed) {
    const a = parsePin(pin);
    const b = parsePin(installed);
    if (!a || !b) return false;
    return a.id === b.id && a.version === b.version && (a.sha8 === null || a.sha8 === b.sha8);
}

// Query string of the overrides the node's own widgets would send (same sentinels as nodes/sampler.py).
export function overridesQuery(values, sigmasLinked) {
    const q = new URLSearchParams();
    if (Number(values.cfg) > 0) q.set("cfg", String(values.cfg));
    if (Number(values.steps) > 0 && !sigmasLinked) q.set("steps", String(Math.trunc(values.steps)));
    if (values.slg_mode && values.slg_mode !== "from_preset") q.set("slg_mode", values.slg_mode);
    if (String(values.slg_layers ?? "").trim() !== "") q.set("slg_layers", String(values.slg_layers));
    if (Number(values.slg_start) >= 0) q.set("slg_start", String(values.slg_start));
    if (Number(values.slg_end) >= 0) q.set("slg_end", String(values.slg_end));
    return q.toString();
}

// The route needs the node class to resolve like the node's own run does (an SLG override on a turbo preset is
// allowed in MossPhotonSamplerAdvanced only). Only a request that carries overrides gets it.
export const NODE_QUERY_PARAM = "node";
export function withNodeClass(query, nodeClass) {
    if (!query || !nodeClass) return query;
    const q = new URLSearchParams(query);
    q.set(NODE_QUERY_PARAM, String(nodeClass));
    return q.toString();
}

function requirementText(r) {
    const extra = REQUIREMENT_FIELDS[r.kind] ? r[REQUIREMENT_FIELDS[r.kind]] : null;
    const what = {
        requires_lora_role: `${extra ?? "?"} LoRA`,
        requires_block: `${extra ?? "?"} block`,
        recommends_input: `${extra ?? "?"} input`,
        forbids_source_latent: "no source latent",
        limits_lora_roles: `LoRA roles limited to ${[].concat(extra ?? []).join(", ")}`,
    }[r.kind] ?? r.kind;
    const verb = r.kind === "recommends_input" ? "recommended"
        : r.kind === "limits_lora_roles" ? "" : "required";
    return `${what} ${verb}${r.severity === "warning" ? " (warning)" : ""}`.replace(/ +/g, " ").trim();
}

function pinLine(pinNote) {
    const t = {
        filled: `preset_pin filled: ${pinNote.pin}`,
        moved: `preset_pin moved ${pinNote.from} -> ${pinNote.pin}`,
        mismatch: `preset_pin ${pinNote.pin} is for ${pinNote.pinId}, preset is ${pinNote.presetId}: not changed, `
            + "the run will stop with preset_version_mismatch",
        older: `preset_pin ${pinNote.pin} differs from installed ${pinNote.installed}: not changed, `
            + "the run will stop with preset_version_mismatch",
        malformed: `preset_pin ${pinNote.pin} is malformed: not changed`,
    }[pinNote.kind];
    if (!t) return null;
    return { text: t, tone: pinNote.kind === "filled" || pinNote.kind === "moved" ? "" : "error" };
}

// Lines of the panel: [{text, tone}], tone in "" | "warn" | "error". `view` is the 200 body, `pinNote` what the pin
// logic decided ({kind: "filled"|"moved"|"mismatch"|"older"|"kept", ...} or null).
export function panelLines(view, pinNote, error) {
    const lines = [];
    if (error) {
        lines.push({ text: error.message ?? "request failed", tone: "error" });
        if (error.fix) lines.push({ text: error.fix, tone: "" });
        const el = pinNote && pinLine(pinNote);
        if (el) lines.push(el);
        return lines;
    }
    const p = view.preset;
    const s = view.effective.sampling;
    const label = p.label ?? p.status;
    const flags = [label];
    if (view.customized) flags.push("customized");
    lines.push({ text: `${p.id}  [${flags.join(", ")}]  ${p.pin}`, tone: "", label });
    const cfgNote = view.effective.sampling.cfg === view.__presetCfg ? "" : " (override)";
    lines.push({
        text: `${s.sampler} / ${s.scheduler}  steps ${fmt(s.steps)}  shift ${fmt(s.shift)}  cfg ${fmt(s.cfg)}${cfgNote}`,
        tone: "",
    });
    const slg = s.slg_mode === "off" || !s.slg_layers.length
        ? "SLG off"
        : `SLG ${s.slg_mode} layers ${s.slg_layers.join(",")} window ${fmt(s.slg_start)}-${fmt(s.slg_end)}`;
    lines.push({ text: slg, tone: "" });
    const cap = view.effective.budget.max_forwards;
    lines.push({
        text: `forwards ${view.forwards.expected_branch_forwards} (${view.forwards.branches_per_step}/step)`
            + (cap === null || cap === undefined ? "" : `  budget max ${cap}`),
        tone: "",
    });
    for (const r of view.requirements ?? []) {
        lines.push({ text: `needs: ${requirementText(r)}`, tone: r.severity === "error" ? "error" : "warn" });
    }
    for (const w of view.warnings ?? []) lines.push({ text: w.message, tone: "warn" });
    const pl = pinNote && pinLine(pinNote);
    if (pl) lines.push(pl);
    return lines;
}

// Widget limits from payload.override_bounds, or null. Only the maximum is applied: 0 stays the "preset value"
// sentinel of the steps and cfg widgets (cfg 0 = the preset's own cfg), so their minimum is never raised.
export function boundsWidgetLimits(bounds, name) {
    const b = bounds?.[name];
    return b && typeof b.max === "number" ? { max: b.max } : null;
}

// Lines flagging the current widget values against payload.override_bounds. 0 (or unset) = the preset value: never flagged.
export function boundsLines(bounds, values) {
    const lines = [];
    if (!bounds) return lines;
    for (const name of BOUNDS_KEYS) {
        const b = bounds[name];
        const v = Number(values[name]);
        if (!b || !(v > 0)) continue;
        if (v < b.min || v > b.max) {
            lines.push({ text: `${name} ${fmt(v)} is outside the allowed ${fmt(b.min)} to ${fmt(b.max)} of this preset: `
                + "the run will stop", tone: "error" });
        } else if (v < b.verified_min || v > b.verified_max) {
            lines.push({ text: `${name} ${fmt(v)} is outside the verified ${fmt(b.verified_min)} to `
                + `${fmt(b.verified_max)}: the run goes on with a warning`, tone: "warn" });
        }
    }
    return lines;
}

function applyBoundsToWidgets(node, bounds) {
    for (const name of BOUNDS_KEYS) {
        const w = widgetOf(node, name);
        if (!w?.options) continue;
        if (w.__mpMax === undefined) w.__mpMax = w.options.max;
        const limit = boundsWidgetLimits(bounds, name);
        w.options.max = limit ? limit.max : w.__mpMax;
    }
}

// ---------------------------------------------------------------- route access

// Short TTL (a package update changes the pin), only successful answers are kept (a failure is retried at once), and
// the whole cache is dropped when the server restarts ("reconnected") or reports a status change.
export const CACHE_TTL_MS = 15000;
const cache = new Map();
function getView(presetId, query) {
    const url = `/mossphoton/presets/${encodeURIComponent(presetId)}/resolved${query ? `?${query}` : ""}`;
    const hit = cache.get(url);
    if (hit && Date.now() - hit.at < CACHE_TTL_MS) return hit.promise;
    const promise = fetch(url)
        .then(async (r) => ({ ok: r.ok, body: await r.json().catch(() => ({})) }))
        .catch((e) => ({ ok: false, body: { error: { message: `route unavailable: ${e}` } } }))
        .then((res) => {
            if (!res.ok) cache.delete(url);
            return res;
        });
    cache.set(url, { at: Date.now(), promise });
    return promise;
}

function invalidateRoutes(refreshPanels) {
    cache.clear();
    if (!refreshPanels) return;
    for (const node of app.graph?._nodes ?? []) {
        if (NODE_CLASSES.includes(node.comfyClass ?? node.type) && node.__mpPanel) schedule(node);
    }
}

// ---------------------------------------------------------------- node side

const widgetOf = (node, name) => node.widgets?.find((w) => w.name === name);
const valueOf = (node, name) => widgetOf(node, name)?.value;

function collectValues(node) {
    const out = {};
    for (const n of ["cfg", "steps", "slg_mode", "slg_layers", "slg_start", "slg_end"]) out[n] = valueOf(node, n);
    return out;
}

const sigmasLinked = (node) => (node.inputs ?? []).some((i) => i.name === "sigmas" && i.link != null);

// Decide what to do with the pin widget. Returns {write: string|null, note}.
async function decidePin(current, presetId, installedPin) {
    const cur = String(current ?? "").trim();
    if (cur === "") return { write: installedPin, note: { kind: "filled", pin: installedPin } };
    const parsed = parsePin(cur);
    if (!parsed) return { write: null, note: { kind: "malformed", pin: cur } };
    if (parsed.id === presetId) {
        if (pinMatches(cur, installedPin)) return { write: null, note: null };
        return { write: null, note: { kind: "older", pin: cur, installed: installedPin } };
    }
    // another preset's pin: ours (installed pin of that preset) follows the combo, a typed one is only reported
    const other = await getView(parsed.id, "");
    if (other.ok && other.body.preset && pinMatches(cur, other.body.preset.pin)) {
        return { write: installedPin, note: { kind: "moved", from: cur, pin: installedPin } };
    }
    return { write: null, note: { kind: "mismatch", pin: cur, pinId: parsed.id, presetId } };
}

const TONE = { "": "var(--input-text, #ddd)", warn: "#e0a33a", error: "#e5604d" };

function render(node, lines) {
    const w = node.__mpPanel;
    const box = w.element;
    box.replaceChildren();
    for (const l of lines) {
        const row = document.createElement("div");
        row.textContent = l.text;
        row.style.cssText = `color:${TONE[l.tone] ?? TONE[""]};white-space:normal;word-break:break-word;`;
        if (l.label) row.style.borderLeft = `3px solid ${STATUS_COLOR[l.label] ?? "#8a9096"}`, row.style.paddingLeft = "5px";
        box.appendChild(row);
    }
    const h = Math.max(1, lines.length + 1) * LINE_H + 8;
    if (w.mpHeight !== h) {
        w.mpHeight = h;
        const min = node.computeSize?.();
        if (min) node.setSize([Math.max(node.size[0], min[0]), Math.max(node.size[1], min[1])]);
    }
    node.setDirtyCanvas?.(true, true);
}

async function refresh(node) {
    if (!node.__mpPanel) return;
    const seq = (node.__mpSeq = (node.__mpSeq ?? 0) + 1);
    const presetId = valueOf(node, "preset");
    if (!presetId) return;
    const base = await getView(presetId, "");
    if (seq !== node.__mpSeq) return;
    if (!base.ok) return render(node, panelLines(null, null, base.body.error ?? { message: "no data" }));
    if (!base.body.preset) return render(node, panelLines(null, null, { message: "unexpected route answer" }));
    const bounds = base.body.override_bounds ?? null;
    applyBoundsToWidgets(node, bounds);
    const values = collectValues(node);
    if (sigmasLinked(node)) values.steps = 0; // steps is ignored with a sigmas input
    const extra = boundsLines(bounds, values);
    const query = withNodeClass(overridesQuery(collectValues(node), sigmasLinked(node)), node.comfyClass ?? node.type);
    const eff = query ? await getView(presetId, query) : base;
    if (seq !== node.__mpSeq) return;

    const pinWidget = widgetOf(node, "preset_pin");
    let note = null;
    if (pinWidget) {
        const decision = await decidePin(pinWidget.value, presetId, base.body.preset.pin);
        if (seq !== node.__mpSeq) return;
        if (decision.write !== null) {
            pinWidget.value = decision.write;
            pinWidget.callback?.(decision.write);
        }
        note = decision.note;
    }
    if (!eff.ok) return render(node, panelLines(null, note, eff.body.error ?? { message: "no data" }).concat(extra));
    eff.body.__presetCfg = base.body.effective.sampling.cfg;
    render(node, panelLines(eff.body, note).concat(extra));
}

function schedule(node) {
    clearTimeout(node.__mpTimer);
    node.__mpTimer = setTimeout(() => refresh(node).catch((e) => console.warn("[MossPhoton] panel:", e)), DEBOUNCE_MS);
}

function hook(node, name) {
    const w = widgetOf(node, name);
    if (!w || w.__mpHooked) return;
    w.__mpHooked = true;
    const prev = w.callback;
    w.callback = function () {
        const r = prev?.apply(this, arguments);
        schedule(node);
        return r;
    };
}

function attach(node) {
    if (node.__mpPanel) return;
    const element = document.createElement("div");
    element.style.cssText = `font:11px/${LINE_H}px var(--comfy-font, sans-serif);box-sizing:border-box;`
        + "padding:3px 4px;overflow:hidden;pointer-events:none;font-variant-numeric:tabular-nums;";
    const w = node.addDOMWidget("mp_effective", "mp_effective", element, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => w.mpHeight,
        getMaxHeight: () => w.mpHeight,
    });
    w.serialize = false; // never part of widgets_values: saved workflows keep their positional values
    w.mpHeight = 4 * LINE_H + 8;
    node.__mpPanel = w;
    for (const n of ["preset", "cfg", "steps", "slg_mode", "slg_layers", "slg_start", "slg_end", "preset_pin"]) {
        hook(node, n);
    }
    // widget values of a loaded workflow are applied after nodeCreated: wait one tick, loadedGraphNode refreshes too
    schedule(node);
}

// After a server restart the installed pin may differ: drop the cache and re-read every open panel.
api.addEventListener("reconnected", () => invalidateRoutes(true));
api.addEventListener("status", () => invalidateRoutes(false));

app.registerExtension({
    name: "MossPhoton.EffectiveSettings",
    nodeCreated(node) {
        if (NODE_CLASSES.includes(node.comfyClass ?? node.type)) attach(node);
    },
    loadedGraphNode(node) {
        if (NODE_CLASSES.includes(node.comfyClass ?? node.type)) {
            attach(node);
            schedule(node);
        }
    },
});
