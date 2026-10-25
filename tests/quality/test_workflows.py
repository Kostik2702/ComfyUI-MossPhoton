# -*- coding: utf-8 -*-
"""Quality checks for the example workflows and the v1 documentation (MP-13; design 6.8).

Offline: the workflows are validated against ``object_info_snapshot.json``, a recording of GET /object_info for the eight
MossPhoton nodes and the native nodes the workflows use (refresh with ``research/refresh_object_info_snapshot.py``).
No server, no ComfyUI import, no generation.

    <embedded python> tests/quality/test_workflows.py            (or: -m unittest tests.quality.test_workflows)

``tests/run_tests.py --suite quality`` runs this module (it is also part of ``--suite all``); it can be run directly too.
Env ``MOSS_PHOTON_WORKFLOWS_DIR`` points the checks at another folder (used to prove the checks go red).

What is checked per workflow: every node type exists (or is a frontend-only Note); ``widgets_values`` have exactly the
widgets the server declares, in order (seed's control_after_generate companion included), each value inside the declared
COMBO list / min / max / type; every link joins an output slot to an input slot of the same declared type and the node
slot lists agree with the link table; required link inputs are connected; the preset pin equals the installed pin; the
LoRA chain matches the workflow's purpose; a Note states versions, weights, preset status and experimental features.
"""
import copy
import glob
import importlib.util
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(PKG)
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from moss_photon.core import presets as presets_mod  # noqa: E402  (stdlib only)
from tests.quality import snapshot_allowlist  # noqa: E402  (the one list of file names the snapshot may hold)

SNAPSHOT_PATH = os.path.join(HERE, "object_info_snapshot.json")
WORKFLOW_DIR = os.environ.get("MOSS_PHOTON_WORKFLOWS_DIR") or os.path.join(PKG, "workflows")
DOCS_DIR = os.path.join(PKG, "docs")

PRIMITIVES = ("INT", "FLOAT", "STRING", "BOOLEAN")
CONTROL_VALUES = ("fixed", "increment", "decrement", "randomize")

# workflow file -> (preset, LoRA roles in apply order)
EXPECTED = {
    "sft_reference": ("reference", []),
    "character_lora": ("identity_safe", ["character"]),
    "multi_lora": ("identity_safe", ["character", "other"]),
    "turbo_8step": ("turbo_8step", ["distill"]),
    "turbo_identity": ("turbo_identity", ["distill", "character"]),
}

NOTE_REQUIRED = (
    "0.42.0", "8f04fd6ec", "frontend 1.55.18", "models/diffusion_models/", "models/text_encoders/", "models/vae/",
    "Preset:", "Experimental", "Base: not verified", "SFT only",
)

# A claim of a gain in quality (the docs must not make one). Negations are written so that they do not match.
QUALITY_CLAIM = re.compile(r"\b(improv\w*|better|higher|superior|outperform\w*|enhanc\w*)\s+(the\s+)?(photo\w*\s+)?"
                           r"(quality|realism|identity|photorealism|results?)\b", re.IGNORECASE)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _widget_specs(snap_node):
    """[(name, spec)] of the widget inputs in server order, plus the control companions."""
    order = snap_node["input_order"]
    specs = []
    for section in ("required", "optional"):
        for name in order.get(section, []):
            specs.append((name, snap_node["input"][section][name]))
    return specs


def _is_widget(spec):
    return isinstance(spec[0], list) or spec[0] in PRIMITIVES or spec[0] == "COMBO"


def _declared_type(spec):
    return "COMBO" if isinstance(spec[0], list) else spec[0]


def check_value(label, spec, value):
    """Return a list of problems for one widget value."""
    kind = spec[0]
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    problems = []
    if isinstance(kind, list):
        if value not in kind:
            problems.append("%s: %r is not one of the %d declared choices" % (label, value, len(kind)))
        return problems
    if kind == "COMBO":
        choices = options.get("options", [])
        if value not in choices:
            problems.append("%s: %r is not one of the declared choices" % (label, value))
        return problems
    if kind == "BOOLEAN":
        if not isinstance(value, bool):
            problems.append("%s: %r is not a boolean" % (label, value))
    elif kind == "STRING":
        if not isinstance(value, str):
            problems.append("%s: %r is not a string" % (label, value))
    elif kind in ("INT", "FLOAT"):
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not number or (kind == "INT" and not isinstance(value, int)):
            problems.append("%s: %r is not %s" % (label, value, "an integer" if kind == "INT" else "a number"))
        else:
            if "min" in options and value < options["min"]:
                problems.append("%s: %r is below the declared minimum %r" % (label, value, options["min"]))
            if "max" in options and value > options["max"]:
                problems.append("%s: %r is above the declared maximum %r" % (label, value, options["max"]))
    return problems


def validate_workflow(wf, snap):
    """Validate one UI-format workflow against the recorded /object_info. Returns a list of problem strings."""
    problems = []
    virtual = set(snap.get("frontend_virtual_types", []))
    nodes = wf.get("nodes")
    links = wf.get("links")
    if not isinstance(nodes, list) or not nodes:
        return ["workflow has no nodes list"]
    if not isinstance(links, list):
        return ["workflow has no links list"]

    by_id = {}
    for node in nodes:
        if node.get("id") in by_id:
            problems.append("duplicate node id %r" % node.get("id"))
        by_id[node.get("id")] = node

    link_by_id = {}
    for link in links:
        if not (isinstance(link, list) and len(link) == 6):
            problems.append("malformed link %r" % (link,))
            continue
        if link[0] in link_by_id:
            problems.append("duplicate link id %r" % link[0])
        link_by_id[link[0]] = link

    for node in nodes:
        ntype = node.get("type")
        nid = node.get("id")
        tag = "node %s (%s)" % (nid, ntype)
        if ntype in virtual:
            continue
        info = snap["nodes"].get(ntype)
        if info is None:
            problems.append("%s: type is not in /object_info" % tag)
            continue
        specs = _widget_specs(info)
        # widgets_values: widget inputs in server order, control_after_generate companion right after its widget
        expected_values = []
        for name, spec in specs:
            if _is_widget(spec):
                expected_values.append((name, spec, False))
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                if opts.get("control_after_generate"):
                    expected_values.append((name + ".control_after_generate", None, True))
        values = node.get("widgets_values", [])
        if not isinstance(values, list) or len(values) != len(expected_values):
            problems.append("%s: widgets_values has %s entries, the server declares %d widgets" % (
                tag, len(values) if isinstance(values, list) else "no list", len(expected_values)))
        else:
            for (name, spec, is_control), value in zip(expected_values, values):
                if is_control:
                    if value != "fixed":
                        problems.append("%s: %s is %r, the shipped workflows use 'fixed'" % (tag, name, value))
                else:
                    problems.extend("%s: %s" % (tag, p) for p in check_value(name, spec, value))
        # inputs list: every declared input exactly once, declared type, no strangers
        declared = {name: spec for name, spec in specs}
        seen = {}
        for slot in node.get("inputs", []):
            name = slot.get("name")
            if name not in declared:
                problems.append("%s: input %r is not declared by the server" % (tag, name))
                continue
            if name in seen:
                problems.append("%s: input %r listed twice" % (tag, name))
            seen[name] = slot
            if slot.get("type") != _declared_type(declared[name]):
                problems.append("%s: input %r has type %r, declared %r" % (
                    tag, name, slot.get("type"), _declared_type(declared[name])))
        for name, spec in specs:
            if name not in seen:
                problems.append("%s: input %r missing from the node" % (tag, name))
        required = info["input_order"].get("required", [])
        for name in required:
            spec = declared[name]
            if not _is_widget(spec) and name in seen and seen[name].get("link") is None:
                problems.append("%s: required input %r is not connected" % (tag, name))
        outputs = node.get("outputs", [])
        if [o.get("type") for o in outputs] != list(info["output"]):
            problems.append("%s: outputs %s differ from the declared %s" % (
                tag, [o.get("type") for o in outputs], list(info["output"])))

    for link in link_by_id.values():
        lid, oid, oslot, tid, tslot, ltype = link
        tag = "link %s" % lid
        src, dst = by_id.get(oid), by_id.get(tid)
        if src is None or dst is None:
            problems.append("%s: endpoint node does not exist" % tag)
            continue
        souts, dins = src.get("outputs", []), dst.get("inputs", [])
        if not (isinstance(oslot, int) and 0 <= oslot < len(souts)):
            problems.append("%s: output slot %r does not exist on node %s" % (tag, oslot, oid))
            continue
        if not (isinstance(tslot, int) and 0 <= tslot < len(dins)):
            problems.append("%s: input slot %r does not exist on node %s" % (tag, tslot, tid))
            continue
        out_type = souts[oslot].get("type")
        in_slot = dins[tslot]
        dst_info = snap["nodes"].get(dst.get("type"))
        declared_in = None
        if dst_info is not None:
            for name, spec in _widget_specs(dst_info):
                if name == in_slot.get("name"):
                    declared_in = _declared_type(spec)
        if not (out_type == ltype == in_slot.get("type")):
            problems.append("%s: type mismatch, output %r -> link %r -> input %r" % (
                tag, out_type, ltype, in_slot.get("type")))
        elif declared_in is not None and declared_in != ltype:
            problems.append("%s: the server declares input %r as %r, not %r" % (
                tag, in_slot.get("name"), declared_in, ltype))
        if lid not in (souts[oslot].get("links") or []):
            problems.append("%s: not listed in the links of output %d of node %s" % (tag, oslot, oid))
        if in_slot.get("link") != lid:
            problems.append("%s: input %r of node %s points at link %r" % (tag, in_slot.get("name"), tid, in_slot.get("link")))
    for node in nodes:
        for slot in node.get("inputs", []):
            if slot.get("link") is not None and slot["link"] not in link_by_id:
                problems.append("node %s input %r points at unknown link %r" % (node.get("id"), slot.get("name"), slot["link"]))
        for out in node.get("outputs", []):
            for lid in out.get("links") or []:
                if lid not in link_by_id:
                    problems.append("node %s output %r lists unknown link %r" % (node.get("id"), out.get("name"), lid))

    problems.extend(_check_note(nodes, virtual))
    return problems


def _check_note(nodes, virtual):
    notes = [n for n in nodes if n.get("type") in virtual]
    if not notes:
        return ["workflow has no Note node"]
    text = "\n".join(str(v) for n in notes for v in n.get("widgets_values", []))
    problems = ["Note lacks %r" % needle for needle in NOTE_REQUIRED if needle not in text]
    if not re.search(r"candidate|stable|experimental|preview_only", text):
        problems.append("Note does not state the preset status")
    m = QUALITY_CLAIM.search(text)
    if m:
        problems.append("Note makes a quality claim: %r" % m.group(0))
    return problems


def chain_of(wf):
    """(preset, pin, [roles in apply order]) following the sampler's model input through the LoRA profile nodes."""
    nodes = {n["id"]: n for n in wf["nodes"]}
    link_src = {l[0]: l for l in wf["links"]}
    samplers = [n for n in wf["nodes"] if n["type"] == "MossPhotonSampler"]
    if len(samplers) != 1:
        return None, None, None, ["expected exactly one MossPhotonSampler, found %d" % len(samplers)]
    sampler = samplers[0]
    problems = []
    inputs = {s["name"]: s for s in sampler["inputs"]}
    widgets = sampler["widgets_values"]
    preset, pin = widgets[2], widgets[5]  # seed, control, preset, cfg, preview, preset_pin
    roles = []
    node = sampler
    while True:
        slot = {s["name"]: s for s in node["inputs"]}["model"]
        if slot.get("link") is None:
            break
        src = nodes[link_src[slot["link"]][1]]
        if src["type"] != "MossPhotonLoRAProfile":
            break
        roles.append(src["widgets_values"][2])  # lora_name, strength_model, role
        prof_in = {s["name"]: s for s in src["inputs"]}["profile_in"]
        node = src
        if prof_in.get("link") is None:
            break
    roles.reverse()
    if roles:
        link = inputs["identity_profile"].get("link")
        if link is None or link_src[link][1] != _last_profile(wf, nodes, link_src, sampler):
            problems.append("sampler identity_profile is not wired from the last LoRA Profile of the chain")
    return preset, pin, roles, problems


def _last_profile(wf, nodes, link_src, sampler):
    slot = {s["name"]: s for s in sampler["inputs"]}["model"]
    return link_src[slot["link"]][1]


def workflow_files():
    return sorted(glob.glob(os.path.join(WORKFLOW_DIR, "*.json")))


LIVE_KEYS = ("category", "display_name", "input", "input_order", "output", "output_name", "output_node")


def _strip_tooltips(value):
    if isinstance(value, dict):
        return {k: _strip_tooltips(v) for k, v in value.items() if k != "tooltip"}
    if isinstance(value, (list, tuple)):
        return [_strip_tooltips(v) for v in value]
    return value


def live_entry(cls, display_name):
    """The snapshot-shaped entry of a node class, as /object_info would record it (tooltips dropped)."""
    inputs = _strip_tooltips(cls.INPUT_TYPES())
    return {
        "category": cls.CATEGORY,
        "display_name": display_name,
        "input": inputs,
        "input_order": {section: list(items) for section, items in inputs.items()},
        "output": list(cls.RETURN_TYPES),
        "output_name": list(getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES)),
        "output_node": bool(getattr(cls, "OUTPUT_NODE", False)),
    }


class TestSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snap = load_json(SNAPSHOT_PATH)

    def test_snapshot_covers_the_eight_nodes_and_the_native_nodes(self):
        for name in ("MossPhotonSampler", "MossPhotonSamplerAdvanced", "MossPhotonSchedule", "MossPhotonLoRAProfile",
                     "MossPhotonPromptLayout", "MossPhotonEmptyLatent", "MossPhotonDiagnostics",
                     "MossPhotonSaveImage", "UNETLoader", "CLIPLoader", "VAELoader", "VAEDecode"):
            self.assertIn(name, self.snap["nodes"])

    def test_snapshot_holds_no_file_name_outside_the_allowlist(self):
        """The shipped snapshot must not carry the recording machine's model file lists (MP-37)."""
        self.assertEqual(snapshot_allowlist.leaks(self.snap), [])

    def test_every_allowlisted_file_widget_is_in_the_snapshot_with_exactly_the_allowlist(self):
        for (node, widget), allowed in snapshot_allowlist.FILE_WIDGET_ALLOWLIST.items():
            with self.subTest(node=node, widget=widget):
                spec = self.snap["nodes"][node]["input"]["required"][widget]
                self.assertEqual(list(snapshot_allowlist.combo_values(spec)), list(allowed))

    def test_the_leak_check_goes_red_on_a_planted_private_name(self):
        """Control: a name inside an allowlisted widget, and a model file in a widget nobody listed, are both caught."""
        for node, widget, planted in (("MossPhotonLoRAProfile", "lora_name", "private_person_lora.safetensors"),
                                      ("MossPhotonSampler", "preset", "private_person_lora.safetensors")):
            bad = copy.deepcopy(self.snap)
            spec = bad["nodes"][node]["input"]["required"][widget]
            snapshot_allowlist.combo_values(spec).append(planted)
            with self.subTest(node=node):
                self.assertEqual(snapshot_allowlist.leaks(bad), [(node, widget, planted)])

    def test_sanitize_removes_a_planted_private_list(self):
        bad = copy.deepcopy(self.snap)
        snapshot_allowlist.combo_values(bad["nodes"]["UNETLoader"]["input"]["required"]["unet_name"]).append("x_private.safetensors")
        self.assertEqual(len(snapshot_allowlist.leaks(bad)), 1)
        self.assertEqual(snapshot_allowlist.leaks(snapshot_allowlist.sanitize(bad)), [])

    def test_snapshot_records_the_supported_comfyui(self):
        self.assertEqual(self.snap["comfyui_version"], "0.42.0")
        self.assertEqual(self.snap["required_frontend_version"], "1.55.18")

    def test_snapshot_preset_combo_equals_the_installed_catalog(self):
        combo = self.snap["nodes"]["MossPhotonSampler"]["input"]["required"]["preset"][0]
        self.assertEqual(sorted(combo), sorted(presets_mod.catalog_ids()),
                         "the recorded preset list is stale: refresh the snapshot")

    def test_moss_entries_equal_the_live_node_classes(self):
        """Each MossPhoton snapshot entry (name, category, inputs, order, outputs) equals what the classes declare."""
        import moss_photon  # noqa: F401  (package import only; no comfy)
        import importlib
        pkg_init = importlib.import_module("moss_photon.nodes")
        classes, names = {}, {}
        for modname in ("sampler", "schedule", "lora", "prompt", "latent", "diagnostics", "save"):
            mod = importlib.import_module("moss_photon.nodes." + modname)
            classes.update(mod.NODE_CLASS_MAPPINGS)
            names.update(mod.NODE_DISPLAY_NAME_MAPPINGS)
        del pkg_init
        self.assertEqual(sorted(classes), sorted(n for n in self.snap["nodes"] if n.startswith("MossPhoton")))
        self.assertEqual(len(classes), 8)
        for name, cls in sorted(classes.items()):
            with self.subTest(node=name):
                # the live file lists are this machine's; the shipped snapshot holds the neutral allowlist (MP-37)
                live = snapshot_allowlist.sanitize({"nodes": {name: live_entry(cls, names[name])}})["nodes"][name]
                self.assertEqual(live, {k: self.snap["nodes"][name][k] for k in LIVE_KEYS},
                                 "the snapshot entry of %s is stale: refresh the snapshot" % name)


class TestWorkflows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snap = load_json(SNAPSHOT_PATH)
        cls.files = {os.path.splitext(os.path.basename(p))[0]: p for p in workflow_files()}

    def test_the_example_workflows_are_present(self):
        self.assertEqual(sorted(self.files), sorted(EXPECTED), "workflow files differ from the expected set")

    def test_every_workflow_validates_against_object_info(self):
        self.assertTrue(self.files)
        for name, path in sorted(self.files.items()):
            with self.subTest(workflow=name):
                problems = validate_workflow(load_json(path), self.snap)
                self.assertEqual(problems, [], "\n".join(problems))

    def test_preset_pin_chain_and_roles(self):
        catalog = presets_mod.catalog()
        for name, (preset_expected, roles_expected) in EXPECTED.items():
            if name not in self.files:
                continue
            with self.subTest(workflow=name):
                preset, pin, roles, problems = chain_of(load_json(self.files[name]))
                self.assertEqual(problems, [])
                self.assertEqual(preset, preset_expected)
                self.assertEqual(pin, presets_mod.pin_of(catalog[preset_expected]),
                                 "stale preset_pin: re-run research/gen_workflows.py")
                self.assertEqual(roles, roles_expected)

    def test_candidate_presets_are_labelled_in_the_note(self):
        catalog = presets_mod.catalog()
        for name, (preset, _roles) in EXPECTED.items():
            if name not in self.files:
                continue
            status = catalog[preset]["status"]
            text = "\n".join(str(v) for n in load_json(self.files[name])["nodes"] if n["type"] == "Note"
                             for v in n["widgets_values"])
            with self.subTest(workflow=name):
                self.assertIn("'%s'" % status, text)

    def test_sampler_cfg_is_the_preset_sentinel(self):
        for name, path in self.files.items():
            for node in load_json(path)["nodes"]:
                if node["type"] == "MossPhotonSampler":
                    with self.subTest(workflow=name):
                        self.assertEqual(node["widgets_values"][3], 0.0)


class TestValidatorGoesRed(unittest.TestCase):
    """The checks above must fail on the defects they claim to catch (an invariant is proven by breaking it)."""

    @classmethod
    def setUpClass(cls):
        cls.snap = load_json(SNAPSHOT_PATH)
        cls.good = load_json(os.path.join(PKG, "workflows", "turbo_identity.json"))

    def mutate(self, fn):
        wf = copy.deepcopy(self.good)
        fn(wf)
        return validate_workflow(wf, self.snap)

    @staticmethod
    def node(wf, ntype, index=0):
        return [n for n in wf["nodes"] if n["type"] == ntype][index]

    def test_clean_copy_is_green(self):
        self.assertEqual(validate_workflow(copy.deepcopy(self.good), self.snap), [])

    def test_bad_combo_value(self):
        problems = self.mutate(lambda wf: self.node(wf, "MossPhotonSampler")["widgets_values"].__setitem__(2, "turbo_9step"))
        self.assertTrue(any("turbo_9step" in p for p in problems), problems)

    def test_missing_lora_file(self):
        problems = self.mutate(lambda wf: self.node(wf, "MossPhotonLoRAProfile")["widgets_values"].__setitem__(0, "moss/none_such.safetensors"))
        self.assertTrue(any("none_such" in p for p in problems), problems)

    def test_unlisted_placeholder_like_name_is_rejected(self):
        problems = self.mutate(lambda wf: self.node(wf, "MossPhotonLoRAProfile")["widgets_values"].__setitem__(
            0, "your_other_lora.safetensors"))
        self.assertTrue(any("your_other_lora" in p for p in problems), problems)

    def test_placeholder_is_accepted_only_in_its_own_widget(self):
        problems = self.mutate(lambda wf: self.node(wf, "UNETLoader")["widgets_values"].__setitem__(
            0, "your_character_lora.safetensors"))
        self.assertTrue(any("your_character_lora" in p for p in problems), problems)

    def test_int_below_min(self):
        problems = self.mutate(lambda wf: self.node(wf, "MossPhotonEmptyLatent")["widgets_values"].__setitem__(0, 8))
        self.assertTrue(any("below the declared minimum" in p for p in problems), problems)

    def test_float_above_max(self):
        problems = self.mutate(lambda wf: self.node(wf, "MossPhotonSampler")["widgets_values"].__setitem__(3, 31.0))
        self.assertTrue(any("above the declared maximum" in p for p in problems), problems)

    def test_bool_is_not_an_int(self):
        problems = self.mutate(lambda wf: self.node(wf, "MossPhotonEmptyLatent")["widgets_values"].__setitem__(2, True))
        self.assertTrue(any("not an integer" in p for p in problems), problems)

    def test_unknown_node_type(self):
        problems = self.mutate(lambda wf: self.node(wf, "VAEDecode").__setitem__("type", "VAEDecodeTypo"))
        self.assertTrue(any("not in /object_info" in p for p in problems), problems)

    def test_widget_count_drift(self):
        problems = self.mutate(lambda wf: self.node(wf, "MossPhotonSaveImage")["widgets_values"].append("extra"))
        self.assertTrue(any("widgets_values has" in p for p in problems), problems)

    def test_link_type_mismatch(self):
        def swap(wf):
            link = wf["links"][0]
            link[5] = "VAE"
        problems = self.mutate(swap)
        self.assertTrue(any("type mismatch" in p for p in problems), problems)

    def test_link_into_wrong_slot(self):
        def retarget(wf):
            # point the CLIP link at the sampler's model input
            link = next(l for l in wf["links"] if l[5] == "CLIP")
            sampler = self.node(wf, "MossPhotonSampler")
            link[3] = sampler["id"]
            link[4] = [i for i, s in enumerate(sampler["inputs"]) if s["name"] == "model"][0]
        problems = self.mutate(retarget)
        self.assertTrue(any("type mismatch" in p or "points at" in p for p in problems), problems)

    def test_required_input_disconnected(self):
        def cut(wf):
            sampler = self.node(wf, "MossPhotonSampler")
            slot = next(s for s in sampler["inputs"] if s["name"] == "positive")
            lid = slot["link"]
            slot["link"] = None
            wf["links"] = [l for l in wf["links"] if l[0] != lid]
            for n in wf["nodes"]:
                for o in n.get("outputs", []):
                    if o.get("links"):
                        o["links"] = [x for x in o["links"] if x != lid]
        problems = self.mutate(cut)
        self.assertTrue(any("required input 'positive' is not connected" in p for p in problems), problems)

    def test_note_removed(self):
        problems = self.mutate(lambda wf: wf.__setitem__("nodes", [n for n in wf["nodes"] if n["type"] != "Note"]))
        self.assertTrue(any("no Note node" in p for p in problems), problems)

    def test_note_without_version(self):
        def strip(wf):
            note = self.node(wf, "Note")
            note["widgets_values"] = [note["widgets_values"][0].replace("8f04fd6ec", "")]
        problems = self.mutate(strip)
        self.assertTrue(any("8f04fd6ec" in p for p in problems), problems)

    def test_quality_claim_in_note_is_caught(self):
        def claim(wf):
            note = self.node(wf, "Note")
            note["widgets_values"] = [note["widgets_values"][0] + "\nImproves quality."]
        problems = self.mutate(claim)
        self.assertTrue(any("quality claim" in p for p in problems), problems)

    def test_stale_pin_is_caught_by_the_chain_check(self):
        wf = copy.deepcopy(self.good)
        self.node(wf, "MossPhotonSampler")["widgets_values"][5] = "turbo_identity@0.0.1#00000000"
        _preset, pin, _roles, _problems = chain_of(wf)
        self.assertNotEqual(pin, presets_mod.pin_of(presets_mod.catalog()["turbo_identity"]))


def _load_doc_generator():
    path = os.path.join(PKG, "tests", "preset_docs.py")
    if not os.path.isfile(path):
        raise AssertionError("tests/preset_docs.py is missing: the preset docs can not be checked")
    spec = importlib.util.spec_from_file_location("gen_preset_docs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDocs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = presets_mod.catalog()

    def read(self, *parts):
        with open(os.path.join(PKG, *parts), "r", encoding="utf-8") as fh:
            return fh.read()

    def test_every_preset_has_a_doc_generated_from_its_file(self):
        gen = _load_doc_generator()
        self.assertEqual(sorted(self.catalog), sorted(presets_mod.catalog_ids()))
        for pid in self.catalog:
            with self.subTest(preset=pid):
                path = os.path.join(DOCS_DIR, "presets", pid + ".md")
                self.assertTrue(os.path.isfile(path), "missing docs/presets/%s.md" % pid)
                self.assertEqual(self.read("docs", "presets", pid + ".md"), gen.render(pid, self.catalog[pid]),
                                 "docs/presets/%s.md differs from its preset file: run tests/preset_docs.py" % pid)

    def test_candidate_docs_carry_label_and_equality_sentence(self):
        for pid, preset in self.catalog.items():
            text = self.read("docs", "presets", pid + ".md")
            with self.subTest(preset=pid):
                self.assertIn("Status: %s" % preset["status"], text)
                if pid in ("photo_safe", "identity_safe", "wardrobe_safe", "scene_safe", "night_photo"):
                    self.assertIn("numerically equal to reference", text)
                self.assertIn("pending MP-12 / calibration", text)

    def test_readme_and_usage_state_the_scope(self):
        readme, usage = self.read("README.md"), self.read("docs", "USAGE.md")
        for text, label in ((readme, "README"), (usage, "USAGE")):
            self.assertIn("Base: not verified in v1", text, label)
            self.assertIn("SFT only", text, label)
        self.assertNotIn("licence: TBD", readme)
        for name in ("MossPhotonSeedDrafts", "MossPhotonPreserveUpscale", "MossPhotonIdentityDetailer",
                     "MossPhotonProtectedRefine"):
            self.assertIn(name, usage, "USAGE must list the out-of-scope node %s" % name)

    def test_license_and_release_metadata_agree(self):
        """MIT licence file, pyproject, version constant, CHANGELOG and README say the same thing (MP-35)."""
        from moss_photon.core import consts
        licence = self.read("LICENSE")
        self.assertTrue(licence.startswith("MIT License"))
        self.assertRegex(licence, r"Copyright \(c\) 2026 \S")
        pyproject = self.read("pyproject.toml")
        self.assertRegex(pyproject, r'(?m)^license = "MIT"$')
        self.assertRegex(pyproject, r'(?m)^version = "%s"$' % re.escape(consts.PACK_VERSION))
        self.assertRegex(pyproject, r'(?m)^requires-python = ">=3\.10"$')
        self.assertRegex(self.read("docs", "CHANGELOG.md"), r"(?m)^## %s\b" % re.escape(consts.PACK_VERSION))
        self.assertIn("MIT", self.read("README.md"))

    def test_docs_make_no_quality_claim(self):
        paths = [os.path.join(PKG, "README.md")] + sorted(glob.glob(os.path.join(DOCS_DIR, "*.md"))) \
            + sorted(glob.glob(os.path.join(DOCS_DIR, "presets", "*.md")))
        self.assertGreaterEqual(len(paths), 8)
        for path in paths:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
            with self.subTest(doc=os.path.relpath(path, PKG)):
                m = QUALITY_CLAIM.search(text)
                self.assertIsNone(m, "quality claim: %r" % (m.group(0) if m else None))

    def test_relative_links_resolve(self):
        link = re.compile(r"\]\((?!https?:|#|mailto:)([^)#\s]+)")
        paths = [os.path.join(PKG, "README.md")] + sorted(glob.glob(os.path.join(DOCS_DIR, "**", "*.md"), recursive=True))
        for path in paths:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
            for target in link.findall(text):
                with self.subTest(doc=os.path.relpath(path, PKG), target=target):
                    self.assertTrue(os.path.exists(os.path.normpath(os.path.join(os.path.dirname(path), target))))

    def test_deviations_doc_lists_all_21_rows(self):
        text = self.read("docs", "DEVIATIONS.md")
        for i in range(1, 22):
            self.assertRegex(text, r"\| DV%d \|" % i)

    def test_deviations_doc_records_the_turbo_and_forbidden_changes_deviations(self):
        text = self.read("docs", "DEVIATIONS.md")
        self.assertIn("forbidden_changes", text)
        self.assertRegex(text, r"(?is)SPEC 1.*distill")


if __name__ == "__main__":
    unittest.main(verbosity=2)
