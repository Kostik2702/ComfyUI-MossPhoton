"""web/mossphoton.js reads only fields the route payload really has (design 4.9, MP-18).

The JS declares what it reads in three literals (READ_FIELDS, REQUIREMENT_FIELDS, ERROR_FIELDS); this test parses
them out of the file and walks every catalog preset's payload (and the 404 / 400 bodies) with them. Pure: no comfy.
"""
import json
import os
import re
import unittest

from moss_photon.core import presets, routes

WEB_JS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                      "web", "mossphoton.js")

# Pinned on purpose: dropping a field from the JS literal must be a conscious edit of this test as well.
PINNED_FIELDS = frozenset((
    "preset.id", "preset.pin", "preset.label", "effective.sampling.steps", "effective.sampling.cfg",
    "effective.sampling.slg_mode", "effective.sampling.slg_layers", "effective.budget.max_forwards",
    "forwards.expected_branch_forwards", "requirements[].kind", "requirements[].severity",
))


def read_js():
    with open(WEB_JS, "r", encoding="utf-8") as f:
        return f.read()


def array_literal(src, name):
    m = re.search(r"export const %s = \[(.*?)\];" % re.escape(name), src, re.S)
    if m is None:
        raise AssertionError("%s literal not found in web/mossphoton.js" % name)
    return re.findall(r'"([^"]+)"', m.group(1))


def object_literal(src, name):
    m = re.search(r"export const %s = \{(.*?)\};" % re.escape(name), src, re.S)
    if m is None:
        raise AssertionError("%s literal not found in web/mossphoton.js" % name)
    return {k: (None if v == "null" else v.strip('"'))
            for k, v in re.findall(r"(\w+):\s*(null|\"[^\"]*\")", m.group(1))}


def missing_paths(payload, paths):
    """Paths (``a.b``, ``a[].b``) the payload lacks; a list segment needs every item to carry the rest."""
    out = []

    def walk(node, parts, trail):
        if not parts:
            return
        head, rest = parts[0], parts[1:]
        if head.endswith("[]"):
            key = head[:-2]
            items = node.get(key) if isinstance(node, dict) else None
            if not isinstance(items, list):
                out.append(trail + key)
                return
            for item in items:
                walk(item, rest, trail + key + "[].")
            return
        if not isinstance(node, dict) or head not in node:
            out.append(trail + head)
            return
        walk(node[head], rest, trail + head + ".")

    for p in paths:
        walk(payload, p.split("."), "")
    return out


class RouteFieldsTest(unittest.TestCase):
    def setUp(self):
        self.src = read_js()
        self.read_fields = array_literal(self.src, "READ_FIELDS")

    def test_literal_is_parsed_and_pinned_fields_present(self):
        self.assertGreaterEqual(len(self.read_fields), len(PINNED_FIELDS))
        self.assertEqual(sorted(PINNED_FIELDS - set(self.read_fields)), [])

    def test_every_catalog_preset_payload_has_every_read_field(self):
        ids = presets.catalog_ids()
        self.assertGreaterEqual(len(ids), 10)
        for pid in ids:
            with self.subTest(pid=pid):
                status, payload = routes.handle_resolved(pid, {})
                self.assertEqual(status, 200)
                self.assertEqual(missing_paths(payload, self.read_fields), [])
                json.dumps(payload)

    def test_override_payload_keeps_the_same_shape(self):
        status, payload = routes.handle_resolved("reference", {"cfg": "4.5", "steps": "30", "slg_layers": "21,22"})
        self.assertEqual(status, 200)
        self.assertEqual(missing_paths(payload, self.read_fields), [])

    def test_requirement_extra_fields_exist_for_their_kind(self):
        extra = object_literal(self.src, "REQUIREMENT_FIELDS")
        seen = set()
        for pid in presets.catalog_ids():
            _, payload = routes.handle_resolved(pid, {})
            for req in payload["requirements"]:
                seen.add(req["kind"])
                self.assertIn(req["kind"], extra, "JS does not know requirement kind %s" % req["kind"])
                if extra[req["kind"]] is not None:
                    self.assertIn(extra[req["kind"]], req)
        self.assertIn("requires_lora_role", seen)  # control: the turbo presets do carry one

    def test_error_bodies_carry_the_error_fields(self):
        fields = array_literal(self.src, "ERROR_FIELDS")
        for query, pid, expect in (({}, "no_such_preset", 404), ({"cfg": "abc"}, "reference", 400)):
            with self.subTest(expect=expect):
                status, payload = routes.handle_resolved(pid, query)
                self.assertEqual(status, expect)
                self.assertEqual(missing_paths(payload, fields), [])

    def test_missing_field_is_detected(self):
        _, payload = routes.handle_resolved("reference", {})
        self.assertEqual(missing_paths(payload, ["preset.nope", "requirements[].kind"]), ["preset.nope"])
        payload = json.loads(json.dumps(payload))
        del payload["forwards"]["expected_branch_forwards"]
        self.assertEqual(missing_paths(payload, ["forwards.expected_branch_forwards"]),
                         ["forwards.expected_branch_forwards"])

    # ---- m11: route cache (source-level: the JS can not run here; a live Playwright check is scheduled by the lead)
    def test_route_cache_has_a_short_ttl_and_drops_failures(self):
        m = re.search(r"export const CACHE_TTL_MS = (\d+);", self.src)
        self.assertIsNotNone(m, "CACHE_TTL_MS literal not found")
        self.assertTrue(1000 <= int(m.group(1)) <= 60000, "TTL must be short (1 s .. 60 s)")
        body = re.search(r"function getView\(.*?\n}\n", self.src, re.S)
        self.assertIsNotNone(body)
        text = body.group(0)
        self.assertIn("CACHE_TTL_MS", text)
        self.assertRegex(text, r"if \(!\w+\.ok\) cache\.delete\(url\)",
                         "a failed answer (ok false) must be removed from the cache")

    def test_route_cache_is_invalidated_on_reconnect_and_status(self):
        self.assertRegex(self.src, r'import \{ api \} from "\.\./\.\./scripts/api\.js";')
        for ev in ("reconnected", "status"):
            self.assertRegex(self.src, r'api\.addEventListener\("%s", ' % ev)
        m = re.search(r"function invalidateRoutes\(.*?\n}\n", self.src, re.S)
        self.assertIsNotNone(m)
        self.assertIn("cache.clear()", m.group(0))

    # ---- m12: override_bounds
    def test_bounds_fields_are_present_in_every_turbo_payload(self):
        fields = array_literal(self.src, "BOUNDS_FIELDS")
        keys = array_literal(self.src, "BOUNDS_KEYS")
        self.assertEqual(sorted(keys), ["cfg", "steps"])
        seen = 0
        for pid in presets.catalog_ids():
            _, payload = routes.handle_resolved(pid, {})
            bounds = payload.get("override_bounds")
            if bounds is None:
                continue
            seen += 1
            for key in keys:
                self.assertEqual([f for f in fields if f not in bounds[key]], [], pid)
        self.assertGreaterEqual(seen, 2)  # control: both turbo presets carry bounds

    def test_panel_reads_override_bounds_and_keeps_cfg_zero_as_preset_value(self):
        self.assertIn("override_bounds", self.src)
        self.assertRegex(self.src, r"export function boundsLines\(")
        self.assertRegex(self.src, r"export function boundsWidgetLimits\(")
        # cfg 0 / steps 0 are the "preset value" sentinels: never flagged, the widget minimum stays 0
        m = re.search(r"export function boundsWidgetLimits\(.*?\n}\n", self.src, re.S)
        self.assertNotRegex(m.group(0), r"\.min\s*=\s*b")

    def test_js_pin_grammar_equals_parse_pin(self):
        m = re.search(r"const PIN_RE = /(.*)/;", self.src)
        self.assertIsNotNone(m)
        js = m.group(1)
        self.assertEqual(js, "^([a-z0-9_]+)@([0-9]+[.][0-9]+[.][0-9]+)(?:#([0-9a-f]{8}))?$")
        py = re.compile(js)
        for pid in presets.catalog_ids():
            _, payload = routes.handle_resolved(pid, {})
            pin = payload["preset"]["pin"]
            self.assertIsNotNone(py.fullmatch(pin))
            self.assertEqual(presets.parse_pin(pin)["id"], pid)
        self.assertIsNone(py.fullmatch("reference@1.0"))

    # ---- MP-71: darkroom presets
    def test_darkroom_fields_are_in_the_payload_of_every_darkroom_preset_and_only_there(self):
        fields = array_literal(self.src, "DARKROOM_FIELDS")
        self.assertGreaterEqual(len(fields), 5)
        seen = []
        for pid in presets.catalog_ids():
            status, payload = routes.handle_resolved(pid, {})
            self.assertEqual(status, 200)
            if "darkroom" in presets.catalog()[pid]:
                seen.append(pid)
                self.assertEqual(missing_paths(payload, fields), [], pid)
                self.assertEqual(missing_paths(payload, self.read_fields), [], pid)
            else:
                self.assertNotIn("darkroom", payload, pid)
                self.assertEqual(missing_paths(payload, fields), ["darkroom"] * len(fields),
                                 "control: the walker does see a payload without the block")
        self.assertEqual(seen, ["turbo_darkroom_portrait", "turbo_darkroom_fullbody"])

    def test_the_cfg_override_of_a_darkroom_preset_is_the_finish_guidance_in_the_route(self):
        status, payload = routes.handle_resolved("turbo_darkroom_portrait", {"cfg": "2.5", "node": "MossPhotonSampler"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["darkroom"]["finish"]["cfg"], 2.5)
        self.assertEqual(payload["effective"]["sampling"]["cfg"], 1.0)
        self.assertEqual(missing_paths(payload, self.read_fields), [])
        status, payload = routes.handle_resolved("turbo_darkroom_portrait", {"cfg": "99"})
        self.assertEqual(status, 200)   # MP-84: moved to the largest guidance (30), with a warning; it was 400
        self.assertEqual(payload["darkroom"]["finish"]["cfg"], 30.0)
        self.assertIn("setting_adjusted", [w["code"] for w in payload["warnings"]])
        status, payload = routes.handle_resolved("turbo_darkroom_portrait", {"cfg": "abc"})
        self.assertEqual(status, 400)
        self.assertEqual(missing_paths(payload, array_literal(self.src, "ERROR_FIELDS")), [])

    def test_the_panel_sends_every_advanced_darkroom_knob_the_route_reads(self):
        sent = array_literal(self.src, "DARKROOM_KNOBS")
        self.assertEqual(sent, list(routes.DARKROOM_PARAMS))
        self.assertIn("...DARKROOM_KNOBS", re.search(r"function collectValues\(.*?\n}\n", self.src, re.S).group(0))
        self.assertIn("...DARKROOM_KNOBS", self.src.split("function attach")[1])  # the widgets refresh the panel
        status, payload = routes.handle_resolved("turbo_darkroom_portrait", {
            "node": "MossPhotonSamplerAdvanced", "compose_cfg": "1.5", "finish_cfg": "2.5", "w_commit": "1.1",
            "compose_slg": "on", "switch_model": "off", "reexpose": "off", "schedule": "native", "eta": "0",
            "slg_mode": "uncond_skip", "slg_layers": "22"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["forwards"]["branches_per_step"], 2)
        d = payload["darkroom"]
        self.assertEqual((d["compose"]["cfg"], d["compose"]["slg"], d["finish"]["cfg"], d["commit"]["w_commit"]),
                         (1.5, True, 2.5, 1.1))
        self.assertTrue(payload["customized"])
        self.assertEqual(missing_paths(payload, self.read_fields), [])
        self.assertIn("darkroom_override_ignored", [w["code"] for w in payload["warnings"]])
        for bad in ({"finish_cfg": "x"}, {"compose_slg": "maybe"}):
            status, payload = routes.handle_resolved("turbo_darkroom_portrait", bad)
            self.assertEqual(status, 400, bad)
            self.assertEqual(missing_paths(payload, array_literal(self.src, "ERROR_FIELDS")), [])
        # a phase order the run sorts is no error either (MP-84): the panel shows the preset's block with a warning
        status, payload = routes.handle_resolved("turbo_darkroom_portrait", {"r_f": "0.1"})
        self.assertEqual(status, 200)
        self.assertIn("setting_adjusted", [w["code"] for w in payload["warnings"]])

    def test_the_panel_query_follows_the_node_sentinels(self):
        """overridesQuery is JS and can not run here: the sentinel rules are read from its source and compared with the node's."""
        m = re.search(r"export function overridesQuery\(.*?\n}\n", self.src, re.S)
        self.assertIsNotNone(m)
        text = m.group(0)
        for rule in ('v !== "from_preset"', "Number(v) > 0", "Number(v) >= 0"):
            self.assertIn(rule, text)
        from moss_photon.nodes import sampler
        opt = sampler.MossPhotonSamplerAdvanced.INPUT_TYPES()["optional"]
        for name in routes.DARKROOM_PARAMS:
            self.assertIn(name, opt)

    def test_the_panel_shows_the_block_and_does_not_bound_the_finish_guidance_by_the_compose_bounds(self):
        self.assertIn("view.darkroom", self.src)
        self.assertRegex(self.src, r"base\.body\.darkroom \? BOUNDS_KEYS\.filter\(\(k\) => k !== \"cfg\"\)")


if __name__ == "__main__":
    unittest.main()
