"""MP-36 NEW-1 (MP-38): the effective-settings route knows which node class asks.

The solver lets an SLG override stand on a turbo preset in MossPhotonSamplerAdvanced only; the route used to resolve
with the flag off for everyone, so the Advanced node's panel showed HTTP 400 for a run that succeeds. The JS sends
the node class as the ``node`` query parameter; the route maps it to ``allow_slg_override``.
"""
import re
import unittest

from moss_photon.core import presets, routes

from .test_route_fields import read_js

ADVANCED = "MossPhotonSamplerAdvanced"
SIMPLE = "MossPhotonSampler"
SLG = {"cfg": "1.5", "slg_mode": "uncond_skip", "slg_layers": "22"}


class RouteNodeClassTest(unittest.TestCase):
    def test_advanced_turbo_slg_override_is_200_with_the_experimental_warning(self):
        status, payload = routes.handle_resolved("turbo_8step", dict(SLG, node=ADVANCED))
        self.assertEqual(status, 200)
        self.assertIn("turbo_slg_experimental", [w["code"] for w in payload["warnings"]])
        self.assertEqual(payload["effective"]["sampling"]["slg_mode"], "uncond_skip")
        self.assertEqual(payload["effective"]["sampling"]["slg_layers"], [22])

    def test_the_simple_node_and_a_missing_node_still_get_400(self):
        for query in (dict(SLG, node=SIMPLE), dict(SLG), dict(SLG, node=""), dict(SLG, node="SomethingElse"),
                      dict(SLG, node=ADVANCED.lower())):
            status, payload = routes.handle_resolved("turbo_8step", query)
            self.assertEqual(status, 400, query)
            self.assertEqual(payload["error"]["code"], "invalid_override")
            self.assertIn(ADVANCED, payload["error"]["message"] + payload["error"]["fix"])

    def test_the_flag_changes_nothing_else_for_the_advanced_node(self):
        status, _ = routes.handle_resolved("turbo_8step", {"cfg": "3.0", "slg_mode": "uncond_skip", "node": ADVANCED})
        self.assertEqual(status, 400)  # cfg 3 is outside the turbo bounds whatever the node is
        status, payload = routes.handle_resolved("reference", {"cfg": "4.0", "node": ADVANCED})
        self.assertEqual(status, 200)
        self.assertEqual(payload, routes.handle_resolved("reference", {"cfg": "4.0"})[1])

    def test_the_node_parameter_is_not_an_override(self):
        self.assertNotIn(routes.NODE_PARAM, routes.QUERY_PARAMS)
        self.assertEqual(routes.NODE_PARAM, "node")
        self.assertEqual(routes.ADVANCED_NODE, ADVANCED)

    def test_effective_view_takes_the_flag(self):
        view = presets.effective_view("turbo_8step", dict(cfg=1.5, slg_mode="uncond_skip", slg_layers=[22]),
                                      allow_slg_override=True)
        self.assertIn("turbo_slg_experimental", [w["code"] for w in view["warnings"]])
        with self.assertRaises(Exception) as cm:
            presets.effective_view("turbo_8step", dict(cfg=1.5, slg_mode="uncond_skip", slg_layers=[22]))
        self.assertEqual(cm.exception.code, "invalid_override")


class JsSendsTheNodeClassTest(unittest.TestCase):
    def setUp(self):
        self.src = read_js()

    def test_the_query_parameter_name_is_the_routes(self):
        m = re.search(r'export const NODE_QUERY_PARAM = "([^"]+)";', self.src)
        self.assertIsNotNone(m, "NODE_QUERY_PARAM literal missing in web/mossphoton.js")
        self.assertEqual(m.group(1), routes.NODE_PARAM)

    def test_the_override_request_of_the_panel_carries_the_node_class(self):
        body = self.src[self.src.index("async function refresh(node)"):]
        # the override query is built from the node's widgets and the node class is added to that very request
        self.assertRegex(body, r"withNodeClass\(\s*overridesQuery\(.*?\)\s*,\s*node\.comfyClass")

    def test_with_node_class_is_defined_and_uses_the_param(self):
        self.assertRegex(self.src, r"export function withNodeClass\(")
        self.assertIn("NODE_QUERY_PARAM", self.src[self.src.index("export function withNodeClass("):][:400])


if __name__ == "__main__":
    unittest.main()
