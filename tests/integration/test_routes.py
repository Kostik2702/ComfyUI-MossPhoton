"""GET /mossphoton/presets/{id}/resolved (design 4.9, 6.4): pure handler, never-raising registration."""
import asyncio
import json
import sys
import types
import unittest
from unittest import mock

from moss_photon.core import errors, presets, routes


class HandlerTest(unittest.TestCase):
    def test_reference_payload_equals_effective_view(self):
        status, payload = routes.handle_resolved("reference", {})
        self.assertEqual(status, 200)
        self.assertEqual(payload, presets.effective_view("reference"))
        self.assertEqual(payload["forwards"]["expected_branch_forwards"], 100)
        self.assertEqual(payload["effective"]["sampling"]["slg_layers"], [22])
        self.assertEqual(payload["preset"]["pin"], presets.pin_of(presets.catalog()["reference"]))
        json.dumps(payload)

    def test_every_catalog_preset_resolves(self):
        for pid in presets.catalog_ids():
            with self.subTest(pid=pid):
                status, payload = routes.handle_resolved(pid, {})
                self.assertEqual((status, payload["preset"]["id"]), (200, pid))

    def test_query_overrides(self):
        status, payload = routes.handle_resolved("reference", {"cfg": "4.5", "steps": "30", "slg_layers": "21, 22"})
        self.assertEqual(status, 200)
        self.assertTrue(payload["customized"])
        self.assertEqual(payload["effective"]["sampling"]["cfg"], 4.5)
        self.assertEqual(payload["effective"]["sampling"]["slg_layers"], [21, 22])
        self.assertEqual(payload["forwards"]["expected_branch_forwards"], 60)
        self.assertEqual(sorted(d["path"] for d in payload["diff"]),
                         ["budget.nominal_steps", "sampling.cfg", "sampling.slg_layers", "sampling.steps"])

    def test_cfg_one_is_one_branch(self):
        status, payload = routes.handle_resolved("reference", {"cfg": "1"})
        self.assertEqual((status, payload["forwards"]["branches_per_step"],
                          payload["forwards"]["expected_branch_forwards"]), (200, 1, 50))

    def test_unknown_id_is_404_invalid_preset(self):
        status, payload = routes.handle_resolved("nope", {})
        self.assertEqual((status, payload["error"]["code"]), (404, "invalid_preset"))

    def test_bad_override_is_400_invalid_override(self):
        for query in ({"cfg": "abc"}, {"steps": "1.5"}, {"cfg": "nan"}, {"steps": "5000"}, {"cfg": "99"},
                      {"slg_mode": "nope"}, {"slg_start": "0.9", "slg_end": "0.1"}):
            with self.subTest(query=query):
                status, payload = routes.handle_resolved("reference", query)
                self.assertEqual((status, payload["error"]["code"]), (400, "invalid_override"))
                self.assertEqual(payload["error"]["severity"], "error")

    def test_empty_and_unknown_query_values_are_ignored(self):
        status, payload = routes.handle_resolved("reference", {"cfg": "", "whatever": "1"})
        self.assertEqual((status, payload["customized"]), (200, False))

    def test_it_is_pure(self):
        before = presets.catalog_sha256()
        routes.handle_resolved("reference", {"cfg": "5"})
        self.assertEqual(presets.catalog_sha256(), before)

    def test_a_broken_catalog_is_500(self):
        def boom(*a, **k):
            raise errors.InvalidPresetFile(errors.make_issue(
                "invalid_preset_file", "presets", "x", "broken", "reinstall"))
        with mock.patch.object(presets, "effective_view", boom):
            status, payload = routes.handle_resolved("reference", {})
        self.assertEqual((status, payload["error"]["code"]), (500, "invalid_preset_file"))


class RegisterTest(unittest.TestCase):
    def setUp(self):
        routes._registered = False
        self.addCleanup(setattr, routes, "_registered", False)

    def fake_server(self):
        registered = []

        class Routes:
            def get(self, path):
                def deco(fn):
                    registered.append((path, fn))
                    return fn
                return deco
        server = types.ModuleType("server")
        server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=Routes()))
        return server, registered

    def test_returns_false_without_a_server_and_never_raises(self):
        with mock.patch.dict(sys.modules, {"server": None}):
            self.assertFalse(routes.register_routes())

    def test_returns_false_when_there_is_no_prompt_server_instance(self):
        server = types.ModuleType("server")
        server.PromptServer = type("PromptServer", (), {})
        with mock.patch.dict(sys.modules, {"server": server}):
            self.assertFalse(routes.register_routes())

    def test_registers_the_get_route_once(self):
        server, registered = self.fake_server()
        aiohttp = types.ModuleType("aiohttp")
        aiohttp.web = types.SimpleNamespace(json_response=lambda payload, status=200: (status, payload))
        with mock.patch.dict(sys.modules, {"server": server, "aiohttp": aiohttp}):
            self.assertTrue(routes.register_routes())
            self.assertTrue(routes.register_routes())
        self.assertEqual([p for p, _ in registered], ["/mossphoton/presets/{id}/resolved"])

    def test_the_registered_handler_serves_the_payload(self):
        server, registered = self.fake_server()
        aiohttp = types.ModuleType("aiohttp")
        aiohttp.web = types.SimpleNamespace(json_response=lambda payload, status=200: (status, payload))
        with mock.patch.dict(sys.modules, {"server": server, "aiohttp": aiohttp}):
            routes.register_routes()
        handler = registered[0][1]
        request = types.SimpleNamespace(match_info={"id": "reference"},
                                        rel_url=types.SimpleNamespace(query={"cfg": "2"}))
        status, payload = asyncio.run(handler(request))
        self.assertEqual((status, payload["effective"]["sampling"]["cfg"]), (200, 2.0))
        bad = types.SimpleNamespace(match_info={"id": "zzz"}, rel_url=types.SimpleNamespace(query={}))
        self.assertEqual(asyncio.run(handler(bad))[0], 404)

    def test_a_failing_server_is_swallowed(self):
        server = types.ModuleType("server")

        class Exploding:
            @property
            def routes(self):
                raise RuntimeError("no routes yet")
        server.PromptServer = types.SimpleNamespace(instance=Exploding())
        with mock.patch.dict(sys.modules, {"server": server}):
            with self.assertLogs("MossPhoton", level="WARNING"):
                self.assertFalse(routes.register_routes())


if __name__ == "__main__":
    unittest.main()
