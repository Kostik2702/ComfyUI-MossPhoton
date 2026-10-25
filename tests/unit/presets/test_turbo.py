"""MP-23: turbo_8step and turbo_identity presets, override bounds, distill/role rules (docs/03-turbo.md 5, Decisions)."""
import copy
import hashlib
import inspect
import json
import os
import unittest

from moss_photon.core import consts, errors, presets

from .test_schema_crosscheck import jsonschema, lite_set, ref_set

TURBO = ("turbo_8step", "turbo_identity")
NON_TURBO = tuple(consts.PRESET_IDS)

# sha256 of the eight pre-turbo preset files at git HEAD 136625b (MP-7); they must stay byte-identical.
SHIPPED_SHA256 = {
    "advanced.json": "c9556fa63274401c0b35913ed4fd268a578fd2f4ab2375f3602f55ce3b8b43db",
    "draft_fast.json": "79e14bf1ef37b3b36c3c4867696569928994ea4d453ddf70f09cc6d1f41b80af",
    "identity_safe.json": "a26b473dd2cbb634e6444ca2e4662e2e8be64f1c9ecb46197cf004f85c57d53f",
    "night_photo.json": "d70bc747e7cfe5755e508c411dd8ff51667c40fb033a07ad6c29d4b0f0bd5b39",
    "photo_safe.json": "01c9456b5f2ebac23abc90664a1be21d6a23550ac60fa1cd6d28f1c36e1901e5",
    "reference.json": "d87c98041e6b993e523078076a71ec1edc54fb7e9ad7769cb27ecaae5908cfa8",
    "scene_safe.json": "6ccb9911820a38e2e3e5d33169b26821842eb443ac477c00f8ff6c931e869fe1",
    "wardrobe_safe.json": "e694ddc0014dcb2a64df4de2b2a4bae183f2db38e1916baa48f62a135b2bb1f5",
}


def pairs(found):
    return [(i.code, i.severity) for i in found]


def req(pid, **ctx):
    return presets.evaluate_requirements(presets.resolve(pid), ctx)


def codes(resolved):
    return [w.code for w in resolved.warnings]


class TurboCatalogTest(unittest.TestCase):
    def test_ids_and_order(self):
        self.assertEqual(presets.TURBO_PRESET_IDS, TURBO)
        self.assertEqual(presets.ALL_PRESET_IDS, tuple(consts.PRESET_IDS) + TURBO)
        self.assertEqual(presets.catalog_ids(), list(presets.ALL_PRESET_IDS))
        self.assertEqual(len(presets.catalog()), 10)
        names = sorted(n for n in os.listdir(presets.preset_dir()) if n.endswith(".json") and n != "schema.json")
        self.assertEqual(names, sorted(pid + ".json" for pid in presets.ALL_PRESET_IDS))

    def test_both_valid_with_cross_rules(self):
        for pid in TURBO:
            self.assertEqual(presets.validate_preset(presets.catalog()[pid], stem=pid), [], pid)

    def test_defaults_are_the_owner_decisions(self):
        for pid in TURBO:
            p = presets.catalog()[pid]
            s = p["sampling"]
            self.assertEqual((s["steps"], s["cfg"], s["sampler"], s["scheduler"], s["shift"]), (8, 1.0, "euler", "simple", 5.0), pid)
            self.assertEqual((s["slg_mode"], s["slg_layers"]), ("off", []), pid)
            self.assertEqual((p["status"], p["ui"]["recommended_final"], p["ui"]["label"]), ("candidate", False, "candidate"), pid)
            self.assertEqual(p["model_families"], ["moss-image-sft"], pid)
            self.assertEqual(p["budget"], {"nominal_steps": 8, "extra_forwards": 0, "max_forwards": 32}, pid)
            self.assertEqual(p["fallback"], "error_on_incompatibility", pid)

    def test_override_bounds_and_verified_range(self):
        for pid in TURBO:
            b = presets.catalog()[pid]["sampling"]["override_bounds"]
            self.assertEqual({k: (v["min"], v["max"], v["verified_min"], v["verified_max"]) for k, v in b.items()},
                             {"steps": (8, 16, 8, 8), "cfg": (1.0, 2.0, 1.0, 1.1)}, pid)
            for v in b.values():
                self.assertEqual(v["source"]["kind"], "owner_practice", pid)

    def test_lora_policy_and_constraints(self):
        cat = presets.catalog()
        a, b = cat["turbo_8step"], cat["turbo_identity"]
        self.assertEqual(a["lora_policy"], "require_distill_profile")
        self.assertEqual([(c["kind"], c.get("role"), c["severity"], c["reason"]) for c in a["constraints"]
                          if c["kind"] == "requires_lora_role"], [("requires_lora_role", "distill", "error", "missing_lora")])
        self.assertEqual([c["roles"] for c in a["constraints"] if c["kind"] == "limits_lora_roles"], [["distill"]])
        self.assertEqual(b["lora_policy"], "require_distill_and_character_profile")
        self.assertEqual(sorted(c["role"] for c in b["constraints"] if c["kind"] == "requires_lora_role"),
                         ["character", "distill"])
        self.assertFalse(any(c["kind"] == "limits_lora_roles" for c in b["constraints"]))

    def test_same_sampling_block_modulo_sources(self):
        a, b = presets.catalog()["turbo_8step"]["sampling"], presets.catalog()["turbo_identity"]["sampling"]
        for x in (a, b):
            for k in x["override_bounds"].values():
                k.pop("source")
        self.assertEqual(a, b)

    def test_existing_eight_files_byte_unchanged_pinned(self):
        for name, want in SHIPPED_SHA256.items():
            with open(os.path.join(presets.preset_dir(), name), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), want, name)

    def test_describe_catalog_shows_candidate(self):
        rows = {r["id"]: r for r in presets.describe_catalog()}
        self.assertEqual([r["id"] for r in presets.describe_catalog()], list(presets.ALL_PRESET_IDS))
        for pid in TURBO:
            self.assertEqual((rows[pid]["label"], rows[pid]["status"], rows[pid]["recommended_final"]),
                             ("candidate", "candidate", False), pid)
            self.assertEqual(rows[pid]["group"], "Candidate")

    def test_effective_view_forward_budget(self):
        for pid in TURBO:
            v = presets.effective_view(pid)
            self.assertEqual((v["forwards"]["branches_per_step"], v["forwards"]["expected_branch_forwards"]), (1, 8), pid)
            self.assertEqual(v["preset"]["label"], "candidate")
            self.assertEqual(v["effective"]["sampling"]["steps"], 8)
            self.assertEqual(v["override_bounds"]["steps"]["max"], 16)
            top = presets.effective_view(pid, {"steps": 16, "cfg": 2.0})
            self.assertEqual((top["forwards"]["branches_per_step"], top["forwards"]["expected_branch_forwards"]), (2, 32), pid)
            self.assertTrue(top["customized"])
        self.assertNotIn("override_bounds", presets.effective_view("reference"))

    def test_effective_keys_are_unchanged(self):
        # override_bounds is preset metadata; the effective sampling keys stay those of every other preset
        want = set(presets.catalog()["reference"]["sampling"])
        for pid in TURBO:
            self.assertEqual(set(presets.resolve(pid).effective["sampling"]), want, pid)
            self.assertIn("override_bounds", presets.resolve(pid).preset["sampling"])

    def test_resolve_signature_is_backward_compatible(self):
        sig = inspect.signature(presets.resolve)
        self.assertEqual(list(sig.parameters), ["preset_id", "overrides", "pin", "model_family", "lora_profile",
                                                "manifest_loader", "directory", "num_layers", "requires_calibration",
                                                "node", "allow_slg_override", "_check_pin"])
        r = presets.resolve("turbo_8step", node="MossPhotonSampler", pin=presets.pin_of(presets.catalog()["turbo_8step"]))
        self.assertTrue(r.pinned)
        self.assertEqual(codes(r), [])


class TurboOverrideTest(unittest.TestCase):
    def test_default_has_no_warnings_and_is_not_customized(self):
        r = presets.resolve("turbo_8step", overrides={"steps": 8, "cfg": 1.0})
        self.assertFalse(r.customized)
        self.assertEqual(codes(r), ["preset_unpinned"])

    def test_inside_verified_cfg_has_no_outside_warning(self):
        for cfg in (1.0, 1.05, 1.1):
            r = presets.resolve("turbo_8step", overrides={"cfg": cfg})
            self.assertNotIn("turbo_setting_outside_verified", codes(r), cfg)
        self.assertTrue(presets.resolve("turbo_8step", overrides={"cfg": 1.1}).customized)

    def test_inside_bounds_outside_verified_warns(self):
        for ov, param in (({"steps": 12}, "steps"), ({"steps": 9}, "steps"), ({"steps": 16}, "steps"),
                          ({"cfg": 1.5}, "cfg"), ({"cfg": 2.0}, "cfg"), ({"cfg": 1.11}, "cfg")):
            r = presets.resolve("turbo_identity", overrides=ov)
            hit = [w for w in r.warnings if w.code == "turbo_setting_outside_verified"]
            self.assertEqual([w.param for w in hit], [param], ov)
            self.assertEqual(hit[0].severity, "warning")
            self.assertEqual(hit[0].detail["param"], param)
            self.assertEqual(hit[0].detail["value"], r.effective["sampling"][param])
            self.assertIn("verified", hit[0].detail)
            self.assertTrue(r.customized)
            self.assertIn("customized_run", codes(r))

    def test_both_outside_gives_two_warnings(self):
        r = presets.resolve("turbo_8step", overrides={"steps": 16, "cfg": 2.0})
        self.assertEqual([w.param for w in r.warnings if w.code == "turbo_setting_outside_verified"], ["steps", "cfg"])
        self.assertEqual(r.effective["budget"]["nominal_steps"], 16)

    def test_outside_bounds_is_invalid_override(self):
        for ov in ({"steps": 7}, {"steps": 4}, {"steps": 17}, {"steps": 50}, {"cfg": 2.01}, {"cfg": 3.0}, {"cfg": 30.0}):
            with self.assertRaises(errors.InvalidOverride, msg=str(ov)) as cm:
                presets.resolve("turbo_8step", overrides=ov)
            self.assertEqual(cm.exception.issue.code, "invalid_override")
            self.assertIn("[", cm.exception.issue.message, ov)

    def test_bounds_message_names_the_range(self):
        with self.assertRaises(errors.InvalidOverride) as cm:
            presets.resolve("turbo_8step", overrides={"steps": 17})
        self.assertIn("[8, 16]", cm.exception.issue.message)
        with self.assertRaises(errors.InvalidOverride) as cm:
            presets.resolve("turbo_8step", overrides={"cfg": 2.5})
        self.assertIn("[1.0, 2.0]", cm.exception.issue.message)

    def test_slg_override_only_in_advanced(self):
        for pid in TURBO:
            for ov in ({"slg_mode": "uncond_skip", "slg_layers": [22]}, {"slg_mode": "uncond_skip"}, {"slg_layers": [22]},
                       {"slg_start": 0.1}, {"slg_end": 0.5}):
                with self.assertRaises(errors.InvalidOverride, msg="%s %s" % (pid, ov)):
                    presets.resolve(pid, overrides=ov)
            # restating the preset value is not an SLG override
            presets.resolve(pid, overrides={"slg_mode": "off", "slg_layers": [], "slg_start": 0.0, "slg_end": 1.0})
        r = presets.resolve("advanced", overrides={"cfg": 1.0})
        self.assertEqual(r.effective["sampling"]["slg_mode"], "uncond_skip")

    def test_the_refusal_names_the_node_that_allows_it(self):
        with self.assertRaises(errors.InvalidOverride) as cm:
            presets.resolve("turbo_8step", overrides={"slg_mode": "uncond_skip", "slg_layers": [22]})
        issue = cm.exception.issue
        self.assertIn("MossPhotonSamplerAdvanced", issue.message + issue.fix)
        self.assertNotIn("advanced preset", issue.message)

    def test_the_advanced_node_may_override_slg_on_a_turbo_preset_as_experimental(self):
        """MP-30 M2 (lead decision): Q6 'Advanced' is the Advanced NODE; the override is allowed there, with a warning."""
        for pid in TURBO:
            r = presets.resolve(pid, overrides={"cfg": 1.5, "slg_mode": "uncond_skip", "slg_layers": [22]},
                                allow_slg_override=True)
            smp = r.effective["sampling"]
            self.assertEqual((smp["slg_mode"], smp["slg_layers"], smp["cfg"]), ("uncond_skip", [22], 1.5), pid)
            hit = [w for w in r.warnings if w.code == "turbo_slg_experimental"]
            self.assertEqual([(w.severity, w.param) for w in hit], [("warning", "slg_mode")], pid)
            self.assertTrue(r.customized)

    def test_the_experimental_warning_is_only_for_a_real_slg_change(self):
        r = presets.resolve("turbo_8step", overrides={"cfg": 1.5}, allow_slg_override=True)
        self.assertNotIn("turbo_slg_experimental", codes(r))
        r = presets.resolve("turbo_8step", overrides={"slg_mode": "off"}, allow_slg_override=True)
        self.assertNotIn("turbo_slg_experimental", codes(r))

    def test_the_flag_does_not_widen_the_bounds(self):
        with self.assertRaises(errors.InvalidOverride):
            presets.resolve("turbo_8step", overrides={"cfg": 3.0, "slg_mode": "uncond_skip", "slg_layers": [22]},
                            allow_slg_override=True)

    def test_slg_at_cfg1_on_turbo_is_inert_and_says_so(self):
        r = presets.resolve("turbo_8step", overrides={"slg_mode": "uncond_skip", "slg_layers": [22]},
                            allow_slg_override=True)
        self.assertIn("slg_ineffective_cfg1", codes(r))

    def test_override_error_fix_names_the_bounds_and_the_preset_value_route(self):
        """MP-30 m6: the fix text of a turbo bounds error names the bounds and cfg 0 (= preset value)."""
        with self.assertRaises(errors.InvalidOverride) as cm:
            presets.resolve("turbo_8step", overrides={"cfg": 3.0})
        fix = cm.exception.issue.fix
        self.assertIn("steps [8, 16]", fix)
        self.assertIn("cfg [1.0, 2.0]", fix)
        self.assertIn("cfg to 0", fix)
        self.assertNotIn("schema.json", fix)

    def test_a_bad_slg_layer_in_the_advanced_node_does_not_get_the_advanced_only_advice(self):
        with self.assertRaises(errors.InvalidOverride) as cm:
            presets.resolve("turbo_8step", overrides={"slg_mode": "uncond_skip", "slg_layers": [99]}, num_layers=24,
                            allow_slg_override=True)
        issue = cm.exception.issue
        self.assertEqual(issue.param, "slg_layers")
        self.assertNotIn("only", issue.fix)
        self.assertNotIn("cfg to 0", issue.fix)

    def test_override_error_fix_of_a_preset_without_bounds_is_unchanged_in_kind(self):
        with self.assertRaises(errors.InvalidOverride) as cm:
            presets.resolve("reference", overrides={"steps": 0})
        self.assertNotIn("cfg to 0", cm.exception.issue.fix)

    def test_slg_ineffective_at_cfg1_in_advanced(self):
        r = presets.resolve("advanced", overrides={"cfg": 1.0})
        hit = [w for w in r.warnings if w.code == "slg_ineffective_cfg1"]
        self.assertEqual([w.severity for w in hit], ["info"])
        self.assertNotIn("slg_ineffective_cfg1", codes(presets.resolve("advanced", overrides={"cfg": 3.5})))
        self.assertNotIn("slg_ineffective_cfg1", codes(presets.resolve("advanced")))
        self.assertNotIn("slg_ineffective_cfg1", codes(presets.resolve("turbo_8step")))

    def test_non_turbo_presets_have_no_bounds_checks(self):
        r = presets.resolve("reference", overrides={"steps": 12, "cfg": 1.5})
        self.assertNotIn("turbo_setting_outside_verified", codes(r))
        self.assertEqual(r.effective["sampling"]["steps"], 12)


class TurboRequirementsTest(unittest.TestCase):
    def test_turbo_8step_needs_distill(self):
        got = req("turbo_8step", lora_roles=[])
        self.assertEqual(pairs(got), [("missing_lora", "error")])
        self.assertIn("role=distill", got[0].fix)
        with self.assertRaises(errors.MissingLora):
            errors.raise_first_error(got)
        self.assertEqual(req("turbo_8step", lora_roles=["distill"]), [])
        self.assertEqual(pairs(req("turbo_8step")), [("missing_lora", "error")])

    def test_turbo_8step_with_character_warns_and_hints_identity(self):
        got = req("turbo_8step", lora_roles=["distill", "character"])
        self.assertEqual(pairs(got), [("preset_lora_roles_exceeded", "warning")])
        self.assertIn("turbo_identity", got[0].fix)
        self.assertIsNone(errors.raise_first_error(got))
        got = req("turbo_8step", lora_roles=["character"])
        self.assertEqual(pairs(got), [("missing_lora", "error"), ("preset_lora_roles_exceeded", "warning")])
        self.assertEqual(pairs(req("turbo_8step", lora_roles=["distill", "style", "scene"])),
                         [("preset_lora_roles_exceeded", "warning")])

    def test_turbo_identity_needs_both(self):
        self.assertEqual(pairs(req("turbo_identity", lora_roles=[])),
                         [("missing_lora", "error"), ("missing_lora", "error"), ("reference_photos_recommended", "info")])
        got = req("turbo_identity", lora_roles=["distill"])
        self.assertEqual(pairs(got), [("missing_lora", "error"), ("reference_photos_recommended", "info")])
        self.assertIn("role=character", got[0].fix)
        got = req("turbo_identity", lora_roles=["character"])
        self.assertEqual(pairs(got), [("missing_lora", "error"), ("reference_photos_recommended", "info")])
        self.assertIn("role=distill", got[0].fix)
        self.assertEqual(pairs(req("turbo_identity", lora_roles=["distill", "character"])), [("reference_photos_recommended", "info")])
        self.assertEqual(req("turbo_identity", lora_roles=["distill", "character"], reference_photos=True), [])

    def test_distill_in_non_turbo_preset(self):
        for pid in NON_TURBO:
            got = req(pid, lora_roles=["distill"])
            if pid == "advanced":
                self.assertEqual(pairs(got), [("distill_lora_non_turbo", "warning")], pid)
                self.assertIsNone(errors.raise_first_error(got))
            else:
                hit = [i for i in got if i.code == "incompatible_lora"]
                self.assertEqual(pairs(hit), [("incompatible_lora", "error")], pid)
                with self.assertRaises(errors.IncompatibleLora):
                    errors.raise_first_error(hit)
                self.assertIn("turbo_8step", hit[0].fix)

    def test_no_distill_no_new_issue_in_non_turbo(self):
        for pid in ("reference", "photo_safe", "night_photo", "draft_fast", "advanced"):
            self.assertEqual(req(pid, lora_roles=["character", "style"]), [], pid)


class TurboCrossRulesTest(unittest.TestCase):
    def broken(self, pid, fn):
        q = copy.deepcopy(presets.catalog()[pid])
        fn(q)
        return presets.validate_preset(q, stem=pid)

    def test_each_cross_rule_fires(self):
        a, b = "turbo_8step", "turbo_identity"
        self.assertTrue(self.broken(a, lambda q: q.update(constraints=[c for c in q["constraints"] if c["kind"] != "requires_lora_role"])))
        self.assertTrue(self.broken(b, lambda q: q.update(constraints=[c for c in q["constraints"] if c.get("role") != "character"])))
        self.assertTrue(self.broken(b, lambda q: q.update(constraints=[c for c in q["constraints"] if c.get("role") != "distill"])))
        self.assertTrue(self.broken(a, lambda q: q["constraints"][1].pop("roles")))
        self.assertTrue(self.broken(a, lambda q: q["constraints"][0].update(kind="requires_lora_role", role="character")))
        b_ = lambda q: q["sampling"]["override_bounds"]
        self.assertTrue(self.broken(a, lambda q: b_(q)["steps"].update(min=9)))
        self.assertTrue(self.broken(a, lambda q: b_(q)["steps"].update(verified_min=9, verified_max=9)))
        self.assertTrue(self.broken(a, lambda q: b_(q)["steps"].update(verified_max=17)))
        self.assertTrue(self.broken(a, lambda q: b_(q)["cfg"].update(verified_max=2.5)))
        self.assertTrue(self.broken(a, lambda q: b_(q)["cfg"].update(verified_min=1.2)))
        self.assertTrue(any("distill preset" in m and "slg_mode" in m for m in self.broken(a, lambda q: q["sampling"].update(slg_mode="uncond_skip"))))
        self.assertTrue(self.broken(a, lambda q: q["budget"].update(max_forwards=31)))
        self.assertTrue(self.broken(a, lambda q: q["budget"].update(max_forwards=None)))
        self.assertTrue(self.broken(a, lambda q: q["sources"].update({"sampling.override_bounds.steps.min": q["sources"]["sampling.cfg"]})))
        self.assertTrue(self.broken(a, lambda q: q.update(status="stable")))
        self.assertTrue(self.broken(a, lambda q: q["ui"].update(recommended_final=True)))
        self.assertEqual(self.broken(a, lambda q: None), [])
        self.assertEqual(self.broken(b, lambda q: None), [])

    def test_turbo_status_candidate_cannot_be_recommended_final(self):
        # Decision Q9: recommended_final false until the turbo gate; candidate presets of the old catalog keep true
        for pid in TURBO:
            self.assertTrue(self.broken(pid, lambda q: q["ui"].update(recommended_final=True)), pid)

    def test_max_forwards_may_exceed_the_corner(self):
        self.assertEqual(self.broken("turbo_8step", lambda q: q["budget"].update(max_forwards=64)), [])


@unittest.skipIf(jsonschema is None, "jsonschema is not importable")
class TurboSchemaAgreementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = presets.load_schema()
        cls.cat = presets.catalog()

    def test_ten_presets_valid_in_both(self):
        self.assertEqual(len(self.cat), 10)
        for pid, p in self.cat.items():
            self.assertEqual(ref_set(p, self.schema), set(), pid)
            self.assertEqual(lite_set(p, self.schema), set(), pid)

    def test_turbo_mutations_same_verdict_and_paths(self):
        muts = {
            "scheduler karras": lambda x: x["sampling"].__setitem__("scheduler", "karras"),
            "sampler dpmpp": lambda x: x["sampling"].__setitem__("sampler", "dpmpp_2m"),
            "restart true": lambda x: x["sampling"].__setitem__("restart", True),
            "cfg below 1": lambda x: x["sampling"].__setitem__("cfg", 0.5),
            "unknown lora_policy": lambda x: x.__setitem__("lora_policy", "require_distill"),
            "unknown id": lambda x: x.__setitem__("id", "turbo_4step"),
            "bounds missing cfg": lambda x: x["sampling"]["override_bounds"].pop("cfg"),
            "bounds float steps": lambda x: x["sampling"]["override_bounds"]["steps"].__setitem__("max", 16.5),
            "bounds string steps": lambda x: x["sampling"]["override_bounds"]["steps"].__setitem__("min", "8"),
            "bounds zero": lambda x: x["sampling"]["override_bounds"]["cfg"].__setitem__("min", 0),
            "bounds bool": lambda x: x["sampling"]["override_bounds"]["steps"].__setitem__("max", True),
            "bounds missing source": lambda x: x["sampling"]["override_bounds"]["cfg"].pop("source"),
            "bounds source bad kind": lambda x: x["sampling"]["override_bounds"]["cfg"]["source"].__setitem__("kind", "hearsay"),
            "bounds extra key": lambda x: x["sampling"]["override_bounds"].__setitem__("shift", {}),
            "bounds not object": lambda x: x["sampling"].__setitem__("override_bounds", []),
            "constraint role bad": lambda x: x["constraints"][0].__setitem__("role", "turbo"),
            "constraint kind bad": lambda x: x["constraints"][0].__setitem__("kind", "forbids_lora_role"),
            "constraint reason bad": lambda x: x["constraints"][0].__setitem__("reason", "no_distill"),
            "roles duplicate": lambda x: x["constraints"].append(
                {"id": "dup", "kind": "limits_lora_roles", "severity": "warning", "reason": "none", "roles": ["distill", "distill"]}),
            "roles empty": lambda x: x["constraints"].append(
                {"id": "dup", "kind": "limits_lora_roles", "severity": "warning", "reason": "none", "roles": []}),
            "roles bad item": lambda x: x["constraints"].append(
                {"id": "dup", "kind": "limits_lora_roles", "severity": "warning", "reason": "none", "roles": ["turbo"]}),
            "postprocess nonempty": lambda x: x.__setitem__("postprocess", ["upscale"]),
            "missing sources": lambda x: x.pop("sources"),
            "missing max_forwards": lambda x: x["budget"].pop("max_forwards"),
            "slg_mode none": lambda x: x["sampling"].__setitem__("slg_mode", "none"),
            "source kind owner_practice typo": lambda x: x["sources"]["sampling.steps"].__setitem__("kind", "owner_practise"),
        }
        for pid in TURBO:
            for name, fn in muts.items():
                q = copy.deepcopy(self.cat[pid])
                fn(q)
                ref, lite = ref_set(q, self.schema), lite_set(q, self.schema)
                self.assertTrue(ref, "%s / %s accepted by jsonschema" % (pid, name))
                self.assertEqual(lite, ref, "%s / %s" % (pid, name))

    def test_old_schema_would_reject_turbo(self):
        # control: the new files really depend on the additive schema changes
        old = copy.deepcopy(self.schema)
        old["properties"]["id"]["enum"] = [x for x in old["properties"]["id"]["enum"] if not x.startswith("turbo")]
        for pid in TURBO:
            self.assertTrue(ref_set(self.cat[pid], old), pid)
            self.assertTrue(lite_set(self.cat[pid], old), pid)


if __name__ == "__main__":
    unittest.main()
