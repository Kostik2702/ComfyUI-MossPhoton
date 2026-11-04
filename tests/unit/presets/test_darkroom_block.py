"""MP-71: the optional ``darkroom`` block of a preset: schema (both validators agree), the cross-field rules, the sources of its
numbers, the two shipped presets and the effective view (the ``cfg`` override of a darkroom preset is the finish guidance).
"""
import copy
import unittest

from moss_photon.core import errors, presets

from .test_schema_crosscheck import jsonschema, lite_set, ref_set

DARKROOM = ("turbo_darkroom_portrait", "turbo_darkroom_fullbody")
NUMERIC = ("darkroom.compose.cfg", "darkroom.commit.r_c", "darkroom.commit.r_k", "darkroom.commit.w_compose",
           "darkroom.commit.w_commit", "darkroom.reexpose.sigma_r", "darkroom.finish.r_f", "darkroom.finish.cfg",
           "darkroom.finish.eta")


def problems(obj):
    return presets.validate_preset(obj, stem=obj["id"])


class ShippedPresetsTest(unittest.TestCase):
    def setUp(self):
        self.cat = presets.catalog()

    def test_only_the_two_darkroom_presets_have_a_block_and_both_are_valid(self):
        self.assertEqual([pid for pid, p in self.cat.items() if "darkroom" in p], list(DARKROOM))
        for pid in DARKROOM:
            self.assertEqual(problems(self.cat[pid]), [], pid)
        self.assertEqual(presets.catalog_issues(), [])

    def test_the_numbers_of_the_block_all_have_a_source(self):
        for pid in DARKROOM:
            self.assertTrue(set(NUMERIC) <= set(self.cat[pid]["sources"]), pid)

    def test_they_are_candidate_turbo_presets_with_the_ticket_sampling(self):
        for pid in DARKROOM:
            p = self.cat[pid]
            self.assertEqual((p["preset_version"], p["status"], p["ui"]["recommended_final"]), ("1.0.0", "candidate", False))
            self.assertEqual(p["lora_policy"], "use_explicit_user_profile_or_none", "MP-84: no LoRA is required")
            self.assertFalse([c for c in p["constraints"] if c["kind"] == "requires_lora_role"])
            self.assertEqual(p["calibration"]["state"], "uncalibrated")
            smp = p["sampling"]
            self.assertEqual((smp["steps"], smp["cfg"], smp["slg_mode"], smp["scheduler"]), (12, 1.0, "off", "simple"))
            self.assertEqual(p["budget"], {"nominal_steps": 12, "extra_forwards": 0, "max_forwards": 32})
            self.assertEqual(p["darkroom"]["compose"]["cfg"], smp["cfg"])

    def test_the_effective_view_carries_the_block_and_the_finish_override(self):
        v = presets.effective_view("turbo_darkroom_portrait")
        self.assertEqual(v["darkroom"], self.cat["turbo_darkroom_portrait"]["darkroom"])
        self.assertIn("finish phase", v["forwards"]["note"])
        v = presets.effective_view("turbo_darkroom_portrait", {"cfg": 3.0})
        self.assertEqual(v["darkroom"]["finish"]["cfg"], 3.0)
        self.assertEqual(v["effective"]["sampling"]["cfg"], 1.0, "sampling.cfg stays the compose guidance")
        self.assertEqual((v["customized"], v["diff"]),
                         (True, [{"path": "darkroom.finish.cfg", "base": 2.0, "effective": 3.0}]))
        v = presets.effective_view("turbo_darkroom_portrait", {"cfg": 2.0})
        self.assertEqual((v["customized"], v["diff"]), (False, []), "a value equal to the preset's is no customization")
        self.assertNotIn("darkroom", presets.effective_view("turbo_identity"))
        self.assertEqual(presets.effective_view("turbo_identity", {"cfg": 1.5})["effective"]["sampling"]["cfg"], 1.5)

    def test_the_view_applies_the_advanced_knobs_and_marks_a_differing_value_customized(self):
        base = presets.catalog()["turbo_darkroom_portrait"]["darkroom"]
        v = presets.effective_view("turbo_darkroom_portrait", darkroom_knobs={"w_commit": 1.1, "finish_cfg": 2.5})
        self.assertEqual((v["darkroom"]["commit"]["w_commit"], v["darkroom"]["finish"]["cfg"]), (1.1, 2.5))
        self.assertTrue(v["customized"])
        self.assertEqual([d["path"] for d in v["diff"]], ["darkroom.commit.w_commit", "darkroom.finish.cfg"])
        same = presets.effective_view("turbo_darkroom_portrait", darkroom_knobs={
            "w_commit": base["commit"]["w_commit"], "finish_cfg": base["finish"]["cfg"], "compose_slg": False})
        self.assertEqual((same["customized"], same["diff"]), (False, []))
        # compose_cfg is the sampling cfg: the branches and the forwards shown follow it
        v = presets.effective_view("turbo_darkroom_portrait", darkroom_knobs={"compose_cfg": 1.5})
        self.assertEqual((v["forwards"]["branches_per_step"], v["forwards"]["expected_branch_forwards"]), (2, 24))
        self.assertEqual(v["darkroom"]["compose"]["cfg"], 1.5)
        self.assertEqual([d["path"] for d in v["diff"]], ["sampling.cfg"])
        # finish_cfg wins over the cfg widget, as in the run
        v = presets.effective_view("turbo_darkroom_portrait", {"cfg": 3.0}, darkroom_knobs={"finish_cfg": 2.5})
        self.assertEqual(v["darkroom"]["finish"]["cfg"], 2.5)

    def test_the_view_shows_the_preset_block_for_knobs_the_run_adjusts_and_drops_what_the_run_drops(self):
        """MP-84: the run moves a knob it cannot use to the nearest usable value (``darkroom.resolve_settings``), so the
        panel does not stop: it shows the preset's block and a ``setting_adjusted`` warning (it was ``invalid_override``)."""
        for knobs in ({"r_f": 0.1}, {"finish_cfg": 0.5}, {"eta": 1.5}, {"r_c": 0.17, "r_k": 0.17}):
            with self.subTest(knobs=knobs):
                v = presets.effective_view("turbo_darkroom_portrait", darkroom_knobs=knobs, node="MossPhotonSamplerAdvanced")
                self.assertEqual(v["darkroom"], self.cat["turbo_darkroom_portrait"]["darkroom"])
                adjusted = [w for w in v["warnings"] if w["code"] == "setting_adjusted"]
                self.assertEqual([(w["severity"], w["node"]) for w in adjusted], [("warning", "MossPhotonSamplerAdvanced")])
        v = presets.effective_view("turbo_darkroom_portrait", darkroom_knobs={"schedule": "sa2_8"})
        self.assertEqual(v["darkroom"]["schedule"], "sa2_8", "sa2_8 needs no step count of 8 any more")
        self.assertEqual(v["warnings"], [])
        v = presets.effective_view("turbo_darkroom_portrait", {"slg_mode": "uncond_skip", "slg_layers": [22]},
                                   allow_slg_override=True)
        self.assertEqual(v["effective"]["sampling"]["slg_mode"], "off")
        codes = [w["code"] for w in v["warnings"]]
        self.assertEqual(codes.count("darkroom_override_ignored"), 1)
        self.assertNotIn("turbo_slg_experimental", codes)
        v = presets.effective_view("turbo_identity", darkroom_knobs={"r_f": 0.3})
        self.assertNotIn("darkroom", v)
        self.assertEqual([w["code"] for w in v["warnings"]], ["darkroom_override_ignored"])

    def test_the_finish_override_is_range_checked(self):
        with self.assertRaises(errors.InvalidOverride):
            presets.effective_view("turbo_darkroom_portrait", {"cfg": True})
        for bad, used in ((0.5, 1.0), (31.0, 30.0)):
            with self.subTest(bad=bad):
                v = presets.effective_view("turbo_darkroom_portrait", {"cfg": bad})
                self.assertEqual(v["darkroom"]["finish"]["cfg"], used)
                self.assertIn("setting_adjusted", [w["code"] for w in v["warnings"]])

    def test_other_overrides_of_a_darkroom_preset_resolve_as_for_any_turbo_preset(self):
        v = presets.effective_view("turbo_darkroom_portrait", {"steps": 16})
        self.assertEqual(v["effective"]["sampling"]["steps"], 16)
        v = presets.effective_view("turbo_darkroom_portrait", {"steps": 17})  # outside the bounds: used, with a warning
        self.assertEqual(v["effective"]["sampling"]["steps"], 17)
        self.assertIn("override_outside_bounds", [w["code"] for w in v["warnings"]])


class BlockRulesTest(unittest.TestCase):
    def base(self):
        return copy.deepcopy(presets.catalog()["turbo_darkroom_portrait"])

    def bad(self, mutate, needle):
        p = self.base()
        mutate(p)
        found = problems(p)
        self.assertTrue(found, "accepted: " + needle)
        self.assertTrue(any(needle in f for f in found), (needle, found))

    def test_the_rules(self):
        self.bad(lambda p: p["darkroom"]["compose"].update(cfg=1.5), "must equal sampling.cfg")
        self.bad(lambda p: p["darkroom"]["commit"].update(r_c=0.5), "darkroom.commit")
        self.bad(lambda p: p["darkroom"]["finish"].update(r_f=0.1), "darkroom.commit")
        self.bad(lambda p: p["darkroom"]["commit"].update(r_c=0.17), "empty commit window")

    def test_no_lora_policy_and_no_step_count_is_demanded(self):
        """MP-84: LoRAs are optional and sa2_8 resamples its knots for any step count: neither is a problem of the block."""
        self.assertEqual(problems(self.base()), [])
        p = self.base()
        p["darkroom"].update(schedule="sa2_8")   # 12 steps
        self.assertEqual(problems(p), [])

    def test_an_equal_weight_needs_no_commit_window(self):
        p = self.base()
        p["darkroom"]["commit"].update(r_c=0.17, w_commit=1.0)
        self.assertEqual(problems(p), [])

    def test_sa2_is_fine_with_8_steps(self):
        p = self.base()
        p["darkroom"]["schedule"] = "sa2_8"
        p["sampling"]["steps"] = 8
        p["budget"]["nominal_steps"] = 8
        p["sampling"]["override_bounds"]["steps"].update(verified_min=8, verified_max=8)
        self.assertEqual(problems(p), [])

    def test_a_number_without_a_source_and_a_source_without_a_number(self):
        p = self.base()
        del p["sources"]["darkroom.finish.cfg"]
        self.assertTrue(any("no entry for numeric leaf darkroom.finish.cfg" in f for f in problems(p)))
        p = self.base()
        p["sources"]["darkroom.finish.nope"] = copy.deepcopy(p["sources"]["darkroom.finish.cfg"])
        self.assertTrue(any("names no numeric leaf" in f for f in problems(p)))

    def test_a_broken_darkroom_file_disables_only_itself(self):
        import json
        import os
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        for name in os.listdir(presets.preset_dir()):
            shutil.copy(os.path.join(presets.preset_dir(), name), tmp)
        path = os.path.join(tmp, "turbo_darkroom_portrait.json")
        with open(path, "rb") as f:
            obj = json.loads(f.read().decode("utf-8"))
        obj["darkroom"]["compose"]["cfg"] = 1.5
        with open(path, "wb") as f:
            f.write(json.dumps(obj).encode("utf-8"))
        self.assertNotIn("turbo_darkroom_portrait", presets.catalog_ids(tmp))
        self.assertIn("turbo_darkroom_fullbody", presets.catalog_ids(tmp))
        self.assertEqual([w.param for w in presets.catalog_issues(tmp)], ["presets/turbo_darkroom_portrait.json"])


@unittest.skipIf(jsonschema is None, "jsonschema is not importable")
class SchemaAgreementTest(unittest.TestCase):
    """schema_lite and the reference implementation give the same verdict and error paths on a mutation corpus of the block."""

    def test_the_mutations(self):
        schema = presets.load_schema()
        base = presets.catalog()["turbo_darkroom_portrait"]
        muts = {
            "missing block part": lambda p: p["darkroom"].pop("finish"),
            "unknown block key": lambda p: p["darkroom"].update(extra=1),
            "unknown compose key": lambda p: p["darkroom"]["compose"].update(extra=1),
            "schedule": lambda p: p["darkroom"].update(schedule="linear"),
            "cfg below 1": lambda p: p["darkroom"]["compose"].update(cfg=0.5),
            "cfg above 30": lambda p: p["darkroom"]["finish"].update(cfg=31),
            "string r_c": lambda p: p["darkroom"]["commit"].update(r_c="0.06"),
            "r_k above 1": lambda p: p["darkroom"]["commit"].update(r_k=1.5),
            "weight above 10": lambda p: p["darkroom"]["commit"].update(w_commit=10.5),
            "negative weight": lambda p: p["darkroom"]["commit"].update(w_compose=-0.1),
            "sigma_r below the floor": lambda p: p["darkroom"]["reexpose"].update(sigma_r=0.0),
            "enabled as number": lambda p: p["darkroom"]["reexpose"].update(enabled=1),
            "switch as string": lambda p: p["darkroom"]["finish"].update(switch="off"),
            "eta above 1": lambda p: p["darkroom"]["finish"].update(eta=1.5),
            "block as list": lambda p: p.update(darkroom=[]),
            "unknown darkroom id": lambda p: p.update(id="turbo_darkroom_other"),
        }
        self.assertEqual(lite_set(base, schema), set())
        self.assertEqual(ref_set(base, schema), set())
        for name, mutate in muts.items():
            with self.subTest(name):
                q = copy.deepcopy(base)
                mutate(q)
                ref = ref_set(q, schema)
                self.assertTrue(ref, "accepted by jsonschema")
                self.assertEqual(lite_set(q, schema), ref)


if __name__ == "__main__":
    unittest.main()
