"""MP-7: resolve / parse_pin / effective_view (design 1.3, 4.9, 6.2)."""
import copy
import importlib.util
import json
import os
import shutil
import tempfile
import unittest

from moss_photon.core import canon, errors, presets

PINNED = {"pin": ""}


def codes(resolved):
    return [w.code for w in resolved.warnings]


def temp_catalog(tmp, manifest="calibration/sft/photo_safe.json", fallback=None):
    for n in os.listdir(presets.preset_dir()):
        shutil.copy(os.path.join(presets.preset_dir(), n), tmp)
    path = os.path.join(tmp, "photo_safe.json")
    p = json.load(open(path, encoding="utf-8"))
    p["calibration"]["manifest"] = manifest
    if fallback:
        p["fallback"] = fallback
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(p, f, indent=2)


class LayerTest(unittest.TestCase):
    def test_default_equals_preset(self):
        for pid, p in presets.catalog().items():
            r = presets.resolve(pid)
            smp = {k: v for k, v in p["sampling"].items() if k != "override_bounds"}  # preset metadata, not a setting
            self.assertEqual(r.effective, {"sampling": smp, "budget": p["budget"]}, pid)
            self.assertFalse(r.customized)
            self.assertEqual(r.diff, ())
            self.assertEqual(r.layers_applied, ("preset",))
            self.assertEqual(r.preset, p)

    def test_layer_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_catalog(tmp)
            manifest = {"schema_version": 1, "overrides": {"cfg": 4.0}}
            r = presets.resolve("photo_safe", directory=tmp, model_family="sft", lora_profile={"stack": []},
                                overrides={"steps": 40}, manifest_loader=lambda fam, pid: manifest)
            self.assertEqual(r.layers_applied, ("preset", "manifest", "lora_profile", "override"))
            self.assertEqual(r.effective["sampling"]["cfg"], 4.0)
            self.assertEqual(r.effective["sampling"]["steps"], 40)

    def test_override_marks_customized_and_diff(self):
        r = presets.resolve("reference", overrides={"cfg": 4.5, "steps": 30})
        self.assertTrue(r.customized)
        self.assertEqual(r.layers_applied, ("preset", "override"))
        self.assertEqual({d["path"]: (d["base"], d["effective"]) for d in r.diff},
                         {"sampling.cfg": (3.0, 4.5), "sampling.steps": (50, 30), "budget.nominal_steps": (50, 30)})
        self.assertIn("customized_run", codes(r))
        self.assertEqual(r.preset["sampling"]["cfg"], 3.0)   # the recipe itself is untouched

    def test_override_equal_to_preset_is_not_customized(self):
        r = presets.resolve("reference", overrides={"cfg": 3, "steps": 50.0, "slg_layers": [22]})
        self.assertFalse(r.customized)
        self.assertEqual(r.diff, ())
        self.assertNotIn("customized_run", codes(r))

    def test_override_none_means_absent(self):
        r = presets.resolve("reference", overrides={"cfg": None})
        self.assertFalse(r.customized)

    def test_resolve_casts_int(self):
        r = presets.resolve("reference", overrides={"steps": 40.0, "slg_layers": [21.0, 22.0], "cfg": 4})
        self.assertIs(type(r.effective["sampling"]["steps"]), int)
        self.assertIs(type(r.effective["budget"]["nominal_steps"]), int)
        self.assertEqual([type(x) for x in r.effective["sampling"]["slg_layers"]], [int, int])
        self.assertIs(type(r.effective["sampling"]["cfg"]), float)

    def test_override_bounds(self):
        bad = [{"steps": 0}, {"steps": 1001}, {"steps": 50.5}, {"steps": True}, {"steps": "50"}, {"cfg": 0.5},
               {"cfg": 31}, {"cfg": True}, {"shift": 0}, {"slg_mode": "wild"}, {"slg_layers": [22, 22]},
               {"slg_layers": [-1]}, {"slg_layers": 22}, {"slg_start": 1.5}, {"slg_end": -0.5}]
        for ov in bad:
            with self.assertRaises(errors.InvalidOverride, msg=str(ov)) as cm:
                presets.resolve("reference", overrides=ov)
            self.assertEqual(cm.exception.code, "invalid_override")
            self.assertEqual(cm.exception.issue.param, next(iter(ov)))

    def test_slg_start_gt_end(self):
        with self.assertRaises(errors.InvalidOverride) as cm:
            presets.resolve("reference", overrides={"slg_start": 0.8, "slg_end": 0.2})
        self.assertEqual(cm.exception.issue.param, "slg_start")
        with self.assertRaises(errors.InvalidOverride):
            presets.resolve("reference", overrides={"slg_start": 0.5, "slg_end": 0.4})
        self.assertFalse(presets.resolve("reference", overrides={"slg_start": 0.5, "slg_end": 0.5}).effective is None)

    def test_slg_mode_needs_layers(self):
        with self.assertRaises(errors.InvalidOverride):
            presets.resolve("reference", overrides={"slg_layers": []})
        r = presets.resolve("reference", overrides={"slg_mode": "off", "slg_layers": []})
        self.assertTrue(r.customized)

    def test_layer_ge_num_layers(self):
        with self.assertRaises(errors.InvalidOverride) as cm:
            presets.resolve("reference", overrides={"slg_layers": [42]}, num_layers=42)
        self.assertEqual(cm.exception.issue.param, "slg_layers")
        presets.resolve("reference", overrides={"slg_layers": [41]}, num_layers=42)
        presets.resolve("reference", overrides={"slg_layers": [200]})   # model unknown: schema bound only

    def test_presets_never_touch_text_seed_size_source_lora_checkpoint(self):
        forbidden = ["prompt", "text", "seed", "width", "height", "size", "batch", "latent", "image", "source",
                     "lora_strength", "strength_model", "strength_clip", "checkpoint", "checkpoint_type", "model",
                     "family", "restart", "ancestral_noise", "sampler", "scheduler", "postprocess", "sampling.steps"]
        for key in forbidden:
            with self.assertRaises(errors.InvalidOverride, msg=key):
                presets.resolve("reference", overrides={key: 1})
        # even a value equal to the preset's own is refused when the name is not an overridable setting
        for ov in ({"restart": False}, {"ancestral_noise": False}, {"sampler": "euler"}, {"scheduler": "sgm_uniform"},
                   {"seed": 0}, {"nominal_steps": 50}, {"max_forwards": None, "width": 1024}):
            with self.assertRaises(errors.InvalidOverride, msg=str(ov)):
                presets.resolve("reference", overrides=ov)
        keys = {"sampling": set(presets.catalog()["reference"]["sampling"]), "budget": set(presets.catalog()["reference"]["budget"])}
        for pid in presets.catalog_ids():
            r = presets.resolve(pid, overrides={"cfg": 1.5 if pid in presets.TURBO_PRESET_IDS else 4.0})
            self.assertEqual(set(r.effective), {"sampling", "budget"})
            for section in r.effective:
                self.assertEqual(set(r.effective[section]), keys[section], pid)
            flat = json.dumps(r.effective)
            for word in ("seed", "width", "height", "prompt", "lora_strength", "checkpoint"):
                self.assertNotIn(word, flat)
            self.assertEqual(r.preset, presets.catalog()[pid])
            self.assertEqual({d["path"] for d in r.diff}, {"sampling.cfg"})

    def test_resolve_does_not_mutate_catalog(self):
        before = presets.catalog()
        r = presets.resolve("reference", overrides={"slg_layers": [3, 4], "cfg": 5})
        r.effective["sampling"]["slg_layers"].append(9)
        r.preset["sampling"]["steps"] = 1
        presets.effective_view("reference", {"steps": 10})
        self.assertEqual(presets.catalog(), before)

    def test_to_dict_has_id_version_hash(self):
        r = presets.resolve("reference")
        d = r.to_dict()
        json.dumps(d)
        self.assertEqual((d["id"], d["version"]), ("reference", "1.0.0"))
        self.assertEqual(d["preset_sha256"], presets.preset_sha256(presets.catalog()["reference"]))
        self.assertIsNone(d["manifest_sha256"])
        self.assertEqual(d["pin"], presets.pin_of(presets.catalog()["reference"]))
        d["effective"]["sampling"]["steps"] = 1
        self.assertEqual(r.effective["sampling"]["steps"], 50)

    def test_unknown_preset(self):
        with self.assertRaises(errors.UnknownPreset) as cm:
            presets.resolve("nope")
        self.assertEqual(cm.exception.code, "invalid_preset")
        self.assertIn("reference", cm.exception.issue.fix)


class ManifestTest(unittest.TestCase):
    def test_default_presets_name_no_manifest(self):
        calls = []
        r = presets.resolve("reference", manifest_loader=lambda *a: calls.append(a))
        self.assertEqual(calls, [])
        self.assertEqual(r.layers_applied, ("preset",))

    def test_missing_manifest_falls_back_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_catalog(tmp)
            r = presets.resolve("photo_safe", directory=tmp, model_family="sft", manifest_loader=lambda f, p: None)
            self.assertIn("manifest_missing", codes(r))
            self.assertIsNone(r.manifest_sha256)
            self.assertEqual(r.layers_applied, ("preset",))
            self.assertEqual(r.effective, presets.resolve("reference").effective)
            self.assertFalse(r.customized)
            self.assertEqual(r.warnings[-1].severity, "warning")

    def test_real_loader_without_file_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_catalog(tmp)
            r = presets.resolve("photo_safe", directory=tmp, model_family="moss-image-sft")
            self.assertIn("manifest_missing", codes(r))

    def test_manifest_required_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_catalog(tmp)
            with self.assertRaises(errors.CalibrationRequired) as cm:
                presets.resolve("photo_safe", directory=tmp, requires_calibration=("tone_curve",),
                                manifest_loader=lambda f, p: None)
            self.assertEqual(cm.exception.code, "calibration_required")
        with tempfile.TemporaryDirectory() as tmp:
            temp_catalog(tmp, fallback="error_on_incompatibility")
            with self.assertRaises(errors.CalibrationRequired):
                presets.resolve("photo_safe", directory=tmp, manifest_loader=lambda f, p: None)

    def test_manifest_applies_and_hashes(self):
        manifest = {"schema_version": 1, "overrides": {"cfg": 3.5, "slg_end": 0.9}}
        with tempfile.TemporaryDirectory() as tmp:
            temp_catalog(tmp)
            seen = []
            r = presets.resolve("photo_safe", directory=tmp, model_family="sft",
                                manifest_loader=lambda f, p: seen.append((f, p)) or manifest)
            self.assertEqual(seen, [("sft", "photo_safe")])
            self.assertEqual(r.layers_applied, ("preset", "manifest"))
            self.assertEqual((r.effective["sampling"]["cfg"], r.effective["sampling"]["slg_end"]), (3.5, 0.9))
            self.assertEqual(r.manifest_sha256, canon.sha256_hex(canon.canonical_json(manifest)))
            self.assertFalse(r.customized)     # a calibration is not a user override
            # an explicit override on top still wins and is the only thing in the diff
            r2 = presets.resolve("photo_safe", directory=tmp, overrides={"cfg": 6.0}, manifest_loader=lambda f, p: manifest)
            self.assertEqual(r2.effective["sampling"]["cfg"], 6.0)
            self.assertEqual([d["path"] for d in r2.diff], ["sampling.cfg"])
            self.assertEqual(r2.diff[0]["base"], 3.5)

    def test_invalid_manifest_is_a_file_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_catalog(tmp)
            with self.assertRaises(errors.InvalidPresetFile):
                presets.resolve("photo_safe", directory=tmp, manifest_loader=lambda f, p: {"overrides": {"cfg": 99}})

    def test_lora_profile_layer_changes_no_numbers(self):
        plain = presets.resolve("identity_safe")
        r = presets.resolve("identity_safe", lora_profile={"kind": "photon_identity_profile"})
        self.assertEqual(r.layers_applied, ("preset", "lora_profile"))
        self.assertEqual(r.effective, plain.effective)
        self.assertFalse(r.customized)

    def test_load_manifest_edges(self):
        self.assertIsNone(presets.load_manifest(None, "reference"))
        self.assertIsNone(presets.load_manifest("moss-image-sft", "reference"))
        for evil in ("..", "../x", "a/b", "a\\b", ".hidden", ""):
            self.assertIsNone(presets.load_manifest(evil, "reference"), evil)
            self.assertIsNone(presets.load_manifest("moss-image-sft", evil), evil)

    def test_load_manifest_reads_a_file_and_rejects_garbage(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "fam"))
            old = presets._calibration_dir
            presets._calibration_dir = lambda: tmp
            try:
                with open(os.path.join(tmp, "fam", "photo_safe.json"), "w", encoding="utf-8") as f:
                    f.write('{"schema_version": 1, "overrides": {}}')
                self.assertEqual(presets.load_manifest("fam", "photo_safe"), {"schema_version": 1, "overrides": {}})
                with open(os.path.join(tmp, "fam", "scene_safe.json"), "w", encoding="utf-8") as f:
                    f.write("{broken")
                with self.assertRaises(errors.InvalidPresetFile):
                    presets.load_manifest("fam", "scene_safe")
            finally:
                presets._calibration_dir = old


class PinTest(unittest.TestCase):
    def setUp(self):
        self.p = presets.catalog()["reference"]
        self.sha8 = presets.preset_sha256(self.p)[:8]

    def test_parse_pin_grammar(self):
        self.assertIsNone(presets.parse_pin(""))
        self.assertEqual(presets.parse_pin("reference@1.0.0"), {"id": "reference", "version": "1.0.0", "sha8": None})
        self.assertEqual(presets.parse_pin("photo_safe@10.20.30#0123abcd"),
                         {"id": "photo_safe", "version": "10.20.30", "sha8": "0123abcd"})
        for bad in ("reference@1.0", "reference@1.0.0#zz", "reference@1.0.0#0123ABCD", "reference@1.0.0#0123abc",
                    "reference@1.0.0#0123abcde", "reference", "@1.0.0", "reference@1.0.0 ", " reference@1.0.0",
                    "reference@1.0.0\n", "reference@v1.0.0", "Reference@1.0.0", "reference@1.0.0#", None, 5, b"x"):
            with self.assertRaises(errors.PresetVersionMismatch, msg=repr(bad)) as cm:
                presets.parse_pin(bad)
            self.assertEqual(cm.exception.code, "preset_version_mismatch")

    def test_pin_match(self):
        for pin in ("reference@1.0.0", "reference@1.0.0#" + self.sha8):
            r = presets.resolve("reference", pin=pin)
            self.assertTrue(r.pinned)
            self.assertNotIn("preset_unpinned", codes(r))
            self.assertEqual(r.pin, "reference@1.0.0#" + self.sha8)

    def test_pin_mismatch_id_version_sha8(self):
        for pin, field in (("photo_safe@1.0.0", "id"), ("reference@1.0.1", "version"), ("reference@2.0.0", "version"),
                           ("reference@1.0.0#00000000", "sha8"), ("reference@1.0", "malformed")):
            with self.assertRaises(errors.PresetVersionMismatch, msg=pin) as cm:
                presets.resolve("reference", pin=pin)
            issue = cm.exception.issue
            self.assertEqual((issue.code, issue.param), ("preset_version_mismatch", "preset_pin"))
            self.assertIn("preset_pin", issue.fix)
            if field != "malformed":
                self.assertIn(field, issue.message)
                self.assertIn("reference@1.0.0#" + self.sha8, issue.message)

    def test_empty_pin_warns_unpinned(self):
        r = presets.resolve("reference", pin="")
        self.assertFalse(r.pinned)
        self.assertIn("preset_unpinned", codes(r))
        w = [x for x in r.warnings if x.code == "preset_unpinned"][0]
        self.assertEqual(w.severity, "warning")
        self.assertIn("reference@1.0.0#" + self.sha8, w.message)
        self.assertEqual(presets.resolve("reference").pin, r.pin)    # the version used is still recorded

    def test_node_name_flows_into_issues(self):
        self.assertEqual(presets.resolve("reference", node="MossPhotonSamplerAdvanced").warnings[0].node,
                         "MossPhotonSamplerAdvanced")


class EffectiveViewTest(unittest.TestCase):
    def test_effective_view_pure(self):
        v = presets.effective_view("reference")
        json.dumps(v)
        self.assertEqual(set(v), {"schema_version", "preset", "effective", "customized", "diff", "forwards", "requirements",
                                  "guards", "warnings", "not_known_before_the_run"})
        self.assertEqual(set(v["preset"]), {"id", "version", "pin", "status", "label", "group", "recommended_final", "summary"})
        self.assertEqual(v["preset"]["version"], "1.0.0")
        self.assertFalse(v["customized"])
        self.assertEqual(v["warnings"], [])          # the panel is not a run: no unpinned warning
        self.assertEqual(v["not_known_before_the_run"], ["live model shift", "checkpoint family", "LoRA stack", "time"])
        self.assertEqual(v, presets.effective_view("reference"))

    def test_effective_view_exact_forwards(self):
        self.assertEqual(presets.effective_view("reference")["forwards"],
                         {"branches_per_step": 2, "expected_branch_forwards": 100,
                          "note": "exact for the steps shown; custom sigmas replace steps"})
        self.assertEqual(presets.effective_view("draft_fast")["forwards"]["expected_branch_forwards"], 50)
        self.assertEqual(presets.effective_view("reference", {"cfg": 1.0})["forwards"]["expected_branch_forwards"], 50)
        self.assertEqual(presets.effective_view("draft_fast", {"cfg": 1})["forwards"]["branches_per_step"], 1)
        self.assertEqual(presets.effective_view("draft_fast", {"cfg": 1})["forwards"]["expected_branch_forwards"], 25)

    def test_overrides_reflected_in_diff(self):
        v = presets.effective_view("reference", {"steps": 40, "cfg": 4.0})
        self.assertTrue(v["customized"])
        self.assertEqual(v["effective"]["sampling"]["steps"], 40)
        self.assertEqual({d["path"] for d in v["diff"]}, {"sampling.steps", "sampling.cfg", "budget.nominal_steps"})
        self.assertEqual([w["code"] for w in v["warnings"]], ["customized_run"])
        self.assertEqual(v["forwards"]["expected_branch_forwards"], 80)

    def test_requirements_and_guards_listed(self):
        v = presets.effective_view("identity_safe")
        self.assertEqual([r["kind"] for r in v["requirements"]], ["requires_lora_role", "recommends_input"])
        self.assertEqual([g["id"] for g in v["guards"]], ["identity_score"])
        v["requirements"].clear()
        self.assertTrue(presets.effective_view("identity_safe")["requirements"])

    def test_errors(self):
        with self.assertRaises(errors.UnknownPreset):
            presets.effective_view("nope")
        with self.assertRaises(errors.InvalidOverride):
            presets.effective_view("reference", {"cfg": 0})

    def test_branches_agree_with_sigma_when_it_exists(self):
        if importlib.util.find_spec("moss_photon.core.sigma") is None:
            self.skipTest("moss_photon.core.sigma is not merged yet (MP-6); presets keeps a local copy")
        from moss_photon.core import sigma
        for cfg in (1.0, 1.0000000001, 3.0, 0.999):
            self.assertEqual(presets._branches_per_step(cfg), sigma.branches_per_step(cfg))


if __name__ == "__main__":
    unittest.main()
