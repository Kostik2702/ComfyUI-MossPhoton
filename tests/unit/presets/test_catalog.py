"""MP-7: the eight shipped presets and the catalog functions (design 2.2, 6.2)."""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from moss_photon.core import canon, consts, errors, presets

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SAFE = ("photo_safe", "identity_safe", "wardrobe_safe", "scene_safe", "night_photo", "advanced")


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)


class CatalogFilesTest(unittest.TestCase):
    def test_eight_files_exact_ids(self):
        # the eight of SPEC 6.1 plus the two turbo presets of MP-23
        names = sorted(n for n in os.listdir(presets.preset_dir()) if n.endswith(".json") and n != "schema.json")
        self.assertEqual(names, sorted(pid + ".json" for pid in presets.ALL_PRESET_IDS))
        self.assertEqual(len(consts.PRESET_IDS), 8)
        self.assertEqual(presets.catalog_ids(), list(presets.ALL_PRESET_IDS))
        self.assertEqual(list(presets.catalog()), list(presets.ALL_PRESET_IDS))
        for pid, p in presets.catalog().items():
            self.assertEqual(p["id"], pid)

    def test_no_gitkeep_left(self):
        self.assertFalse(os.path.exists(os.path.join(presets.preset_dir(), ".gitkeep")))

    def test_reason_enum_matches_errors(self):
        # The schema is static JSON, the registry is code. Only RAISABLE codes (those with an exception class) are
        # constraint reasons; status codes (warnings/infos) are not.
        enum = presets.load_schema()["$defs"]["constraint"]["properties"]["reason"]["enum"]
        raisable = {code for code, info in errors.REASONS.items() if info.exc is not None}
        self.assertEqual(set(enum), raisable | {"none"})
        self.assertEqual(len(enum), len(set(enum)))

    def test_all_valid_with_cross_rules(self):
        for pid, p in presets.catalog().items():
            self.assertEqual(presets.validate_preset(p, stem=pid), [], pid)

    def test_safe_presets_numerically_equal_reference(self):
        cat = presets.catalog()
        for pid in SAFE:
            self.assertEqual(cat[pid]["sampling"], cat["reference"]["sampling"], pid)
            self.assertEqual(cat[pid]["budget"], cat["reference"]["budget"], pid)
            r = presets.resolve(pid, pin="")
            self.assertEqual(r.effective, presets.resolve("reference", pin="").effective, pid)

    def test_statuses_and_flags(self):
        cat = presets.catalog()
        self.assertEqual(cat["reference"]["status"], "stable")
        for pid in ("photo_safe", "identity_safe", "wardrobe_safe", "scene_safe"):
            self.assertEqual(cat[pid]["status"], "candidate", pid)
            self.assertTrue(cat[pid]["ui"]["recommended_final"], pid)
        self.assertEqual(cat["night_photo"]["status"], "experimental")
        self.assertEqual(cat["draft_fast"]["status"], "preview_only")
        self.assertEqual(cat["advanced"]["status"], "experimental")
        for pid in ("night_photo", "draft_fast", "advanced"):
            self.assertFalse(cat[pid]["ui"]["recommended_final"], pid)
        self.assertTrue(cat["reference"]["ui"]["recommended_final"])

    def test_draft_fast_budget(self):
        d = presets.catalog()["draft_fast"]
        self.assertEqual(d["sampling"]["steps"], 25)
        self.assertEqual(d["budget"]["nominal_steps"], 25)
        self.assertEqual(presets.effective_view("draft_fast")["forwards"]["expected_branch_forwards"], 50)

    def test_slg_window_fields(self):
        for pid, p in presets.catalog().items():
            s = p["sampling"]
            if pid in presets.TURBO_PRESET_IDS:  # SLG is off in turbo (owner Q6)
                self.assertEqual((s["slg_mode"], s["slg_layers"], s["slg_start"], s["slg_end"]), ("off", [], 0.0, 1.0), pid)
                continue
            self.assertEqual((s["slg_mode"], s["slg_layers"], s["slg_start"], s["slg_end"]),
                             ("uncond_skip", [22], 0.0, 1.0), pid)

    def test_identity_safe_requires_character_lora(self):
        p = presets.catalog()["identity_safe"]
        self.assertEqual(p["lora_policy"], "require_explicit_user_profile")
        c = [c for c in p["constraints"] if c["kind"] == "requires_lora_role"]
        self.assertEqual([(x["role"], x["severity"], x["reason"]) for x in c], [("character", "error", "missing_lora")])

    def test_block_constraints(self):
        cat = presets.catalog()
        for pid, block in (("wardrobe_safe", "wardrobe"), ("scene_safe", "scene")):
            kinds = {c["kind"]: c for c in cat[pid]["constraints"]}
            self.assertEqual(kinds["requires_block"]["block"], block)
            self.assertEqual(kinds["requires_block"]["severity"], "error")
            self.assertEqual(kinds["forbids_source_latent"]["severity"], "error")
            self.assertEqual(cat[pid]["lora_policy"], "use_explicit_user_profile_or_none")

    def test_postprocess_empty_restart_false(self):
        for pid, p in presets.catalog().items():
            self.assertEqual(p["postprocess"], [], pid)
            self.assertIs(p["sampling"]["restart"], False)
            self.assertIs(p["sampling"]["ancestral_noise"], False)

    def test_sources_cover_every_numeric_leaf(self):
        for pid, p in presets.catalog().items():
            leaves = set(presets._numeric_leaves({"sampling": p["sampling"], "budget": p["budget"]}))
            if p["budget"]["max_forwards"] is None:
                leaves.discard("budget.max_forwards")
            self.assertEqual(leaves, set(p["sources"]), pid)

    def test_numeric_leaf_excludes_bool(self):
        self.assertEqual(presets._numeric_leaves({"a": True, "b": 1, "c": [True], "d": [1, 2], "e": []}), ["b", "d"])

    def test_catalog_returns_copies(self):
        a = presets.catalog()
        a["reference"]["sampling"]["steps"] = 1
        a["reference"]["sampling"]["slg_layers"].append(3)
        del a["photo_safe"]
        b = presets.catalog()
        self.assertEqual(b["reference"]["sampling"]["steps"], 50)
        self.assertEqual(b["reference"]["sampling"]["slg_layers"], [22])
        self.assertIn("photo_safe", b)

    def test_describe_catalog(self):
        rows = presets.describe_catalog()
        self.assertEqual([r["id"] for r in rows], list(presets.ALL_PRESET_IDS))
        self.assertEqual(set(rows[0]), {"id", "status", "label", "group", "recommended_final", "summary"})
        self.assertEqual({r["id"]: r["recommended_final"] for r in rows}["draft_fast"], False)

    def test_catalog_sha256_changes_with_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in os.listdir(presets.preset_dir()):
                shutil.copy(os.path.join(presets.preset_dir(), n), tmp)
            before = presets.catalog_sha256(tmp)
            self.assertEqual(before, presets.catalog_sha256())
            p = json.load(open(os.path.join(tmp, "reference.json"), encoding="utf-8"))
            p["description"] += " x"
            write_json(os.path.join(tmp, "reference.json"), p)
            self.assertNotEqual(presets.catalog_sha256(tmp), before)

    def test_pin_of_and_hash(self):
        p = presets.catalog()["reference"]
        sha = presets.preset_sha256(p)
        self.assertEqual(sha, canon.sha256_hex(canon.canonical_json(p)))
        self.assertEqual(presets.pin_of(p), "reference@1.0.0#" + sha[:8])

    def test_validate_preset_cross_rules_fire(self):
        base = presets.catalog()["identity_safe"]

        def broken(fn):
            q = copy.deepcopy(base)
            fn(q)
            return presets.validate_preset(q, stem="identity_safe")

        self.assertTrue(broken(lambda q: q["budget"].update(nominal_steps=49)))
        self.assertTrue(broken(lambda q: q["sampling"].update(slg_start=0.9, slg_end=0.1)))
        self.assertTrue(broken(lambda q: q["sampling"].update(slg_layers=[])))
        self.assertTrue(broken(lambda q: q.update(constraints=[])))
        self.assertTrue(broken(lambda q: q["sources"].pop("sampling.cfg")))
        self.assertTrue(broken(lambda q: q["sources"].update({"sampling.restart": q["sources"]["sampling.cfg"]})))
        self.assertTrue(broken(lambda q: q.update(status="stable")))
        self.assertTrue(broken(lambda q: q.update(status="experimental")))
        self.assertTrue(presets.validate_preset(copy.deepcopy(base), stem="other"))
        self.assertEqual(presets.validate_preset(copy.deepcopy(base), stem="identity_safe"), [])


class PurityTest(unittest.TestCase):
    def test_no_comfy_or_torch_after_import_and_use(self):
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "from moss_photon.core import presets\n"
            "presets.effective_view('reference', {'steps': 40})\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in ('comfy', 'torch', 'numpy', 'jsonschema', 'folder_paths', 'nodes')]\n"
            "assert not bad, bad\n" % PKG_ROOT)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
