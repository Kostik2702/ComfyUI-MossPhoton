"""MP-7 / MP-30 m22: broken preset files give invalid_preset_file naming the file (design 3.1, 6.2).

A broken SCHEMA disables the whole catalog (nothing can be validated). A broken PRESET file disables only that
preset: the others keep working and the broken file is reported as a warning (``catalog_issues``)."""
import json
import os
import shutil
import tempfile
import unittest

from moss_photon.core import errors, presets


class InvalidFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for n in os.listdir(presets.preset_dir()):
            shutil.copy(os.path.join(presets.preset_dir(), n), self.tmp)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def rewrite(self, name, fn):
        obj = json.load(open(self.path(name), encoding="utf-8"))
        fn(obj)
        with open(self.path(name), "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f)

    def expect_whole_catalog_error(self, name_in_param):
        with self.assertRaises(errors.InvalidPresetFile) as cm:
            presets.catalog(self.tmp)
        self.assertEqual(cm.exception.code, "invalid_preset_file")
        self.assertIn(name_in_param, cm.exception.issue.param)
        return cm.exception.issue

    def expect_one_preset_disabled(self, preset_id, file_name):
        """The broken file disables ``preset_id`` only; returns the error ``resolve`` raises for it."""
        cat = presets.catalog(self.tmp)
        self.assertNotIn(preset_id, cat)
        self.assertEqual(sorted(cat), sorted(set(presets.ALL_PRESET_IDS) - {preset_id}))
        self.assertEqual(presets.catalog_ids(self.tmp), [p for p in presets.ALL_PRESET_IDS if p != preset_id])
        self.assertEqual(len(presets.describe_catalog(self.tmp)), len(presets.ALL_PRESET_IDS) - 1)
        issues = presets.catalog_issues(self.tmp)
        self.assertEqual(len(issues), 1)
        self.assertEqual((issues[0].code, issues[0].severity), ("invalid_preset_file", "warning"))
        self.assertIn(file_name, issues[0].param)
        self.assertIn("disabled", issues[0].fix)
        other = next(p for p in presets.ALL_PRESET_IDS if p != preset_id)
        self.assertEqual(presets.resolve(other, directory=self.tmp).preset["id"], other)  # the others keep working
        with self.assertRaises(errors.InvalidPresetFile) as cm:
            presets.resolve(preset_id, directory=self.tmp)
        self.assertIn(file_name, cm.exception.issue.param)
        self.assertEqual(cm.exception.issue.severity, "error")
        return cm.exception.issue

    def test_pristine_copy_is_valid(self):
        self.assertEqual(len(presets.catalog(self.tmp)), len(presets.ALL_PRESET_IDS))
        self.assertEqual(presets.catalog_issues(self.tmp), [])

    def test_broken_json_disables_only_that_preset(self):
        with open(self.path("reference.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        self.expect_one_preset_disabled("reference", "reference.json")

    def test_missing_schema(self):
        os.remove(self.path("schema.json"))
        self.expect_whole_catalog_error("schema.json")

    def test_unparseable_schema(self):
        with open(self.path("schema.json"), "w", encoding="utf-8") as f:
            f.write("[")
        self.expect_whole_catalog_error("schema.json")

    def test_schema_with_unsupported_keyword(self):
        s = json.load(open(self.path("schema.json"), encoding="utf-8"))
        s["oneOf"] = []
        with open(self.path("schema.json"), "w", encoding="utf-8") as f:
            json.dump(s, f)
        self.expect_whole_catalog_error("schema.json")

    def test_a_user_added_broken_file_disables_nothing_and_is_reported(self):
        shutil.copy(self.path("reference.json"), self.path("copy.json"))
        self.assertEqual(len(presets.catalog(self.tmp)), len(presets.ALL_PRESET_IDS))
        issues = presets.catalog_issues(self.tmp)
        self.assertEqual([i.param for i in issues], ["presets/copy.json"])
        self.assertIn("id", issues[0].message)
        self.assertEqual(presets.resolve("reference", directory=self.tmp).preset["id"], "reference")

    def test_id_differs_from_stem(self):
        self.rewrite("photo_safe.json", lambda o: o.update(id="scene_safe"))
        self.expect_one_preset_disabled("photo_safe", "photo_safe.json")
        self.assertEqual(presets.resolve("scene_safe", directory=self.tmp).preset["id"], "scene_safe")

    def test_schema_violation_names_json_path(self):
        self.rewrite("draft_fast.json", lambda o: o["sampling"].update(steps=0))
        issue = self.expect_one_preset_disabled("draft_fast", "draft_fast.json")
        self.assertIn("sampling.steps", issue.message)

    def test_cross_rule_violation(self):
        self.rewrite("draft_fast.json", lambda o: o["budget"].update(nominal_steps=24))
        issue = self.expect_one_preset_disabled("draft_fast", "draft_fast.json")
        self.assertIn("nominal_steps", issue.message)

    def test_two_broken_files_disable_two_presets(self):
        self.rewrite("draft_fast.json", lambda o: o["sampling"].update(steps=0))
        self.rewrite("night_photo.json", lambda o: o["budget"].update(nominal_steps=24))
        cat = presets.catalog(self.tmp)
        self.assertEqual(sorted(set(presets.ALL_PRESET_IDS) - set(cat)), ["draft_fast", "night_photo"])
        self.assertEqual(sorted(i.param for i in presets.catalog_issues(self.tmp)),
                         ["presets/draft_fast.json", "presets/night_photo.json"])

    def test_catalog_sha256_still_works_and_changes_when_a_file_is_repaired(self):
        self.rewrite("draft_fast.json", lambda o: o["sampling"].update(steps=0))
        broken = presets.catalog_sha256(self.tmp)
        shutil.copy(os.path.join(presets.preset_dir(), "draft_fast.json"), self.path("draft_fast.json"))
        self.assertNotEqual(presets.catalog_sha256(self.tmp), broken)

    def test_resolve_uses_the_same_error(self):
        os.remove(self.path("schema.json"))
        with self.assertRaises(errors.InvalidPresetFile):
            presets.resolve("reference", directory=self.tmp)

    def test_missing_preset_file_is_unknown_preset_not_a_crash(self):
        os.remove(self.path("scene_safe.json"))
        self.assertNotIn("scene_safe", presets.catalog_ids(self.tmp))
        with self.assertRaises(errors.UnknownPreset):
            presets.resolve("scene_safe", directory=self.tmp)

    def test_missing_directory(self):
        with self.assertRaises(errors.InvalidPresetFile):
            presets.catalog(os.path.join(self.tmp, "nope"))


if __name__ == "__main__":
    unittest.main()
