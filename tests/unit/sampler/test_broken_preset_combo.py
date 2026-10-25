"""MP-36 NEW-2 (MP-38): a broken preset file keeps its id in the sampler's ``preset`` combo.

Without the id, a saved workflow that uses the preset fails in ComfyUI's own validation ("value not in list") and never
reaches the typed ``invalid_preset_file`` error naming the file. A MISSING file is not listed (nothing to explain)."""
import os
import shutil
import tempfile
import unittest
from unittest import mock

from moss_photon.core import errors, presets
from moss_photon.nodes import sampler


class BrokenPresetComboTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for name in os.listdir(presets.preset_dir()):
            shutil.copy(os.path.join(presets.preset_dir(), name), self.tmp)
        patch = mock.patch.object(presets, "preset_dir", lambda: self.tmp)
        patch.start()
        self.addCleanup(patch.stop)

    def break_file(self, name):
        with open(os.path.join(self.tmp, name), "w", encoding="utf-8") as f:
            f.write("{not json")

    def combo(self, cls=sampler.MossPhotonSampler):
        return cls.INPUT_TYPES()["required"]["preset"][0]

    def test_a_healthy_catalog_lists_every_preset_in_order(self):
        self.assertEqual(self.combo(), list(presets.ALL_PRESET_IDS))

    def test_a_broken_file_keeps_its_id_in_both_samplers(self):
        self.break_file("photo_safe.json")
        for cls in (sampler.MossPhotonSampler, sampler.MossPhotonSamplerAdvanced):
            with self.subTest(node=cls.__name__):
                self.assertEqual(self.combo(cls), list(presets.ALL_PRESET_IDS))
        self.assertNotIn("photo_safe", presets.catalog_ids())  # control: the catalog itself does drop it

    def test_selecting_it_raises_the_typed_error_naming_the_file(self):
        self.break_file("photo_safe.json")
        with self.assertRaises(errors.InvalidPresetFile) as cm:
            sampler.effective_cfg("photo_safe", 0.0, "", "MossPhotonSampler")
        self.assertIn("photo_safe.json", cm.exception.issue.param)

    def test_the_other_presets_still_work(self):
        self.break_file("photo_safe.json")
        self.assertEqual(sampler.effective_cfg("reference", 0.0, "", "MossPhotonSampler")[1], "preset")

    def test_a_missing_file_is_not_listed(self):
        os.remove(os.path.join(self.tmp, "scene_safe.json"))
        self.assertNotIn("scene_safe", self.combo())
        self.assertEqual(len(self.combo()), len(presets.ALL_PRESET_IDS) - 1)

    def test_a_user_added_broken_file_is_not_a_choice(self):
        shutil.copy(os.path.join(self.tmp, "reference.json"), os.path.join(self.tmp, "copy.json"))
        self.assertEqual(self.combo(), list(presets.ALL_PRESET_IDS))

    def test_a_broken_schema_falls_back_to_all_ids(self):
        os.remove(os.path.join(self.tmp, "schema.json"))
        self.assertEqual(self.combo(), list(presets.ALL_PRESET_IDS))


if __name__ == "__main__":
    unittest.main()
