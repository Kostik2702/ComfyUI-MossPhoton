"""Sidecar <stem>.photon.json (design 2.4, A10): valid, invalid, curated flag."""
import json
import os
import tempfile
import unittest

from moss_photon.core import lora

VALID = {"schema_version": 1, "verified_strength": {"min": 0.6, "max": 0.9, "recommended": 0.8},
         "dataset": {"images": 42}, "recommended_prompt": "a portrait photo of Anna", "trained_on": "sft"}


class SidecarTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.lora_path = os.path.join(self._tmp.name, "character_a.safetensors")
        with open(self.lora_path, "wb") as f:
            f.write(b"x")

    def write(self, obj, raw=None):
        with open(lora.sidecar_path(self.lora_path), "wb") as f:
            f.write(raw if raw is not None else json.dumps(obj).encode("utf-8"))

    def test_path_is_the_stem_plus_photon_json(self):
        self.assertTrue(lora.sidecar_path(self.lora_path).endswith("character_a.photon.json"))
        self.assertEqual(os.path.dirname(lora.sidecar_path(self.lora_path)), self._tmp.name)

    def test_absent_sidecar_is_none_and_not_curated(self):
        self.assertIsNone(lora.read_sidecar(self.lora_path))
        block = lora.sidecar_block(None)
        self.assertEqual(block, {"found": False, "curated": False, "verified_strength": None, "dataset": None,
                                 "recommended_prompt": None})

    def test_valid_sidecar_is_curated(self):
        self.write(VALID)
        sc = lora.read_sidecar(self.lora_path)
        self.assertTrue(sc["found"] and sc["curated"])
        self.assertEqual(sc["verified_strength"], {"min": 0.6, "max": 0.9, "recommended": 0.8})
        self.assertEqual(sc["trained_on"], "sft")
        block = lora.sidecar_block(sc)
        self.assertEqual(set(block), {"found", "curated", "verified_strength", "dataset", "recommended_prompt"})
        self.assertEqual(block["recommended_prompt"], "a portrait photo of Anna")

    def test_minimal_sidecar(self):
        self.write({})
        sc = lora.read_sidecar(self.lora_path)
        self.assertEqual((sc["found"], sc["curated"], sc["verified_strength"], sc["trained_on"]), (True, True, None, None))

    def test_utf8_bom_is_accepted(self):
        self.write(None, raw=b"\xef\xbb\xbf" + json.dumps(VALID).encode("utf-8"))
        self.assertIsNotNone(lora.read_sidecar(self.lora_path))

    def test_invalid_sidecars_raise_value_error(self):
        bad = [
            (None, b"{not json"),
            (None, b"[1, 2]"),
            ({"unexpected": 1}, None),
            ({"schema_version": 2}, None),
            ({"verified_strength": {"min": 0.9, "max": 0.6, "recommended": 0.8}}, None),
            ({"verified_strength": {"min": 0.6, "max": 0.9}}, None),
            ({"verified_strength": {"min": 0.6, "max": 0.9, "recommended": "x"}}, None),
            ({"verified_strength": {"min": True, "max": 0.9, "recommended": 0.8}}, None),
            ({"dataset": "folder"}, None),
            ({"recommended_prompt": 5}, None),
            ({"trained_on": "both"}, None),
            (None, b"\xff\xfe\x00"),
        ]
        for obj, raw in bad:
            with self.subTest(obj=obj, raw=raw):
                self.write(obj, raw=raw)
                with self.assertRaises(ValueError) as ctx:
                    lora.read_sidecar(self.lora_path)
                self.assertIn("photon.json", str(ctx.exception))
                self.assertNotIn(self._tmp.name, str(ctx.exception))  # basename only, never the folder

    def test_known_distill_supplies_the_verified_strength_without_a_file(self):
        row = lora.distill_entry("7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2")
        block = lora.sidecar_block(None, row)
        self.assertEqual(block, {"found": False, "curated": True,
                                 "verified_strength": {"min": 1.0, "max": 1.0, "recommended": 1.0},
                                 "dataset": None, "recommended_prompt": None})

    def test_found_sidecar_wins_over_the_table(self):
        self.write({"verified_strength": {"min": 0.5, "max": 1.5, "recommended": 1.0}})
        sc = lora.read_sidecar(self.lora_path)
        row = lora.distill_entry("7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2")
        self.assertEqual(lora.sidecar_block(sc, row)["verified_strength"]["max"], 1.5)


if __name__ == "__main__":
    unittest.main()
