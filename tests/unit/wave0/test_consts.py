"""Wave 0: core/consts.py against design 1.1 / SPEC 4.1 / SPEC 6.1."""
import ast
import os
import re
import tomllib
import unittest

from moss_photon.core import consts

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ConstsTest(unittest.TestCase):
    def test_pack_identity_matches_pyproject(self):
        with open(os.path.join(PKG_ROOT, "pyproject.toml"), "rb") as f:
            proj = tomllib.load(f)
        self.assertEqual(consts.PACK_VERSION, proj["project"]["version"])
        self.assertEqual(consts.PACK_NAME, "ComfyUI-MossPhoton")
        self.assertEqual(os.path.basename(PKG_ROOT), consts.PACK_NAME)

    def test_schema_versions(self):
        for name in ("REPORT_SCHEMA", "PRESET_SCHEMA", "SCHEDULE_INFO_SCHEMA", "CONSTRAINTS_SCHEMA",
                     "IDENTITY_PROFILE_SCHEMA", "COMPILER_VERSION"):
            with self.subTest(name=name):
                self.assertEqual(getattr(consts, name), 1)
                self.assertIs(type(getattr(consts, name)), int)

    def test_node_ids_are_the_eight_of_spec_4_1_in_order(self):
        self.assertEqual(tuple(consts.NODE_IDS), (
            "MossPhotonSampler", "MossPhotonSamplerAdvanced", "MossPhotonSchedule", "MossPhotonLoRAProfile",
            "MossPhotonPromptLayout", "MossPhotonEmptyLatent", "MossPhotonDiagnostics", "MossPhotonSaveImage"))

    def test_categories(self):
        self.assertEqual(consts.CAT_SETUP, "MossPhoton/1 Setup")
        self.assertEqual(consts.CAT_SAMPLING, "MossPhoton/2 Sampling")
        self.assertEqual(consts.CAT_DIAGNOSTICS, "MossPhoton/3 Diagnostics")
        self.assertEqual(consts.CAT_OUTPUT, "MossPhoton/4 Output")

    def test_attachment_key(self):
        self.assertEqual(consts.ATTACH_LORA_STACK, "moss_photon.lora_stack")

    def test_preset_ids_spec_6_1_order(self):
        self.assertEqual(tuple(consts.PRESET_IDS), (
            "reference", "photo_safe", "identity_safe", "wardrobe_safe", "scene_safe", "night_photo",
            "draft_fast", "advanced"))

    def test_reference_weight_dtypes(self):
        self.assertEqual(tuple(consts.REFERENCE_WEIGHT_DTYPES), ("torch.bfloat16",))

    def test_native_sigma_hash_literal(self):
        # independent copy: INV 1.3 / design 2.6
        self.assertEqual(consts.REFERENCE_SIGMAS_SHA256,
                         "9b2c66b3fb335d998fc03e73ddbafe997fe2d3cf3b187f51629bdf0332f5a536")

    def test_sft_official_hash_literal(self):
        # independent copy: INV 7 / design 1.8 (equals the HF LFS oid of moss_image_sft_bf16.safetensors)
        self.assertEqual(consts.SFT_BF16_SHA256,
                         "f96738d8116d78b7316b57c663d8c98591ddafab7488f48e3099b7655e34ab6f")

    def test_hashes_are_lowercase_hex64(self):
        for name in ("REFERENCE_SIGMAS_SHA256", "SFT_BF16_SHA256"):
            self.assertRegex(getattr(consts, name), HEX64)

    def test_hash_constants_are_literals_in_source(self):
        """The hashes must be written as string literals, not computed or read from a file."""
        path = os.path.join(PKG_ROOT, "moss_photon", "core", "consts.py")
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        found = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ("REFERENCE_SIGMAS_SHA256", "SFT_BF16_SHA256"):
                    found[node.targets[0].id] = node.value
        self.assertEqual(sorted(found), ["REFERENCE_SIGMAS_SHA256", "SFT_BF16_SHA256"])
        for name, value in found.items():
            self.assertIsInstance(value, ast.Constant, name)
            self.assertIsInstance(value.value, str, name)

    def test_consts_is_a_leaf_that_does_not_import_errors(self):
        path = os.path.join(PKG_ROOT, "moss_photon", "core", "consts.py")
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(("." * node.level) + (node.module or ""))
        self.assertEqual(imported, [], "consts must stay a leaf with no imports")
        self.assertFalse(hasattr(consts, "REASON_CODES"))


if __name__ == "__main__":
    unittest.main()
