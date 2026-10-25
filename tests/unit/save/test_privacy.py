"""Private prompts and the workflow are not written unless embed_workflow is on (design 4.8, Q13, 6.6 test_privacy)."""
import json
import os
import unittest

import torch

from .common import CONTRACTS, SaveTestCase

SENTINEL = "SENTINEL-prompt-7f3a91-do-not-leak"
PROMPT = {"3": {"class_type": "CLIPTextEncode", "inputs": {"text": SENTINEL}}}
EXTRA = {"workflow": {"nodes": [{"id": 3, "widgets_values": [SENTINEL]}]}}


def constraints_with_sentinel():
    with open(os.path.join(CONTRACTS, "constraints.json"), "r", encoding="utf-8") as f:
        c = json.load(f)
    c["user_text"] = SENTINEL
    c["compiled_positive"] = SENTINEL + " compiled"
    return c


class PrivacyTest(SaveTestCase):
    def setUp(self):
        super().setUp()
        self.image = torch.rand((1, 16, 16, 3), generator=torch.Generator().manual_seed(1))

    def all_bytes(self):
        return b"".join(self.read(n) for n in self.listing())

    def test_sentinel_prompt_is_absent_from_every_file_by_default(self):
        self.report["text_encoder"] = constraints_with_sentinel()["text_encoder"]  # the allowed copy only
        for mode in ("both", "embedded", "sidecar"):
            for fmt in ("png", "jpeg", "webp"):
                with self.subTest(mode=mode, format=fmt):
                    self.run_save(self.image, report=json.loads(json.dumps(self.report)), filename_prefix="p",
                                  metadata=mode, format=fmt, prompt=PROMPT, extra_pnginfo=EXTRA)
        self.assertTrue(self.listing())
        self.assertNotIn(SENTINEL.encode(), self.all_bytes())
        self.assertNotIn(b"CLIPTextEncode", self.all_bytes())

    def test_embed_workflow_is_the_explicit_opt_in_and_the_sidecar_says_so(self):
        self.run_save(self.image, filename_prefix="p", metadata="both", prompt=PROMPT, extra_pnginfo=EXTRA,
                      embed_workflow=True)
        self.assertIn(SENTINEL.encode(), self.read("p_00001_.png"))
        side = self.read_json("p_00001_.json")
        self.assertTrue(side["file"]["workflow_embedded"])
        self.assertNotIn(SENTINEL, json.dumps(side))  # the sidecar itself never holds the workflow

    def test_embed_workflow_without_png_embedding_writes_nothing_private(self):
        self.run_save(self.image, filename_prefix="p", metadata="sidecar", prompt=PROMPT, extra_pnginfo=EXTRA,
                      embed_workflow=True)
        self.assertNotIn(SENTINEL.encode(), self.all_bytes())
        self.assertFalse(self.read_json("p_00001_.json")["file"]["workflow_embedded"])

    def test_objects_that_list_private_paths_are_redacted_out_of_files(self):
        # defensive layer: the report schema has no such object, but a free-form diagnostics block could carry one
        self.report["diagnostics"] = {"extra": {"secret": SENTINEL, "kept": 1, "private_paths": ["secret"]}}
        self.run_save(self.image, report=self.report, filename_prefix="p", metadata="both")
        self.assertNotIn(SENTINEL.encode(), self.all_bytes())
        self.assertNotIn(b"private_paths", self.all_bytes())
        self.assertEqual(self.read_json("p_00001_.json")["report"]["diagnostics"], {"extra": {"kept": 1}})
        self.assertIn("secret", self.report["diagnostics"]["extra"])  # the caller's report is untouched


if __name__ == "__main__":
    unittest.main()
