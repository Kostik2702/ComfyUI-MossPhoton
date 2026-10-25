"""PNG text metadata: which chunks exist for which setting (design 4.8, 6.6 test_png_metadata)."""
import io
import json
import os
import unittest

import numpy as np
import torch
from PIL import Image

from moss_photon.core import saveio

from .common import SaveTestCase, load_report

PROMPT = {"1": {"inputs": {"text": "a quiet harbour"}}}
EXTRA = {"workflow": {"nodes": [{"id": 1}]}}
INFO = {"name": "img_00001_.png", "format": "png", "width": 8, "height": 8, "lossless": True}


def decode(data):
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


class ChunkTest(unittest.TestCase):
    def chunks(self, mode, embed=False, **kw):
        return saveio.png_text_chunks(load_report(), mode, PROMPT, EXTRA, embed, file_info=INFO, **kw)

    def test_the_photon_chunk_exists_for_embedded_and_both_only(self):
        for mode, expected in (("embedded", True), ("both", True), ("sidecar", False), ("none", False)):
            with self.subTest(mode=mode):
                self.assertEqual(saveio.EMBED_KEY in self.chunks(mode), expected)
                if not expected:
                    self.assertEqual(self.chunks(mode), {})

    def test_prompt_and_workflow_chunks_only_with_embed_workflow(self):
        self.assertEqual(sorted(self.chunks("both")), ["moss_photon"])
        self.assertEqual(sorted(self.chunks("both", embed=True)), ["moss_photon", "prompt", "workflow"])
        self.assertEqual(self.chunks("sidecar", embed=True), {})  # nothing is embedded in these modes
        self.assertEqual(json.loads(self.chunks("both", embed=True)["prompt"]), PROMPT)

    def test_disable_metadata_suppresses_embedding(self):
        self.assertEqual(self.chunks("both", embed=True, disable_metadata=True), {})

    def test_chunk_payload_is_the_sidecar_object_without_bytes_and_hash(self):
        info = dict(INFO, bytes=5, sha256="0" * 64)
        chunks = saveio.png_text_chunks(load_report(), "both", None, None, False, file_info=info, batch_index=2)
        obj = json.loads(chunks["moss_photon"])
        self.assertEqual(obj["kind"], "photon_sidecar")
        self.assertEqual(obj["file"], dict(INFO, batch_index=2, workflow_embedded=False))
        self.assertEqual(obj["report"]["preset"]["id"], "reference")

    def test_embedded_workflow_flag_is_recorded_only_when_chunks_are_written(self):
        flag = lambda **kw: json.loads(self.chunks("both", **kw)["moss_photon"])["file"]["workflow_embedded"]
        self.assertFalse(flag())
        self.assertTrue(flag(embed=True))

    def test_pil_round_trip_reads_plain_text_chunks_even_for_non_ascii_text(self):
        rep = load_report()
        rep["model"]["identity"]["basename"] = "möss_模型.safetensors"
        chunks = saveio.png_text_chunks(rep, "both", None, None, False, file_info=INFO)
        data = saveio.encode_image(np.zeros((8, 8, 3), np.uint8), "png", text_chunks=chunks)
        self.assertIn(b"tEXt", data)
        self.assertNotIn(b"iTXt", data)
        img = decode(data)
        self.assertEqual(json.loads(img.text["moss_photon"])["report"]["model"]["identity"]["basename"],
                         rep["model"]["identity"]["basename"])

    def test_encode_without_chunks_has_no_text_at_all(self):
        data = saveio.encode_image(np.zeros((8, 8, 3), np.uint8), "png")
        self.assertNotIn(b"tEXt", data)
        self.assertEqual(decode(data).text, {})


class NodeMetadataTest(SaveTestCase):
    def image(self):
        return torch.rand((1, 16, 16, 3), generator=torch.Generator().manual_seed(5))

    def png_text(self):
        return decode(self.read("img_00001_.png")).text

    def test_chunk_present_for_embedded_and_both_absent_for_sidecar_and_none(self):
        for mode, expected in (("embedded", True), ("both", True), ("sidecar", False), ("none", False)):
            with self.subTest(mode=mode):
                self.run_save(self.image(), filename_prefix="img", metadata=mode)
                self.assertEqual("moss_photon" in self.png_text(), expected)
                for name in self.listing():
                    os.remove(os.path.join(self.out, name))

    def test_prompt_and_workflow_chunks_follow_embed_workflow(self):
        self.run_save(self.image(), filename_prefix="img", metadata="both", prompt=PROMPT, extra_pnginfo=EXTRA)
        self.assertEqual(sorted(self.png_text()), ["moss_photon"])
        self.run_save(self.image(), filename_prefix="img", metadata="both", prompt=PROMPT, extra_pnginfo=EXTRA,
                      embed_workflow=True)
        self.assertEqual(sorted(decode(self.read("img_00002_.png")).text), ["moss_photon", "prompt", "workflow"])

    def test_disable_metadata_suppresses_the_png_chunks_but_not_the_sidecar(self):
        self.disable_metadata = True
        self.run_save(self.image(), filename_prefix="img", metadata="both", prompt=PROMPT, embed_workflow=True)
        self.assertEqual(self.png_text(), {})
        self.assertEqual(self.listing(), ["img_00001_.json", "img_00001_.png"])
        self.assertFalse(self.read_json("img_00001_.json")["file"]["workflow_embedded"])


if __name__ == "__main__":
    unittest.main()
