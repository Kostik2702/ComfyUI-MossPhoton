"""MP-30 m19 (SPEC 8 / R5): a full tiny run -- prompt node, LoRA node, sampler, save node -- leaves no prompt text and no
prompt-derived word in any log record, in the saved PNG chunk, in the sidecar or in the report, with the defaults.

The half without ComfyUI (prompt node alone, and the harness controls) is ``tests/unit/privacy/test_prompt_logs.py``.
"""
import json
import logging
import os
import shutil
import tempfile
import unittest
from unittest import mock

from tests.unit.privacy.logcapture import COLOUR_WORDS, SENTINELS, Capture, leaked_words, prompt_fields

from . import _comfy_env, lora_synth, tiny_moss
from . import sampler_support as ss


class TinyClip:
    """Encodes to the tiny model's conditioning; the text only passes through ``tokenize`` like the native CLIP."""

    def __init__(self, tiny):
        self.tiny = tiny

    def tokenize(self, text):
        return {"n": 23 if "woman" in text else 9}

    def encode_from_tokens_scheduled(self, tokens):
        return self.tiny.make_cond(tokens["n"])


class PrivacyFullRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import folder_paths
        from moss_photon.nodes import prompt, save
        cls.folder_paths, cls.prompt, cls.save = folder_paths, prompt, save

    def setUp(self):
        self.env = lora_synth.install(self)
        self.env.write("char.safetensors", lora_synth.make_lora_sd(self.tiny.patcher, seed=31, modules=("qkv", "o")),
                       {"ss_base_model_version": "moss_image"})
        self.out = tempfile.mkdtemp(prefix="mp30_privacy_")
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)
        saved = self.folder_paths.get_output_directory()
        self.folder_paths.set_output_directory(self.out)
        self.addCleanup(self.folder_paths.set_output_directory, saved)

    def full_run(self):
        """Prompt node -> LoRA node (trigger word) -> sampler -> save node, all under one capture."""
        words = list(SENTINELS.values()) + list(COLOUR_WORDS)
        with Capture() as cap:
            out = self.prompt.MossPhotonPromptLayout().encode(TinyClip(self.tiny), **prompt_fields())
            positive, negative, constraints = out["result"]
            model, profile, _ = lora_synth.run_node(self.tiny.patcher, "char.safetensors", trigger=SENTINELS["trigger"])
            res = ss.run(self.tiny, model=model, positive=positive, negative=negative, identity_profile=profile,
                         constraints=constraints, overrides={"steps": 4})
            image = self.tiny.latent.new_zeros((1, 16, 16, 3)) + 0.5
            self.save.MossPhotonSaveImage().save(image, res.report, filename_prefix="priv/img", format="png",
                                                 metadata="both")
        return cap, res, constraints, words

    def saved(self):
        folder = os.path.join(self.out, "priv")
        blobs = {}
        for name in sorted(os.listdir(folder)):
            with open(os.path.join(folder, name), "rb") as f:
                blobs[name] = f.read().decode("latin-1")
        return blobs

    def test_a_full_default_run_leaks_no_prompt_word_into_logs_report_png_or_sidecar(self):
        cap, res, constraints, words = self.full_run()
        # controls: there WAS something to leak, and the files exist
        self.assertIn(SENTINELS["positive"], constraints["compiled_positive"])
        self.assertEqual([c["attribute"] for c in constraints["conflicts"]], ["dress"])
        files = self.saved()
        self.assertEqual(sorted(files), ["img_00001_.json", "img_00001_.png"])
        self.assertIn("attribute_conflict", [w["code"] for w in res.report["warnings"]])
        self.assertIn("dress colour conflict between text and wardrobe", files["img_00001_.json"])
        self.assertTrue(cap.records, "control: the capture saw the run's own log lines")
        # the verdict
        self.assertEqual(leaked_words(cap.text, words), [], "logs")
        self.assertEqual(leaked_words(json.dumps(res.report), words), [], "report")
        for name, blob in files.items():
            self.assertEqual(leaked_words(blob, words), [], name)

    def test_the_harness_sees_a_prompt_logged_in_the_middle_of_the_run(self):
        """M3 of the review: a planted prompt log survived 1140 tests. The same run with the plant must go red."""
        from moss_photon.core import solver
        real = solver.install_counter
        compiled = prompt_fields()["positive"]

        def leaky(clone, stats):
            logging.getLogger("MossPhoton").info("[MossPhoton] prompt: %s", compiled)
            return real(clone, stats)
        with mock.patch.object(solver, "install_counter", leaky):
            cap, _res, _constraints, words = self.full_run()
        self.assertIn(SENTINELS["positive"], leaked_words(cap.text, words))


if __name__ == "__main__":
    unittest.main()
