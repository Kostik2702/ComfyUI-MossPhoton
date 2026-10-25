"""MP-30 m19 (SPEC 8 / R5 "no prompts in logs"): the prompt node logs nothing that comes from a prompt.

The full run (prompt node + sampler + save node on the tiny MOSS) is ``tests/integration/test_privacy_logs_tiny.py``;
this half needs no ComfyUI. Mutant M3 of the MP-30 review (a planted ``_log.info`` of the prompt) survived every test
before this file existed, so the harness is proven first: it must see a leak when one is planted.
"""
import json
import logging
import unittest

from moss_photon.nodes import prompt as P
from .logcapture import COLOUR_WORDS, SENTINELS, Capture, leaked_words, prompt_fields


class FakeClip:
    def tokenize(self, text):
        return {"t": text}

    def encode_from_tokens_scheduled(self, tokens):
        return [["cond", {}]]


def run_prompt_node():
    return P.MossPhotonPromptLayout().encode(FakeClip(), **prompt_fields())


class HarnessSeesLeaksTest(unittest.TestCase):
    """The control: without these the other tests could pass because the harness looks at nothing."""

    def leak(self, emit):
        with Capture() as cap:
            emit()
        return leaked_words(cap.text, SENTINELS.values())

    def test_a_message_argument_is_seen(self):
        word = SENTINELS["positive"]
        self.assertEqual(self.leak(lambda: logging.getLogger("MossPhoton").info("prompt: %s", word)), [word])

    def test_a_debug_message_of_a_quiet_logger_is_seen(self):
        word = SENTINELS["scene"]
        quiet = logging.getLogger("MossPhoton.quiet")
        quiet.setLevel(logging.ERROR)
        self.addCleanup(quiet.setLevel, logging.NOTSET)
        self.assertEqual(self.leak(lambda: quiet.debug(word)), [word])

    def test_a_print_is_seen(self):
        word = SENTINELS["camera"]
        self.assertEqual(self.leak(lambda: print("prompt", word)), [word])

    def test_a_clean_message_is_not_a_leak(self):
        self.assertEqual(self.leak(lambda: logging.getLogger("MossPhoton").info("nothing to see")), [])


class PromptNodeLogsTest(unittest.TestCase):
    def test_the_prompt_node_logs_no_prompt_derived_word(self):
        with Capture() as cap:
            out = run_prompt_node()
        constraints = out["result"][2]
        # control: the run did compile the prompt and found the conflict, so there was something to leak
        self.assertIn(SENTINELS["positive"], constraints["compiled_positive"])
        self.assertEqual([c["attribute"] for c in constraints["conflicts"]], ["dress"])
        self.assertEqual(leaked_words(cap.text, list(SENTINELS.values()) + list(COLOUR_WORDS)), [])

    def test_the_node_ui_carries_no_sentinel(self):
        ui = json.dumps(run_prompt_node()["ui"])
        self.assertEqual(leaked_words(ui, SENTINELS.values()), [])

    def test_the_conflict_warning_names_fields_not_colours(self):
        """m10: the warnings of the constraints are copied into the report, the PNG chunk and the sidecar."""
        warnings = run_prompt_node()["result"][2]["warnings"]
        self.assertEqual([w["code"] for w in warnings], ["attribute_conflict"])
        self.assertEqual(leaked_words(json.dumps(warnings), COLOUR_WORDS), [])
        self.assertIn("dress", warnings[0]["message"])


if __name__ == "__main__":
    unittest.main()
