"""MossPhotonPromptSplit: marker parsing, degrade-with-warning fallbacks, node schema, no prompt text in report/log."""
import unittest

from moss_photon.nodes import prompt
from moss_photon.nodes import prompt_split as S

FULL = (
    "<<<POSITIVE>>>\nA medium shot of a woman on a wet street.\n"
    "<<<NEGATIVE>>>\nblurry face\n"
    "<<<IDENTITY>>>\nwoman in her late twenties, brown eyes\n"
    "<<<WARDROBE>>>\nred wool coat\n"
    "<<<SCENE>>>\nnarrow street after rain\n"
    "<<<POSE>>>\nlooking back over the shoulder\n"
    "<<<CAMERA>>>\n50mm lens, eye level\n"
    "<<<LIGHTING>>>\nwarm shop-window light from the left\n"
    "<<<FORBIDDEN_CHANGES>>>\nred coat, brown eyes\n"
    "<<<END>>>\n"
)
EXPECTED = {
    "positive": "A medium shot of a woman on a wet street.", "negative": "blurry face",
    "identity": "woman in her late twenties, brown eyes", "wardrobe": "red wool coat",
    "scene": "narrow street after rain", "pose": "looking back over the shoulder", "camera": "50mm lens, eye level",
    "lighting": "warm shop-window light from the left", "forbidden_changes": "red coat, brown eyes",
}


class SplitFieldsTest(unittest.TestCase):
    def test_full_answer_maps_every_field_without_warnings(self):
        self.assertEqual(S.split_fields(FULL), (EXPECTED, []))

    def test_empty_field_between_markers_is_empty_string(self):
        fields, warnings = S.split_fields(FULL.replace("blurry face\n", ""))
        self.assertEqual(fields["negative"], "")
        self.assertEqual(fields["identity"], EXPECTED["identity"])
        self.assertEqual(warnings, [])

    def test_preamble_and_text_after_end_are_ignored(self):
        fields, warnings = S.split_fields("Sure, here it is:\n" + FULL + "Hope this helps! <<<SCENE>>> x")
        self.assertEqual((fields, warnings), (EXPECTED, []))

    def test_missing_end_runs_last_field_to_the_end(self):
        fields, warnings = S.split_fields(FULL.replace("<<<END>>>\n", ""))
        self.assertEqual((fields, warnings), (EXPECTED, []))

    def test_crlf_code_fences_case_and_spaces_are_tolerated(self):
        text = "```text\r\n" + FULL.replace("<<<FORBIDDEN_CHANGES>>>", "<<< forbidden changes >>>").replace(
            "<<<SCENE>>>", "<<<Scene>>>").replace("\n", "\r\n") + "```\r\n"
        self.assertEqual(S.split_fields(text), (EXPECTED, []))

    def test_reasoning_block_is_skipped_even_when_it_quotes_markers(self):
        think = "<think>I will write <<<SCENE>>> a beach and <<<POSE>>> sitting</think>\n"
        self.assertEqual(S.split_fields(think + FULL), (EXPECTED, []))
        self.assertEqual(S.split_fields("I will write <<<SCENE>>> a beach</think>" + FULL), (EXPECTED, []))

    def test_inner_line_breaks_are_kept_verbatim(self):
        fields, _ = S.split_fields(FULL.replace("red wool coat", "red wool coat,\n  black boots"))
        self.assertEqual(fields["wardrobe"], "red wool coat,\n  black boots")

    def test_missing_marker_is_empty_with_warning(self):
        fields, warnings = S.split_fields(FULL.replace("<<<POSE>>>\nlooking back over the shoulder\n", ""))
        self.assertEqual(fields["pose"], "")
        self.assertEqual(warnings, ["marker <<<POSE>>> missing, left empty"])

    def test_no_markers_sends_whole_text_as_positive(self):
        fields, warnings = S.split_fields("  a cat on a sofa \n")
        self.assertEqual(fields, dict({n: "" for n in S.FIELDS}, positive="a cat on a sofa"))
        self.assertEqual(warnings, ["no field markers found: the whole text is sent as positive"])

    def test_repeated_marker_keeps_the_first(self):
        fields, warnings = S.split_fields(FULL.replace("<<<END>>>", "<<<SCENE>>>\nbeach\n<<<END>>>"))
        self.assertEqual(fields["scene"], EXPECTED["scene"])
        self.assertEqual(warnings, ["marker <<<SCENE>>> repeated: the first one is used"])

    def test_unknown_marker_is_ignored_and_ends_the_previous_field(self):
        fields, warnings = S.split_fields(FULL.replace("<<<END>>>", "<<<STYLE>>>\nnoir\n<<<END>>>"))
        self.assertEqual(fields, EXPECTED)
        self.assertEqual(warnings, ["unknown marker <<<STYLE>>> ignored"])

    def test_placeholders_become_empty_with_warning(self):
        for placeholder in ("none", "N/A", "-", "None."):
            with self.subTest(placeholder=placeholder):
                fields, warnings = S.split_fields(FULL.replace("blurry face", placeholder))
                self.assertEqual(fields["negative"], "")
                self.assertEqual(warnings, ["<<<NEGATIVE>>> held a placeholder, left empty"])

    def test_empty_or_none_input_gives_empty_fields_without_warnings(self):
        for text in ("", "   \n", None):
            with self.subTest(text=text):
                self.assertEqual(S.split_fields(text), ({n: "" for n in S.FIELDS}, []))


class NodeTest(unittest.TestCase):
    def test_outputs_are_the_prompt_layout_text_inputs_plus_report(self):
        layout = prompt.MossPhotonPromptLayout.INPUT_TYPES()
        layout_text = [n for part in ("required", "optional") for n, spec in layout[part].items()
                       if spec[0] == "STRING" and n != "layout_json"]
        self.assertEqual(list(S.FIELDS), layout_text)
        self.assertEqual(S.MossPhotonPromptSplit.RETURN_NAMES, S.FIELDS + ("report",))
        self.assertEqual(len(S.MossPhotonPromptSplit.RETURN_TYPES), len(S.FIELDS) + 1)
        self.assertEqual(len(S.MossPhotonPromptSplit.OUTPUT_TOOLTIPS), len(S.FIELDS) + 1)

    def test_registration(self):
        self.assertEqual(S.NODE_CLASS_MAPPINGS, {"MossPhotonPromptSplit": S.MossPhotonPromptSplit})
        self.assertEqual(S.NODE_DISPLAY_NAME_MAPPINGS, {"MossPhotonPromptSplit": "Moss Photon Prompt Split"})
        self.assertEqual(S.MossPhotonPromptSplit.CATEGORY, "MossPhoton/1 Setup")

    def test_split_returns_fields_in_order_and_ok_report(self):
        out = S.MossPhotonPromptSplit().split(FULL)
        self.assertEqual(out["result"], tuple(EXPECTED[n] for n in S.FIELDS) + ("ok",))
        self.assertEqual(out["ui"]["text"], ["filled: %s | warnings: 0" % ",".join(S.FIELDS)])

    def test_report_ui_and_log_never_carry_prompt_text(self):
        secret = "zebra-velvet-lighthouse"
        text = FULL.replace("<<<POSE>>>\nlooking back over the shoulder\n", "") + "<<<STYLE>>> %s" % secret
        with self.assertLogs("MossPhoton", level="WARNING") as logs:
            out = S.MossPhotonPromptSplit().split("<<<SCENE>>>%s\n<<<SCENE>>>%s\n<<<NEGATIVE>>>none" % (secret, secret))
            out2 = S.MossPhotonPromptSplit().split(text.replace("<<<END>>>\n", ""))
        for o in (out, out2):
            self.assertNotIn(secret, o["result"][-1])
            self.assertNotIn(secret, o["ui"]["text"][0])
            self.assertNotEqual(o["result"][-1], "ok")
        self.assertNotIn(secret, "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
