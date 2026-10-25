"""MP-8 detect_conflicts (design 4.3 step 2, 6.3): deterministic heuristic, lists the conflicts, never edits text."""
import json
import re
import unittest

from moss_photon.core import constraints as c


class DetectTest(unittest.TestCase):
    def test_red_vs_blue_dress_is_one_conflict(self):
        found = c.detect_conflicts({"text": "a woman in a red dress", "wardrobe": "blue dress"})
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["attribute"], "dress")
        self.assertEqual(found[0]["kind"], "color_mismatch")
        self.assertEqual(found[0]["sources"], [{"field": "text", "value": "red"}, {"field": "wardrobe", "value": "blue"}])
        self.assertIn("red", found[0]["message"])
        self.assertIn("blue", found[0]["message"])

    def test_red_dress_vs_a_red_dress_is_none(self):
        self.assertEqual(c.detect_conflicts({"text": "red dress", "wardrobe": "a red dress"}), [])

    def test_hair_and_eye_nouns(self):
        found = c.detect_conflicts({"text": "blonde hair, green eyes", "identity": "Anna, black hair and brown eyes"})
        self.assertEqual([f["attribute"] for f in found], ["eyes", "hair"])

    def test_grey_gray_and_case_are_the_same(self):
        self.assertEqual(c.detect_conflicts({"text": "GREY coat", "wardrobe": "a gray coat"}), [])

    def test_overlapping_sets_are_not_a_conflict(self):
        # window of 2 words: 'red white dress' gives {red, white}, 'white dress' gives {white}: overlap, no conflict
        self.assertEqual(c.detect_conflicts({"text": "red white dress", "wardrobe": "white dress"}), [])
        self.assertEqual(len(c.detect_conflicts({"text": "red white dress", "wardrobe": "blue dress"})), 1)

    def test_colour_does_not_leak_across_punctuation_or_nouns(self):
        # 'red' belongs to the shirt, never to the jeans that follow after a comma or directly
        self.assertEqual(c.detect_conflicts({"text": "red shirt, jeans", "wardrobe": "blue jeans"}), [])
        self.assertEqual(c.detect_conflicts({"text": "red shirt jeans", "wardrobe": "blue jeans"}), [])

    def test_window_of_two_words(self):
        self.assertEqual(len(c.detect_conflicts({"text": "red leather jacket", "wardrobe": "blue jacket"})), 1)
        self.assertEqual(c.detect_conflicts({"text": "red very long old jacket", "wardrobe": "blue jacket"}), [])

    def test_same_source_twice_is_no_conflict(self):
        self.assertEqual(c.detect_conflicts({"text": "red dress and blue dress"}), [])

    def test_different_nouns_do_not_conflict(self):
        self.assertEqual(c.detect_conflicts({"text": "red dress", "wardrobe": "blue jacket"}), [])

    def test_three_sources_one_conflict_listing_all(self):
        found = c.detect_conflicts({"scene": "green coat", "wardrobe": "blue coat", "text": "red coat"})
        self.assertEqual(len(found), 1)
        self.assertEqual([s["field"] for s in found[0]["sources"]], ["text", "wardrobe", "scene"])

    def test_deterministic_and_independent_of_dict_order(self):
        a = {"text": "red dress, blonde hair", "wardrobe": "blue dress", "identity": "black hair"}
        b = dict(reversed(list(a.items())))
        self.assertEqual(c.detect_conflicts(a), c.detect_conflicts(b))
        self.assertEqual([f["attribute"] for f in c.detect_conflicts(a)], ["dress", "hair"])
        self.assertEqual(c.detect_conflicts(a), c.detect_conflicts(a))

    def test_non_ascii_and_empty_are_safe(self):
        self.assertEqual(c.detect_conflicts({"text": "красное платье", "wardrobe": ""}), [])
        self.assertEqual(c.detect_conflicts({}), [])
        self.assertEqual(c.detect_conflicts({"text": "éred dress", "wardrobe": "blue dress"})[0]["attribute"], "dress")

    def test_input_not_mutated(self):
        src = {"text": "red dress", "wardrobe": "blue dress"}
        before = dict(src)
        c.detect_conflicts(src)
        self.assertEqual(src, before)

    def test_unknown_field_name_sorts_after_known_ones(self):
        found = c.detect_conflicts({"zeta": "red hat", "text": "blue hat", "extra": "green hat"})
        self.assertEqual([s["field"] for s in found[0]["sources"]], ["text", "extra", "zeta"])


class WarningTest(unittest.TestCase):
    def test_one_attribute_conflict_warning_per_conflict_listing_the_sources(self):
        found = c.detect_conflicts({"text": "red dress, blonde hair", "wardrobe": "blue dress", "identity": "black hair"})
        issues = c.conflict_issues(found)
        self.assertEqual(len(issues), 2)
        for issue, conflict in zip(issues, found):
            self.assertEqual(issue["code"], "attribute_conflict")
            self.assertEqual(issue["severity"], "warning")
            self.assertEqual(issue["detail"]["attribute"], conflict["attribute"])
            self.assertIn(conflict["attribute"], issue["message"])
            for source in conflict["sources"]:
                self.assertIn(source["field"], issue["message"])

    def test_the_warning_carries_field_names_only_never_the_prompt_words(self):
        """m10 (owner 2026-10-25): the warning reaches the PNG chunk and the sidecar, so no colour word may be in it."""
        found = c.detect_conflicts({"text": "red dress, blonde hair", "wardrobe": "blue dress", "identity": "black hair"})
        self.assertEqual(len(found), 2)
        words = {s["value"] for f in found for s in f["sources"]}
        self.assertTrue(words >= {"red", "blue", "blonde", "black"})  # control: the values exist in the conflict
        blob = json.dumps(c.conflict_issues(found)).lower()
        for word in words:
            self.assertIsNone(re.search(r"\b%s\b" % word, blob), word)
        self.assertIn("dress colour conflict between text and wardrobe", blob)

    def test_no_conflicts_no_warnings(self):
        self.assertEqual(c.conflict_issues([]), [])


if __name__ == "__main__":
    unittest.main()
