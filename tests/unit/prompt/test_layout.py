"""MP-8 parse_layout (design 4.3 step 3, 6.3): native MOSS layout schema, abs_centralized, yxyx, errors name the path."""
import copy
import json
import unittest

from moss_photon.core import constraints as c
from moss_photon.core.errors import InvalidLayout

GOLDEN = (
    '{"summary":{"caption":"A woman by a window"},"items":['
    '{"label":"character","caption":"a woman","bbox":[-300,-200,400,100]},'
    '{"label":"text","caption":"a sign","text":"OPEN ü","palette":["#ff0000"],"bbox":[-500,100,-400,300]}],'
    '"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}')

ITEM_TEXT = {"label": "text", "caption": "a sign", "text": "OPEN ü", "palette": ["#ff0000"],
             "bbox": [-500, 100, -400, 300]}
ITEM_CHAR = {"label": "character", "caption": "a woman", "bbox": [-300, -200, 400, 100]}


def doc(**over):
    d = {"items": [copy.deepcopy(ITEM_CHAR), copy.deepcopy(ITEM_TEXT)]}
    d.update(over)
    return d


def parse(d, text="A woman by a window"):
    return c.parse_layout(d if isinstance(d, str) else json.dumps(d), compiled_text=text)


class ValidTest(unittest.TestCase):
    def test_valid_document_and_golden_native_format(self):
        # key order of the input is scrambled on purpose; the output is the native order and separators
        raw = ('{"bbox_format": "yxyx", "items": [%s, %s], "bbox_coord_space": "abs_centralized"}'
               % (json.dumps(ITEM_CHAR), json.dumps({k: ITEM_TEXT[k] for k in ("bbox", "palette", "text", "caption", "label")}, ensure_ascii=False)))
        res = parse(raw)
        self.assertEqual(res.text, GOLDEN)
        self.assertEqual(res.text, c.layout_prompt(res.document))
        self.assertEqual(list(res.document), ["summary", "items", "bbox_coord_space", "bbox_format"])

    def test_caption_is_stripped_compiled_text_also_with_surrounding_whitespace(self):
        res = parse(doc(), text="  \tA woman by a window \r\n")
        self.assertEqual(res.document["summary"]["caption"], "A woman by a window")
        self.assertEqual(res.text, GOLDEN)

    def test_optional_keys_default_in(self):
        res = parse({"items": [ITEM_CHAR]})
        self.assertEqual(res.document["bbox_coord_space"], "abs_centralized")
        self.assertEqual(res.document["bbox_format"], "yxyx")

    def test_layout_info_for_constraints(self):
        res = parse(doc())
        self.assertEqual(res.layout, {
            "items": 2, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
            "bboxes": [[-300, -200, 400, 100], [-500, 100, -400, 300]], "extent": [-500, -200, 400, 300],
            "experimental": True})

    def test_summary_meta_and_canvas_kept(self):
        res = parse(doc(summary={"caption": "A woman by a window", "meta": ["a", 1], "canvas": "1024x1024"}))
        self.assertEqual(res.document["summary"],
                         {"caption": "A woman by a window", "meta": ["a", 1], "canvas": "1024x1024"})

    def test_caption_equal_to_compiled_text_is_fine_and_caption_only_in_json_is_fine(self):
        parse(doc(summary={"caption": "A woman by a window"}))
        res = parse(doc(summary={"caption": "only here"}), text="")
        self.assertEqual(res.document["summary"]["caption"], "only here")

    def test_zero_items_is_no_layout(self):
        for text in ("A woman", "  A woman \n"):
            res = parse({"items": []}, text=text)
            self.assertEqual((res.text, res.document, res.layout), (text, None, None))

    def test_empty_palette_dropped_like_native(self):
        item = dict(ITEM_CHAR, palette=[])
        self.assertNotIn("palette", parse({"items": [item]}).document["items"][0])

    def test_unicode_caption_not_escaped(self):
        res = parse({"items": [dict(ITEM_CHAR, caption="женщина")]}, text="ф")
        self.assertIn("женщина", res.text)
        self.assertNotIn("\\u", res.text)


class ErrorTest(unittest.TestCase):
    def assertPath(self, path, d, text="A woman by a window"):
        with self.assertRaises(InvalidLayout) as ctx:
            parse(d, text)
        self.assertEqual(ctx.exception.code, "invalid_layout")
        self.assertEqual(ctx.exception.issue.detail["path"], path)
        self.assertIn(path, str(ctx.exception))
        self.assertIn("layout_json", str(ctx.exception))

    def mutated(self, index, **changes):
        d = doc()
        d["items"][index].update(changes)
        return d

    def test_bad_json_path_is_root(self):
        self.assertPath("$", "{not json")
        self.assertPath("$", "[1, 2]")
        self.assertPath("$", '{"items": [], "x": NaN}')

    def test_unknown_key(self):
        self.assertPath("surprise", doc(surprise=1))
        self.assertPath("summary.surprise", doc(summary={"surprise": 1}))
        self.assertPath("items[1].surprise", self.mutated(1, surprise=1))

    def test_coord_space_and_format(self):
        self.assertPath("bbox_coord_space", doc(bbox_coord_space="normalized_0_1000"))
        self.assertPath("bbox_format", doc(bbox_format="xyxy"))

    def test_items_required_and_typed(self):
        self.assertPath("items", {"summary": {}})
        self.assertPath("items", {"items": {"a": 1}})
        self.assertPath("items[0]", {"items": ["x"]})

    def test_missing_or_empty_caption(self):
        bad = doc()
        del bad["items"][1]["caption"]
        self.assertPath("items[1].caption", bad)
        self.assertPath("items[0].caption", self.mutated(0, caption=""))

    def test_label(self):
        self.assertPath("items[0].label", self.mutated(0, label="person"))

    def test_text_rules(self):
        self.assertPath("items[0].text", self.mutated(0, text="HELLO"))  # on a non-text label
        bad = doc()
        del bad["items"][1]["text"]
        self.assertPath("items[1].text", bad)  # missing on a text label
        self.assertPath("items[1].text", self.mutated(1, text=5))

    def test_palette(self):
        self.assertPath("items[1].palette", self.mutated(1, palette="#fff"))
        self.assertPath("items[1].palette[1]", self.mutated(1, palette=["#fff", 3]))

    def test_float_bbox(self):
        self.assertPath("items[0].bbox[1]", self.mutated(0, bbox=[-300, -200.5, 400, 100]))
        self.assertPath("items[0].bbox[0]", self.mutated(0, bbox=[-300.0, -200, 400, 100]))  # 3.0 is a float too
        self.assertPath("items[0].bbox[2]", self.mutated(0, bbox=[-300, -200, True, 100]))

    def test_short_or_long_bbox(self):
        self.assertPath("items[1].bbox", self.mutated(1, bbox=[1, 2, 3]))
        self.assertPath("items[1].bbox", self.mutated(1, bbox=[1, 2, 3, 4, 5]))
        self.assertPath("items[1].bbox", self.mutated(1, bbox="1,2,3,4"))

    def test_degenerate_boxes(self):
        self.assertPath("items[0].bbox[2]", self.mutated(0, bbox=[10, 0, 10, 5]))
        self.assertPath("items[0].bbox[2]", self.mutated(0, bbox=[10, 0, 5, 5]))
        self.assertPath("items[0].bbox[3]", self.mutated(0, bbox=[0, 5, 10, 5]))

    def test_first_violation_wins(self):
        bad = doc()
        bad["items"][0]["label"] = "nope"
        bad["items"][1]["bbox"] = [1]
        self.assertPath("items[0].label", bad)

    def test_caption_given_twice(self):
        self.assertPath("summary.caption", doc(summary={"caption": "something else"}))
        self.assertPath("summary.caption", doc(summary={"caption": 5}))

    def test_summary_types(self):
        self.assertPath("summary", doc(summary="x"))
        self.assertPath("summary.meta", doc(summary={"meta": "x"}))
        self.assertPath("summary.canvas", doc(summary={"canvas": 3}))

    def test_error_is_a_registered_typed_error(self):
        with self.assertRaises(InvalidLayout) as ctx:
            parse("nope")
        issue = ctx.exception.issue
        self.assertEqual((issue.code, issue.severity, issue.node, issue.param),
                         ("invalid_layout", "error", "MossPhotonPromptLayout", "layout_json"))
        self.assertIn("abs_centralized", issue.fix)


class NotMyJobTest(unittest.TestCase):
    def test_extent_beyond_any_canvas_is_not_checked_here(self):
        # A23: the canvas size is unknown to MP-8; check_layout_extent belongs to MP-9
        res = parse({"items": [dict(ITEM_CHAR, bbox=[-10**6, -10**6, 10**6, 10**6])]})
        self.assertEqual(res.layout["extent"], [-10**6, -10**6, 10**6, 10**6])
        self.assertFalse(hasattr(c, "check_layout_in_canvas"))
        self.assertFalse(hasattr(c, "check_layout_extent"))


if __name__ == "__main__":
    unittest.main()
