"""MP-8 PHOTON_CONSTRAINTS (design 2.5, 6.3): output validates against the contract schema and the fixture."""
import copy
import json
import os
import unittest

from jsonschema import Draft202012Validator

from moss_photon.core import constraints as c
from moss_photon.core.canon import sha256_hex
from moss_photon.core.consts import COMPILER_VERSION

CONTRACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "fixtures", "contracts")


def _load(*parts):
    with open(os.path.join(CONTRACTS, *parts), "rb") as f:
        return json.loads(f.read().decode("utf-8"))


SCHEMA = Draft202012Validator(_load("schemas", "constraints.schema.json"))
FIXTURE = _load("constraints.json")

BLOCKS = {"identity": "Anna, short dark hair", "wardrobe": "red wool sweater", "scene": "small kitchen, wooden table"}
USER = "A woman in a kitchen, morning light."
TEXT_ENCODER = {"basename": "te.safetensors", "size": 5, "sha256": None, "official": None,
                "fingerprint_source": "path_only"}


def errors(instance):
    return [e.message for e in SCHEMA.iter_errors(instance)]


def build(user=USER, negative="", blocks=BLOCKS, layout=None, text_encoder=None):
    found = c.detect_conflicts(dict({"text": user}, **{n: blocks[n] for n in c.COMPILED_BLOCKS if blocks.get(n)}))
    return c.build_constraints(user, negative, blocks, layout, found, text_encoder)


class SchemaTest(unittest.TestCase):
    def test_fixture_is_valid_and_reproduced_key_by_key(self):
        self.assertEqual(errors(FIXTURE), [])
        built = build()
        self.assertEqual(errors(built), [])
        self.assertEqual(set(built), set(FIXTURE))
        for key in FIXTURE:
            if key in ("prompt_sha256", "text_encoder"):  # fixture carries placeholders / an example TE
                continue
            with self.subTest(key=key):
                self.assertEqual(built[key], FIXTURE[key])

    def test_fixture_passes_validate_constraints_except_its_placeholder_hashes(self):
        problems = c.validate_constraints(FIXTURE)
        self.assertEqual(sorted(problems), ["prompt_sha256.negative does not match negative_text",
                                            "prompt_sha256.positive does not match compiled_positive"])

    def test_built_object_passes_validate_constraints(self):
        self.assertEqual(c.validate_constraints(build()), [])

    def test_with_layout_conflicts_and_text_encoder_still_valid(self):
        layout = c.parse_layout(
            json.dumps({"items": [{"label": "object", "caption": "a lamp", "bbox": [-10, -10, 10, 10]}]}),
            compiled_text=c.compile_prompt("a red dress", {"wardrobe": "blue dress"}))
        built = build(user="a red dress", blocks={"wardrobe": "blue dress"}, layout=layout, text_encoder=TEXT_ENCODER)
        self.assertEqual(errors(built), [])
        self.assertEqual(c.validate_constraints(built), [])
        self.assertEqual(built["layout"]["extent"], [-10, -10, 10, 10])
        self.assertEqual(built["compiled_positive"], layout.text)
        self.assertEqual([w["code"] for w in built["warnings"]], ["attribute_conflict", "layout_experimental"])
        self.assertEqual(built["conflicts"][0]["attribute"], "dress")
        self.assertEqual(built["text_encoder"], TEXT_ENCODER)

    def test_zero_item_layout_means_no_layout(self):
        layout = c.parse_layout('{"items": []}', compiled_text=USER)
        built = build(layout=layout)
        self.assertIsNone(built["layout"])
        self.assertEqual(built["warnings"], [])

    def test_blocks_present_correct(self):
        built = build(blocks={"identity": "  ", "wardrobe": "x", "forbidden_changes": "no hats"})
        self.assertEqual(built["blocks_present"], {
            "identity": False, "wardrobe": True, "scene": False, "pose": False, "camera": False, "lighting": False,
            "forbidden_changes": True})

    def test_empty_block_requirement_data_for_the_sampler(self):
        # SPEC 6.3: wardrobe_safe needs a wardrobe block, scene_safe a scene block; the sampler reads this flag
        self.assertFalse(build(blocks={"scene": "x"})["blocks_present"]["wardrobe"])
        self.assertFalse(build(blocks={"wardrobe": "x"})["blocks_present"]["scene"])
        self.assertFalse(build(blocks={"wardrobe": " \n\t"})["blocks_present"]["wardrobe"])

    def test_private_paths_as_specified_and_exist(self):
        built = build()
        self.assertEqual(built["private_paths"], ["fields", "user_text", "compiled_positive", "negative_text"])
        for key in built["private_paths"]:
            self.assertIn(key, built)

    def test_prompt_hashes(self):
        built = build(negative="blurry")
        self.assertEqual(built["prompt_sha256"]["positive"], sha256_hex(built["compiled_positive"]))
        self.assertEqual(built["prompt_sha256"]["negative"], sha256_hex("blurry"))
        self.assertNotEqual(built["prompt_sha256"]["positive"], built["prompt_sha256"]["negative"])

    def test_compiler_version_is_the_consts_one(self):
        self.assertEqual(build()["compiler_version"], COMPILER_VERSION)

    def test_user_and_negative_text_verbatim(self):
        built = build(user="  odd \r\n text  ", negative=" \tneg ")
        self.assertEqual(built["user_text"], "  odd \r\n text  ")
        self.assertEqual(built["negative_text"], " \tneg ")

    def test_forbidden_changes_only_in_fields(self):
        blocks = dict(BLOCKS, forbidden_changes="NO-TATTOOS-XYZ")
        built = build(blocks=blocks, negative="blurry")
        self.assertEqual(built["fields"]["forbidden_changes"], "NO-TATTOOS-XYZ")
        self.assertTrue(built["blocks_present"]["forbidden_changes"])
        self.assertNotIn("NO-TATTOOS-XYZ", built["compiled_positive"])
        self.assertNotIn("NO-TATTOOS-XYZ", built["negative_text"])
        self.assertNotIn("NO-TATTOOS-XYZ", json.dumps(built["conflicts"] + built["warnings"]))

    def test_inputs_not_mutated(self):
        blocks = dict(BLOCKS)
        before = copy.deepcopy(blocks)
        build(blocks=blocks)
        self.assertEqual(blocks, before)


class ValidatorTest(unittest.TestCase):
    """validate_constraints goes red on each planted defect (an invariant is proven by trying to break it)."""

    def broken(self, mutate):
        built = build()
        mutate(built)
        return c.validate_constraints(built)

    def test_plants(self):
        cases = {
            "missing key": lambda b: b.pop("layout"),
            "version": lambda b: b.update(schema_version=2),
            "kind": lambda b: b.update(kind="x"),
            "compiler_version": lambda b: b.update(compiler_version=0),
            "private path absent": lambda b: b["private_paths"].append("nope"),
            "blocks_present lies": lambda b: b["blocks_present"].update(scene=False),
            "fields lacks block": lambda b: b["fields"].pop("pose"),
            "hash stale": lambda b: b["prompt_sha256"].update(positive="0" * 64),
            "layout wrong space": lambda b: b.update(layout={"items": 0, "bboxes": [], "extent": [0, 0, 0, 0],
                                                             "bbox_coord_space": "x", "bbox_format": "yxyx"}),
            "layout items mismatch": lambda b: b.update(layout={"items": 3, "bboxes": [], "extent": [0, 0, 0, 0],
                                                                "bbox_coord_space": "abs_centralized",
                                                                "bbox_format": "yxyx"}),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                self.assertNotEqual(self.broken(mutate), [], name)

    def test_non_dict(self):
        self.assertNotEqual(c.validate_constraints(None), [])

    def test_jsonschema_rejects_what_the_schema_forbids(self):
        built = build()
        built["surprise"] = 1
        self.assertNotEqual(errors(built), [])


if __name__ == "__main__":
    unittest.main()
