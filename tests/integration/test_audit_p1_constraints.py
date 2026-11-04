"""MP-60 / audit F02 on the tiny 24-layer MOSS with the real MossPhotonPromptLayout node and the real sampler.

Audit repro C: ``constraints={'blocks_present': {'wardrobe': True}}`` was accepted by ``wardrobe_safe`` although the
validator listed 14 missing fields. And the usual graph mistake: constraints from Prompt A, conditioning from Prompt B
(or conditioning changed after the prompt node): the wardrobe / scene / layout / trigger checks then described a
prompt that was not sampled."""
import json
import unittest
import zlib

from . import tiny_moss
from . import sampler_support as ss

MISMATCH = "constraints_conditioning_mismatch"


class PromptClip:
    """Enough CLIP for MossPhotonPromptLayout: the conditioning depends on the text, so two prompts differ."""

    def __init__(self, torch):
        self.torch = torch

    def tokenize(self, text):
        return {"text": text}

    def encode_from_tokens_scheduled(self, tokens):
        gen = self.torch.Generator().manual_seed(zlib.crc32(tokens["text"].encode("utf-8")))
        return [[self.torch.randn(1, 23, tiny_moss.TXT, generator=gen), {"pooled_output": None}]]


class ConstraintsAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny = tiny_moss.build_tiny_moss()
        import torch
        from moss_photon.core import errors
        from moss_photon.nodes import prompt
        cls.torch, cls.errors, cls.prompt = torch, errors, prompt

    def node(self, positive="a woman in a kitchen", negative="blur", **blocks):
        out = self.prompt.MossPhotonPromptLayout().encode(PromptClip(self.torch), positive, negative, **blocks)
        return out["result"]

    def sample(self, preset_id="reference", **kw):
        with ss.NoSample():
            return ss.run(self.tiny, preset_id=preset_id, overrides={"steps": 4}, **kw)

    # ---------------------------------------------------------------- repro C
    def test_repro_c_the_partial_object_is_a_typed_error(self):
        with self.assertRaises(self.errors.InvalidConditioning) as cm:
            self.sample("wardrobe_safe", constraints={"blocks_present": {"wardrobe": True}})
        self.assertEqual(cm.exception.issue.detail["problems_total"], 14)

    def test_the_audit_probe_object_validates_nothing_else_first(self):
        # control: the same call without the broken object reaches the sampler (wardrobe_safe then asks for its block)
        with self.assertRaises(self.errors.MissingRequiredInput):
            self.sample("wardrobe_safe")

    # ---------------------------------------------------------------- the right branch
    def test_constraints_of_the_same_node_are_used_and_raise_no_warning(self):
        pos, neg, constraints = self.node(wardrobe="red wool sweater")
        res = self.sample("wardrobe_safe", positive=pos, negative=neg, constraints=constraints)
        self.assertNotIn(MISMATCH, ss.codes(res.report))

    def test_the_empty_block_of_the_same_node_is_refused(self):
        pos, neg, constraints = self.node()
        with self.assertRaises(self.errors.MissingRequiredInput):
            self.sample("wardrobe_safe", positive=pos, negative=neg, constraints=constraints)

    def test_a_layout_outside_the_canvas_of_the_same_node_is_refused(self):
        layout = json.dumps({"items": [{"label": "object", "caption": "a lamp", "bbox": [-5000, -10, 10, 10]}]})
        pos, neg, constraints = self.node(layout_json=layout)
        with self.assertRaises(self.errors.InvalidLayout):
            self.sample(positive=pos, negative=neg, constraints=constraints)

    # ---------------------------------------------------------------- another branch
    def test_prompt_a_constraints_with_prompt_b_conditioning_skip_the_checks_and_warn(self):
        _pos_a, _neg_a, constraints_a = self.node(positive="prompt A")  # no wardrobe block: wardrobe_safe would refuse
        pos_b, neg_b, _constraints_b = self.node(positive="prompt B", wardrobe="a blue coat")
        res = self.sample("wardrobe_safe", positive=pos_b, negative=neg_b, constraints=constraints_a)
        w = [x for x in res.report["warnings"] if x["code"] == MISMATCH]
        self.assertEqual(len(w), 1)
        self.assertEqual(w[0]["severity"], "warning")
        self.assertEqual(w[0]["detail"]["branches"], ["positive"])
        self.assertIn("blocks_present", w[0]["detail"]["skipped_checks"])

    def test_the_layout_of_the_wrong_prompt_is_not_judged(self):
        layout = json.dumps({"items": [{"label": "object", "caption": "a lamp", "bbox": [-5000, -10, 10, 10]}]})
        _pos_a, _neg_a, constraints_a = self.node(positive="prompt A", layout_json=layout)
        pos_b, neg_b, _ = self.node(positive="prompt B")
        res = self.sample(positive=pos_b, negative=neg_b, constraints=constraints_a)
        self.assertIn(MISMATCH, ss.codes(res.report))

    def test_conditioning_changed_after_the_prompt_node_is_a_mismatch(self):
        pos, neg, constraints = self.node(wardrobe="red wool sweater")
        later = [[pos[0][0], dict(pos[0][1], start_percent=0.5)]]  # what ConditioningSetTimestepRange does
        res = self.sample("wardrobe_safe", positive=later, negative=neg, constraints=constraints)
        w = [x for x in res.report["warnings"] if x["code"] == MISMATCH]
        self.assertEqual([x["detail"]["branches"] for x in w], [["positive"]])

    def test_a_mismatch_does_not_forward_the_foreign_warnings_or_text_encoder_section(self):
        pos_a, neg_a, constraints_a = self.node(positive="a red dress", wardrobe="a blue dress")
        self.assertTrue(constraints_a["warnings"])  # control: prompt A has an attribute_conflict
        constraints_a["text_encoder"] = {"basename": "te.safetensors", "size": 5, "sha256": None, "official": None,
                                         "fingerprint_source": "path_only"}
        pos_b, neg_b, _ = self.node(positive="prompt B")
        res = self.sample(positive=pos_b, negative=neg_b, constraints=constraints_a)
        self.assertNotIn("attribute_conflict", ss.codes(res.report))
        self.assertIsNone(res.report["text_encoder"])
        same = self.sample(positive=pos_a, negative=neg_a, constraints=constraints_a)
        self.assertIn("attribute_conflict", ss.codes(same.report))
        self.assertEqual(same.report["text_encoder"]["basename"], "te.safetensors")

    # ---------------------------------------------------------------- no numeric change
    def test_binding_changes_no_latent(self):
        pos, neg, constraints = self.node(wardrobe="red wool sweater")
        with_constraints = ss.run(self.tiny, positive=pos, negative=neg, constraints=constraints,
                                  overrides={"steps": 4})
        without = ss.run(self.tiny, positive=pos, negative=neg, overrides={"steps": 4})
        mismatched = ss.run(self.tiny, positive=pos, negative=neg, overrides={"steps": 4},
                            constraints=self.node(positive="prompt A")[2])
        shas = {r.report["output"]["latent_sha256"] for r in (with_constraints, without, mismatched)}
        self.assertEqual(len(shas), 1)
        self.assertEqual(with_constraints.report["recipe_hash"], without.report["recipe_hash"])


if __name__ == "__main__":
    unittest.main()
