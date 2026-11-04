"""MP-60 (audit F02): PHOTON_CONSTRAINTS carries the fingerprint of the conditioning it describes.

The fingerprint covers the embedding bytes AND the metadata of every entry (area, mask, strength, start/end percent,
hooks), unlike the sampler's report hash, which is embeddings only (audit F04: a ``start_percent`` change left the hash
unchanged). The sampler fingerprints what it receives and compares (``tests/unit/sampler/test_constraints_gate.py``)."""
import copy
import json
import os
import unittest

import torch
from jsonschema import Draft202012Validator

from moss_photon.core import constraints, constraints_contract as cc
from moss_photon.nodes import prompt as P
from .test_node import FakeClip, run

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "fixtures", "contracts", "schemas", "constraints.schema.json")
with open(SCHEMA_PATH, "rb") as _f:
    VALIDATOR = Draft202012Validator(json.loads(_f.read().decode("utf-8")))


def cond(value=1.0, **meta):
    return [[torch.full((1, 5, 8), value), dict(meta)]]


class FingerprintTest(unittest.TestCase):
    def test_same_content_same_fingerprint_and_it_is_a_sha256(self):
        a, b = cc.conditioning_fingerprint(cond(pooled_output=torch.ones(3))), cc.conditioning_fingerprint(
            cond(pooled_output=torch.ones(3)))
        self.assertEqual(a, b)
        self.assertRegex(a, r"^[0-9a-f]{64}$")

    def test_the_key_order_of_the_metadata_does_not_matter(self):
        one = [[torch.zeros(1, 2), {"a": 1, "b": 2}]]
        two = [[torch.zeros(1, 2), {"b": 2, "a": 1}]]
        self.assertEqual(cc.conditioning_fingerprint(one), cc.conditioning_fingerprint(two))

    def test_every_change_of_the_embedding_or_of_its_metadata_is_seen(self):
        base = cc.conditioning_fingerprint(cond())
        changed = {
            "embedding value": cond(2.0),
            "start_percent (audit F04 repro D)": cond(start_percent=0.5),
            "end_percent": cond(end_percent=0.9),
            "strength (audit F04 repro E)": cond(strength=0.75),
            "area": cond(area=(8, 8, 0, 0)),
            "mask": cond(mask=torch.ones(1, 8, 8)),
            "pooled output": cond(pooled_output=torch.ones(1, 4)),
            "a hook object": cond(hooks=object()),
        }
        seen = {}
        for name, c in changed.items():
            with self.subTest(name):
                fp = cc.conditioning_fingerprint(c)
                self.assertNotEqual(fp, base)
                self.assertNotIn(fp, seen)
                seen[fp] = name
        self.assertNotEqual(cc.conditioning_fingerprint(cond(strength=0.25)), cc.conditioning_fingerprint(cond(strength=0.75)))
        self.assertNotEqual(cc.conditioning_fingerprint(cond(mask=torch.ones(1, 8, 8))),
                            cc.conditioning_fingerprint(cond(mask=torch.zeros(1, 8, 8))))

    def test_shape_dtype_and_entry_count_are_seen(self):
        base = cc.conditioning_fingerprint(cond())
        self.assertNotEqual(base, cc.conditioning_fingerprint([[torch.full((1, 4, 10), 1.0), {}]]))
        self.assertNotEqual(base, cc.conditioning_fingerprint([[torch.full((1, 5, 8), 1.0, dtype=torch.float64), {}]]))
        self.assertNotEqual(base, cc.conditioning_fingerprint(cond() + cond()))

    def test_anything_that_is_not_a_conditioning_list_never_equals_a_real_one(self):
        real = cc.conditioning_fingerprint(cond())
        for odd in (None, object(), "text", 5, {}):
            self.assertNotEqual(cc.conditioning_fingerprint(odd), real)

    def test_the_fingerprint_does_not_modify_or_retain_the_input(self):
        c = cond(pooled_output=torch.ones(2))
        before = copy.deepcopy(c)
        cc.conditioning_fingerprint(c)
        self.assertTrue(torch.equal(c[0][0], before[0][0]))
        self.assertEqual(sorted(c[0][1]), ["pooled_output"])


class BindingStateTest(unittest.TestCase):
    def constraints(self, pos, neg):
        built = constraints.build_constraints("a", "b", {}, None, [])
        return cc.bind_conditioning(built, pos, neg)

    def test_the_conditioning_it_was_bound_to_matches(self):
        pos, neg = cond(1.0), cond(2.0)
        self.assertIsNone(cc.binding_state(self.constraints(pos, neg), pos, neg))

    def test_a_copy_of_the_same_values_matches_too(self):
        self.assertIsNone(cc.binding_state(self.constraints(cond(1.0), cond(2.0)), cond(1.0), cond(2.0)))

    def test_another_branch_names_the_branch_that_differs(self):
        c = self.constraints(cond(1.0), cond(2.0))
        self.assertEqual(cc.binding_state(c, cond(9.0), cond(2.0)), {"reason": "mismatch", "branches": ["positive"]})
        self.assertEqual(cc.binding_state(c, cond(1.0), cond(9.0)), {"reason": "mismatch", "branches": ["negative"]})
        self.assertEqual(cc.binding_state(c, cond(9.0), cond(9.0))["branches"], ["positive", "negative"])

    def test_a_conditioning_transformed_after_the_prompt_node_is_a_mismatch(self):
        pos = cond(1.0)
        c = self.constraints(pos, cond(2.0))
        transformed = [[pos[0][0], dict(pos[0][1], start_percent=0.5)]]
        self.assertEqual(cc.binding_state(c, transformed, cond(2.0)), {"reason": "mismatch", "branches": ["positive"]})

    def test_swapped_positive_and_negative_is_a_mismatch(self):
        pos, neg = cond(1.0), cond(2.0)
        self.assertEqual(cc.binding_state(self.constraints(pos, neg), neg, pos)["reason"], "mismatch")

    def test_constraints_without_a_binding_are_unbound(self):
        built = constraints.build_constraints("a", "b", {}, None, [])
        self.assertIsNone(built["conditioning"])
        self.assertEqual(cc.binding_state(built, cond(), cond()), {"reason": "unbound", "branches": ["positive", "negative"]})

    def test_bind_conditioning_returns_a_copy(self):
        built = constraints.build_constraints("a", "b", {}, None, [])
        bound = cc.bind_conditioning(built, cond(), cond())
        self.assertIsNone(built["conditioning"])
        self.assertIsNotNone(bound["conditioning"])


class NodeOutputTest(unittest.TestCase):
    def test_the_node_binds_the_constraints_to_the_conditioning_it_outputs(self):
        (out, _clip) = run(wardrobe="red coat")
        pos, neg, c = out["result"]
        self.assertEqual(c["conditioning"], cc.conditioning_binding(pos, neg))
        self.assertIsNone(cc.binding_state(c, pos, neg))
        self.assertEqual(cc.validate_constraints(c), [])
        self.assertEqual([e.message for e in VALIDATOR.iter_errors(c)], [])

    def test_the_two_branches_are_fingerprinted_separately(self):
        out, _ = run(positive="one", negative="two")
        c = out["result"][2]
        self.assertNotEqual(c["conditioning"]["positive_sha256"], c["conditioning"]["negative_sha256"])

    def test_constraints_of_another_prompt_do_not_fit_this_conditioning(self):
        a, _ = run(positive="prompt A")
        b, _ = run(positive="prompt B")
        pos_b, neg_b, _ = b["result"]
        constraints_a = a["result"][2]
        self.assertEqual(cc.binding_state(constraints_a, pos_b, neg_b), {"reason": "mismatch", "branches": ["positive"]})

    def test_a_change_after_the_node_is_seen(self):
        out, _ = run()
        pos, neg, c = out["result"]
        later = [[pos[0][0], dict(pos[0][1], area=(4, 4, 0, 0))]]
        self.assertEqual(cc.binding_state(c, later, neg)["branches"], ["positive"])

    def test_the_binding_is_not_prompt_text(self):
        out, _ = run(positive="zkq_secret_prompt")
        self.assertNotIn("zkq_secret_prompt", json.dumps(out["result"][2]["conditioning"]))


class ContractModuleTest(unittest.TestCase):
    def test_constraints_reexports_the_one_validator(self):
        self.assertIs(constraints.validate_constraints, cc.validate_constraints)
        self.assertEqual(constraints.BLOCK_NAMES, cc.BLOCK_NAMES)

    def test_the_binding_schema_accepts_null_and_two_hashes_only(self):
        built = constraints.build_constraints("a", "b", {}, None, [])
        self.assertEqual([e.message for e in VALIDATOR.iter_errors(built)], [])
        built["conditioning"] = {"positive_sha256": "a" * 64, "negative_sha256": "b" * 64}
        self.assertEqual([e.message for e in VALIDATOR.iter_errors(built)], [])
        built["conditioning"] = {"positive_sha256": "a" * 64}
        self.assertNotEqual([e.message for e in VALIDATOR.iter_errors(built)], [])

    def test_the_skipped_checks_are_named(self):
        self.assertEqual(set(cc.SKIPPED_CHECKS), {"blocks_present", "layout_extent", "trigger_words",
                                                  "constraint_warnings", "text_encoder"})


if __name__ == "__main__":
    unittest.main()
