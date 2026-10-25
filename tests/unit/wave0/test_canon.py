"""Wave 0: core/canon.py against design 1.1 (byte-exact definitions)."""
import hashlib
import itertools
import unittest

from moss_photon.core import canon
from moss_photon.core.canon import canonical_json, deep_diff, sha256_hex, short


class CanonicalJsonTest(unittest.TestCase):
    def test_canonical_json_sorted_keys_compact(self):
        self.assertEqual(canonical_json({"b": 1, "a": [1, 2], "c": {"z": None, "y": True}}),
                         '{"a":[1,2],"b":1,"c":{"y":true,"z":null}}')

    def test_canonical_json_key_order_independent(self):
        items = [("k%d" % i, i) for i in range(5)]
        outs = {canonical_json(dict(p)) for p in itertools.permutations(items)}
        self.assertEqual(len(outs), 1)

    def test_canonical_json_keeps_utf8(self):
        self.assertEqual(canonical_json({"k": "éя"}), '{"k":"éя"}')

    def test_canonical_json_float_formatting(self):
        self.assertEqual(canonical_json([0.1, 1.0, 3, -0.0, 1e-07, 1e22, 0.9959554672241211]),
                         "[0.1,1.0,3,-0.0,1e-07,1e+22,0.9959554672241211]")

    def test_canonical_json_int_and_float_are_distinct(self):
        self.assertNotEqual(canonical_json({"cfg": 3}), canonical_json({"cfg": 3.0}))

    def test_canonical_json_tuple_equals_list(self):
        self.assertEqual(canonical_json({"a": (1, 2)}), canonical_json({"a": [1, 2]}))

    def test_canonical_json_rejects_nan_and_inf(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    canonical_json({"x": bad})

    def test_canonical_json_rejects_non_json_value(self):
        with self.assertRaises(TypeError):
            canonical_json({"x": object()})

    def test_canonical_json_is_ascii_free_of_whitespace(self):
        self.assertNotIn(" ", canonical_json({"a": [1, {"b": 2}]}))


class HashTest(unittest.TestCase):
    def test_sha256_known_vectors(self):
        self.assertEqual(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        self.assertEqual(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_sha256_str_is_utf8(self):
        self.assertEqual(sha256_hex("é"), hashlib.sha256("é".encode("utf-8")).hexdigest())
        self.assertEqual(sha256_hex("abc"), sha256_hex(b"abc"))

    def test_sha256_of_canonical_json_vector(self):
        # canonical form of {"a":1} is the 7 bytes {"a":1}
        self.assertEqual(sha256_hex(canonical_json({"a": 1})), hashlib.sha256(b'{"a":1}').hexdigest())

    def test_sha256_rejects_other_types(self):
        for bad in (123, None, bytearray(b"x"), ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    sha256_hex(bad)

    def test_short(self):
        h = sha256_hex("x")
        self.assertEqual(short(h), h[:8])
        self.assertEqual(short(h, 12), h[:12])
        self.assertEqual(len(short(h, 64)), 64)
        with self.assertRaises(ValueError):
            short(h, 0)


class DeepDiffTest(unittest.TestCase):
    def test_identical_is_empty(self):
        self.assertEqual(deep_diff({"a": {"b": 1}}, {"a": {"b": 1}}), [])

    def test_paths_are_dotted_and_sorted(self):
        base = {"sampling": {"cfg": 3.0, "steps": 50}, "name": "x", "budget": {"max_forwards": 100}}
        eff = {"sampling": {"cfg": 4.0, "steps": 25}, "name": "x", "budget": {"max_forwards": 120}}
        self.assertEqual(deep_diff(base, eff), [
            {"path": "budget.max_forwards", "base": 100, "effective": 120},
            {"path": "sampling.cfg", "base": 3.0, "effective": 4.0},
            {"path": "sampling.steps", "base": 50, "effective": 25},
        ])

    def test_lists_compared_whole(self):
        d = deep_diff({"slg": {"layers": [22]}}, {"slg": {"layers": [22, 23]}})
        self.assertEqual(d, [{"path": "slg.layers", "base": [22], "effective": [22, 23]}])

    def test_prefix(self):
        self.assertEqual(deep_diff({"a": 1}, {"a": 2}, prefix="preset"),
                         [{"path": "preset.a", "base": 1, "effective": 2}])

    def test_added_and_removed_keys_report_none_on_the_absent_side(self):
        d = deep_diff({"a": 1}, {"b": 2})
        self.assertEqual(d, [{"path": "a", "base": 1, "effective": None},
                             {"path": "b", "base": None, "effective": 2}])

    def test_absent_key_and_explicit_none_are_the_same(self):
        self.assertEqual(deep_diff({"a": 1}, {"a": 1, "b": None}), [])

    def test_dict_vs_scalar_is_a_leaf_diff(self):
        self.assertEqual(deep_diff({"a": {"b": 1}}, {"a": 5}),
                         [{"path": "a", "base": {"b": 1}, "effective": 5}])

    def test_bool_is_not_a_number(self):
        self.assertEqual(len(deep_diff({"a": True}, {"a": 1})), 1)
        self.assertEqual(len(deep_diff({"a": False}, {"a": 0})), 1)

    def test_int_equals_equal_float(self):
        self.assertEqual(deep_diff({"cfg": 3}, {"cfg": 3.0}), [])

    def test_does_not_mutate_inputs(self):
        a, b = {"x": {"y": [1]}}, {"x": {"y": [2]}}
        deep_diff(a, b)
        self.assertEqual((a, b), ({"x": {"y": [1]}}, {"x": {"y": [2]}}))


class PurityTest(unittest.TestCase):
    def test_module_has_public_names(self):
        for name in ("canonical_json", "sha256_hex", "short", "deep_diff"):
            self.assertTrue(callable(getattr(canon, name)))


if __name__ == "__main__":
    unittest.main()
