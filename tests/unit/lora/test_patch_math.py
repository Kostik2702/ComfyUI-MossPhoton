"""MP-60 (audit F01): the factors of a LoRA are checked against the weight they land on BEFORE anything is registered.

The native adapters swallow a delta that does not fit (log + weight unchanged), so a file that maps by name but does not
fit used to count as fully applied. ``check_patch_math`` is pure and duck-typed; these tests use real tensors and
adapter look-alikes (``name`` + ``weights``), no ComfyUI. The integration twin is
``tests/integration/test_audit_p1_lora.py`` (repro A on the tiny MOSS, with the native weight read back)."""
import types
import unittest

import torch

from moss_photon.core import consts, lora

DMD2 = "7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2"
W = "diffusion_model.layers.0.proj.qkv.weight"


def adapter(name, *weights):
    return types.SimpleNamespace(name=name, weights=tuple(weights))


def lora_adapter(out=384, in_=128, rank=4, *, up=None, down=None, alpha=4.0, mid=None, reshape=None):
    return adapter("lora", up if up is not None else torch.ones(out, rank),
                   down if down is not None else torch.ones(rank, in_), alpha, mid, None, reshape)


def check(patch_dict, shapes):
    return lora.check_patch_math(patch_dict, shapes.get)


class LoraShapeTest(unittest.TestCase):
    SHAPES = {W: (384, 128)}

    def test_a_fitting_lora_is_ok_and_counted_as_verified(self):
        bad, nonfinite, unchecked = check({W: lora_adapter()}, self.SHAPES)
        self.assertEqual((bad, nonfinite, unchecked), ({}, {}, []))

    def test_audit_repro_a_input_dim_127_instead_of_128(self):
        bad, nonfinite, unchecked = check({W: lora_adapter(in_=127)}, self.SHAPES)
        self.assertEqual(list(bad), [W])
        self.assertIn("does not fit", bad[W])
        self.assertIn("384", bad[W])

    def test_a_transposed_delta_of_the_same_size_is_refused(self):
        # native reshape() would accept 128x384 for a 384x128 weight and scramble the delta
        patch = lora_adapter(up=torch.ones(128, 4), down=torch.ones(4, 384))
        self.assertEqual(list(check({W: patch}, self.SHAPES)[0]), [W])

    def test_up_and_down_must_share_the_rank(self):
        patch = lora_adapter(up=torch.ones(384, 4), down=torch.ones(8, 128))
        self.assertIn("rank differs", check({W: patch}, self.SHAPES)[0][W])

    def test_rank_zero_and_one_dimensional_factors_are_refused(self):
        self.assertIn("rank is 0", check({W: lora_adapter(up=torch.ones(384, 0), down=torch.ones(0, 128))},
                                         self.SHAPES)[0][W])
        self.assertIn("2 or more dims", check({W: lora_adapter(up=torch.ones(384), down=torch.ones(128))},
                                              self.SHAPES)[0][W])

    def test_fused_qkv_slices_use_the_slice_not_the_whole_weight(self):
        q, k = (W, (0, 0, 128)), (W, (0, 128, 128))
        ok = lora_adapter(out=128, in_=128)
        whole = lora_adapter(out=384, in_=128)  # the full qkv delta for one third: wrong
        bad, _, _ = check({q: ok, k: whole}, self.SHAPES)
        self.assertEqual(list(bad), [k])

    def test_a_slice_outside_the_weight_is_refused(self):
        bad, _, _ = check({(W, (0, 300, 128)): lora_adapter(out=128)}, self.SHAPES)
        self.assertIn("slice", next(iter(bad.values())))
        bad, _, _ = check({(W, (5, 0, 128)): lora_adapter(out=128)}, self.SHAPES)
        self.assertEqual(len(bad), 1)

    def test_a_key_that_is_not_in_the_model_is_not_checked(self):
        self.assertEqual(check({"no.such.weight": lora_adapter(in_=1)}, self.SHAPES), ({}, {}, []))

    def test_a_reshape_entry_is_checked_against_the_padded_shape(self):
        patch = lora_adapter(out=400, in_=128, reshape=[400, 128])
        self.assertEqual(check({W: patch}, self.SHAPES)[0], {})

    def test_tucker_mid_needs_chaining_ranks(self):
        shapes = {"conv.weight": (16, 8, 3, 3)}
        good = adapter("lora", torch.ones(16, 4), torch.ones(4, 8, 1, 1), 4.0, torch.ones(4, 4, 3, 3), None, None)
        broken = adapter("lora", torch.ones(16, 4), torch.ones(5, 8, 1, 1), 4.0, torch.ones(4, 4, 3, 3), None, None)
        self.assertEqual(check({"conv.weight": good}, shapes)[0], {})
        self.assertEqual(list(check({"conv.weight": broken}, shapes)[0]), ["conv.weight"])


class FiniteTest(unittest.TestCase):
    SHAPES = {W: (384, 128)}

    def test_nan_or_inf_in_a_factor_is_reported_not_applied(self):
        for name, mutate in (("nan up", lambda up, down: up.__setitem__((3, 1), float("nan"))),
                             ("inf down", lambda up, down: down.__setitem__((0, 0), float("inf")))):
            with self.subTest(name):
                up, down = torch.ones(384, 4), torch.ones(4, 128)
                mutate(up, down)
                bad, nonfinite, _ = check({W: lora_adapter(up=up, down=down)}, self.SHAPES)
                self.assertEqual((bad, list(nonfinite)), ({}, [W]))

    def test_alpha_and_dora_scale_are_checked_too(self):
        _, nonfinite, _ = check({W: lora_adapter(alpha=float("nan"))}, self.SHAPES)
        self.assertEqual(list(nonfinite), [W])
        patch = adapter("lora", torch.ones(384, 4), torch.ones(4, 128), 4.0, None, torch.tensor([float("inf")]), None)
        self.assertEqual(list(check({W: patch}, self.SHAPES)[1]), [W])

    def test_a_shape_problem_is_reported_as_shape_not_also_as_nonfinite(self):
        up = torch.full((384, 4), float("nan"))
        bad, nonfinite, _ = check({W: lora_adapter(in_=127, up=up)}, self.SHAPES)
        self.assertEqual((list(bad), nonfinite), ([W], {}))


class OtherKindsTest(unittest.TestCase):
    SHAPES = {W: (384, 128), "b": (384,)}

    def test_diff_and_set_must_equal_the_weight(self):
        ok = {W: ("diff", (torch.zeros(384, 128),)), "b": ("set", (torch.zeros(384),))}
        self.assertEqual(check(ok, self.SHAPES), ({}, {}, []))
        bad = {W: ("diff", (torch.zeros(384, 127),)), "b": ("set", (torch.zeros(383),))}
        self.assertEqual(sorted(check(bad, self.SHAPES)[0]), sorted([W, "b"]))

    def test_a_padding_diff_is_not_judged(self):
        patch = {W: ("diff", (torch.zeros(400, 128), {"pad_weight": True}))}
        self.assertEqual(check(patch, self.SHAPES), ({}, {}, [W]))

    def test_loha_and_lokr_get_their_own_rule(self):
        loha = adapter("loha", torch.ones(384, 2), torch.ones(2, 128), 2.0, torch.ones(384, 2), torch.ones(2, 128),
                       None, None, None)
        self.assertEqual(check({W: loha}, self.SHAPES), ({}, {}, []))
        loha_bad = adapter("loha", torch.ones(384, 2), torch.ones(2, 127), 2.0, torch.ones(384, 2), torch.ones(2, 127),
                           None, None, None)
        self.assertEqual(list(check({W: loha_bad}, self.SHAPES)[0]), [W])
        lokr = adapter("lokr", torch.ones(24, 8), torch.ones(16, 16), None, None, None, None, None, None, None)
        self.assertEqual(check({W: lokr}, self.SHAPES), ({}, {}, []))
        lokr_bad = adapter("lokr", torch.ones(24, 8), torch.ones(16, 15), None, None, None, None, None, None, None)
        self.assertEqual(list(check({W: lokr_bad}, self.SHAPES)[0]), [W])

    def test_a_kind_without_a_rule_is_unchecked_but_still_finite_checked(self):
        patch = adapter("oft", torch.ones(4, 4), torch.ones(1), None, None)
        self.assertEqual(check({W: patch}, self.SHAPES), ({}, {}, [W]))
        nan_patch = adapter("oft", torch.full((4, 4), float("nan")), torch.ones(1), None, None)
        bad, nonfinite, unchecked = check({W: nan_patch}, self.SHAPES)
        self.assertEqual((bad, list(nonfinite), unchecked), ({}, [W], []))


class MappingCountsTest(unittest.TestCase):
    """key_found, shape_ok and applied are three numbers; a mismatching module never reaches coverage."""

    KEYS = ["diffusion_model.layers.%d.proj.o.%s" % (i, s) for i in range(4) for s in ("lora_A.weight", "lora_B.weight")]
    ACCEPTED = ["diffusion_model.layers.%d.proj.o.weight" % i for i in range(4)]

    def mapping(self, accepted, **kw):
        return lora.summarize_mapping(self.KEYS, accepted_model=accepted, accepted_te=[], mapper_names=set(),
                                      num_layers=4, te_active=False, **kw)

    def test_all_good(self):
        m = self.mapping(self.ACCEPTED)
        self.assertEqual((m.key_found, m.shape_ok, m.modules_mapped, m.shape_unchecked), (4, 4, 4, 0))
        self.assertEqual(lora.shape_verdict(m)["verdict"], "ok")

    def test_mismatches_are_found_but_not_applied(self):
        m = self.mapping(self.ACCEPTED[:3], shape_mismatch=["layers.3.proj.o.weight: delta 1x1 does not fit"])
        self.assertEqual((m.key_found, m.shape_ok, m.modules_mapped), (4, 3, 3))
        self.assertEqual(m.shape_mismatch, ("layers.3.proj.o.weight: delta 1x1 does not fit",))

    def test_unchecked_entries_are_applied_but_not_shape_ok(self):
        m = self.mapping(self.ACCEPTED, unchecked=1)
        self.assertEqual((m.key_found, m.shape_ok, m.shape_unchecked, m.modules_mapped), (4, 3, 1, 4))

    def test_coverage_block_carries_the_three_counts(self):
        m = self.mapping(self.ACCEPTED[:3], shape_mismatch=["x: y"])
        block = lora.coverage_block(m, lora.coverage_verdict(m.modules_total, m.modules_mapped))
        self.assertEqual((block["key_found"], block["shape_ok"], block["applied"], block["modules_mapped"]), (4, 3, 3, 3))
        self.assertEqual(block["shape_mismatch"], ["x: y"])


class ShapeVerdictThresholdTest(unittest.TestCase):
    def verdict(self, found, mismatched, applied=None, nonfinite=0):
        m = lora.MappingResult(
            tensors_total=0, modules_total=found, modules_mapped=found - mismatched - nonfinite if applied is None
            else applied, model_entries=0, keys_patched=0, te_entries=0, te_modules_ignored=0, te_modules_in_file=0,
            te_active=False, layers_total=0, layers_covered=0, layers_missing=(), unmapped_examples=(),
            key_format="unknown", key_found=found, shape_mismatch=tuple("m%d" % i for i in range(mismatched)),
            nonfinite=tuple("n%d" % i for i in range(nonfinite)))
        return lora.shape_verdict(m)

    def test_table(self):
        self.assertEqual(self.verdict(24, 0)["verdict"], "ok")
        self.assertEqual(self.verdict(24, 1)["verdict"], "warn")       # 4.2% < 5%
        self.assertEqual(self.verdict(24, 1)["code"], "partial_lora_coverage")
        self.assertEqual(self.verdict(24, 2)["verdict"], "error")      # 8.3%
        self.assertEqual(self.verdict(100, 4)["verdict"], "warn")
        self.assertEqual(self.verdict(100, 5)["verdict"], "error")     # the share is inclusive
        self.assertEqual(self.verdict(24, 24)["verdict"], "error")     # audit repro A: nothing applied
        self.assertEqual(self.verdict(24, 0, nonfinite=1)["verdict"], "error")

    def test_the_threshold_mirrors_the_coverage_ok_value(self):
        self.assertAlmostEqual(lora.SHAPE_MISMATCH_ERROR_SHARE, 1.0 - lora.COVERAGE_OK)


class EffectiveRoleTest(unittest.TestCase):
    def test_a_known_distill_hash_is_a_distill_whatever_the_role(self):
        for role in lora.ROLES:
            self.assertEqual(lora.effective_role(role, DMD2), "distill")

    def test_an_unknown_missing_or_odd_hash_keeps_the_role(self):
        for sha in ("f" * 64, None, "", 5, ["x"]):
            self.assertEqual(lora.effective_role("character", sha), "character", sha)

    def test_the_table_is_the_one_in_consts(self):
        self.assertIn(DMD2, consts.KNOWN_DISTILL_LORAS)


if __name__ == "__main__":
    unittest.main()
