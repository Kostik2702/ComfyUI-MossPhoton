"""MP-84: the LoRA Phase plan (``core/lora_plan.py``) and the MossPhotonLoRAPhase node schema, on fakes (no comfy, no torch).

One list of phases (``consts.LORA_PHASES``): the plan table, the node's combo and the report schema agree with it. The mapping of a
phase to the darkroom phases is written out here from the ticket, so a change in the module is caught.
"""
import copy
import json
import os
import sys
import unittest
from unittest import mock

from moss_photon.core import consts, darkroom, errors, lora_plan
from moss_photon.nodes import lora_plan as node

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def input_types(cls):
    """``INPUT_TYPES`` with ``folder_paths`` unavailable (the unit suite keeps ComfyUI out of the process)."""
    with mock.patch.dict(sys.modules, {"folder_paths": None}):
        return cls.INPUT_TYPES()

TICKET_TABLE = {
    "all": ("compose", "commit", "hold", "finish"),
    "composition": ("compose",),
    "identity": ("compose", "commit", "hold", "finish"),   # MP-86: MP-85 showed compose sets hair and silhouette
    "texture": ("finish",),
    "composition+identity": ("compose", "commit", "hold"),
}
BOOSTED = ("identity",)   # MP-86: the only phase that takes the commit boost (A/B research/v2_normalize_ab: the boost is additive)


class FakeModel:
    """The part of ``ModelPatcher`` the plan touches: ``clone`` deep-copies the options, ``add_patches`` records."""

    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.added = []
        self.attachments = {}

    def clone(self):
        other = FakeModel()
        other.model_options = copy.deepcopy(self.model_options)
        return other

    def add_patches(self, patches, strength):
        self.added.append((patches, strength))

    def set_attachments(self, key, value):
        self.attachments[key] = value


def entry(phase="identity", strength=1.0, name="a.safetensors"):
    return lora_plan.make_entry(name, strength, phase, name, 10, None)


def resolved(phase, strength=1.0, index=0, patches=None, metadata=None, sha256=None, **extra):
    return lora_plan.Resolved(index=index, basename="x.safetensors", size=1, sha256=sha256, strength=strength, phase=phase,
                              patch_dict={"k": index} if patches is None else patches, metadata=metadata, **extra)


class TableTest(unittest.TestCase):
    def test_the_phase_list_is_the_one_in_consts(self):
        self.assertEqual(lora_plan.PHASES, consts.LORA_PHASES)
        self.assertEqual(tuple(lora_plan.PHASE_SEGMENTS), consts.LORA_PHASES)
        self.assertEqual(consts.LORA_PHASES, ("all", "composition", "identity", "texture", "composition+identity"))

    def test_the_mapping_is_the_owners_table(self):
        self.assertEqual(lora_plan.PHASE_SEGMENTS, TICKET_TABLE)

    def test_the_segments_are_darkroom_phases_in_darkroom_order(self):
        for phase, segments in lora_plan.PHASE_SEGMENTS.items():
            with self.subTest(phase=phase):
                self.assertTrue(set(segments) <= set(darkroom.PHASE_NAMES))
                self.assertEqual(list(segments), [p for p in darkroom.PHASE_NAMES if p in segments])

    def test_all_and_identity_cover_the_whole_run(self):
        """MP-86: identity acts in every darkroom segment, so on a preset without phases it is the whole run too."""
        self.assertEqual([p for p in lora_plan.PHASES if lora_plan.covers_whole_run(p)], ["all", "identity"])
        self.assertEqual([p for p in lora_plan.PHASES if set(TICKET_TABLE[p]) == set(darkroom.PHASE_NAMES)],
                         ["all", "identity"], "derived from the table, not a name")
        self.assertEqual(set(lora_plan.PHASE_SEGMENTS["all"]), set(darkroom.PHASE_NAMES))

    def test_the_node_combo_and_the_report_schema_follow_the_list(self):
        self.assertEqual(input_types(node.MossPhotonLoRAPhase)["required"]["phase"][0], list(consts.LORA_PHASES))
        with open(os.path.join(PKG, "tests", "fixtures", "contracts", "schemas", "report.schema.json"), "r",
                  encoding="utf-8") as f:
            schema = json.load(f)
        enum = schema["properties"]["lora"]["properties"]["plan"]["items"]["properties"]["phase"]["enum"]
        self.assertEqual(enum, list(consts.LORA_PHASES))


class EntryTest(unittest.TestCase):
    def test_make_entry_is_a_plain_json_safe_dict(self):
        e = lora_plan.make_entry("sub/a.safetensors", 0.8, "identity", "a.safetensors", 12, "f" * 64, warnings=[{"code": "x"}])
        self.assertEqual(e, {"lora_name": "sub/a.safetensors", "strength": 0.8, "phase": "identity",
                             "basename": "a.safetensors", "size": 12, "sha256": "f" * 64, "warnings": [{"code": "x"}]})
        json.dumps(e)

    def test_a_phase_outside_the_list_and_a_bad_strength_are_refused(self):
        for phase, strength in (("compose", 1.0), ("", 1.0), ("identity", float("nan")), ("identity", float("inf")),
                                ("identity", 20.01), ("identity", -20.01), ("identity", True), ("identity", "1")):
            with self.subTest(phase=phase, strength=strength):
                with self.assertRaises(errors.InvalidOverride):
                    lora_plan.make_entry("a", strength, phase, "a", 1, None)

    def test_the_strength_bounds_are_the_widgets(self):
        spec = input_types(node.MossPhotonLoRAPhase)["required"]["strength"][1]
        self.assertEqual((spec["min"], spec["max"], spec["default"]), (-20.0, 20.0, 1.0))
        lora_plan.make_entry("a", 20.0, "all", "a", 1, None)
        lora_plan.make_entry("a", -20.0, "all", "a", 1, None)


class PlanOnTheModelTest(unittest.TestCase):
    def test_add_to_plan_clones_and_never_touches_the_input(self):
        base = FakeModel()
        one = lora_plan.add_to_plan(base, entry("identity", name="a"))
        two = lora_plan.add_to_plan(one, entry("texture", 0.5, name="b"))
        self.assertIsNot(one, base)
        self.assertNotIn(lora_plan.PLAN_KEY, base.model_options, "the plan never leaks into the model it came from")
        self.assertEqual([e["lora_name"] for e in lora_plan.entries_of(one)], ["a"])
        self.assertEqual([e["lora_name"] for e in lora_plan.entries_of(two)], ["a", "b"])
        self.assertEqual([e["lora_name"] for e in lora_plan.entries_of(one)], ["a"], "the earlier model keeps its plan")

    def test_two_branches_of_one_model_do_not_share_a_plan(self):
        base = FakeModel()
        left = lora_plan.add_to_plan(base, entry(name="left"))
        right = lora_plan.add_to_plan(base, entry(name="right"))
        self.assertEqual([e["lora_name"] for e in lora_plan.entries_of(left)], ["left"])
        self.assertEqual([e["lora_name"] for e in lora_plan.entries_of(right)], ["right"])

    def test_a_clone_carries_the_plan_along_by_copy(self):
        plan = lora_plan.add_to_plan(FakeModel(), entry(name="a"))
        clone = plan.clone()
        clone.model_options[lora_plan.PLAN_KEY].append({"lora_name": "intruder"})
        self.assertEqual([e["lora_name"] for e in lora_plan.entries_of(plan)], ["a"])

    def test_entries_of_hands_out_copies(self):
        plan = lora_plan.add_to_plan(FakeModel(), lora_plan.make_entry("a", 1.0, "all", "a", 1, None, warnings=[{"code": "w"}]))
        got = lora_plan.entries_of(plan)
        got[0]["strength"] = 99.0
        got[0]["warnings"].append({"code": "z"})
        self.assertEqual(lora_plan.entries_of(plan)[0]["strength"], 1.0)
        self.assertEqual(lora_plan.entries_of(plan)[0]["warnings"], [{"code": "w"}])

    def test_a_model_without_a_plan_or_without_options_has_none(self):
        self.assertEqual(lora_plan.entries_of(FakeModel()), ())
        self.assertEqual(lora_plan.entries_of(object()), ())
        self.assertEqual(lora_plan.entries_of(None), ())


class ActiveSetsTest(unittest.TestCase):
    def test_active_follows_the_table(self):
        entries = [entry(p, name=p) for p in lora_plan.PHASES]
        for darkroom_phase in darkroom.PHASE_NAMES:
            with self.subTest(phase=darkroom_phase):
                want = tuple(i for i, p in enumerate(lora_plan.PHASES) if darkroom_phase in TICKET_TABLE[p])
                self.assertEqual(lora_plan.active(entries, darkroom_phase), want)

    def test_active_reads_resolved_entries_too_and_skips_zero_strength(self):
        entries = [resolved("all", 1.0, 0), resolved("all", 0.0, 1), resolved("texture", -2.0, 2)]
        self.assertEqual(lora_plan.active(entries, "compose"), (0,))
        self.assertEqual(lora_plan.active(entries, "finish"), (0, 2))

    def test_only_a_live_identity_entry_takes_the_boost(self):
        """MP-86: the boost is additive (+0.35), so it suits a character LoRA near 1.0 and not a style LoRA or a slider:
        ``all`` is exactly an ordinary loader and takes none, and so does ``composition+identity`` (the place of style and
        slider LoRAs)."""
        self.assertEqual(lora_plan.PHASES, ("all", "composition", "identity", "texture", "composition+identity"))
        entries = [resolved(p, 1.0, i) for i, p in enumerate(lora_plan.PHASES)] + [resolved("identity", 0.0, 9)]
        self.assertEqual(lora_plan.boost_indices(entries), (2,))
        self.assertEqual(lora_plan.boost_indices([]), ())

    def test_the_boosted_phases_are_one_list_next_to_the_table(self):
        self.assertEqual(lora_plan.BOOSTED_PHASES, BOOSTED)
        for phase in lora_plan.BOOSTED_PHASES:
            with self.subTest(phase=phase):
                self.assertIn(phase, lora_plan.PHASES)
                self.assertIn("commit", lora_plan.PHASE_SEGMENTS[phase], "a boosted phase must act in the commit")
        self.assertNotIn("all", lora_plan.BOOSTED_PHASES)

    def test_identity_acts_from_the_first_step_and_all_is_an_ordinary_loader(self):
        self.assertEqual(lora_plan.PHASE_SEGMENTS["identity"], ("compose", "commit", "hold", "finish"))
        self.assertEqual(lora_plan.PHASE_SEGMENTS["identity"], lora_plan.PHASE_SEGMENTS["all"])
        self.assertEqual(lora_plan.active([entry("identity")], "compose"), (0,))

    def test_plain_entry_dicts_take_the_boost_by_the_same_rule(self):
        entries = [entry(p, name=p) for p in lora_plan.PHASES]
        self.assertEqual(lora_plan.boost_indices(entries), (2,))

    def test_a_known_distill_file_never_takes_the_boost_whatever_its_phase(self):
        distill = next(iter(consts.KNOWN_DISTILL_LORAS))
        entries = [resolved(p, 1.0, i, sha256=distill) for i, p in enumerate(lora_plan.PHASES)]
        self.assertEqual(lora_plan.boost_indices(entries), ())
        plain = [dict(entry(p, name=p), sha256=distill) for p in lora_plan.PHASES]
        self.assertEqual(lora_plan.boost_indices(plain), ())
        other = resolved("identity", 1.0, 0, sha256="0" * 64)
        self.assertEqual(lora_plan.boost_indices([other]), (0,), "control: an unknown sha is an ordinary LoRA")

    def test_a_negative_strength_never_takes_the_boost(self):
        """MP-86 review: the node default is ``identity``, so a slider left on it at -2 would become -1.65 and -0.2 would flip
        to +0.15; the boost is for a positive character strength only."""
        entries = [resolved("identity", -2.0, 0), resolved("identity", -0.2, 1), resolved("identity", 1.0, 2),
                   resolved("identity", 0.0, 3), resolved("all", -2.0, 4)]
        self.assertEqual(lora_plan.boost_indices(entries), (2,))
        self.assertEqual(lora_plan.negative_boost_indices(entries), (0, 1), "the entries that asked for a phase with a boost")
        plain = [entry("identity", name="a", strength=-0.2), entry("identity", name="b", strength=0.3)]
        self.assertEqual((lora_plan.boost_indices(plain), lora_plan.negative_boost_indices(plain)), ((1,), (0,)))


class ApplyTest(unittest.TestCase):
    def test_a_constant_run_adds_every_entry_in_plan_order_at_its_strength(self):
        model = FakeModel()
        entries = [resolved("texture", 0.5, 0), resolved("identity", 1.5, 1), resolved("composition", -1.0, 2)]
        self.assertIs(lora_plan.apply_to(model, entries), model)
        self.assertEqual(model.added, [({"k": 0}, 0.5), ({"k": 1}, 1.5), ({"k": 2}, -1.0)])

    def test_a_phase_adds_only_the_entries_that_act_in_it(self):
        entries = [resolved(p, 1.0, i) for i, p in enumerate(lora_plan.PHASES)]
        for darkroom_phase in darkroom.PHASE_NAMES:
            with self.subTest(phase=darkroom_phase):
                model = FakeModel()
                lora_plan.apply_to(model, entries, darkroom_phase)
                want = [({"k": i}, 1.0) for i, p in enumerate(lora_plan.PHASES) if darkroom_phase in TICKET_TABLE[p]]
                self.assertEqual(model.added, want)

    def test_the_boost_goes_to_identity_only(self):
        model = FakeModel()
        entries = [resolved("identity", 1.0, 0), resolved("texture", 1.0, 1), resolved("all", 1.0, 2),
                   resolved("composition+identity", 0.5, 3), resolved("composition", 1.0, 4)]
        lora_plan.apply_to(model, entries, "finish", boost=0.35)
        self.assertEqual(model.added, [({"k": 0}, 1.35), ({"k": 1}, 1.0), ({"k": 2}, 1.0)], "finish: identity, texture, all act")
        model = FakeModel()
        lora_plan.apply_to(model, entries, "commit", boost=0.35)
        self.assertEqual(model.added, [({"k": 0}, 1.35), ({"k": 2}, 1.0), ({"k": 3}, 0.5)])

    def test_a_slider_and_a_style_lora_keep_their_strength_in_every_segment_they_act_in(self):
        """MP-86 review: an additive boost would turn a slider at -2 into -1.65 (weaker, the wrong way) and a style LoRA at
        .65 into 1.0; ``all`` and ``composition+identity`` (the recommended place for them) take no boost."""
        distill = next(iter(consts.KNOWN_DISTILL_LORAS))
        for phase in ("all", "composition+identity"):
            entries = [resolved(phase, -2.0, 0), resolved(phase, 0.65, 1)]
            for darkroom_phase in darkroom.PHASE_NAMES:
                if darkroom_phase not in TICKET_TABLE[phase]:
                    continue
                with self.subTest(phase=phase, darkroom_phase=darkroom_phase):
                    model = FakeModel()
                    lora_plan.apply_to(model, entries, darkroom_phase, boost=0.35)
                    self.assertEqual(model.added, [({"k": 0}, -2.0), ({"k": 1}, 0.65)])
        model = FakeModel()
        lora_plan.apply_to(model, [resolved("identity", 1.0, 0, sha256=distill)], "commit", boost=0.35)
        self.assertEqual(model.added, [({"k": 0}, 1.0)], "a known distill under identity keeps its strength too")

    def test_a_negative_identity_strength_keeps_its_value_in_every_segment(self):
        entries = [resolved("identity", -2.0, 0), resolved("identity", -0.2, 1), resolved("identity", 1.0, 2)]
        for darkroom_phase in darkroom.PHASE_NAMES:
            with self.subTest(phase=darkroom_phase):
                model = FakeModel()
                lora_plan.apply_to(model, entries, darkroom_phase, boost=0.35)
                self.assertEqual(model.added, [({"k": 0}, -2.0), ({"k": 1}, -0.2), ({"k": 2}, 1.35)])

    def test_no_boost_leaves_the_strength_bit_equal(self):
        model = FakeModel()
        lora_plan.apply_to(model, [resolved("identity", 0.7, 0)], "commit", boost=0.0)
        self.assertEqual(model.added[0][1], 0.7)
        self.assertIs(type(model.added[0][1]), float)

    def test_zero_strength_is_never_added_and_metadata_rides_along(self):
        model = FakeModel()
        lora_plan.apply_to(model, [resolved("all", 0.0, 0), resolved("all", 1.0, 1, metadata={"m": 1})])
        self.assertEqual(model.added, [({"k": 1}, 1.0)])
        self.assertEqual(model.attachments, {"lora_metadata": {"m": 1}})


class NodeSchemaTest(unittest.TestCase):
    def test_the_node_is_a_helper_and_has_one_model_output(self):
        self.assertEqual(node.NODE_ID, "MossPhotonLoRAPhase")
        self.assertIn(node.NODE_ID, consts.NODE_IDS_HELPERS)
        self.assertEqual(node.NODE_CLASS_MAPPINGS, {"MossPhotonLoRAPhase": node.MossPhotonLoRAPhase})
        self.assertEqual(node.NODE_DISPLAY_NAME_MAPPINGS, {"MossPhotonLoRAPhase": "Moss Photon LoRA Phase"})
        self.assertEqual(node.MossPhotonLoRAPhase.RETURN_TYPES, ("MODEL",))
        self.assertEqual(node.MossPhotonLoRAPhase.CATEGORY, consts.CAT_SETUP)

    def test_inputs_and_defaults(self):
        spec = input_types(node.MossPhotonLoRAPhase)
        self.assertEqual(list(spec["required"]), ["model", "lora_name", "strength", "phase"])
        self.assertNotIn("optional", spec)
        self.assertEqual(spec["required"]["phase"][1]["default"], "identity", "the character phase (Krea's default)")
        self.assertEqual(spec["required"]["lora_name"][1]["default"], "none")
        self.assertEqual(spec["required"]["lora_name"][0][0], "none")

    def test_none_and_zero_strength_pass_the_same_model_through(self):
        model = FakeModel()
        n = node.MossPhotonLoRAPhase()
        self.assertIs(n.apply(model, "none", 1.0, "identity")[0], model)
        self.assertIs(n.apply(model, "a.safetensors", 0.0, "identity")[0], model)
        self.assertIs(n.apply(model, "", 1.0, "texture")[0], model)
        self.assertEqual(lora_plan.entries_of(model), ())

    def test_the_old_profile_node_is_deprecated_and_untouched_otherwise(self):
        from moss_photon.nodes import lora as old
        self.assertIs(old.MossPhotonLoRAProfile.DEPRECATED, True)
        self.assertTrue(old.NODE_DISPLAY_NAME_MAPPINGS["MossPhotonLoRAProfile"].endswith(" (deprecated)"))
        self.assertIn("DEPRECATED", old.MossPhotonLoRAProfile.DESCRIPTION)
        self.assertEqual(list(input_types(old.MossPhotonLoRAProfile)["required"]),
                         ["model", "lora_name", "strength_model", "role", "profile"])
        self.assertEqual(old.MossPhotonLoRAProfile.RETURN_TYPES, ("MODEL", "PHOTON_IDENTITY_PROFILE", "CLIP"))


if __name__ == "__main__":
    unittest.main()
