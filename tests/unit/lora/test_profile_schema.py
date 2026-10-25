"""PHOTON_IDENTITY_PROFILE (design 2.4): builder output validates against the contract schema and the fixtures."""
import copy
import json
import os
import unittest

from jsonschema import Draft202012Validator

from moss_photon.core import canon, consts, errors, lora

from .builders import DMD2, SHA_A, SHA_B, SFT_BF16, entry, full_mapping, profile_of

CONTRACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "fixtures", "contracts")


def _load(*parts):
    with open(os.path.join(CONTRACTS, *parts), "rb") as f:
        return json.loads(f.read().decode("utf-8"))


SCHEMA = Draft202012Validator(_load("schemas", "identity_profile.schema.json"))
FIXTURE = _load("identity_profile.json")
FIXTURE_TURBO = _load("identity_profile_turbo.json")
PLACEHOLDER_DIGEST = "digest does not match the stack"


def schema_errors(instance):
    return [e.message for e in SCHEMA.iter_errors(instance)]


class FixtureTest(unittest.TestCase):
    def test_fixtures_are_valid_and_pass_the_validator_except_the_placeholder_digest(self):
        for name, fx in (("character", FIXTURE), ("turbo", FIXTURE_TURBO)):
            with self.subTest(name=name):
                self.assertEqual(schema_errors(fx), [])
                self.assertEqual(lora.validate_identity_profile(fx), [PLACEHOLDER_DIGEST])

    def test_builder_reproduces_the_character_fixture_key_by_key(self):
        e = entry(0, "character", "character_a.safetensors", "0" * 64, 0.8, size=148697664,
                  metadata={"ss_base_model_version": "moss_image", "software": "ai-toolkit 0.13.34"})
        built = profile_of([e], uuid="00000000-0000-0000-0000-000000000000",
                           model_identity={"family": "moss-image-sft", "sha256": SFT_BF16})
        self.assertEqual(schema_errors(built), [])
        self.assertEqual(set(built), set(FIXTURE))
        for key in FIXTURE:
            if key == "digest":
                continue
            with self.subTest(key=key):
                got = built[key]
                if key == "summary":  # the v1 fixture predates has_distill (docs/03-turbo.md 5.4, optional in the schema)
                    self.assertFalse(got.pop("has_distill"))
                self.assertEqual(got, FIXTURE[key])

    def test_builder_reproduces_the_turbo_fixture_stack_shape(self):
        distill = entry(0, "distill", "lodestones-8-steps-dmd2-lora.safetensors", DMD2, 1.0, size=1189176800,
                        trained_on="base", base_match=False,
                        sidecar=lora.sidecar_block(None, lora.distill_entry(DMD2)))
        char = entry(1, "character", "character_a.safetensors", "0" * 64, 0.8, size=148697664,
                     metadata={"ss_base_model_version": "moss_image", "software": "ai-toolkit 0.13.34"})
        built = profile_of([distill, char], uuid="00000000-0000-0000-0000-000000000000",
                           model_identity={"family": "moss-image-sft", "sha256": SFT_BF16})
        self.assertEqual(schema_errors(built), [])
        fx = copy.deepcopy(FIXTURE_TURBO)
        for key in ("summary", "attachment", "private_paths", "model_identity", "phase_plan"):
            self.assertEqual(built[key], fx[key], key)
        for got, want in zip(built["stack"], fx["stack"]):
            want = dict(want, warnings=[])  # the warnings come from the node, not from the builder
            if want["role"] == "distill":
                want["key_format"] = got["key_format"]  # the fixture says peft_split_qkv, see test below
            self.assertEqual(got, want)

    def test_fixture_key_format_of_the_distill_example_is_only_an_example(self):
        # the example says peft_split_qkv for lora_down/lora_up keys; classify_key_format says kohya (ASSUMED, reported)
        self.assertEqual(FIXTURE_TURBO["stack"][0]["key_format"], "peft_split_qkv")
        self.assertIn(FIXTURE_TURBO["stack"][0]["key_format"], lora.KEY_FORMATS)


class BuilderTest(unittest.TestCase):
    def test_empty_stack_is_valid(self):
        p = profile_of([], baseline=0, total=0)
        self.assertEqual(schema_errors(p), [])
        self.assertEqual(lora.validate_identity_profile(p), [])
        self.assertEqual(p["summary"], {"count": 0, "roles": [], "has_character": False, "has_distill": False,
                                        "te_applied": False, "patch_entries_total": 0})

    def test_summary_flags_for_distill_plus_character(self):
        p = profile_of([entry(0, "distill", "d", DMD2, 1.0), entry(1, "character", "c", SHA_A, 0.8)])
        self.assertEqual(p["summary"], {"count": 2, "roles": ["distill", "character"], "has_character": True,
                                        "has_distill": True, "te_applied": False, "patch_entries_total": 504})
        self.assertEqual(lora.validate_identity_profile(p), [])

    def test_chain_order_is_node_order(self):
        a, b = entry(0, "distill", "d", DMD2, 1.0), entry(1, "character", "c", SHA_A, 0.8)
        fwd = profile_of([a, b])
        swapped_a, swapped_b = entry(0, "character", "c", SHA_A, 0.8), entry(1, "distill", "d", DMD2, 1.0)
        rev = profile_of([swapped_a, swapped_b])
        self.assertEqual([e["role"] for e in fwd["stack"]], ["distill", "character"])
        self.assertEqual([e["basename"] for e in fwd["attachment"]["entries"]], ["d", "c"])
        self.assertEqual([e["basename"] for e in rev["attachment"]["entries"]], ["c", "d"])
        self.assertNotEqual(fwd["digest"], rev["digest"])  # the order is part of the recipe
        self.assertNotEqual(lora.attachment_json(fwd["attachment"]), lora.attachment_json(rev["attachment"]))

    def test_digest_ignores_private_trigger_hints_but_not_other_fields(self):
        base = profile_of([entry(0, "character", "c", SHA_A, 0.8, hints=["alpha"])])
        other_hints = profile_of([entry(0, "character", "c", SHA_A, 0.8, hints=["beta", "gamma"])])
        self.assertEqual(base["digest"], other_hints["digest"])
        stripped = copy.deepcopy(base)
        for e in stripped["stack"]:
            e["trigger_hints"] = []
        self.assertEqual(lora.profile_digest(stripped), base["digest"])
        self.assertNotEqual(base["digest"], profile_of([entry(0, "character", "c", SHA_A, 0.9)])["digest"])
        self.assertNotEqual(base["digest"], profile_of([entry(0, "character", "c", SHA_B, 0.8)])["digest"])

    def test_digest_is_the_sha_of_the_canonical_stack_without_hints(self):
        p = profile_of([entry(0, "character", "c", SHA_A, 0.8, hints=["alpha"])])
        stack = [{k: v for k, v in e.items() if k != "trigger_hints"} for e in p["stack"]]
        self.assertEqual(p["digest"], canon.sha256_hex(canon.canonical_json(stack)))

    def test_trigger_hints_stay_in_memory_and_are_declared_private(self):
        p = profile_of([entry(0, "character", "c", SHA_A, 0.8, hints=["", "alpha"])])
        self.assertEqual(p["stack"][0]["trigger_hints"], ["alpha"])
        self.assertTrue(p["stack"][0]["trigger_present"])
        self.assertEqual(p["private_paths"], ["stack[*].trigger_hints"])
        self.assertFalse(profile_of([entry(0, "character", "c", SHA_A, 0.8)])["stack"][0]["trigger_present"])

    def test_the_profile_is_canonically_serialisable(self):
        p = profile_of([entry(0, "character", "c", SHA_A, 0.8)])
        self.assertEqual(json.loads(canon.canonical_json(p)), p)

    def test_stack_entry_values(self):
        e = entry(0, "character", "c", SHA_A, 0.8)
        m = full_mapping()
        self.assertEqual((e["keys_patched"], e["patch_entries"]), (168, 252))
        self.assertEqual(e["applied_to"], {"transformer": True, "text_encoder": False})
        self.assertEqual(e["weight_profile"], {"kind": "constant", "w": 0.8})
        self.assertEqual(e["coverage"]["modules_mapped"], m.modules_mapped)
        self.assertEqual(e["coverage"]["verdict"], "ok")
        self.assertEqual(e["key_format"], "peft_split_qkv")

    def test_forced_low_coverage_is_error_forced_in_the_profile(self):
        from . import golden_keys as gk
        keys = gk.peft_split()
        accepted = ["diffusion_model.layers.%d.proj.o.weight" % n for n in range(40)]
        m = lora.summarize_mapping(keys, accepted_model=accepted, accepted_te=[], mapper_names=set(), num_layers=42,
                                   te_active=False)
        v = lora.coverage_verdict(m.modules_total, m.modules_mapped, allow_low=True)
        self.assertEqual(lora.coverage_block(m, v)["verdict"], "error-forced")
        self.assertEqual(schema_errors(profile_of([entry(0, mapping=m, allow_low=True)])), [])


class ValidatorTest(unittest.TestCase):
    def good(self):
        return profile_of([entry(0, "distill", "d", DMD2, 1.0), entry(1, "character", "c", SHA_A, 0.8)])

    def test_good_profile(self):
        self.assertEqual(lora.validate_identity_profile(self.good()), [])

    def test_each_corruption_is_found(self):
        def mutate(fn):
            p = self.good()
            fn(p)
            return lora.validate_identity_profile(p)

        cases = {
            "kind": lambda p: p.update(kind="x"),
            "schema_version": lambda p: p.update(schema_version=2),
            "phase_plan": lambda p: p.update(phase_plan={"kind": "phased"}),
            "private_paths": lambda p: p.update(private_paths=[]),
            "role": lambda p: p["stack"][0].update(role="boss"),
            "index": lambda p: p["stack"][1].update(index=5),
            "key_format": lambda p: p["stack"][0].update(key_format="zip"),
            "weight_profile": lambda p: p["stack"][0]["weight_profile"].update(kind="phased"),
            "summary_roles": lambda p: p["summary"].update(roles=["character", "distill"]),
            "summary_count": lambda p: p["summary"].update(count=3),
            "attachment_len": lambda p: p["attachment"]["entries"].pop(),
            "attachment_total": lambda p: p["attachment"].update(patch_entries_total=1),
            "attachment_baseline": lambda p: p["attachment"].update(baseline_entries=7),
            "digest": lambda p: p.update(digest="f" * 64),
            "missing_attachment": lambda p: p.pop("attachment"),
            "entry_key": lambda p: p["stack"][0].pop("coverage"),
        }
        for name, fn in cases.items():
            with self.subTest(case=name):
                self.assertTrue(mutate(fn), "corruption %s went unnoticed" % name)

    def test_non_dict(self):
        self.assertEqual(lora.validate_identity_profile([]), ["profile is not an object"])


class CheckProfileInTest(unittest.TestCase):
    def prof(self):
        return profile_of([entry(0, "character", "c", SHA_A, 0.8)], uuid="u-1")

    def test_matching_attachment_passes(self):
        p = self.prof()
        self.assertIsNone(lora.check_profile_in(p, lora.attachment_json(p["attachment"])))

    def test_empty_profile_matches_a_model_without_attachment(self):
        p = profile_of([], baseline=3, total=3)
        self.assertIsNone(lora.check_profile_in(p, None))

    def test_foreign_chain_is_incompatible_lora(self):
        p = self.prof()
        other = profile_of([entry(0, "character", "other", SHA_B, 0.8)])
        for att in (None, lora.attachment_json(other["attachment"])):
            with self.subTest(att=att):
                with self.assertRaises(errors.IncompatibleLora) as ctx:
                    lora.check_profile_in(p, att)
                self.assertEqual(ctx.exception.code, "incompatible_lora")
                self.assertIn("does not belong to this MODEL", str(ctx.exception))

    def test_not_a_profile(self):
        for bad in ({}, {"kind": "x"}, "text", 5):
            with self.subTest(bad=bad):
                with self.assertRaises(errors.IncompatibleLora):
                    lora.check_profile_in(bad, None)

    def test_newer_schema_and_unknown_weight_profiles_are_unsupported_features(self):
        p = self.prof()
        newer = dict(copy.deepcopy(p), schema_version=2)
        unknown_w = copy.deepcopy(p)
        unknown_w["stack"][0]["weight_profile"] = {"kind": "ramp"}
        for bad in (newer, unknown_w):
            with self.subTest(keys=sorted(bad)[:2]):
                with self.assertRaises(errors.UnsupportedFeature):
                    lora.check_profile_in(bad, lora.attachment_json(p["attachment"]))

    def test_a_half_phased_profile_is_malformed_not_silently_constant(self):
        p = self.prof()
        for name, bad in (("plan without an entry", dict(copy.deepcopy(p), phase_plan={"x": 1})),):
            with self.subTest(name):
                with self.assertRaises(errors.IncompatibleLora) as ctx:
                    lora.check_profile_in(bad, lora.attachment_json(p["attachment"]))
                self.assertIn("malformed", str(ctx.exception))
        phased_w = copy.deepcopy(p)
        phased_w["stack"][0]["weight_profile"] = {"kind": "phased", "w0": 0.3, "w1": 0.9, "a": 0.2, "b": 0.6}
        phased_w["digest"] = lora.profile_digest(phased_w)
        self.assertTrue(any("phase_plan is null" in line for line in lora.validate_identity_profile(phased_w)))

    def test_malformed_profile_is_incompatible_lora(self):
        p = self.prof()
        p["summary"]["count"] = 9
        with self.assertRaises(errors.IncompatibleLora) as ctx:
            lora.check_profile_in(p, lora.attachment_json(p["attachment"]))
        self.assertIn("malformed", str(ctx.exception))


class ProvenanceTest(unittest.TestCase):
    def att(self, uuid="u1", total=252, baseline=0):
        return {"version": 1, "patches_uuid": uuid, "patch_entries_total": total, "baseline_entries": baseline,
                "entries": []}

    def test_first_node_takes_every_existing_entry_as_baseline(self):
        self.assertEqual(lora.provenance_baseline(None, patches_uuid="x", patch_entries=0), 0)
        self.assertEqual(lora.provenance_baseline(None, patches_uuid="x", patch_entries=40), 40)

    def test_same_uuid_keeps_the_baseline(self):
        self.assertEqual(lora.provenance_baseline(self.att(baseline=5, total=257), patches_uuid="u1",
                                                  patch_entries=257), 5)

    def test_foreign_loader_between_nodes_grows_the_baseline_by_its_entries(self):
        self.assertEqual(lora.provenance_baseline(self.att(), patches_uuid="u2", patch_entries=252 + 168), 168)
        self.assertEqual(lora.provenance_baseline(self.att(baseline=5, total=257), patches_uuid="u2",
                                                  patch_entries=257 + 10), 15)

    def test_a_removal_never_makes_the_baseline_negative(self):
        self.assertEqual(lora.provenance_baseline(self.att(), patches_uuid="u2", patch_entries=10), 0)


class AttachmentTest(unittest.TestCase):
    def test_attachment_json_is_canonical_and_stable(self):
        att = lora.build_attachment([lora.attachment_entry_of(entry(0, "character", "c", SHA_A, 0.8))],
                                    patches_uuid="u", patch_entries_total=252, baseline_entries=0)
        self.assertEqual(lora.attachment_json(att), canon.canonical_json(att))
        self.assertIsInstance(lora.attachment_json(att), str)
        self.assertEqual(list(att), ["version", "patches_uuid", "patch_entries_total", "baseline_entries", "entries"])
        self.assertEqual(consts.ATTACH_LORA_STACK, "moss_photon.lora_stack")

    def test_entry_without_a_key_is_refused(self):
        with self.assertRaises(ValueError):
            lora.build_attachment([{"basename": "x"}], patches_uuid="u", patch_entries_total=0, baseline_entries=0)

    def test_uuid_objects_are_stringified(self):
        import uuid
        u = uuid.uuid4()
        att = lora.build_attachment([], patches_uuid=u, patch_entries_total=0, baseline_entries=0)
        self.assertEqual(att["patches_uuid"], str(u))


if __name__ == "__main__":
    unittest.main()
