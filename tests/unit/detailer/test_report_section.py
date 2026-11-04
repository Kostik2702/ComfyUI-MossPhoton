"""(g) The detailer section of the PHOTON_REPORT validates against both contracts (the hand validator of core/report.py
and the strict JSON schema), and both reject the same broken sections (MP-58)."""
import copy
import json
import os
import unittest

import numpy as np
from jsonschema import Draft202012Validator

from moss_photon.core import canon, errors
from moss_photon.core import report as report_mod
from moss_photon.finishing import face_detailer as fd

from . import fakes

CONTRACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "fixtures", "contracts")


def load(*parts):
    with open(os.path.join(CONTRACTS, *parts), "rb") as f:
        return json.loads(f.read().decode("utf-8"))


SCHEMA = Draft202012Validator(load("schemas", "report.schema.json"))
H, W = 240, 320
FACE_A = ("A", 130, 90, 180, 150)
FACE_C = ("B", 30, 150, 80, 205)
FACE_BIG = ("A", 200, 40, 312, 200)
SENTINEL_TRIGGER = "zq7sentinel"


def det(face):
    return (float(face[1]), float(face[2]), float(face[3]), float(face[4]), 0.9)


SHA = "a" * 64


def recipe_for(settings, seed, **over):
    args = dict(model={"sha256": SHA, "basename": "moss_image_sft_bf16.safetensors", "size": 10164655808},
                lora=[{"role": "distill", "basename": "d.safetensors", "size": 1, "sha256": SHA, "strength_model": 1.0},
                      {"role": "character", "basename": "c.safetensors", "size": 2, "sha256": None, "strength_model": 0.8}],
                vae_sha256=SHA, input_sha256=[SHA], prompt_sha256="b" * 64, reference_sha256="c" * 64)
    args.update(over)
    return fd.recipe_block(settings, seed, **args)


def section_for(faces, redraw, characters, settings=None, arcface=True, seed=3):
    settings = settings or fd.Settings()
    img = fakes.make_scene(H, W, faces)
    backend = fakes.FakeBackend([det(f) for f in faces], redraw)
    _, recs, issues = fd.detail_image(img, settings, backend, characters, seed=seed)
    return fd.build_section(
        settings, seed, images=1, faces=recs, issues=issues,
        detector={"available": True, "model": "face_yolov8m.pt"},
        arcface={"available": arcface, "pack": "buffalo_l" if arcface else None},
        references={"connected": bool(characters), "consent": bool(characters), "used": bool(characters),
                    "n_images": 3 if characters else 0, "n_with_face": 3 if characters else 0,
                    "n_characters": len(characters)},
        prompt_sha256=canon.sha256_hex("close-up photo of %s's face, natural skin texture" % SENTINEL_TRIGGER),
        trigger_used=True, time_s=1.25, recipe=recipe_for(settings, seed))


SCENARIOS = {
    "accepted": lambda: section_for([FACE_A], fakes.sharpen, [fakes.reference("A")]),
    "rejected": lambda: section_for([FACE_A], fakes.flatten, [fakes.reference("A")]),
    "large": lambda: section_for([FACE_BIG], fakes.sharpen, [fakes.reference("A")]),
    "unscorable_no_refs": lambda: section_for([FACE_A], fakes.sharpen, [], settings=fd.Settings(accept_without_references=True)),
    "pass_through_no_refs": lambda: section_for([FACE_A], fakes.sharpen, []),
    "profile_after": lambda: section_for([FACE_A], fakes.sharpen, [fakes.reference("A")]),
    "mixed": lambda: section_for([FACE_BIG, FACE_A, FACE_C], fakes.sharpen, [fakes.reference("A"), fakes.reference("B")]),
    "no_faces": lambda: section_for([], fakes.sharpen, []),
    "no_colour_match": lambda: section_for([FACE_A], fakes.sharpen, [], settings=fd.Settings(
        color_match=False, accept_without_references=True)),
}


def schema_errors(rep):
    return sorted(e.message for e in SCHEMA.iter_errors(rep))


class SectionValidatesTest(unittest.TestCase):
    def setUp(self):
        self.base = load("report.json")

    def test_every_scenario_validates_in_both_contracts(self):
        for name, build in SCENARIOS.items():
            with self.subTest(scenario=name):
                rep = report_mod.with_detailer(self.base, build())
                self.assertEqual(report_mod.validate_report(rep), [])
                self.assertEqual(schema_errors(rep), [])

    def test_a_report_without_the_section_still_validates(self):
        self.assertEqual(report_mod.validate_report(self.base), [])
        self.assertEqual(schema_errors(self.base), [])
        self.assertNotIn("detailer", self.base)

    def test_the_section_carries_what_the_lead_asked_for(self):
        sec = SCENARIOS["mixed"]()
        keys = {"index", "bbox", "size_gate", "score_before", "score_after", "decision", "denoise", "time_s",
                "color_match", "unscorable"}
        for face in sec["faces"]:
            self.assertTrue(keys <= set(face), keys - set(face))
        by_decision = {f["decision"]: f for f in sec["faces"]}
        self.assertEqual(set(by_decision), {"skipped_large", "accepted", "skipped_other_character"})
        accepted = by_decision["accepted"]
        self.assertEqual(len(accepted["bbox"]), 4)
        self.assertEqual(accepted["denoise"], 0.25)
        self.assertEqual(accepted["steps"], 8)
        self.assertTrue(accepted["size_gate"]["passed"])
        self.assertIsNotNone(accepted["score_before"])
        self.assertIsNotNone(accepted["score_after"])
        self.assertIsNotNone(accepted["color_match"])
        self.assertIn("delta_e_after", accepted["color_match"])
        self.assertGreaterEqual(accepted["time_s"], 0.0)
        self.assertEqual(sec["counts"], {"faces": 3, "accepted": 1, "rejected": 0, "skipped": 2, "unscorable": 0})

    def test_counts_follow_the_decisions(self):
        self.assertEqual(SCENARIOS["rejected"]()["counts"],
                         {"faces": 1, "accepted": 0, "rejected": 1, "skipped": 0, "unscorable": 0})
        self.assertEqual(SCENARIOS["unscorable_no_refs"]()["counts"],
                         {"faces": 1, "accepted": 1, "rejected": 0, "skipped": 0, "unscorable": 1})
        self.assertEqual(SCENARIOS["pass_through_no_refs"]()["counts"],
                         {"faces": 1, "accepted": 0, "rejected": 0, "skipped": 1, "unscorable": 1})
        self.assertEqual(SCENARIOS["no_faces"]()["counts"],
                         {"faces": 0, "accepted": 0, "rejected": 0, "skipped": 0, "unscorable": 0})

    def test_the_input_report_is_not_mutated_and_the_recipe_hash_is_kept(self):
        before = copy.deepcopy(self.base)
        out = report_mod.with_detailer(self.base, SCENARIOS["accepted"]())
        self.assertEqual(self.base, before)
        self.assertIsNot(out, self.base)
        self.assertEqual(out["recipe_hash"], self.base["recipe_hash"])
        self.assertEqual(report_mod.recipe_hash(out), report_mod.recipe_hash(self.base))

    def test_the_summary_mentions_the_detailer_only_when_it_ran(self):
        out = report_mod.with_detailer(self.base, SCENARIOS["mixed"]())
        self.assertIn("detailer: 3 face(s): 1 accepted, 0 rejected, 2 skipped, 0 unscorable",
                      report_mod.format_summary(out))
        self.assertNotIn("detailer", report_mod.format_summary(self.base))

    def test_every_issue_code_in_the_section_is_registered(self):
        for name, build in SCENARIOS.items():
            for w in build()["warnings"]:
                with self.subTest(scenario=name, code=w["code"]):
                    self.assertIn(w["code"], errors.REASONS)
                    self.assertIn(w["severity"], ("info", "warning"))

    def test_the_decision_enum_of_the_report_equals_the_pipeline(self):
        spec = report_mod._DET_FACE.req["decision"]
        self.assertEqual(tuple(spec.values), fd.DECISIONS)
        schema_enum = SCHEMA.schema["properties"]["detailer"]["properties"]["faces"]["items"]["properties"]["decision"]["enum"]
        self.assertEqual(tuple(schema_enum), fd.DECISIONS)

    def test_no_prompt_text_reference_data_or_path_is_written(self):
        text = json.dumps(report_mod.with_detailer(self.base, SCENARIOS["accepted"]())["detailer"])
        self.assertNotIn(SENTINEL_TRIGGER, text)
        self.assertNotIn("close-up photo", text)
        for banned in ("embedding", "E:/", "C:/"):
            self.assertNotIn(banned, text)

    def test_redact_hides_the_basenames_of_the_detailer_recipe_too(self):
        rep = report_mod.with_detailer(self.base, SCENARIOS["accepted"]())
        self.assertIn("moss_image_sft_bf16.safetensors", json.dumps(rep["detailer"]))
        hidden = report_mod.redact(rep, keep_basenames=False)
        self.assertNotIn(".safetensors", json.dumps(hidden["detailer"]))
        self.assertEqual(hidden["detailer"]["recipe"]["model"]["sha256"], SHA)
        self.assertIn(".safetensors", json.dumps(report_mod.redact(rep)["detailer"]))   # kept by default
        self.assertEqual(report_mod.validate_report(hidden), [])


class BothContractsRejectTheSameBrokenSectionsTest(unittest.TestCase):
    MUTATIONS = {
        "extra key": lambda s: s.update(extra=1),
        "missing prompt": lambda s: s.pop("prompt"),
        "missing counts": lambda s: s.pop("counts"),
        "bad version": lambda s: s.update(version=2),
        "recipe missing": lambda s: s.pop("recipe"),
        "recipe extra": lambda s: s["recipe"].update(extra=1),
        "recipe bad hash": lambda s: s["recipe"].update(recipe_hash="nope"),
        "recipe input not a list": lambda s: s["recipe"].update(input_sha256="abc"),
        "recipe lora missing key": lambda s: s["recipe"]["lora"][0].pop("sha256"),
        "recipe model size string": lambda s: s["recipe"]["model"].update(size="big"),
        "rollback missing": lambda s: s["faces"][0].pop("rollback"),
        "settings new key missing": lambda s: s["settings"].pop("min_delta"),
        "bad sha": lambda s: s["prompt"].update(sha256="xyz"),
        "string time": lambda s: s.update(time_s="1.2"),
        "bad decision": lambda s: s["faces"][0].update(decision="maybe"),
        "face missing key": lambda s: s["faces"][0].pop("size_gate"),
        "face extra key": lambda s: s["faces"][0].update(extra=1),
        "bbox of three": lambda s: s["faces"][0].update(bbox=[1.0, 2.0, 3.0]),
        "gate not bool": lambda s: s["faces"][0]["size_gate"].update(passed="yes"),
        "score as string": lambda s: s["faces"][0].update(score_before="0.5"),
        "settings missing": lambda s: s["settings"].pop("denoise"),
        "settings extra": lambda s: s["settings"].update(turbo=True),
        "dependency flag": lambda s: s["dependencies"]["detector"].update(available="yes"),
        "references missing": lambda s: s["references"].pop("used"),
        "warning without fix": lambda s: s["warnings"][0].pop("fix"),
        "warning bad severity": lambda s: s["warnings"][0].update(severity="fatal"),
    }

    def test_each_mutation_is_rejected_by_both(self):
        base = load("report.json")
        for name, mutate in self.MUTATIONS.items():
            with self.subTest(mutation=name):
                section = SCENARIOS["rejected"]()
                self.assertTrue(section["warnings"], "control: the section needs a warning to mutate")
                mutate(section)
                rep = copy.deepcopy(base)
                rep["detailer"] = section
                self.assertTrue(report_mod.validate_report(rep), "validate_report accepted: " + name)
                self.assertTrue(schema_errors(rep), "the schema accepted: " + name)

    def test_the_hand_validator_also_refuses_non_finite_numbers(self):
        """JSON has no NaN, so only validate_report (which with_detailer uses) can see this one."""
        rep = load("report.json")
        rep["detailer"] = SCENARIOS["accepted"]()
        rep["detailer"]["time_s"] = float("nan")
        self.assertTrue(report_mod.validate_report(rep))

    def test_with_detailer_refuses_an_invalid_section(self):
        section = SCENARIOS["accepted"]()
        section["faces"][0]["decision"] = "maybe"
        with self.assertRaises(errors.OutputError):
            report_mod.with_detailer(load("report.json"), section)

    def test_a_newer_report_schema_is_refused(self):
        rep = load("report.json")
        rep["schema_version"] = 2
        with self.assertRaises(errors.MossPhotonError) as ctx:
            report_mod.with_detailer(rep, SCENARIOS["accepted"]())
        self.assertEqual(ctx.exception.code, "unsupported_feature")


class DetailerRecipeTest(unittest.TestCase):
    """Audit F05: the output has its own recipe, separate from the sampler's recipe_hash."""

    def base(self):
        return fd.Settings(), 7

    def test_it_is_deterministic_and_has_every_part(self):
        s, seed = self.base()
        a, b = recipe_for(s, seed), recipe_for(s, seed)
        self.assertEqual(a, b)
        self.assertRegex(a["recipe_hash"], "^[0-9a-f]{64}$")
        self.assertEqual(set(a), {"settings", "model", "lora", "vae_sha256", "input_sha256", "prompt_sha256",
                                  "reference_sha256", "recipe_hash"})

    def test_every_input_of_the_output_changes_the_hash(self):
        s, seed = self.base()
        ref = recipe_for(s, seed)["recipe_hash"]
        changes = {
            "seed": recipe_for(s, seed + 1),
            "setting": recipe_for(fd.Settings(denoise=0.3), seed),
            "opt-in": recipe_for(fd.Settings(accept_without_references=True), seed),
            "input image": recipe_for(s, seed, input_sha256=["d" * 64]),
            "second image": recipe_for(s, seed, input_sha256=[SHA, SHA]),
            "vae": recipe_for(s, seed, vae_sha256="e" * 64),
            "model": recipe_for(s, seed, model={"sha256": "f" * 64, "basename": "x", "size": 1}),
            "prompt": recipe_for(s, seed, prompt_sha256="1" * 64),
            "references": recipe_for(s, seed, reference_sha256=None),
            "lora strength": recipe_for(s, seed, lora=[{"role": "character", "basename": "c", "size": 2, "sha256": None,
                                                         "strength_model": 0.9}]),
        }
        for name, other in changes.items():
            with self.subTest(change=name):
                self.assertNotEqual(other["recipe_hash"], ref)

    def test_it_is_not_the_sampler_recipe_and_does_not_replace_it(self):
        base = load("report.json")
        s, seed = self.base()
        section = SCENARIOS["accepted"]()
        rep = report_mod.with_detailer(base, section)
        self.assertEqual(rep["recipe_hash"], base["recipe_hash"])            # the sampler's, untouched
        self.assertNotEqual(rep["detailer"]["recipe"]["recipe_hash"], rep["recipe_hash"])
        self.assertEqual(report_mod.recipe_hash(rep), report_mod.recipe_hash(base))

    def test_nothing_private_is_in_the_recipe(self):
        text = json.dumps(recipe_for(fd.Settings(), 1))
        for banned in ("close-up", SENTINEL_TRIGGER, "E:/", "embedding"):
            self.assertNotIn(banned, text)


class MoreScenariosTest(unittest.TestCase):
    def test_the_new_decisions_validate(self):
        base = load("report.json")
        for name in ("pass_through_no_refs", "profile_after"):
            rep = report_mod.with_detailer(base, SCENARIOS[name]())
            self.assertEqual(report_mod.validate_report(rep), [])
            self.assertEqual(schema_errors(rep), [])


if __name__ == "__main__":
    unittest.main()
