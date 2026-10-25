"""MP-7: schema_lite agrees with the jsonschema package (tests only, design R6) on the 8 presets and a mutation corpus."""
import copy
import json
import os
import unittest

from moss_photon.core import consts, presets, schema_lite

try:
    import jsonschema
except ImportError:  # the only legal skip: the reference implementation is absent
    jsonschema = None


def _path(parts):
    out = ""
    for p in parts:
        out += "[%d]" % p if isinstance(p, int) else ("." if out else "") + str(p)
    return out


def lite_set(instance, schema):
    return {(e.path, e.keyword) for e in schema_lite.validate(instance, schema)}


def ref_set(instance, schema):
    v = jsonschema.Draft202012Validator(schema)
    return {(_path(e.absolute_path), e.validator) for e in v.iter_errors(instance)}


def build_mutations(base):
    muts = []

    def m(fn):
        q = copy.deepcopy(base)
        fn(q)
        muts.append(q)

    # the 23 of research/design_mp4/build_contracts.py
    m(lambda q: q.pop("sources"))
    m(lambda q: q["sampling"].update(steps=0))
    m(lambda q: q["sampling"].update(steps=50.5))
    m(lambda q: q["sampling"].update(steps=True))
    m(lambda q: q["sampling"].update(cfg=0.5))
    m(lambda q: q["sampling"].update(shift=0))
    m(lambda q: q["sampling"].update(sampler="euler_ancestral"))
    m(lambda q: q["sampling"].update(restart=True))
    m(lambda q: q["sampling"].update(slg_layers=[22, 22]))
    m(lambda q: q["sampling"].update(slg_layers=[-1]))
    m(lambda q: q["sampling"].update(slg_start=1.5))
    m(lambda q: q["sampling"].update(extra=1))
    m(lambda q: q.update(id="nope"))
    m(lambda q: q.update(preset_version="1.0"))
    m(lambda q: q.update(schema_version=2))
    m(lambda q: q.update(postprocess=["upscale"]))
    m(lambda q: q.update(model_families=[]))
    m(lambda q: q.update(status="stable-ish"))
    m(lambda q: q["budget"].update(max_forwards=0))
    m(lambda q: q["ui"].pop("label"))
    m(lambda q: q["calibration"].update(state="done"))
    m(lambda q: q["sources"]["sampling.steps"].update(kind="vibes"))
    m(lambda q: q.update(unknown_top_level=1))
    # type confusions and further bounds
    m(lambda q: q["sampling"].update(steps="50"))
    m(lambda q: q["sampling"].update(steps=None))
    m(lambda q: q["sampling"].update(steps=1001))
    m(lambda q: q["sampling"].update(cfg="3"))
    m(lambda q: q["sampling"].update(cfg=True))
    m(lambda q: q["sampling"].update(shift=-1.0))
    m(lambda q: q["sampling"].update(shift=100.5))
    m(lambda q: q["sampling"].update(restart=0))
    m(lambda q: q["sampling"].update(ancestral_noise=None))
    m(lambda q: q["sampling"].update(slg_layers="22"))
    m(lambda q: q["sampling"].update(slg_layers=[22.5]))
    m(lambda q: q["sampling"].update(slg_layers=[True]))
    m(lambda q: q["sampling"].update(slg_layers=list(range(17))))
    m(lambda q: q["sampling"].update(slg_end=-0.1))
    m(lambda q: q["sampling"].pop("cfg"))
    m(lambda q: q.update(description=""))
    m(lambda q: q.update(description="x" * 401))
    m(lambda q: q.update(description=5))
    m(lambda q: q.update(model_families=["moss-image-sft", "moss-image-sft"]))
    m(lambda q: q.update(model_families=["moss-image-sft", "moss-image-base", "moss-image-sft"]))
    m(lambda q: q.update(model_families="moss-image-sft"))
    m(lambda q: q.update(preset_version="1.0.0 "))
    m(lambda q: q.update(preset_version=1))
    m(lambda q: q.update(constraints=[{"id": "Bad Id", "kind": "nope", "severity": "fatal", "reason": "x"}]))
    m(lambda q: q.update(constraints=[{"id": "a", "kind": "requires_block", "severity": "error", "reason": "none",
                                       "block": "x", "extra": 1}]))
    m(lambda q: q.update(guards=[{"id": "photo_artifacts", "required": "no", "requires_dependency": "none",
                                  "implemented_in": "v1"}]))
    m(lambda q: q.update(guards=[{"id": "photo_artifacts"}]))
    m(lambda q: q.update(constraints={}))
    m(lambda q: q["budget"].update(max_forwards="10"))
    m(lambda q: q["budget"].update(max_forwards=1.5))
    m(lambda q: q["budget"].update(extra_forwards=-1))
    m(lambda q: q["budget"].update(nominal_steps=0))
    m(lambda q: q["calibration"].update(manifest=3))
    m(lambda q: q["calibration"].update(reference=""))
    m(lambda q: q["ui"].update(recommended_final="yes"))
    m(lambda q: q["ui"].update(group="Other"))
    m(lambda q: q["ui"].update(summary="s" * 161))
    m(lambda q: q["sources"].update(x={"unit": "u", "kind": "spec"}))
    m(lambda q: q["sources"].update(x=5))
    m(lambda q: q.update(sources=[]))
    m(lambda q: q.update(checkpoint_policy="any"))
    m(lambda q: q.update(lora_policy=None))
    m(lambda q: q.update(fallback=False))
    return muts


@unittest.skipIf(jsonschema is None, "jsonschema is not importable")
class CrossCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = presets.load_schema()
        cls.cat = presets.catalog()

    def test_reference_validator_accepts_the_schema(self):
        jsonschema.Draft202012Validator.check_schema(self.schema)

    def test_all_eight_presets_valid_in_both(self):
        self.assertEqual(sorted(self.cat), sorted(presets.ALL_PRESET_IDS))  # the eight plus the two turbo presets
        for pid, p in self.cat.items():
            self.assertEqual(ref_set(p, self.schema), set(), pid)
            self.assertEqual(lite_set(p, self.schema), set(), pid)

    def test_mutation_corpus_same_verdict_and_same_error_paths(self):
        muts = build_mutations(self.cat["reference"])
        self.assertGreaterEqual(len(muts), 40)
        for i, q in enumerate(muts):
            ref, lite = ref_set(q, self.schema), lite_set(q, self.schema)
            self.assertTrue(ref, "mutation %d accepted by jsonschema" % i)
            self.assertEqual(lite, ref, "mutation %d: %s" % (i, json.dumps(q)[:200]))

    def test_mutations_apply_to_every_preset_alike(self):
        # every preset as the mutation base, not only reference: the verdict stays identical
        for pid, p in self.cat.items():
            for i, q in enumerate(build_mutations(p)):
                if pid == "reference" or ref_set(q, self.schema):
                    self.assertEqual(lite_set(q, self.schema), ref_set(q, self.schema), (pid, i))

    def test_schema_uses_only_the_supported_subset(self):
        # the Validator raises on any keyword outside the subset, so constructing it is the proof
        schema_lite.Validator(self.schema)
        raw = open(os.path.join(presets.preset_dir(), "schema.json"), encoding="utf-8").read()
        self.assertNotIn('"oneOf"', raw)


if __name__ == "__main__":
    unittest.main()
