"""PHOTON_REPORT builder, validator, recipe hash, redaction (design 2.3, 6.1)."""
import copy
import json
import os
import unittest

from jsonschema import Draft202012Validator

from moss_photon.core import consts, errors, report

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTRACTS = os.path.join(TESTS_DIR, "fixtures", "contracts")


def load(*parts):
    with open(os.path.join(CONTRACTS, *parts), "r", encoding="utf-8") as f:
        return json.load(f)


def fixture():
    return load("report.json")


SCHEMA_VALIDATOR = Draft202012Validator(load("schemas", "report.schema.json"))


def schema_errors(rep):
    return sorted(e.message for e in SCHEMA_VALIDATOR.iter_errors(rep))


def get_path(obj, path):
    for part in path.replace("[*]", ".0").split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def set_path(obj, path, value):
    parts = path.replace("[*]", ".0").split(".")
    for part in parts[:-1]:
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    last = parts[-1]
    if isinstance(obj, list):
        obj[int(last)] = value
    else:
        obj[last] = value


def mutated(value):
    """A different JSON value of the same kind."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, list):
        return value + ["x"]
    if isinstance(value, dict):
        return dict(value, changed=True)
    return "now-set"  # None -> something


def producer_stack(**lora_fields):
    """``lora.stack`` exactly as the sampler emits it (``solver._lora_stack``) from the contract profile fixture.
    MP-30 M3: a hand-built item hid that the real producer never wrote ``size``."""
    from moss_photon.core import solver
    profile = load("identity_profile.json")
    profile["stack"][0]["lora"].update(lora_fields)
    return solver._lora_stack(profile)


def lora_item():
    return producer_stack(basename="a.safetensors", sha256="1" * 64)[0]


class ExampleTest(unittest.TestCase):
    def test_example_report_validates(self):
        self.assertEqual(report.validate_report(fixture()), [])
        self.assertEqual(schema_errors(fixture()), [])  # the strict schema agrees on the fixture

    def test_fixture_recipe_hash_is_not_trusted_but_computable(self):
        rep = fixture()
        self.assertRegex(report.recipe_hash(rep), "^[0-9a-f]{64}$")

    def test_summary_is_text_without_prompts(self):
        text = report.format_summary(fixture())
        self.assertIn("reference@1.0.0", text)
        self.assertIn("reference_exact", text)
        self.assertNotIn("\n\n", text)


class CrossCheckAgainstSchemaTest(unittest.TestCase):
    """validate_report is a hand transcription of the schema: it must reject what the schema rejects."""

    def walk_paths(self, obj, prefix=()):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield prefix + (k,)
                yield from self.walk_paths(v, prefix + (k,))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from self.walk_paths(v, prefix + (i,))

    def test_deleting_any_key_is_rejected_by_both(self):
        base = fixture()
        checked = 0
        for path in self.walk_paths(base):
            rep = copy.deepcopy(base)
            parent = rep
            for p in path[:-1]:
                parent = parent[p]
            if path[-1] in ("base", "effective"):
                continue
            del parent[path[-1]]
            with self.subTest(path=".".join(map(str, path))):
                self.assertTrue(schema_errors(rep), "schema accepted a report without it")
                self.assertTrue(report.validate_report(rep), "validate_report accepted a report without it")
            checked += 1
        self.assertGreater(checked, 100)  # control: the walk really visited the whole tree

    def test_wrong_types_are_rejected_by_both(self):
        for path, bad in (("inputs.seed", "42"), ("inputs.batch", True), ("inputs.width", 1000), ("sampling.cfg", "3"),
                          ("label.calibrated", 0), ("guards.guard_passed", True), ("privacy.prompts_logged", 1),
                          ("model.family", "moss"), ("preset.id", "nope"), ("recipe_hash", "abc"),
                          ("created_utc", "yesterday"), ("pack.version", "1.0"), ("inputs.latent_shape", [1, 2, 3]),
                          ("timing.estimate_s", 1.0), ("lora.foreign_patches.keys_sample", ["k"] * 21),
                          ("env.attention.global", "basic"), ("schema_version", 2), ("conditioning.positive.entries", 0)):
            rep = fixture()
            set_path(rep, path, bad)
            with self.subTest(path=path):
                self.assertTrue(schema_errors(rep))
                self.assertTrue(report.validate_report(rep))

    def test_unexpected_key_is_rejected_by_both(self):
        rep = fixture()
        rep["model"]["extra"] = 1
        self.assertTrue(schema_errors(rep))
        self.assertTrue(report.validate_report(rep))
        rep = fixture()
        rep["extra_top"] = 1
        self.assertTrue(report.validate_report(rep))

    def test_optional_blocks_are_accepted(self):
        rep = fixture()
        rep["diagnostics"] = {"anything": 1}
        rep["timing"]["step_s"] = [1.0, 2.0]
        self.assertEqual(report.validate_report(rep), [])
        self.assertEqual(schema_errors(rep), [])

    def test_problems_carry_json_paths(self):
        rep = fixture()
        del rep["sampling"]["slg"]["mode"]
        self.assertEqual(report.validate_report(rep), ["report.sampling.slg.mode: required key is missing"])

    def test_non_dict_report(self):
        self.assertEqual(len(report.validate_report([])), 1)

    def test_future_schema_version_is_a_problem(self):
        rep = fixture()
        rep["schema_version"] = consts.REPORT_SCHEMA + 1
        self.assertTrue(report.validate_report(rep))


DESIGNED_RECIPE_PATHS = (
    "pack.version", "env.comfyui_version", "env.torch", "env.cuda", "env.device_name", "env.attention",
    "env.inference_dtype", "env.weights_streaming", "env.deterministic_flag",
    "preset.id", "preset.version", "preset.sha256", "preset.manifest_sha256", "label.modifiers",
    "model.family", "model.identity.sha256", "model.identity.basename", "model.identity.size", "sampling",
    "inputs.seed", "inputs.batch", "inputs.width", "inputs.height", "inputs.noise_policy", "inputs.noise_sha256",
    "conditioning.positive.sha256", "conditioning.negative.sha256",
    "lora.stack[*].sha256", "lora.stack[*].basename", "lora.stack[*].size", "lora.stack[*].strength_model",
    "lora.stack[*].strength_clip", "lora.stack[*].applied_to",
    "lora.foreign_patches.entries_observed", "lora.foreign_patches.uuid_match",
)


class RecipeHashTest(unittest.TestCase):
    def test_recipe_paths_are_exactly_the_designed_ones(self):
        """Design 2.3 literally: the sensitivity test below iterates the module's list, so the list itself is pinned."""
        self.assertEqual(report.RECIPE_HASH_PATHS, DESIGNED_RECIPE_PATHS)

    def base(self):
        rep = fixture()
        rep["lora"]["stack"] = [lora_item()]
        return rep

    def test_recipe_hash_sensitivity(self):
        base = self.base()
        h0 = report.recipe_hash(base)
        self.assertEqual(h0, report.recipe_hash(copy.deepcopy(base)))
        for path in report.RECIPE_HASH_PATHS:
            rep = copy.deepcopy(base)
            if path in ("model.identity.basename", "model.identity.size"):
                rep["model"]["identity"]["sha256"] = None  # counted only for an unhashed checkpoint (m3)
                h_ref = report.recipe_hash(rep)
            elif path in ("lora.stack[*].basename", "lora.stack[*].size"):
                rep["lora"]["stack"][0]["sha256"] = None  # the size field is the one the producer emitted
                h_ref = report.recipe_hash(rep)
            else:
                h_ref = h0
            current = get_path(rep, path)
            set_path(rep, path, mutated(current))
            with self.subTest(path=path):
                self.assertNotEqual(report.recipe_hash(rep), h_ref)

    def test_the_real_producer_stack_is_a_valid_report_and_carries_the_size(self):
        base = self.base()
        item = base["lora"]["stack"][0]
        self.assertIsInstance(item["size"], int)
        self.assertEqual(item["size"], load("identity_profile.json")["stack"][0]["lora"]["size"])
        self.assertEqual(report.validate_report(base), [])
        self.assertEqual(schema_errors(base), [])

    def test_two_unhashed_files_with_one_basename_get_different_recipe_hashes(self):
        """MP-30 M3 (design 2.3): a LoRA without sha256 enters the recipe by basename AND size."""
        def hashed(size):
            rep = fixture()
            rep["lora"]["stack"] = producer_stack(basename="same.safetensors", sha256=None, size=size)
            self.assertEqual(report.validate_report(rep), [])
            return report.recipe_hash(rep)
        self.assertNotEqual(hashed(1000), hashed(2000))
        self.assertEqual(hashed(1000), hashed(1000))

    def test_the_size_does_not_count_when_the_lora_hash_is_known(self):
        a, b = copy.deepcopy(self.base()), copy.deepcopy(self.base())
        b["lora"]["stack"][0]["size"] += 1
        self.assertEqual(report.recipe_hash(a), report.recipe_hash(b))

    def test_a_stack_item_without_size_is_invalid_in_both_validators(self):
        rep = self.base()
        del rep["lora"]["stack"][0]["size"]
        self.assertTrue(report.validate_report(rep))
        self.assertTrue(schema_errors(rep))

    def test_identity_basename_does_not_count_when_the_hash_is_known(self):
        base = self.base()
        rep = copy.deepcopy(base)
        rep["model"]["identity"]["basename"] = "renamed.safetensors"
        rep["lora"]["stack"][0]["basename"] = "renamed.safetensors"
        self.assertEqual(report.recipe_hash(rep), report.recipe_hash(base))

    def test_excluded_blocks_do_not_change_the_hash(self):
        base = self.base()
        h0 = report.recipe_hash(base)
        for path in ("timing.total_s", "timing.preflight_s", "memory.peak_vram_bytes", "memory.measured",
                     "preview.enabled", "created_utc", "warnings", "fallbacks", "output.latent_sha256",
                     "forwards.model_calls", "guards.finite_check", "node", "label.ui_label", "recipe_hash",
                     "lora.stack[*].coverage_ratio", "lora.stack[*].trigger_present", "inputs.subseeds"):
            rep = copy.deepcopy(base)
            set_path(rep, path, mutated(get_path(rep, path)))
            with self.subTest(path=path):
                self.assertEqual(report.recipe_hash(rep), h0)

    def test_stack_length_and_order_matter(self):
        base = self.base()
        two = copy.deepcopy(base)
        other = lora_item()
        other["sha256"] = "2" * 64
        two["lora"]["stack"].append(other)
        self.assertNotEqual(report.recipe_hash(two), report.recipe_hash(base))
        swapped = copy.deepcopy(two)
        swapped["lora"]["stack"].reverse()
        self.assertNotEqual(report.recipe_hash(swapped), report.recipe_hash(two))


class BuilderTest(unittest.TestCase):
    def filled(self, builder=None):
        builder = builder or report.ReportBuilder("MossPhotonSampler", clock=lambda: 1761393600.0, perf=iter(
            [10.0, 11.0, 15.0, 16.0, 17.0]).__next__)
        rep = fixture()
        for name in ("env", "label", "preset", "model", "conditioning", "inputs", "sampling", "lora", "guards",
                     "forwards", "memory", "preview", "output"):
            builder.section(name, **rep[name])
        builder.section("text_encoder", **{"basename": "te.safetensors"})
        return builder

    def test_finalize_produces_a_valid_report(self):
        builder = self.filled()
        builder.mark_phase("preflight_done")   # perf 11.0
        builder.mark_phase("sampling_done")    # perf 15.0
        out = builder.finalize()               # perf 16.0
        self.assertEqual(report.validate_report(out), [])
        self.assertEqual(schema_errors(out), [])
        self.assertEqual(out["created_utc"], "2025-10-25T12:00:00Z")
        self.assertEqual(out["timing"], {"total_s": 6.0, "preflight_s": 1.0, "sampling_s": 4.0, "step_s_mean": None,
                                         "step_s_max": None, "estimate_s": None})
        self.assertEqual(out["recipe_hash"], report.recipe_hash(out))
        self.assertEqual(out["pack"], {"name": consts.PACK_NAME, "version": consts.PACK_VERSION})
        self.assertEqual(out["schema_version"], consts.REPORT_SCHEMA)
        self.assertEqual(out["text_encoder"], {"basename": "te.safetensors"})

    def test_finalized_turbo_report_validates_against_schema(self):
        for pid in consts.TURBO_PRESET_IDS:
            with self.subTest(preset=pid):
                builder = self.filled()
                preset = dict(fixture()["preset"], id=pid)
                label = dict(fixture()["label"], parity_class="native_turbo")
                builder.section("preset", **preset)
                builder.section("label", **label)
                out = builder.finalize()
                self.assertEqual(report.validate_report(out), [])
                self.assertEqual(schema_errors(out), [])

    def test_finalized_report_with_digit_pin_and_distill_stack_validates(self):
        builder = self.filled()
        builder.section("preset", **dict(fixture()["preset"], id="turbo_8step", pin="turbo_8step@1.0.0#0a1b2c3d"))
        stack = [dict(lora_item(), role=role, sha256=None) for role in ("distill", "character")]
        builder.section("lora", **dict(fixture()["lora"], stack=stack))
        out = builder.finalize()
        self.assertEqual(report.validate_report(out), [])
        self.assertEqual(schema_errors(out), [])

    def test_pin_id_class_agrees_with_parse_pin(self):
        from moss_photon.core import presets
        ids = {"reference", "photo_safe", "identity_safe", "wardrobe_safe", "scene_safe", "night_photo",
               "draft_fast", "advanced", "turbo_8step", "turbo_identity"}
        self.assertEqual(ids, set(presets.ALL_PRESET_IDS))
        for pid in sorted(ids):
            with self.subTest(preset=pid):
                pin = "%s@1.0.0#0a1b2c3d" % pid
                self.assertEqual(presets.parse_pin(pin)["id"], pid)
                rep = fixture()
                rep["preset"]["pin"] = pin
                self.assertEqual(schema_errors(rep), [])
                self.assertEqual(report.validate_report(rep), [])

    def test_finalize_without_marks_has_null_phase_times(self):
        out = self.filled().finalize()
        self.assertIsNone(out["timing"]["preflight_s"])
        self.assertIsNone(out["timing"]["sampling_s"])

    def test_incomplete_report_does_not_finalize(self):
        builder = report.ReportBuilder("MossPhotonSampler")
        with self.assertRaises(ValueError) as cm:
            builder.finalize()
        self.assertIn("env", str(cm.exception))

    def test_section_merges_shallowly_and_copies(self):
        builder = report.ReportBuilder("n")
        payload = {"a": [1]}
        builder.section("x", **payload)
        payload["a"].append(2)
        builder.section("x", b=2)
        self.assertEqual(builder._report["x"], {"a": [1], "b": 2})

    def test_builder_owned_names_are_refused(self):
        builder = report.ReportBuilder("n")
        for name in ("recipe_hash", "created_utc", "warnings", "fallbacks", "kind"):
            with self.assertRaises(ValueError):
                builder.section(name, x=1)

    def test_warn_dedupes_by_code_and_param_and_refuses_errors(self):
        builder = self.filled()
        w = errors.make_issue("foreign_modifiers", "MossPhotonSampler", "model", "m", "f")
        builder.warn(w)
        builder.warn(w)
        builder.warn(errors.make_issue("foreign_modifiers", "MossPhotonSampler", "model", "other text", "f"))
        builder.warn(errors.make_issue("foreign_modifiers", "MossPhotonSampler", "other", "m", "f"))
        builder.warn(w.to_dict())
        out = builder.finalize()
        self.assertEqual([(x["code"], x["param"]) for x in out["warnings"]],
                         [("foreign_modifiers", "model"), ("foreign_modifiers", "other")])
        with self.assertRaises(ValueError):
            builder.warn(errors.make_issue("invalid_schedule", "n", "p", "m", "f"))

    def test_warn_keeps_the_same_warning_of_two_different_scopes(self):
        """MP-30 m8: two LoRAs that both skip a layer share code and param; the second one must not be dropped."""
        builder = self.filled()
        w = errors.make_issue("lora_layers_missing", "MossPhotonLoRAProfile", "lora_name", "layer 22 skipped", "none")
        builder.warn(w, scope="lora[0]")
        builder.warn(w, scope="lora[0]")  # still deduped inside one scope
        builder.warn(w, scope="lora[1]")
        builder.warn(w)                   # the unscoped one is its own key
        out = builder.finalize()
        self.assertEqual([x["code"] for x in out["warnings"]], ["lora_layers_missing"] * 3)

    def test_the_unknown_attention_backend_is_valid_and_a_made_up_name_is_not(self):
        """MP-30 m15: a backend the table does not know is reported as ``unknown:<name>``."""
        rep = fixture()
        rep["env"]["attention"]["global"] = "unknown:attention_new_backend"
        self.assertEqual(report.validate_report(rep), [])
        self.assertEqual(schema_errors(rep), [])
        for bad in ("unknown:", "unknown", "basic", "xunknown:y"):
            rep["env"]["attention"]["global"] = bad
            with self.subTest(value=bad):
                self.assertTrue(report.validate_report(rep))
                self.assertTrue(schema_errors(rep))

    def test_the_fixture_warnings_use_registered_codes(self):
        for w in fixture()["warnings"]:
            self.assertIn(w["code"], errors.REASONS)

    def test_fallback_and_phase_names(self):
        builder = self.filled()
        builder.fallback("manifest_missing", "reference configuration used")
        self.assertEqual(builder.finalize()["fallbacks"], [{"reason": "manifest_missing",
                                                           "action": "reference configuration used"}])
        with self.assertRaises(ValueError):
            builder.mark_phase("later")

    def test_step_times_given_by_the_producer_are_kept(self):
        builder = self.filled()
        builder.section("timing", step_s_mean=1.5, step_s_max=2.0, estimate_s=99)
        timing = builder.finalize()["timing"]
        self.assertEqual((timing["step_s_mean"], timing["step_s_max"], timing["estimate_s"]), (1.5, 2.0, None))


class PrivacyTest(unittest.TestCase):
    SENTINEL = "PRIVATE-SENTINEL-d41d8cd9"

    def constraints(self):
        c = load("constraints.json")
        c["user_text"] = self.SENTINEL
        c["compiled_positive"] = self.SENTINEL + " more"
        c["fields"]["identity"] = self.SENTINEL
        return c

    def test_report_never_contains_private_text(self):
        c = self.constraints()
        builder = BuilderTest().filled()
        te = c["text_encoder"]
        builder.section("text_encoder", **te)  # the one block copied from a constraints object: no prompt in it
        out = builder.finalize()
        self.assertNotIn(self.SENTINEL, json.dumps(out))
        self.assertNotIn(self.SENTINEL, report.format_summary(out))

    def test_a_whole_private_object_is_refused(self):
        builder = BuilderTest().filled()
        with self.assertRaises(ValueError):
            builder.section("text_encoder", wrapped=self.constraints())
        with self.assertRaises(ValueError):
            builder.section("lora", stack=[{"private_paths": ["x"]}])

    def test_redact_removes_private_paths(self):
        rep = fixture()
        profile = load("identity_profile.json")
        profile["stack"][0]["trigger_hints"] = ["zq-trigger-word-7f3"]
        rep["diagnostics"] = {"constraints": self.constraints(), "profile": profile}
        out = report.redact(rep)
        text = json.dumps(out)
        self.assertNotIn(self.SENTINEL, text)
        self.assertNotIn("private_paths", text)
        # the identity profile's trigger hints are dropped through the [*] path
        for item in out["diagnostics"]["profile"]["stack"]:
            self.assertNotIn("trigger_hints", item)
        # control: the same fixtures do contain the sentinel / trigger words before redaction
        self.assertIn(self.SENTINEL, json.dumps(rep))
        hints = [h for item in rep["diagnostics"]["profile"]["stack"] for h in item.get("trigger_hints", [])]
        self.assertTrue(hints)  # control: the planted trigger words exist before redaction
        for h in hints:
            self.assertNotIn(h, text)

    def test_redact_basenames(self):
        rep = fixture()
        rep["lora"]["stack"] = [lora_item()]
        kept = report.redact(rep)
        self.assertEqual(kept["model"]["identity"]["basename"], rep["model"]["identity"]["basename"])
        out = report.redact(rep, keep_basenames=False)
        self.assertEqual(out["model"]["identity"]["basename"], "redacted")
        self.assertEqual(out["lora"]["stack"][0]["basename"], "redacted")
        self.assertEqual(rep["lora"]["stack"][0]["basename"], "a.safetensors")  # original untouched

    def test_finalize_catches_private_object_smuggled_past_section(self):
        builder = BuilderTest().filled()
        builder._report["text_encoder"] = {"private_paths": ["x"]}  # bypass section() on purpose
        with self.assertRaises(ValueError) as cm:
            builder.finalize()
        self.assertIn("private_paths", str(cm.exception))


class DiagnosticsTest(unittest.TestCase):
    def test_with_diagnostics_does_not_mutate(self):
        rep = fixture()
        snapshot = copy.deepcopy(rep)
        diag = {"image_stats": {"mean": 0.5}}
        diag_snapshot = copy.deepcopy(diag)
        out = report.with_diagnostics(rep, diag)
        self.assertEqual(rep, snapshot)
        self.assertEqual(diag, diag_snapshot)
        self.assertEqual(out["diagnostics"]["image_stats"], {"mean": 0.5})
        self.assertEqual(out["diagnostics"]["parent_recipe_hash"], rep["recipe_hash"])
        self.assertEqual(out["recipe_hash"], rep["recipe_hash"])
        self.assertEqual(report.validate_report(out), [])
        out["diagnostics"]["image_stats"]["mean"] = 9
        self.assertEqual(diag, diag_snapshot)

    def test_diagnostics_may_set_its_own_parent(self):
        out = report.with_diagnostics(fixture(), {"parent_recipe_hash": "x"})
        self.assertEqual(out["diagnostics"]["parent_recipe_hash"], "x")

    def test_future_schema_version_rejected(self):
        rep = fixture()
        rep["schema_version"] = consts.REPORT_SCHEMA + 1
        for fn in (lambda: report.with_diagnostics(rep, {}), lambda: report.format_summary(rep),
                   lambda: report.redact(rep)):
            with self.assertRaises(errors.UnsupportedFeature) as cm:
                fn()
            self.assertEqual(cm.exception.code, "unsupported_feature")

    def test_summary_lists_modifiers_and_warnings(self):
        rep = fixture()
        rep["label"]["modifiers"] = ["cfg_function"]
        text = report.format_summary(rep)
        self.assertIn("modifiers: cfg_function", text)
        self.assertIn("warnings: family_not_verified", text)


class PurityTest(unittest.TestCase):
    def test_report_module_is_pure(self):
        import subprocess
        import sys
        code = ("import sys\nsys.path.insert(0, sys.argv[1])\nfrom moss_photon.core import report\n"
                "assert not [m for m in sys.modules if m.split('.')[0] in ('torch', 'comfy', 'numpy')]\nprint('OK')\n")
        proc = subprocess.run([sys.executable, "-I", "-c", code, os.path.dirname(TESTS_DIR)],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])


if __name__ == "__main__":
    unittest.main()
