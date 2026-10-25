"""MP-8 node (design 4.3, 6.3): fake CLIP records the native call sequence; INPUT_TYPES; no prompt text in the UI."""
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from jsonschema import Draft202012Validator

from moss_photon.core import fingerprint
from moss_photon.core.errors import InvalidLayout, MissingRequiredInput
from moss_photon.nodes import prompt as P

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "fixtures", "contracts", "schemas", "constraints.schema.json")
with open(SCHEMA_PATH, "rb") as _f:
    VALIDATOR = Draft202012Validator(json.loads(_f.read().decode("utf-8")))


def load_clip_model_patcher(paths, *a):  # stand-ins named like the ComfyUI CLIP loaders (the guard checks names)
    raise AssertionError("must never be called")


def load_checkpoint_clip_patcher(path, *a):
    raise AssertionError("must never be called")


def load_diffusion_model(path, *a):
    raise AssertionError("must never be called")


class FakeClip:
    """Records calls like the native CLIP; mirrors CLIPTextEncode.encode (CUI/nodes.py:75-79)."""

    def __init__(self, init=None):
        self.calls = []
        if init is not None:
            self.patcher = type("Patcher", (), {"cached_patcher_init": init})()

    def tokenize(self, text):
        self.calls.append(("tokenize", text))
        return {"tokens-of": text}

    def encode_from_tokens_scheduled(self, tokens):
        self.calls.append(("encode", tokens["tokens-of"]))
        return [["cond-of:" + tokens["tokens-of"], {}]]


def stock_encode(clip, text):
    """The stock CLIPTextEncode.encode body."""
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


def run(clip=None, **kw):
    clip = clip if clip is not None else FakeClip()
    args = {"positive": "A woman in a kitchen.", "negative": ""}
    args.update(kw)
    return P.MossPhotonPromptLayout().encode(clip, **args), clip


class InterfaceTest(unittest.TestCase):
    def test_input_types_equal_the_design(self):
        it = P.MossPhotonPromptLayout.INPUT_TYPES()
        self.assertEqual(list(it["required"]), ["clip", "positive", "negative"])
        self.assertEqual(it["required"]["clip"][0], "CLIP")
        for name in ("positive", "negative"):
            self.assertEqual(it["required"][name][0], "STRING")
            self.assertEqual({k: v for k, v in it["required"][name][1].items() if k != "tooltip"},
                             {"multiline": True, "default": ""})
        self.assertEqual(list(it["optional"]), ["identity", "wardrobe", "scene", "pose", "camera", "lighting",
                                                "forbidden_changes", "layout_json", "fingerprint_text_encoder"])
        for name in list(it["optional"])[:-1]:
            self.assertEqual(it["optional"][name][0], "STRING")
            self.assertEqual({k: v for k, v in it["optional"][name][1].items() if k != "tooltip"},
                             {"multiline": True, "default": ""})
        te = it["optional"]["fingerprint_text_encoder"]
        self.assertEqual((te[0], te[1]["default"]), ("BOOLEAN", False))

    def test_every_input_and_output_has_a_tooltip(self):
        """m3 (design 4: every tooltip says what the widget changes), like the other seven nodes."""
        it = P.MossPhotonPromptLayout.INPUT_TYPES()
        for section in ("required", "optional"):
            for name, spec in it[section].items():
                with self.subTest(input=name):
                    self.assertTrue(len(spec) == 2 and str(spec[1].get("tooltip", "")).strip(), "no tooltip")
        n = P.MossPhotonPromptLayout
        self.assertEqual(len(n.OUTPUT_TOOLTIPS), len(n.RETURN_TYPES))
        self.assertTrue(all(t.strip() for t in n.OUTPUT_TOOLTIPS))

    def test_display_name_follows_the_other_nodes(self):
        self.assertEqual(P.NODE_DISPLAY_NAME_MAPPINGS, {"MossPhotonPromptLayout": "Moss Photon Prompt & Layout"})

    def test_outputs_function_category_and_mapping(self):
        n = P.MossPhotonPromptLayout
        self.assertEqual(n.RETURN_TYPES, ("CONDITIONING", "CONDITIONING", "PHOTON_CONSTRAINTS"))
        self.assertEqual(n.RETURN_NAMES, ("positive", "negative", "constraints"))
        self.assertEqual(n.FUNCTION, "encode")
        self.assertEqual(n.CATEGORY, "MossPhoton/1 Setup")
        self.assertTrue(callable(getattr(n, n.FUNCTION)))
        self.assertEqual(P.NODE_CLASS_MAPPINGS, {"MossPhotonPromptLayout": n})
        self.assertIn("MossPhotonPromptLayout", P.NODE_DISPLAY_NAME_MAPPINGS)

    def test_node_id_matches_consts(self):
        from moss_photon.core import consts
        self.assertIn(P._NODE, consts.NODE_IDS)

    def test_no_comfy_import(self):
        self.assertNotIn("comfy", sys.modules)


class EncodeTest(unittest.TestCase):
    def test_per_text_exactly_tokenize_then_encode_like_the_stock_node(self):
        out, clip = run(negative="blurry")
        stock = FakeClip()
        stock_encode(stock, "A woman in a kitchen.")
        stock_encode(stock, "blurry")
        self.assertEqual(clip.calls, stock.calls)
        self.assertEqual([k for k, _ in clip.calls], ["tokenize", "encode", "tokenize", "encode"])

    def test_without_blocks_conditioning_equals_stock(self):
        out, _ = run(positive="  odd \r\n", negative=" neg ")
        stock = FakeClip()
        self.assertEqual(out["result"][0], stock_encode(stock, "  odd \r\n"))
        self.assertEqual(out["result"][1], stock_encode(stock, " neg "))

    def test_negative_passed_verbatim_and_blocks_never_reach_it(self):
        _, clip = run(negative="  blurry \n", wardrobe="red sweater", forbidden_changes="NO-HATS")
        self.assertEqual(clip.calls[2], ("tokenize", "  blurry \n"))

    def test_blocks_compiled_into_the_positive_text_only(self):
        out, clip = run(wardrobe="red sweater", scene="kitchen", forbidden_changes="NO-HATS")
        sent = clip.calls[0][1]
        self.assertEqual(sent, "A woman in a kitchen.\n\nClothing: red sweater\nScene: kitchen")
        self.assertNotIn("NO-HATS", sent)
        self.assertEqual(out["result"][2]["compiled_positive"], sent)
        self.assertEqual(out["result"][2]["fields"]["forbidden_changes"], "NO-HATS")

    def test_layout_text_goes_to_the_encoder(self):
        layout = json.dumps({"items": [{"label": "object", "caption": "a lamp", "bbox": [-10, -10, 10, 10]}]})
        out, clip = run(positive="  A room \n", layout_json=layout)
        sent = clip.calls[0][1]
        self.assertEqual(json.loads(sent)["summary"]["caption"], "A room")
        self.assertEqual(sent, out["result"][2]["compiled_positive"])
        self.assertEqual(out["result"][2]["layout"]["extent"], [-10, -10, 10, 10])
        self.assertEqual([w["code"] for w in out["result"][2]["warnings"]], ["layout_experimental"])

    def test_blank_layout_json_is_no_layout(self):
        out, _ = run(layout_json="  \n ")
        self.assertIsNone(out["result"][2]["layout"])

    def test_bad_layout_raises_with_path_before_any_encode(self):
        clip = FakeClip()
        bad = json.dumps({"items": [{"label": "object", "caption": "x", "bbox": [0, 0, 1.5, 2]}]})
        with self.assertRaises(InvalidLayout) as ctx:
            run(clip, layout_json=bad)
        self.assertIn("items[0].bbox[2]", str(ctx.exception))
        self.assertEqual(clip.calls, [])

    def test_missing_clip(self):
        with self.assertRaises(MissingRequiredInput):
            P.MossPhotonPromptLayout().encode(None, positive="a", negative="")

    def test_none_optionals_are_empty(self):
        out, _ = run(identity=None, wardrobe=None, layout_json=None)
        self.assertEqual(out["result"][2]["compiled_positive"], "A woman in a kitchen.")

    def test_conflict_gives_a_warning_listing_it(self):
        out, _ = run(positive="a woman in a red dress", wardrobe="blue dress")
        constraints = out["result"][2]
        self.assertEqual(len(constraints["conflicts"]), 1)
        self.assertEqual([w["code"] for w in constraints["warnings"]], ["attribute_conflict"])
        self.assertIn("dress", constraints["warnings"][0]["message"])
        self.assertIn("dress", out["ui"]["text"][0])

    def test_forbidden_changes_is_not_a_conflict_source(self):
        out, _ = run(positive="a woman in a red dress", forbidden_changes="blue dress")
        self.assertEqual(out["result"][2]["conflicts"], [])

    def test_constraints_validate_against_the_contract_schema(self):
        out, _ = run(identity="Anna", wardrobe="red wool sweater", scene="kitchen", forbidden_changes="no hats")
        self.assertEqual([e.message for e in VALIDATOR.iter_errors(out["result"][2])], [])


class UiTest(unittest.TestCase):
    def test_ui_text_contains_no_prompt_text(self):
        secret = "SECRET-PROSE-123"
        out, _ = run(positive="woman " + secret, negative="NEG-" + secret, identity="ID-" + secret,
                     forbidden_changes="FORB-" + secret)
        text = json.dumps(out["ui"], ensure_ascii=False)
        self.assertNotIn(secret, text)
        self.assertEqual(len(out["ui"]["text"]), 1)
        self.assertIn("identity", out["ui"]["text"][0])
        self.assertIn("forbidden_changes", out["ui"]["text"][0])


class TextEncoderInfoTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "te.safetensors")
        with open(self.path, "wb") as f:
            f.write(b"abc")
        # folder_paths faked as unavailable: the hash goes to the in-memory cache, never to comfy.
        patcher = mock.patch.dict(sys.modules, {"folder_paths": None})
        patcher.start()
        self.addCleanup(patcher.stop)
        saved = fingerprint._default_cache
        fingerprint._default_cache = None
        self.addCleanup(setattr, fingerprint, "_default_cache", saved)

    def info(self, init, flag=False):
        out, _ = run(FakeClip(init), fingerprint_text_encoder=flag)
        return out["result"][2]["text_encoder"]

    def test_info_without_hash_when_flag_off(self):
        info = self.info((load_clip_model_patcher, ([self.path],)))
        self.assertEqual(info["basename"], "te.safetensors")
        self.assertEqual(info["size"], 3)
        self.assertIsNone(info["sha256"])

    def test_plain_str_path_of_a_bundled_clip(self):
        self.assertEqual(self.info((load_checkpoint_clip_patcher, (self.path,)))["basename"], "te.safetensors")

    def test_hash_when_flag_on(self):
        import hashlib
        info = self.info((load_clip_model_patcher, ([self.path],)), flag=True)
        self.assertEqual(info["sha256"], hashlib.sha256(b"abc").hexdigest())

    def test_null_hash_reports_path_only_and_warns(self):
        with mock.patch.object(P, "sha256_file", return_value=None):
            out, _ = run(FakeClip((load_clip_model_patcher, ([self.path],))), fingerprint_text_encoder=True)
        info = out["result"][2]["text_encoder"]
        self.assertIsNone(info["sha256"])
        self.assertEqual(info["fingerprint_source"], P.SOURCE_PATH_ONLY)
        self.assertIn("fingerprint_unavailable", [w["code"] for w in out["result"][2]["warnings"]])
        self.assertEqual([e.message for e in VALIDATOR.iter_errors(out["result"][2])], [])

    def test_basename_only_never_the_directory(self):
        self.assertNotIn(self.dir.name, json.dumps(self.info((load_clip_model_patcher, ([self.path],)))))

    def test_guarded_when_unavailable(self):
        for init in (None, (load_clip_model_patcher,), (load_clip_model_patcher, ()), (load_clip_model_patcher, ([],)),
                     (load_clip_model_patcher, ([self.path, self.path],)), (load_clip_model_patcher, (["/nonexistent/x"],)),
                     (load_clip_model_patcher, (5,))):
            with self.subTest(init=init):
                self.assertIsNone(self.info(init))
        self.assertIsNone(run()[0]["result"][2]["text_encoder"])  # clip without a patcher attribute

    def test_path_comes_from_the_shared_fingerprint_helper_with_its_loader_guard(self):
        """m2: no local copy of the discovery; a MODEL loader's file is not a text encoder."""
        self.assertIsNone(self.info((load_diffusion_model, ([self.path],))))
        self.assertIsNone(self.info((lambda: None, ([self.path],))))  # unknown loader name
        self.assertFalse(hasattr(P, "_text_encoder_path"))
        with mock.patch.object(P, "text_encoder_path_of", return_value=self.path) as helper:
            self.assertEqual(self.info((load_diffusion_model, ([self.path],)))["basename"], "te.safetensors")
        helper.assert_called_once()

    def test_the_long_hash_is_cancellable(self):
        """m2: a 19 GB text encoder must be interruptible; the poll is ComfyUI's, found without importing comfy."""
        class Stop(Exception):
            pass

        def poll():
            raise Stop()
        mm = types.ModuleType("comfy.model_management")
        mm.throw_exception_if_processing_interrupted = poll
        init = (load_clip_model_patcher, ([self.path],))
        with mock.patch.dict(sys.modules, {"comfy.model_management": mm}):
            with self.assertRaises(Stop):
                run(FakeClip(init), fingerprint_text_encoder=True)
        self.assertIsNone(fingerprint.default_cache().get(fingerprint.cache_key(self.path)))  # nothing cached half-way
        self.assertEqual(run(FakeClip(init), fingerprint_text_encoder=True)[0]["result"][2]["text_encoder"]["sha256"],
                         __import__("hashlib").sha256(b"abc").hexdigest())  # without ComfyUI it hashes as before
        self.assertNotIn("comfy", [m for m in sys.modules if m == "comfy"])

    def test_text_encoder_info_validates(self):
        out, _ = run(FakeClip((load_clip_model_patcher, ([self.path],))))
        self.assertEqual([e.message for e in VALIDATOR.iter_errors(out["result"][2])], [])


if __name__ == "__main__":
    unittest.main()
