"""MP-8 integration (design 6.3, CPU, no weights): the node against the native ComfyUI code.

* Layout string == the output of the native ``MOSSImageLayoutPrompt`` (same prompt, same boxes), also for text with
  leading/trailing whitespace (A24, m8).
* Conditioning goes through the real native path: a real ``comfy.sd.CLIP`` with the real ``MOSSImageTokenizer``
  (bundled tokenizer files, no weights) and a tiny deterministic stand-in for the Qwen3.5-9B module (the 9B text
  encoder weights are not used here). The node's CONDITIONING is compared with the stock ``CLIPTextEncode``.
"""
import copy
import json
import unittest

from . import _comfy_env


def _tiny_clip():
    import torch
    import comfy.model_management
    import comfy.model_patcher
    import comfy.sd
    import comfy.text_encoders.moss_image as moss

    class TinyTE(torch.nn.Module):
        """Deterministic stand-in: embeds the token ids; implements the three methods CLIP calls."""

        def __init__(self):
            super().__init__()
            gen = torch.Generator().manual_seed(8)
            self.emb = torch.nn.Embedding(300000, 8)
            with torch.no_grad():
                self.emb.weight.copy_(torch.randn(300000, 8, generator=gen))
            self.dtypes = {torch.float32}

        def reset_clip_options(self):
            pass

        def set_clip_options(self, options):
            pass

        def encode_token_weights(self, token_weight_pairs):
            (pairs,) = token_weight_pairs["qwen35_9b"]
            ids = torch.tensor([[tok for tok, _ in pairs]])
            return self.emb(ids).detach(), None

    clip = comfy.sd.CLIP(no_init=True)
    clip.cond_stage_model = TinyTE()
    clip.tokenizer = moss.MOSSImageTokenizer()
    cpu = torch.device("cpu")
    clip.patcher = comfy.model_patcher.ModelPatcher(clip.cond_stage_model, load_device=cpu, offload_device=cpu)
    clip.patcher.is_clip = True
    clip.layer_idx = None
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None
    clip.tokenizer_options = {}
    return clip


def _same_cond(a, b):
    import torch
    if len(a) != len(b):
        return False
    for (ta, da), (tb, db) in zip(a, b):
        if not torch.equal(ta, tb) or da.keys() != db.keys():
            return False
    return True


class NativeLayoutGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _comfy_env.setup()

    ELEMENTS = [
        {"type": "object", "desc": "a woman", "bbox": [120, 80, 960, 420], "color_palette": ["#aa0000"]},
        {"type": "text", "desc": "a sign", "text": "OPEN ü", "bbox": [10, 500, 90, 940]},
        {"type": "object", "desc": "a lamp", "bbox": [333, 333, 667, 667]},
    ]

    def native(self, prompt, width=1024, height=768, characters=""):
        from comfy_extras.nodes_moss_image import MOSSImageLayoutPrompt
        return MOSSImageLayoutPrompt.execute(prompt, copy.deepcopy(self.ELEMENTS), width, height, characters).args[0]

    def ours_from_native_document(self, native_text, compiled_text):
        from moss_photon.core import constraints as c
        return c.parse_layout(native_text, compiled_text=compiled_text)

    def test_our_layout_text_equals_native_output(self):
        for prompt in ("A woman at a door", "  A woman at a door \n", "\tженщина\r\n"):
            with self.subTest(prompt=prompt):
                native_text = self.native(prompt, characters="1")
                self.assertTrue(native_text.startswith('{"summary"'))  # control: native produced a layout
                res = self.ours_from_native_document(native_text, prompt)
                self.assertEqual(res.text, native_text)
                self.assertEqual(res.layout["items"], 3)

    def test_native_output_with_character_labels_validates(self):
        native_text = self.native("A", characters="1, 3")
        labels = [i["label"] for i in json.loads(native_text)["items"]]
        self.assertEqual(labels, ["character", "text", "character"])
        self.ours_from_native_document(native_text, "A")

    def test_node_sends_the_native_layout_text_to_the_encoder(self):
        from moss_photon.nodes.prompt import MossPhotonPromptLayout
        native_text = self.native("  A woman at a door \n")
        sent = []

        class Spy:
            def tokenize(self, text):
                sent.append(text)
                return text

            def encode_from_tokens_scheduled(self, tokens):
                return [[tokens, {}]]

        MossPhotonPromptLayout().encode(Spy(), positive="  A woman at a door \n", negative="", layout_json=native_text)
        self.assertEqual(sent[0], native_text)


class NativeEncodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _comfy_env.setup()
        cls.clip = _tiny_clip()

    def stock(self, text):
        import nodes
        return nodes.CLIPTextEncode().encode(self.clip, text)[0]

    def node(self, **kw):
        from moss_photon.nodes.prompt import MossPhotonPromptLayout
        args = {"positive": "A woman in a kitchen, morning light.", "negative": ""}
        args.update(kw)
        return MossPhotonPromptLayout().encode(self.clip, **args)["result"]

    def test_control_the_stand_in_reacts_to_the_text(self):
        self.assertFalse(_same_cond(self.stock("a"), self.stock("b")))

    def test_without_blocks_and_layout_conditioning_is_bit_equal_to_the_stock_node(self):
        for pos, neg in (("A woman in a kitchen, morning light.", ""), ("  spaced ж \n", "blurry, low quality")):
            with self.subTest(pos=pos):
                p, n, constraints = self.node(positive=pos, negative=neg)
                self.assertTrue(_same_cond(p, self.stock(pos)))
                self.assertTrue(_same_cond(n, self.stock(neg)))
                self.assertEqual(constraints["compiled_positive"], pos)

    def test_with_blocks_conditioning_equals_stock_of_the_compiled_text(self):
        p, n, constraints = self.node(identity="Anna", wardrobe="red wool sweater", forbidden_changes="no hats",
                                      negative="blurry")
        text = "A woman in a kitchen, morning light.\n\nSubject: Anna\nClothing: red wool sweater"
        self.assertEqual(constraints["compiled_positive"], text)
        self.assertTrue(_same_cond(p, self.stock(text)))
        self.assertTrue(_same_cond(n, self.stock("blurry")))
        self.assertFalse(_same_cond(p, self.stock("A woman in a kitchen, morning light.")))

    def test_layout_conditioning_equals_stock_of_the_layout_text(self):
        layout = json.dumps({"items": [{"label": "object", "caption": "a lamp", "bbox": [-10, -10, 10, 10]}]})
        p, _, constraints = self.node(layout_json=layout)
        self.assertTrue(_same_cond(p, self.stock(constraints["compiled_positive"])))
        self.assertTrue(constraints["compiled_positive"].startswith('{"summary"'))

    def test_inputs_conditioning_independent_of_forbidden_changes(self):
        a = self.node(forbidden_changes="no hats", negative="x")
        b = self.node(forbidden_changes="", negative="x")
        self.assertTrue(_same_cond(a[0], b[0]))
        self.assertTrue(_same_cond(a[1], b[1]))


if __name__ == "__main__":
    unittest.main()
