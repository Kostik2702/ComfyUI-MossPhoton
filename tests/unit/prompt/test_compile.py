"""MP-8 compile_prompt (design 4.3 step 1, 6.3): verbatim user text, fixed order, OD1, nothing added."""
import random
import unittest

from moss_photon.core import constraints as c

ALL = {"identity": "Anna, short dark hair", "wardrobe": "red wool sweater", "scene": "small kitchen",
       "pose": "standing", "camera": "35mm", "lighting": "window light", "forbidden_changes": "NO-TATTOOS-XYZ"}


def random_unicode(rng, n):
    out = []
    while len(out) < n:
        cp = rng.choice((rng.randrange(0x20, 0x7F), rng.randrange(0xA0, 0x2FF), rng.randrange(0x400, 0x4FF),
                         rng.randrange(0x4E00, 0x4F00), rng.randrange(0x1F300, 0x1F400), rng.randrange(0, 0x20),
                         0x2028, 0x00A0, 0x200B))
        out.append(chr(cp))
    return "".join(out)


class UserTextVerbatimTest(unittest.TestCase):
    CASES = ("A woman", "Привет, мир", "emoji \U0001F600\U0001F468‍\U0001F469 end", "trailing   ", "\ttabs\t",
             "crlf\r\nline\r\n", "  leading", "", "   ", "\n", "é combining", "\x00nul")

    def test_user_text_verbatim_prefix(self):
        for text in self.CASES:
            for blocks in ({}, ALL, {"scene": "kitchen"}):
                with self.subTest(text=text, blocks=sorted(blocks)):
                    out = c.compile_prompt(text, blocks)
                    self.assertTrue(out.startswith(text))

    def test_arbitrary_unicode_never_lost_or_reordered(self):
        rng = random.Random(8)
        for _ in range(300):
            text = random_unicode(rng, rng.randrange(0, 40))
            blocks = {n: random_unicode(rng, rng.randrange(0, 12)) for n in ALL}
            out = c.compile_prompt(text, blocks)
            self.assertTrue(out.startswith(text), repr((text, out)))
            self.assertEqual(len(out) >= len(text), True)

    def test_no_mutation_of_inputs(self):
        blocks = dict(ALL)
        before = dict(blocks)
        c.compile_prompt("x", blocks)
        self.assertEqual(blocks, before)


class StructureTest(unittest.TestCase):
    def test_no_blocks_returns_input(self):
        for text in ("a", " a ", "", "x\r\n"):
            self.assertEqual(c.compile_prompt(text, {}), text)
        self.assertEqual(c.compile_prompt("a", {n: "" for n in c.BLOCK_NAMES}), "a")

    def test_blank_user_text_has_no_leading_separator(self):
        self.assertEqual(c.compile_prompt("", {"scene": "kitchen"}), "Scene: kitchen")
        self.assertEqual(c.compile_prompt("  ", {"scene": "kitchen"}), "  Scene: kitchen")

    def test_whitespace_blocks_omitted(self):
        self.assertEqual(c.compile_prompt("t", {"wardrobe": "  \n\t ", "scene": ""}), "t")
        self.assertEqual(c.compile_prompt("t", {"wardrobe": "  ", "scene": "s"}), "t\n\nScene: s")

    def test_fixed_order_independent_of_dict_order(self):
        forward = dict(ALL)
        backward = dict(reversed(list(ALL.items())))
        self.assertEqual(c.compile_prompt("t", forward), c.compile_prompt("t", backward))
        lines = c.compile_prompt("t", forward).split("\n")
        self.assertEqual([l.split(":")[0] for l in lines[2:]],
                         ["Subject", "Clothing", "Scene", "Pose", "Camera", "Lighting"])

    def test_labels_exact(self):
        self.assertEqual(c.BLOCK_LABELS, {"identity": "Subject", "wardrobe": "Clothing", "scene": "Scene",
                                          "pose": "Pose", "camera": "Camera", "lighting": "Lighting"})
        self.assertEqual(c.COMPILED_BLOCKS, ("identity", "wardrobe", "scene", "pose", "camera", "lighting"))
        self.assertEqual(c.BLOCK_NAMES, c.COMPILED_BLOCKS + ("forbidden_changes",))

    def test_golden_full_compile(self):
        self.assertEqual(
            c.compile_prompt("A woman in a kitchen, morning light.", ALL),
            "A woman in a kitchen, morning light.\n\nSubject: Anna, short dark hair\nClothing: red wool sweater\n"
            "Scene: small kitchen\nPose: standing\nCamera: 35mm\nLighting: window light")

    def test_block_inner_text_untouched(self):
        out = c.compile_prompt("t", {"scene": "  line one\r\n  line two\té  "})
        self.assertEqual(out, "t\n\nScene: line one\r\n  line two\té")

    def test_nothing_added_beyond_labels(self):
        user = "A portrait, no jewellery"
        out = c.compile_prompt(user, ALL)
        rest = out
        for name in c.COMPILED_BLOCKS:
            rest = rest.replace("%s: %s" % (c.BLOCK_LABELS[name], ALL[name]), "", 1)
        rest = rest.replace(user, "", 1)
        self.assertEqual(rest.strip("\n"), "")  # only separators remain
        self.assertEqual(rest.replace("\n", ""), "")

    def test_forbidden_changes_never_compiled(self):
        only = c.compile_prompt("t", {"forbidden_changes": "NO-TATTOOS-XYZ"})
        self.assertEqual(only, "t")
        full = c.compile_prompt("t", ALL)
        self.assertNotIn("NO-TATTOOS-XYZ", full)
        self.assertNotIn("Forbidden", full)
        self.assertEqual(sum(label + ":" in full for label in c.BLOCK_LABELS.values()), 6)

    def test_unknown_block_keys_ignored(self):
        self.assertEqual(c.compile_prompt("t", {"jewellery": "gold necklace"}), "t")

    def test_none_block_treated_as_empty(self):
        self.assertEqual(c.compile_prompt("t", {"scene": None}), "t")


if __name__ == "__main__":
    unittest.main()
