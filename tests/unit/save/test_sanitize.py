"""saveio.sanitize_prefix (design 6.6 test_sanitize)."""
import unittest

from moss_photon.core import errors, saveio


class SanitizeTest(unittest.TestCase):
    def test_keeps_valid_prefixes_and_subfolders_unchanged(self):
        for prefix in ("MossPhoton/img", "img", "a/b/c", "run_%width%x%height%/img", "a b/c", "a\\b", "dir.name/x",
                       "a..b/c", "...x"):
            with self.subTest(prefix=prefix):
                self.assertEqual(saveio.sanitize_prefix(prefix), prefix)

    def test_rejects_unsafe_prefixes(self):
        bad = ["", "   ", "..", "../x", "a/../x", "a/..", "a\\..\\x", "..\\x", "C:/x", "c:x", "Z:\\x", "/abs",
               "\\\\server\\share", "\\x", "a\x00b", "a\nb", None, 5]
        for prefix in bad:
            with self.subTest(prefix=prefix):
                with self.assertRaises(errors.OutputError) as ctx:
                    saveio.sanitize_prefix(prefix)
                self.assertEqual(ctx.exception.issue.code, "output_error")
                self.assertEqual(ctx.exception.issue.param, "filename_prefix")


if __name__ == "__main__":
    unittest.main()
