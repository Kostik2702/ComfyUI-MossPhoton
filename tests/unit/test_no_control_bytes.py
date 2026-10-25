"""Byte guard (design 6.0): no control byte below 32, except TAB/LF/CR, in the text files of the pack.

Same rule as research/check_bytes.py. Why it exists: type-check, lint and prettier all stayed green over a file
holding a raw U+0000 (CLAUDE.md incident 2026-09-03). The scan works on raw bytes, so it does not depend on how a
tool decodes the file. This file builds its control bytes from numbers; it must not contain one itself.
"""
import os
import tempfile
import unittest

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUFFIXES = (".py", ".md", ".json", ".toml", ".txt", ".js", ".yaml", ".yml")
ALLOWED = frozenset((9, 10, 13))
SKIP_DIRS = frozenset(("__pycache__", ".git"))


def find_control_bytes(root):
    """Return ``(scanned_relpaths, findings)``; a finding is ``(relpath, [(offset, byte), ...first 5])``.

    Raises ``FileNotFoundError`` when ``root`` is not a directory: a guard with nothing to scan must fail, not pass.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError("byte guard root does not exist: %s" % root)
    scanned, findings = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            if not name.endswith(SUFFIXES):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            scanned.append(rel)
            with open(path, "rb") as f:
                data = f.read()
            hits = [(i, b) for i, b in enumerate(data) if b < 32 and b not in ALLOWED]
            if hits:
                findings.append((rel, hits[:5]))
    return scanned, findings


class PackByteGuardTest(unittest.TestCase):
    def test_pack_has_no_control_bytes(self):
        scanned, findings = find_control_bytes(PKG_ROOT)
        self.assertEqual(findings, [], "control bytes found (offset, byte)")
        # control specimen: the scan must really have walked the pack and seen known files
        self.assertGreater(len(scanned), 10)
        for known in ("moss_photon/core/errors.py", "pyproject.toml", "tests/run_tests.py"):
            self.assertIn(known, scanned)


class ByteGuardSelfTest(unittest.TestCase):
    """The guard is proven by planting the violation in a scratch tree and watching it go red."""

    def _write(self, root, rel, data):
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def test_planted_nul_is_found_with_offset(self):
        with tempfile.TemporaryDirectory() as root:
            self._write(root, "a/clean.py", b"x = 1\n")
            self._write(root, "a/bad.py", b"x = " + bytes([0]) + b"1\n")
            scanned, findings = find_control_bytes(root)
            self.assertEqual(sorted(scanned), ["a/bad.py", "a/clean.py"])
            self.assertEqual(findings, [("a/bad.py", [(4, 0)])])

    def test_every_scanned_suffix_is_guarded(self):
        with tempfile.TemporaryDirectory() as root:
            for suffix in SUFFIXES:
                self._write(root, "f" + suffix, b"a" + bytes([27]) + b"b")
            _, findings = find_control_bytes(root)
            self.assertEqual(sorted(f[0] for f in findings), sorted("f" + s for s in SUFFIXES))

    def test_tab_lf_cr_are_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            self._write(root, "ok.md", b"a\tb\r\nc\n")
            _, findings = find_control_bytes(root)
            self.assertEqual(findings, [])

    def test_other_suffixes_and_cache_dirs_are_not_scanned(self):
        with tempfile.TemporaryDirectory() as root:
            self._write(root, "img.png", bytes([0, 1, 2]))
            self._write(root, "__pycache__/m.py", bytes([0]))
            scanned, findings = find_control_bytes(root)
            self.assertEqual((scanned, findings), ([], []))

    def test_missing_root_fails_instead_of_passing(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                find_control_bytes(os.path.join(root, "nope"))


if __name__ == "__main__":
    unittest.main()
