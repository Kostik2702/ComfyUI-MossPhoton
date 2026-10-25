"""Noise policy bookkeeping that needs no torch (design 1.2, 6.1); the tensors are tested in integration."""
import os
import subprocess
import sys
import unittest

from moss_photon.core import errors, rng

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SubseedTest(unittest.TestCase):
    def test_subseed_wraps_at_2_pow_64(self):
        self.assertEqual(rng.SEED_MASK, 2 ** 64 - 1)
        self.assertEqual(rng.subseed(5, 3), 8)
        self.assertEqual(rng.subseed(2 ** 64 - 1, 0), 2 ** 64 - 1)
        self.assertEqual(rng.subseed(2 ** 64 - 1, 1), 0)
        self.assertEqual(rng.subseed(2 ** 64 - 1, 2), 1)

    def test_subseeds_per_policy(self):
        self.assertEqual(rng.subseeds(42, 3, "native"), [42])  # one stream for the whole batch
        self.assertEqual(rng.subseeds(42, 3, "seed_plus_i"), [42, 43, 44])
        self.assertEqual(rng.subseeds(2 ** 64 - 2, 3, "seed_plus_i"), [2 ** 64 - 2, 2 ** 64 - 1, 0])

    def test_unknown_policy_rejected_everywhere(self):
        for fn in (lambda: rng.subseeds(1, 1, "nope"), lambda: rng.seed_warnings(1, 1, "nope"),
                   lambda: rng.make_noise(None, 1, "nope")):
            with self.assertRaises(ValueError):
                fn()
        self.assertEqual(rng.NOISE_POLICIES, ("native", "seed_plus_i"))

    def test_seed_wrapped_warning_only_when_it_wraps(self):
        self.assertEqual(rng.seed_warnings(5, 100, "seed_plus_i"), [])
        self.assertEqual(rng.seed_warnings(2 ** 64 - 1, 4, "native"), [])  # native never adds i
        (w,) = rng.seed_warnings(2 ** 64 - 1, 2, "seed_plus_i")
        self.assertEqual((w.code, w.severity), ("seed_wrapped", "info"))
        self.assertEqual(rng.seed_warnings(2 ** 64 - 1, 1, "seed_plus_i"), [])  # item 0 is the seed itself

    def test_seed_plus_i_rejects_batch_index_before_importing_torch(self):
        code = ("import sys\nsys.path.insert(0, sys.argv[1])\nfrom moss_photon.core import rng, errors\n"
                "class L:\n    shape = (2, 4, 2, 2)\n    dtype = None\n"
                "try:\n    rng.make_noise(L(), 1, 'seed_plus_i', batch_index=[0, 1])\n"
                "except errors.UnsupportedFeature as e:\n    assert e.code == 'unsupported_feature'\n"
                "else:\n    raise SystemExit('no error')\n"
                "assert 'torch' not in sys.modules\nprint('OK')\n")
        proc = subprocess.run([sys.executable, "-I", "-c", code, PKG_ROOT], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:] + proc.stdout)

    def test_module_import_is_pure(self):
        code = ("import sys\nsys.path.insert(0, sys.argv[1])\nfrom moss_photon.core import rng\n"
                "assert not [m for m in sys.modules if m.split('.')[0] in ('torch', 'comfy')]\nprint('OK')\n")
        proc = subprocess.run([sys.executable, "-I", "-c", code, PKG_ROOT], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertTrue(issubclass(errors.UnsupportedFeature, errors.MossPhotonError))


if __name__ == "__main__":
    unittest.main()
