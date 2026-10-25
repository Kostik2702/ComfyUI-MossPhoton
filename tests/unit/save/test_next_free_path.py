"""saveio.next_free_path: exclusive create, never overwrites, race safe (MP-11 acceptance 1; design 6.6)."""
import os
import shutil
import tempfile
import threading
import unittest
from unittest import mock

from moss_photon.core import errors, saveio


class NextFreePathTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="mp11_nfp_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.dir, name)

    def test_first_free_name_follows_the_stock_pattern_and_is_reserved(self):
        paths, counter = saveio.next_free_path(self.dir, "img", ".png", 1)
        self.assertEqual((paths, counter), ((self.path("img_00001_.png"),), 1))
        self.assertTrue(os.path.exists(paths[0]))  # reserved empty, so a second caller cannot take it
        self.assertEqual(os.path.getsize(paths[0]), 0)

    def test_an_existing_file_is_never_overwritten(self):
        with open(self.path("img_00001_.png"), "wb") as f:
            f.write(b"PRECIOUS")
        paths, counter = saveio.next_free_path(self.dir, "img", ".png", 1)
        self.assertEqual(counter, 2)
        self.assertEqual(os.path.basename(paths[0]), "img_00002_.png")
        with open(self.path("img_00001_.png"), "rb") as f:
            self.assertEqual(f.read(), b"PRECIOUS")

    def test_a_stale_sidecar_advances_the_counter_of_the_pair(self):
        with open(self.path("img_00001_.json"), "wb") as f:
            f.write(b"STALE")
        paths, counter = saveio.next_free_path(self.dir, "img", (".png", ".json"), 1)
        self.assertEqual(counter, 2)
        self.assertEqual([os.path.basename(p) for p in paths], ["img_00002_.png", "img_00002_.json"])
        # the image name of the failed attempt was given back, not left behind as an empty file
        self.assertEqual(sorted(os.listdir(self.dir)),
                         ["img_00001_.json", "img_00002_.json", "img_00002_.png"])
        with open(self.path("img_00001_.json"), "rb") as f:
            self.assertEqual(f.read(), b"STALE")

    def test_eight_threads_racing_for_one_stem_get_eight_distinct_names(self):
        barrier = threading.Barrier(8)
        got, failures = [], []

        def worker(i):
            try:
                barrier.wait()
                paths, counter = saveio.next_free_path(self.dir, "race", (".png", ".json"), 1)
                for path in paths:
                    with open(path, "wb") as f:
                        f.write(b"worker-%d" % i)
                got.append((i, counter, paths))
            except BaseException as exc:  # noqa: BLE001 - reported below
                failures.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(failures, [])
        self.assertEqual(sorted(c for _, c, _ in got), list(range(1, 9)))
        for i, _, paths in got:
            for path in paths:  # nobody wrote into somebody else's file
                with open(path, "rb") as f:
                    self.assertEqual(f.read(), b"worker-%d" % i)
        self.assertEqual(len(os.listdir(self.dir)), 16)

    def test_exhausted_counter_range_is_a_typed_error(self):
        for n in (1, 2, 3):
            open(self.path("img_%05d_.png" % n), "wb").close()
        with mock.patch.object(saveio, "MAX_COUNTER_TRIES", 3):
            with self.assertRaises(errors.OutputError):
                saveio.next_free_path(self.dir, "img", ".png", 1)
        self.assertEqual(len(os.listdir(self.dir)), 3)

    def test_missing_directory_is_a_typed_error(self):
        with self.assertRaises(errors.OutputError):
            saveio.next_free_path(os.path.join(self.dir, "nope"), "img", ".png", 1)

    def test_no_extension_is_a_programmer_error(self):
        with self.assertRaises(ValueError):
            saveio.next_free_path(self.dir, "img", (), 1)


if __name__ == "__main__":
    unittest.main()
