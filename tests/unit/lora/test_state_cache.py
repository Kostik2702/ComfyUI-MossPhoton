"""MP-36 m17 (MP-38): the LoRA file cache is a small LRU, so a chain of several LoRA nodes (turbo_identity: distill +
character) re-run reads no file twice. Pure: a read-count spy stands in for ``comfy.utils.load_torch_file``."""
import unittest

from moss_photon.core import lora


class Reader:
    """Read-count spy: ``reads`` lists the keys the cache had to load."""

    def __init__(self):
        self.reads = []

    def __call__(self, key):
        def load():
            self.reads.append(key)
            return ({"sd": key}, {"meta": key})
        return load


class StateCacheTest(unittest.TestCase):
    def test_slots_cover_a_turbo_identity_chain(self):
        # distill + character is the shipped maximum for turbo_identity; 4 also fits two more adapters (style, scene)
        self.assertGreaterEqual(lora.STATE_CACHE_SLOTS, 3)
        self.assertEqual(lora.STATE_CACHE_SLOTS, 4)

    def test_a_two_lora_chain_re_run_reads_no_file_twice(self):
        cache, read = lora.StateCache(lora.STATE_CACHE_SLOTS), Reader()
        for _ in range(3):  # three executions of the whole chain, both nodes each time
            cache.get("distill", read("distill"))
            cache.get("character", read("character"))
        self.assertEqual(read.reads, ["distill", "character"])

    def test_the_old_one_slot_behaviour_would_have_read_every_time(self):
        cache, read = lora.StateCache(1), Reader()  # control: the same chain on one slot
        for _ in range(3):
            cache.get("distill", read("distill"))
            cache.get("character", read("character"))
        self.assertEqual(len(read.reads), 6)

    def test_the_least_recently_used_file_is_evicted(self):
        cache, read = lora.StateCache(2), Reader()
        cache.get("a", read("a"))
        cache.get("b", read("b"))
        cache.get("a", read("a"))          # a is now the most recent
        cache.get("c", read("c"))          # evicts b
        cache.get("a", read("a"))
        cache.get("b", read("b"))          # b was evicted: read again
        self.assertEqual(read.reads, ["a", "b", "c", "b"])

    def test_the_cache_never_holds_more_than_its_slots(self):
        cache, read = lora.StateCache(3), Reader()
        for k in "abcdef":
            cache.get(k, read(k))
        self.assertEqual(len(cache), 3)

    def test_state_is_shared_and_metadata_is_a_copy(self):
        cache = lora.StateCache(2)
        sd, meta = cache.get("a", lambda: ({"w": 1}, {"m": 1}))
        meta["poison"] = 1
        sd2, meta2 = cache.get("a", lambda: self.fail("must be cached"))
        self.assertIs(sd, sd2)
        self.assertEqual(meta2, {"m": 1})

    def test_a_failing_loader_caches_nothing(self):
        cache = lora.StateCache(2)

        def boom():
            raise OSError("unreadable")
        with self.assertRaises(OSError):
            cache.get("a", boom)
        self.assertEqual(len(cache), 0)

    def test_slots_must_be_positive(self):
        with self.assertRaises(ValueError):
            lora.StateCache(0)


if __name__ == "__main__":
    unittest.main()
