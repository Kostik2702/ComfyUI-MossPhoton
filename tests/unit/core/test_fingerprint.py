"""Checkpoint identity: tables, hash cache and guarded path discovery (design 1.2, 1.8, 6.1)."""
import builtins
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from moss_photon.core import consts, fingerprint

SFT = "f96738d8116d78b7316b57c663d8c98591ddafab7488f48e3099b7655e34ab6f"
BASE = "cacd158728a1fb9e57c408a922b54752ec71fceb566b3821269927964e62b41f"


def load_diffusion_model(path, options=None):  # stand-ins named like the ComfyUI loaders (the guard checks names)
    raise AssertionError("must never be called")


def load_checkpoint_guess_config(path, *a):
    raise AssertionError("must never be called")


def load_clip_model_patcher(paths, *a):
    raise AssertionError("must never be called")


def load_checkpoint_clip_patcher(path, *a):
    raise AssertionError("must never be called")


def some_other_loader(path):
    raise AssertionError("must never be called")


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def make_file(self, name="a.safetensors", size=5 * 1024 * 1024 + 123, fill=b"abc"):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write((fill * (size // len(fill) + 1))[:size])
        return path

    @staticmethod
    def sha(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()


class KnownTableTest(unittest.TestCase):
    def test_known_table_has_sft_and_base_oids(self):
        families = {k: v["family"] for k, v in fingerprint.KNOWN_CHECKPOINTS.items()}
        self.assertEqual(families, {
            SFT: "moss-image-sft", BASE: "moss-image-base",
            "b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f": "moss-image-sft",
            "f601c1e1161915fb3fea003da0c1782e374dd960fbb3978bc0e4e3fa1f2d8e6f": "moss-image-base"})
        self.assertEqual(consts.SFT_BF16_SHA256, SFT)

    def test_only_sft_bf16_is_verified(self):
        verified = [k for k, v in fingerprint.KNOWN_CHECKPOINTS.items() if v["verified_matrix"]]
        self.assertEqual(verified, [SFT])

    def test_text_encoder_table(self):
        self.assertEqual(len(fingerprint.KNOWN_TEXT_ENCODERS), 3)
        self.assertEqual(fingerprint.KNOWN_TEXT_ENCODERS[
            "7e6e9f08d598f829cb940e60ac0c698e1f1c27a47daffd7e598cd78c78b4cc53"]["name"], "qwen3.5_9b_bf16")

    def test_classify_hash(self):
        self.assertEqual(fingerprint.classify_hash(SFT, fingerprint.KNOWN_CHECKPOINTS)["variant"], "bf16")
        self.assertIsNone(fingerprint.classify_hash("0" * 64, fingerprint.KNOWN_CHECKPOINTS))
        self.assertIsNone(fingerprint.classify_hash(None, fingerprint.KNOWN_CHECKPOINTS))
        entry = fingerprint.classify_hash(SFT, fingerprint.KNOWN_CHECKPOINTS)
        entry["family"] = "mutated"  # a copy: the table stays intact
        self.assertEqual(fingerprint.KNOWN_CHECKPOINTS[SFT]["family"], "moss-image-sft")


class Sha256FileTest(TmpCase):
    def test_sha256_file_matches_hashlib(self):
        path = self.make_file()
        cache = fingerprint.JsonCache(None)
        self.assertEqual(fingerprint.sha256_file(path, cache=cache, chunk=1 << 20), self.sha(path))

    def test_cache_hit_skips_hashing(self):
        path = self.make_file()
        cache = fingerprint.JsonCache(os.path.join(self.dir, "c.json"))
        first = fingerprint.sha256_file(path, cache=cache)
        opened = []
        real_open = builtins.open

        def spy(file, *a, **k):
            opened.append(os.fspath(file))
            return real_open(file, *a, **k)
        with mock.patch.object(builtins, "open", spy):
            again = fingerprint.sha256_file(path, cache=cache)
        self.assertEqual(again, first)
        self.assertNotIn(path, opened)
        # control: a changed file is opened again
        with open(path, "ab") as f:
            f.write(b"x")
        opened.clear()
        with mock.patch.object(builtins, "open", spy):
            changed = fingerprint.sha256_file(path, cache=cache)
        self.assertIn(path, opened)
        self.assertNotEqual(changed, first)

    def test_cache_survives_a_new_process_object(self):
        path = self.make_file()
        cache_path = os.path.join(self.dir, "c.json")
        sha = fingerprint.sha256_file(path, cache=fingerprint.JsonCache(cache_path))
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(set(data), {"version", "entries"})
        entry = data["entries"][fingerprint.cache_key(path)]
        self.assertEqual(set(entry), {"size", "mtime_ns", "sha256"})  # hashes only
        self.assertEqual(entry["sha256"], sha)
        self.assertEqual(fingerprint.sha256_file(path, cache=fingerprint.JsonCache(cache_path), mode="cached_only"), sha)

    def test_cache_invalidated_by_size_or_mtime(self):
        path = self.make_file()
        cache = fingerprint.JsonCache(None)
        sha = fingerprint.sha256_file(path, cache=cache)
        key = fingerprint.cache_key(path)
        entry = cache.get(key)
        cache.put(key, dict(entry, size=entry["size"] + 1))
        self.assertIsNone(fingerprint.sha256_file(path, cache=cache, mode="cached_only"))
        cache.put(key, dict(entry, mtime_ns=entry["mtime_ns"] + 1))
        self.assertIsNone(fingerprint.sha256_file(path, cache=cache, mode="cached_only"))
        cache.put(key, entry)
        self.assertEqual(fingerprint.sha256_file(path, cache=cache, mode="cached_only"), sha)

    def test_modes(self):
        path = self.make_file()
        cache = fingerprint.JsonCache(None)
        self.assertIsNone(fingerprint.sha256_file(path, cache=cache, mode="cached_only"))  # miss
        self.assertIsNone(fingerprint.sha256_file(path, cache=cache, mode="off"))
        self.assertIsNone(cache.get(fingerprint.cache_key(path)))  # off and cached_only never hash
        with self.assertRaises(ValueError):
            fingerprint.sha256_file(path, cache=cache, mode="sometimes")

    def test_missing_file_is_none(self):
        self.assertIsNone(fingerprint.sha256_file(os.path.join(self.dir, "nope"), cache=fingerprint.JsonCache(None)))
        self.assertIsNone(fingerprint.sha256_file(None, cache=fingerprint.JsonCache(None)))

    def test_cache_corrupt_or_unwritable_is_swallowed(self):
        path = self.make_file(size=1000)
        corrupt = os.path.join(self.dir, "corrupt.json")
        with open(corrupt, "w", encoding="utf-8") as f:
            f.write("{not json")
        with self.assertLogs("MossPhoton", level="WARNING"):
            sha = fingerprint.sha256_file(path, cache=fingerprint.JsonCache(corrupt))
        self.assertEqual(sha, self.sha(path))
        # unwritable: the cache "file" is below a path component that is a regular file
        blocker = os.path.join(self.dir, "blocker")
        with open(blocker, "w", encoding="utf-8") as f:
            f.write("x")
        bad = fingerprint.JsonCache(os.path.join(blocker, "sub", "c.json"))
        with self.assertLogs("MossPhoton", level="WARNING") as logs:
            self.assertEqual(fingerprint.sha256_file(path, cache=bad), sha)
            self.assertEqual(fingerprint.sha256_file(path, cache=bad, mode="cached_only"), sha)  # still memory-cached
        self.assertEqual(len(logs.output), 1)  # logged once

    def test_interrupt_check_called_between_chunks(self):
        path = self.make_file(size=5 * 1024 * 1024)
        calls = []
        fingerprint.sha256_file(path, cache=fingerprint.JsonCache(None), chunk=1 << 20,
                                interrupt_check=lambda: calls.append(1))
        self.assertEqual(len(calls), 5)

    def test_interrupt_exception_propagates_and_nothing_is_cached(self):
        path = self.make_file(size=3 * 1024 * 1024)
        cache = fingerprint.JsonCache(None)

        class Stop(Exception):
            pass

        def stop():
            raise Stop()
        with self.assertRaises(Stop):
            fingerprint.sha256_file(path, cache=cache, chunk=1 << 20, interrupt_check=stop)
        self.assertIsNone(cache.get(fingerprint.cache_key(path)))

    def test_progress_reports_bytes(self):
        path = self.make_file(size=2 * 1024 * 1024 + 5)
        seen = []
        fingerprint.sha256_file(path, cache=fingerprint.JsonCache(None), chunk=1 << 20,
                                progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen[-1], (2 * 1024 * 1024 + 5, 2 * 1024 * 1024 + 5))

    def test_cache_temp_file_name_is_unique_per_write_and_per_process(self):
        """m14: two ComfyUI processes sharing the user directory must not race on one fixed ``.tmp`` name."""
        cache = fingerprint.JsonCache(os.path.join(self.dir, "c.json"))
        sources = []
        real_replace = os.replace

        def spy(src, dst):
            sources.append(os.path.basename(src))
            return real_replace(src, dst)
        with mock.patch.object(os, "replace", spy):
            cache.put("a", {"size": 1})
            cache.put("b", {"size": 2})
        self.assertEqual(len(sources), 2)
        self.assertNotEqual(sources[0], sources[1])
        self.assertNotEqual(sources[0], "c.json.tmp")
        for name in sources:
            self.assertIn(str(os.getpid()), name)
            self.assertTrue(name.startswith("c.json.") and name.endswith(".tmp"), name)
        self.assertEqual(sorted(os.listdir(self.dir)), ["c.json"])  # nothing left behind

    def test_failed_write_removes_its_own_temp_file(self):
        cache = fingerprint.JsonCache(os.path.join(self.dir, "c.json"))
        with mock.patch.object(os, "replace", side_effect=OSError("denied")), self.assertLogs("MossPhoton", "WARNING"):
            cache.put("a", {"size": 1})
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_junction_and_the_real_path_share_one_entry(self):
        """m14: the key is realpath + normcase, so a linked path does not hash a 10 GB file twice."""
        path = self.make_file(size=1000)
        link_dir = os.path.join(self.dir, "link")
        if sys.platform == "win32":
            import subprocess
            subprocess.run(["cmd", "/c", "mklink", "/J", link_dir, self.dir], check=True, capture_output=True)
        else:
            os.symlink(self.dir, link_dir)
        self.addCleanup(lambda: os.rmdir(link_dir) if sys.platform == "win32" else os.unlink(link_dir))
        linked = os.path.join(link_dir, os.path.basename(path))
        self.assertNotEqual(os.path.abspath(linked), os.path.abspath(path))  # control: abspath would differ
        self.assertEqual(fingerprint.cache_key(linked), fingerprint.cache_key(path))
        cache = fingerprint.JsonCache(None)
        sha = fingerprint.sha256_file(path, cache=cache)
        self.assertEqual(fingerprint.sha256_file(linked, cache=cache, mode="cached_only"), sha)


def patcher_with(fn, args):
    return types.SimpleNamespace(cached_patcher_init=(fn, args))


class PathDiscoveryTest(TmpCase):
    def test_checkpoint_path_of_is_guarded(self):
        path = self.make_file(size=10)
        self.assertEqual(fingerprint.checkpoint_path_of(patcher_with(load_diffusion_model, (path, {}))), path)
        self.assertEqual(fingerprint.checkpoint_path_of(
            types.SimpleNamespace(cached_patcher_init=(load_checkpoint_guess_config, (path, False), 0))), path)
        self.assertIsNone(fingerprint.checkpoint_path_of(types.SimpleNamespace()))  # no attribute
        self.assertIsNone(fingerprint.checkpoint_path_of(patcher_with(some_other_loader, (path,))))  # wrong name
        self.assertIsNone(fingerprint.checkpoint_path_of(patcher_with(load_diffusion_model, [])))  # empty args
        self.assertIsNone(fingerprint.checkpoint_path_of(patcher_with(load_diffusion_model, None)))
        self.assertIsNone(fingerprint.checkpoint_path_of(patcher_with("load_diffusion_model", (path,))))  # not callable
        self.assertIsNone(fingerprint.checkpoint_path_of(
            patcher_with(load_diffusion_model, (os.path.join(self.dir, "missing"),))))
        self.assertIsNone(fingerprint.checkpoint_path_of(patcher_with(load_diffusion_model, (5,))))
        self.assertIsNone(fingerprint.checkpoint_path_of(types.SimpleNamespace(cached_patcher_init=None)))
        self.assertIsNone(fingerprint.checkpoint_path_of(types.SimpleNamespace(cached_patcher_init=(load_diffusion_model,))))

    def test_text_encoder_path_of_handles_list_and_str(self):
        path = self.make_file(size=10)
        standalone = types.SimpleNamespace(patcher=patcher_with(load_clip_model_patcher, ([path], None, 1, {})))
        bundled = types.SimpleNamespace(patcher=patcher_with(load_checkpoint_clip_patcher, (path, None, {}, {})))
        self.assertEqual(fingerprint.text_encoder_path_of(standalone), path)
        self.assertEqual(fingerprint.text_encoder_path_of(bundled), path)
        two = types.SimpleNamespace(patcher=patcher_with(load_clip_model_patcher, ([path, path], None)))
        self.assertIsNone(fingerprint.text_encoder_path_of(two))  # ambiguous
        self.assertIsNone(fingerprint.text_encoder_path_of(types.SimpleNamespace()))
        self.assertIsNone(fingerprint.text_encoder_path_of(
            types.SimpleNamespace(patcher=patcher_with(load_diffusion_model, ([path],)))))  # a model loader, not a CLIP one

    def test_missing_cached_patcher_init_warns_once_and_names_the_comfyui_version(self):
        """m16: a ComfyUI that renamed the attribute must not silently turn every run into fingerprint_unavailable."""
        fake = types.ModuleType("comfyui_version")
        fake.__version__ = "9.8.7"
        patcher = mock.patch.dict(sys.modules, {"comfyui_version": fake})
        patcher.start()
        self.addCleanup(patcher.stop)
        saved = fingerprint._init_missing_warned
        fingerprint._init_missing_warned = False
        self.addCleanup(setattr, fingerprint, "_init_missing_warned", saved)
        with self.assertLogs("MossPhoton", level="WARNING") as logs:
            self.assertIsNone(fingerprint.checkpoint_path_of(types.SimpleNamespace()))
            self.assertIsNone(fingerprint.text_encoder_path_of(types.SimpleNamespace(patcher=types.SimpleNamespace())))
            self.assertIsNone(fingerprint.checkpoint_path_of(types.SimpleNamespace()))
        self.assertEqual(len(logs.output), 1)  # once per process
        self.assertIn("9.8.7", logs.output[0])
        self.assertIn("cached_patcher_init", logs.output[0])

    def test_present_cached_patcher_init_does_not_warn(self):
        path = self.make_file(size=10)
        saved = fingerprint._init_missing_warned
        fingerprint._init_missing_warned = False
        self.addCleanup(setattr, fingerprint, "_init_missing_warned", saved)
        with self.assertNoLogs("MossPhoton", level="WARNING"):
            fingerprint.checkpoint_path_of(patcher_with(load_diffusion_model, (path, {})))
            fingerprint.checkpoint_path_of(patcher_with(some_other_loader, (path,)))  # a known attribute, other loader
            fingerprint.text_encoder_path_of(types.SimpleNamespace())  # no patcher at all: not this problem

    def test_file_basic_info(self):
        path = self.make_file("x.bin", size=77)
        info = fingerprint.file_basic_info(path)
        self.assertEqual((info["basename"], info["size"]), ("x.bin", 77))
        self.assertIsInstance(info["mtime_ns"], int)


class IdentifyTest(TmpCase):
    def identify(self, path, **kw):
        patcher = patcher_with(load_diffusion_model, (path, {})) if path else types.SimpleNamespace()
        kw.setdefault("cache", fingerprint.JsonCache(None))
        return fingerprint.identify_checkpoint(patcher, **kw)

    def test_identify_unknown_hash(self):
        path = self.make_file("kromoss_v10_bf16.safetensors", size=1000)
        out = self.identify(path)
        self.assertEqual(out["family"], "unknown-moss")
        self.assertEqual([w.code for w in out["warnings"]], ["unverified_checkpoint"])
        ident = out["identity"]
        self.assertEqual((ident["basename"], ident["size"], ident["hash_source"], ident["classified_by"]),
                         ("kromoss_v10_bf16.safetensors", 1000, "computed", "none"))
        self.assertEqual(ident["sha256"], self.sha(path))
        self.assertEqual(set(ident), {"sha256", "size", "basename", "hash_source", "classified_by", "official",
                                      "variant", "verified_matrix"})
        self.assertNotIn(self.dir, json.dumps(out["identity"]))  # basename only, never the path

    def test_known_hashes_classify(self):
        path = self.make_file(size=100)
        cache = fingerprint.JsonCache(None)
        fingerprint.sha256_file(path, cache=cache)
        for sha, family, variant, verified, codes in (
                (SFT, "moss-image-sft", "bf16", True, []),
                (BASE, "moss-image-base", "bf16", False, ["family_not_verified"]),
                ("b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f", "moss-image-sft", "int8",
                 False, ["family_not_verified", "quantized_checkpoint_unverified"])):
            entry = cache.get(fingerprint.cache_key(path))
            cache.put(fingerprint.cache_key(path), dict(entry, sha256=sha))
            out = self.identify(path, cache=cache)
            with self.subTest(family=family, variant=variant):
                self.assertEqual((out["family"], out["identity"]["variant"], out["identity"]["verified_matrix"],
                                  out["identity"]["hash_source"], out["identity"]["classified_by"]),
                                 (family, variant, verified, "cache", "known_table"))
                self.assertEqual([w.code for w in out["warnings"]], codes)

    def test_unavailable_path_returns_unknown_plus_warning(self):
        out = self.identify(None)
        self.assertEqual(out["family"], "unknown-moss")
        self.assertEqual(out["identity"]["hash_source"], "unavailable")
        self.assertIsNone(out["identity"]["sha256"])
        self.assertEqual(out["identity"]["basename"], "unknown")
        self.assertEqual([(w.code, w.severity) for w in out["warnings"]], [("fingerprint_unavailable", "warning")])

    def test_off_and_cached_only_miss_are_unavailable_with_basename_kept(self):
        path = self.make_file("m.safetensors", size=50)
        for mode in ("off", "cached_only"):
            with self.subTest(mode=mode):
                out = self.identify(path, mode=mode)
                self.assertEqual([w.code for w in out["warnings"]], ["fingerprint_unavailable"])
                self.assertEqual((out["identity"]["basename"], out["identity"]["size"]), ("m.safetensors", 50))
                self.assertEqual(out["identity"]["hash_source"], "unavailable")


class DefaultCacheTest(unittest.TestCase):
    """folder_paths is faked: the embedded interpreter can import the real one, which drags comfy into the unit run."""

    def fresh(self, fake_module):
        saved = fingerprint._default_cache
        fingerprint._default_cache = None
        self.addCleanup(setattr, fingerprint, "_default_cache", saved)
        patcher = mock.patch.dict(sys.modules, {"folder_paths": fake_module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_without_comfy_the_default_cache_is_in_memory(self):
        self.fresh(None)  # None in sys.modules makes "import folder_paths" raise ImportError
        cache = fingerprint.default_cache()
        self.assertIsNone(cache.path)
        self.assertIs(fingerprint.default_cache(), cache)

    def test_with_comfy_the_cache_lives_in_the_system_user_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = types.SimpleNamespace(get_system_user_directory=lambda name: os.path.join(tmp, "__" + name))
            self.fresh(fake)
            cache = fingerprint.default_cache()
            self.assertEqual(cache.path, os.path.join(tmp, "__moss_photon", "hash_cache.json"))


if __name__ == "__main__":
    unittest.main()
