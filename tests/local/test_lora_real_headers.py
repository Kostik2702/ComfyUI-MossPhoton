"""Real LoRA headers (design 6.5, 1.9 / check C6; docs/03-turbo.md 2): HEADERS ONLY, no tensor byte is read.

Every ``.safetensors`` header becomes a dict of zero-filled meta tensors (as ``research/spike_mp2/lora_native_mapping.py``;
``.alpha`` scalars get a 1.0 stand-in because the native mapper calls ``.item()`` on them). The native mapper then runs
against a real-shaped 42-layer MOSS built on the meta device (no weights), through ``lora.map_lora``.

Opt-in (MOSS_PHOTON_LOCAL=1, suite ``local``). The real file names come from the untracked
``tests/local/local_models.json`` (see ``_config.py``), keyed by ROLE below. With the flag set, a missing config, a missing
folder or a missing named file FAILS: a check must not switch itself off by losing its specimen.
"""
import collections
import json
import os
import struct
import unittest

from tests.integration import _comfy_env
from . import _config

LORA_DIR_REL = os.path.join("models", "loras", "moss")  # default; config "lora_dir" overrides

# role -> (modules_total, modules_mapped, keys_patched, layers_covered, key_format, verdict) measured on the real headers
# (design 1.9 C6 and docs/03-turbo.md 2; keys_patched = distinct model weight keys, fused q/k/v count once).
EXPECTED = {
    "split_qkv_full": (252, 252, 168, 42, "peft_split_qkv", "ok"),
    "fused_qkv_gap22": (164, 164, 164, 41, "peft_fused_qkv", "ok"),
    "kohya_a": (246, 246, 164, 41, "kohya", "ok"),
    "lokr": (164, 164, 164, 41, "lokr", "ok"),
    "fused_qkv_b": (164, 164, 164, 41, "peft_fused_qkv", "ok"),
    "kohya_distill_sft": (252, 252, 168, 42, "kohya", "ok"),
    "foreign_a": (240, 0, 0, 0, "peft_split_qkv", "error"),
    "foreign_b": (264, 0, 0, 0, "kohya", "error"),
    # the two 8-step distill files (docs/03-turbo.md 2): same shape, 252 modules, 168 keys, layer 22 covered
    "distill_8_dmd2": (252, 252, 168, 42, "kohya", "ok"),
    "distill_8_sft": (252, 252, 168, 42, "kohya", "ok"),
}
DT = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64", "I64": "int64", "I32": "int32",
      "I8": "int8", "U8": "uint8", "BOOL": "bool"}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
    return header, header.pop("__metadata__", {}) or {}


class RealHeadersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _config.require_flag()
        cfg = _config.load()
        cls.files = cfg["files"]
        try:
            root = _comfy_env.setup()
        except unittest.SkipTest as e:  # opted in: a missing ComfyUI is a failure, not a skip
            raise _config.LocalConfigError(str(e))
        cls.lora_dir = cfg.get("lora_dir") or os.path.join(root, LORA_DIR_REL)
        if not os.path.isdir(cls.lora_dir):
            raise _config.LocalConfigError("local LoRA folder does not exist: %s" % cls.lora_dir)
        import torch
        import comfy.model_base
        import comfy.supported_models
        from moss_photon.core import lora
        cls.torch, cls.lora = torch, lora
        ucfg = {"image_model": "moss_image", "dim": 3072, "ffn_dim": 8192, "num_heads": 24, "num_layers": 42,
                "fused_qkv": True}  # == model_detection output for the SFT header (spike MP-2)
        inner = comfy.model_base.MOSSImage(comfy.supported_models.MOSSImage(ucfg), device="meta")
        import types
        cls.model = types.SimpleNamespace(model=inner)  # map_lora only reads .model

    def header_sd(self, name):
        name = self.files.get(name, name)  # a role from the config, or a literal file name (sweep)
        header, meta = read_header(os.path.join(self.lora_dir, name))
        torch = self.torch
        sd = {}
        for key, info in header.items():
            if key.endswith(".alpha"):
                sd[key] = torch.tensor(1.0)
            else:
                sd[key] = torch.empty(info["shape"], dtype=getattr(torch, DT[info["dtype"]]), device="meta")
        return sd, meta

    def map_file(self, name):
        sd, meta = self.header_sd(name)
        m = self.lora.map_lora(self.model, sd)
        v = self.lora.coverage_verdict(m.modules_total, m.modules_mapped)
        return m, v, self.lora.parse_lora_metadata(meta)

    def test_the_named_files_reproduce_the_measured_numbers(self):
        for role, (total, mapped, keys, layers, fmt, verdict) in EXPECTED.items():
            with self.subTest(role=role):
                name = self.files[role]
                self.assertTrue(os.path.isfile(os.path.join(self.lora_dir, name)), "specimen missing: %s" % name)
                m, v, _meta = self.map_file(role)
                self.assertEqual((m.modules_total, m.modules_mapped, m.keys_patched, m.layers_covered, m.key_format, v["verdict"]),
                                 (total, mapped, keys, layers, fmt, verdict))
                self.assertEqual(m.model_entries, mapped)
                self.assertEqual(m.layers_total, 42)
                self.assertEqual(m.te_modules_ignored, 0)

    def test_six_compatible_files_are_ok_and_two_foreign_architectures_are_errors(self):
        verdicts = {name: self.map_file(name)[1]["verdict"] for name in list(EXPECTED)[:8]}
        self.assertEqual(sorted(v for v in verdicts.values()), ["error", "error", "ok", "ok", "ok", "ok", "ok", "ok"])
        self.assertEqual({n for n, v in verdicts.items() if v == "error"}, {"foreign_a", "foreign_b"})

    def test_layer_gaps_are_reported_not_penalised(self):
        m, v, _ = self.map_file("fused_qkv_gap22")
        self.assertEqual((m.layers_missing, v["verdict"]), ((22,), "ok"))
        m, v, _ = self.map_file("split_qkv_full")
        self.assertEqual((m.layers_missing, v["verdict"]), ((), "ok"))
        m, v, _ = self.map_file("distill_8_dmd2")
        self.assertNotIn(22, m.layers_missing)  # the distill files cover layer 22 (docs/03-turbo.md 2)

    def test_distill_files_are_told_apart_by_hash_not_by_header(self):
        a, _, _ = self.map_file("distill_8_dmd2")
        b, _, _ = self.map_file("distill_8_sft")
        self.assertEqual(a, b)  # identical mapping: only the sha256 table can tell them apart

    def test_metadata_architecture_agrees_with_the_mapping_for_every_local_file(self):
        """Sweep over ALL local files: a header that names another architecture maps nothing, and the other way round."""
        names = sorted(n for n in os.listdir(self.lora_dir) if n.endswith(".safetensors"))
        self.assertGreater(len(names), 8)
        seen = collections.Counter()
        for name in names:
            with self.subTest(name=name):
                m, v, meta = self.map_file(name)
                self.assertTrue(0.0 <= v["ratio"] <= 1.0)
                self.assertLessEqual(m.layers_covered, 42)
                self.assertEqual(len(m.layers_missing) + m.layers_covered, 42)
                self.assertEqual(m.tensors_total, len(read_header(os.path.join(self.lora_dir, name))[0]))
                seen[(meta["architecture"], v["verdict"])] += 1
                if meta["architecture"] == "other":
                    self.assertEqual(v["verdict"], "error")
                if v["verdict"] == "ok":
                    self.assertGreater(m.keys_patched, 0)
        # the sweep must have exercised both sides (a zero here would mean the loop checked nothing)
        self.assertGreater(seen[("other", "error")], 0)
        self.assertGreater(sum(c for (arch, verdict), c in seen.items() if verdict == "ok"), 20)


if __name__ == "__main__":
    unittest.main()
