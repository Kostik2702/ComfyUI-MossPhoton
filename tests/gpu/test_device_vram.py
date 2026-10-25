"""Device-level VRAM figures of the stability gate (MP-42) on plain text and numbers. No GPU, no torch, no comfy.

An unavailable figure must never read as "no growth": ``None`` stays ``None``.
"""
import unittest

from . import _env

extras = None


def setUpModule():
    global extras
    _env.require_parity_dir()
    import extras as module
    extras = module


SMI = "pid, used_gpu_memory [MiB]\n1234, 2048 MiB\n5678, [N/A]\n"


class ParseSmiApps(unittest.TestCase):
    def test_numeric_row_is_bytes(self):
        self.assertEqual(extras.parse_smi_apps(SMI, 1234), 2048 * 1024 * 1024)

    def test_na_row_is_none_not_zero(self):
        self.assertIsNone(extras.parse_smi_apps(SMI, 5678))

    def test_missing_pid_is_none(self):
        self.assertIsNone(extras.parse_smi_apps(SMI, 99))

    def test_pid_is_matched_whole(self):
        self.assertIsNone(extras.parse_smi_apps(SMI, 123))

    def test_empty_output_is_none(self):
        self.assertIsNone(extras.parse_smi_apps("", 1234))


class DeviceGrowth(unittest.TestCase):
    def test_growth(self):
        self.assertAlmostEqual(extras.device_growth(1000, 1050), 0.05)
        self.assertAlmostEqual(extras.device_growth(1000, 900), -0.1)

    def test_missing_figure_is_none(self):
        self.assertIsNone(extras.device_growth(None, 1000))
        self.assertIsNone(extras.device_growth(1000, None))
        self.assertIsNone(extras.device_growth(0, 1000))


if __name__ == "__main__":
    unittest.main()
