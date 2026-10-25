"""Shared helpers of the MP-11 unit tests: a real report fixture, a fake ``folder_paths``, a temp output folder."""
import copy
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from moss_photon.core import report as report_mod
from moss_photon.nodes import save

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTRACTS = os.path.join(TESTS_DIR, "fixtures", "contracts")


def load_report():
    """The contract example report with a real ``recipe_hash`` (the fixture carries zeros)."""
    with open(os.path.join(CONTRACTS, "report.json"), "r", encoding="utf-8") as f:
        rep = json.load(f)
    rep["recipe_hash"] = report_mod.recipe_hash(rep)
    return rep


class FakeFolderPaths:
    """Just enough of ComfyUI's ``folder_paths``. ``get_save_image_path`` always answers counter 1, so every test
    exercises the collision path of the node instead of relying on a directory scan."""

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def get_output_directory(self):
        return self.output_dir

    def get_save_image_path(self, prefix, output_dir, width=0, height=0):
        norm = os.path.normpath(prefix)
        subfolder, filename = os.path.dirname(norm), os.path.basename(norm)
        folder = os.path.join(output_dir, subfolder)
        os.makedirs(folder, exist_ok=True)
        return folder, filename, 1, subfolder, prefix


class SaveTestCase(unittest.TestCase):
    """Temp output folder + patched ``folder_paths`` / ``--disable-metadata`` seams of ``nodes.save``."""

    disable_metadata = False

    def setUp(self):
        self.out = tempfile.mkdtemp(prefix="mp11_")
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)
        fake = FakeFolderPaths(self.out)
        for patcher in (mock.patch.object(save, "_folder_paths", lambda: fake),
                        mock.patch.object(save, "_disable_metadata", lambda: self.disable_metadata)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.report = load_report()

    def run_save(self, image, **kw):
        args = dict(report=copy.deepcopy(self.report))
        args.update(kw)
        return save.MossPhotonSaveImage().save(image, **args)

    def listing(self, sub=""):
        return sorted(os.listdir(os.path.join(self.out, sub)))

    def read(self, name, sub=""):
        with open(os.path.join(self.out, sub, name), "rb") as f:
            return f.read()

    def read_json(self, name, sub=""):
        return json.loads(self.read(name, sub).decode("utf-8"))
