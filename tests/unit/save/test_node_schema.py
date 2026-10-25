"""MossPhotonSaveImage INPUT_TYPES equal design 4.8 (design 6.6 test_node_schema)."""
import unittest

from moss_photon.core import consts
from moss_photon.nodes import save


class SaveSchemaTest(unittest.TestCase):
    def setUp(self):
        self.spec = save.MossPhotonSaveImage.INPUT_TYPES()

    def test_required_inputs(self):
        req = self.spec["required"]
        self.assertEqual(list(req), ["image", "report", "filename_prefix", "format", "metadata"])
        self.assertEqual(req["image"][0], "IMAGE")
        self.assertEqual(req["report"][0], "PHOTON_REPORT")
        self.assertEqual(req["filename_prefix"][0], "STRING")
        self.assertEqual(req["filename_prefix"][1]["default"], "MossPhoton/img")
        self.assertEqual(req["format"][0], ["png", "jpeg", "webp"])
        self.assertEqual(req["format"][1]["default"], "png")
        self.assertEqual(req["metadata"][0], ["both", "embedded", "sidecar", "none"])
        self.assertEqual(req["metadata"][1]["default"], "both")  # Q13

    def test_optional_and_hidden_inputs(self):
        opt = self.spec["optional"]
        self.assertEqual(list(opt), ["quality", "embed_workflow"])
        self.assertEqual((opt["quality"][0], opt["quality"][1]["default"], opt["quality"][1]["min"],
                          opt["quality"][1]["max"]), ("INT", 95, 1, 100))
        self.assertEqual((opt["embed_workflow"][0], opt["embed_workflow"][1]["default"]), ("BOOLEAN", False))
        self.assertEqual(self.spec["hidden"], {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"})

    def test_node_attributes(self):
        cls = save.MossPhotonSaveImage
        self.assertEqual(cls.RETURN_TYPES, ())
        self.assertTrue(cls.OUTPUT_NODE)
        self.assertEqual((cls.FUNCTION, cls.CATEGORY), ("save", consts.CAT_OUTPUT))
        self.assertTrue(callable(getattr(cls, cls.FUNCTION)))
        self.assertEqual(save.NODE_DISPLAY_NAME_MAPPINGS, {"MossPhotonSaveImage": "Moss Photon Save Image"})
        self.assertIn("MossPhotonSaveImage", consts.NODE_IDS)

    def test_function_signature_accepts_every_declared_input(self):
        import inspect
        params = set(inspect.signature(save.MossPhotonSaveImage.save).parameters)
        declared = {k for group in self.spec.values() for k in group}
        self.assertLessEqual(declared, params)


if __name__ == "__main__":
    unittest.main()
