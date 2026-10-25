"""saveio.to_uint8 equals the literal native SaveImage conversion (design 1.7, 6.6)."""
import unittest

import numpy as np
import torch

from moss_photon.core import saveio


def native(image):
    """CUI/nodes.py SaveImage.save_images, lines 1707-1708, verbatim."""
    i = 255. * image.cpu().numpy()
    return np.clip(i, 0, 255).astype(np.uint8)


class ToUint8Test(unittest.TestCase):
    def test_equals_the_native_formula_on_random_images_including_out_of_range_values(self):
        gen = torch.Generator().manual_seed(11)
        for shape in ((16, 16, 3), (7, 9, 3), (8, 8, 4)):
            with self.subTest(shape=shape):
                image = torch.rand(shape, generator=gen) * 2.5 - 0.75  # -0.75 .. 1.75
                image[0, 0, 0], image[0, 1, 0], image[1, 0, 0] = -3.0, 7.0, 1.0
                got = saveio.to_uint8(image)
                self.assertEqual(got.dtype, np.uint8)
                self.assertTrue(np.array_equal(got, native(image)))
                self.assertEqual(got[0, 0, 0], 0)
                self.assertEqual(got[0, 1, 0], 255)

    def test_rounding_is_truncation_like_native_not_round_half_up(self):
        image = torch.tensor([[[0.5, 254.9 / 255.0, 1.0 / 255.0]]], dtype=torch.float32)
        got = saveio.to_uint8(image)
        self.assertEqual(got.tolist(), [[[127, 254, 1]]])
        self.assertTrue(np.array_equal(got, native(image)))

    def test_numpy_input_gives_the_same_result_as_a_tensor(self):
        image = torch.rand((5, 6, 3), generator=torch.Generator().manual_seed(2))
        self.assertTrue(np.array_equal(saveio.to_uint8(image.numpy()), saveio.to_uint8(image)))

    def test_input_is_not_modified(self):
        image = torch.rand((4, 4, 3)) * 3 - 1
        before = image.clone()
        saveio.to_uint8(image)
        self.assertTrue(torch.equal(image, before))


if __name__ == "__main__":
    unittest.main()
