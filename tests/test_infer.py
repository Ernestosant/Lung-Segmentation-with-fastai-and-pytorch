from __future__ import annotations

import unittest

import numpy as np

from lung_segmentation.infer import prediction_to_mask


class InferenceTests(unittest.TestCase):
    def test_prediction_to_mask_argmaxes_channel_dimension(self) -> None:
        prediction = np.zeros((2, 3, 3), dtype=np.float32)
        prediction[1, 1:, :] = 0.9
        prediction[0, :1, :] = 0.9

        mask = prediction_to_mask(prediction)

        self.assertEqual(mask.shape, (3, 3))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(set(np.unique(mask)).issubset({0, 255}), True)
        self.assertEqual(mask[2, 2], 255)


if __name__ == "__main__":
    unittest.main()
