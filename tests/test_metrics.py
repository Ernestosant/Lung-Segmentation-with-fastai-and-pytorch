from __future__ import annotations

import unittest

import numpy as np

from lung_segmentation.metrics import dice_score, iou_score, pixel_accuracy


class MetricTests(unittest.TestCase):
    def test_binary_metrics_match_expected_values(self) -> None:
        prediction = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        target = np.array([[1, 0], [1, 0]], dtype=np.uint8)

        self.assertAlmostEqual(dice_score(prediction, target), 0.5)
        self.assertAlmostEqual(iou_score(prediction, target), 1 / 3)
        self.assertAlmostEqual(pixel_accuracy(prediction, target), 0.5)

    def test_empty_masks_are_perfect_match(self) -> None:
        prediction = np.zeros((4, 4), dtype=np.uint8)
        target = np.zeros((4, 4), dtype=np.uint8)

        self.assertEqual(dice_score(prediction, target), 1.0)
        self.assertEqual(iou_score(prediction, target), 1.0)


if __name__ == "__main__":
    unittest.main()
