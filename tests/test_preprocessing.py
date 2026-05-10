from __future__ import annotations

import unittest

import numpy as np

from lung_segmentation.preprocessing import (
    apply_mask,
    binarize_mask,
    equalize_histogram_rgb,
    overlay_mask,
    postprocess_mask,
)


class PreprocessingTests(unittest.TestCase):
    def test_binarize_mask_returns_class_ids(self) -> None:
        mask = np.array([[0, 127], [128, 255]], dtype=np.uint8)
        binary = binarize_mask(mask, threshold=127)

        np.testing.assert_array_equal(binary, np.array([[0, 0], [1, 1]], dtype=np.uint8))

    def test_binarize_mask_respects_opencv_bgr_channel_order(self) -> None:
        blue_in_bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)

        bgr_binary = binarize_mask(blue_in_bgr, threshold=50, channel_order="BGR")
        rgb_binary = binarize_mask(blue_in_bgr, threshold=50, channel_order="RGB")

        np.testing.assert_array_equal(bgr_binary, np.array([[0]], dtype=np.uint8))
        np.testing.assert_array_equal(rgb_binary, np.array([[1]], dtype=np.uint8))

    def test_equalize_histogram_preserves_rgb_shape(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, 4:] = 180

        equalized = equalize_histogram_rgb(image)

        self.assertEqual(equalized.shape, image.shape)
        self.assertEqual(equalized.dtype, np.uint8)

    def test_postprocess_mask_resizes_to_requested_shape(self) -> None:
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        resized = postprocess_mask(mask, output_shape=(6, 4), close_kernel=0, blur_kernel=0)

        self.assertEqual(resized.shape, (6, 4))
        self.assertEqual(set(np.unique(resized)).issubset({0, 255}), True)

    def test_overlay_and_segmented_outputs_match_input_shape(self) -> None:
        image = np.full((4, 4, 3), 100, dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:, :2] = 255

        self.assertEqual(overlay_mask(image, mask).shape, image.shape)
        self.assertEqual(apply_mask(image, mask).shape, image.shape)


if __name__ == "__main__":
    unittest.main()
