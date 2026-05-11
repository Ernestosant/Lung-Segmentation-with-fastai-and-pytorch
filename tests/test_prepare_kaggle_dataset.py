from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from lung_segmentation.preprocessing import read_mask
from scripts.prepare_kaggle_dataset import prepare_manifest


class KaggleManifestPreparationTests(unittest.TestCase):
    def test_matches_shenzhen_and_montgomery_mask_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "Lung Segmentation" / "CXR_png"
            masks = root / "Lung Segmentation" / "masks"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)

            for stem, mask_name in [
                ("CHNCXR_0001_0", "CHNCXR_0001_0_mask.png"),
                ("MCUCXR_0001_0", "MCUCXR_0001_0.png"),
                ("CHNCXR_0002_0", "CHNCXR_0002_0_mask.png"),
            ]:
                (images / f"{stem}.png").write_bytes(b"fake-image")
                (masks / mask_name).write_bytes(b"fake-mask")

            manifest = root / "manifest.csv"
            entries = prepare_manifest(root, manifest, seed=42, preprocess_size=None)

            self.assertEqual(len(entries), 3)
            self.assertEqual({entry.source for entry in entries}, {"shenzhen", "montgomery"})
            self.assertEqual({entry.split for entry in entries}, {"train", "val", "test"})
            self.assertIn("image_path,mask_path,source,patient_id,split", manifest.read_text())

    def test_preprocessed_masks_remain_threshold_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "Lung Segmentation" / "CXR_png"
            masks = root / "Lung Segmentation" / "masks"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)

            image = np.full((4, 4, 3), 128, dtype=np.uint8)
            mask = np.array(
                [
                    [0, 0, 255, 255],
                    [0, 0, 255, 255],
                    [0, 0, 255, 255],
                    [0, 0, 255, 255],
                ],
                dtype=np.uint8,
            )

            for index in range(3):
                stem = f"CHNCXR_{index:04d}_0"
                cv2.imwrite(str(images / f"{stem}.png"), image)
                cv2.imwrite(str(masks / f"{stem}_mask.png"), mask)

            manifest = root / "manifest.csv"
            entries = prepare_manifest(root, manifest, seed=42, preprocess_size=8)

            loaded_mask = read_mask(entries[0].mask_path)
            self.assertEqual(set(np.unique(loaded_mask)).issubset({0, 1}), True)
            self.assertGreater(int(loaded_mask.sum()), 0)


if __name__ == "__main__":
    unittest.main()
