from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
