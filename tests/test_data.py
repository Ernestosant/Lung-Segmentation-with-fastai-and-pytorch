from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from lung_segmentation.data import load_manifest, split_summary, validate_manifest


class ManifestTests(unittest.TestCase):
    def test_manifest_loads_required_columns_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            mask = root / "mask.png"
            image.write_bytes(b"fake")
            mask.write_bytes(b"fake")
            manifest = root / "manifest.csv"

            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["image_path", "mask_path", "source", "patient_id", "split"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": "image.png",
                        "mask_path": "mask.png",
                        "source": "montgomery",
                        "patient_id": "p001",
                        "split": "train",
                    }
                )
                writer.writerow(
                    {
                        "image_path": "image.png",
                        "mask_path": "mask.png",
                        "source": "montgomery",
                        "patient_id": "p002",
                        "split": "val",
                    }
                )

            entries = load_manifest(manifest, root=root)
            validate_manifest(entries, require_files=True)

            self.assertEqual(len(entries), 2)
            self.assertEqual(
                split_summary(entries),
                {"train": {"montgomery": 1}, "val": {"montgomery": 1}},
            )

    def test_manifest_requires_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            manifest.write_text("image_path,mask_path\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_manifest(manifest, root=tmp)


if __name__ == "__main__":
    unittest.main()
