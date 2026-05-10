from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from lung_segmentation.data import (
    ManifestEntry,
    build_manifest_from_directories,
    load_manifest,
    split_summary,
    validate_manifest,
    write_manifest,
)


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

    def test_write_manifest_uses_relative_paths_when_possible(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            image = root / "image.png"
            mask = root / "mask.png"
            image.write_bytes(b"fake")
            mask.write_bytes(b"fake")
            manifest = root / "manifest.csv"

            write_manifest(
                manifest,
                [
                    ManifestEntry(
                        image_path=image,
                        mask_path=mask,
                        source="montgomery",
                        patient_id="p001",
                        split="train",
                    )
                ],
            )

            content = manifest.read_text(encoding="utf-8")
            self.assertIn("image.png", content)
            self.assertNotIn(str(root), content)

    def test_build_manifest_validates_ratios_and_non_empty_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()

            for index in range(3):
                (images / f"{index}.png").write_bytes(b"fake")
                (masks / f"{index}.png").write_bytes(b"fake")

            entries = build_manifest_from_directories(
                images,
                masks,
                root / "manifest.csv",
                source="montgomery",
                train_ratio=0.70,
                val_ratio=0.15,
            )

            self.assertEqual({entry.split for entry in entries}, {"train", "val", "test"})

            with self.assertRaises(ValueError):
                build_manifest_from_directories(
                    images,
                    masks,
                    root / "invalid.csv",
                    source="montgomery",
                    train_ratio=0.90,
                    val_ratio=0.10,
                )


if __name__ == "__main__":
    unittest.main()
