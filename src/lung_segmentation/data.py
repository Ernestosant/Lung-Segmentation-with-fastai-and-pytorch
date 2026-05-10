"""Manifest-based dataset utilities."""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import resolve_path

REQUIRED_MANIFEST_COLUMNS = ("image_path", "mask_path", "source", "patient_id", "split")
VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class ManifestEntry:
    image_path: Path
    mask_path: Path
    source: str
    patient_id: str
    split: str


def load_manifest(manifest_path: str | Path, root: str | Path | None = None) -> list[ManifestEntry]:
    """Load and validate a CSV manifest."""

    base_root = Path(root) if root is not None else Path.cwd()
    manifest = resolve_path(manifest_path, base_root)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    with manifest.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [
            column
            for column in REQUIRED_MANIFEST_COLUMNS
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
        entries = [_entry_from_row(row, base_root) for row in reader if any(row.values())]

    for entry in entries:
        if entry.split not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{entry.split}' for image {entry.image_path}")
    return entries


def validate_manifest(
    entries: list[ManifestEntry],
    require_files: bool = True,
    require_test: bool = False,
) -> None:
    """Validate manifest consistency and file availability."""

    if not entries:
        raise ValueError("Manifest has no data rows.")

    if require_files:
        missing = [
            str(path)
            for entry in entries
            for path in (entry.image_path, entry.mask_path)
            if not path.exists()
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"Missing image or mask files: {preview}")

    splits = {entry.split for entry in entries}
    required = {"train", "val"} | ({"test"} if require_test else set())
    missing_splits = required - splits
    if missing_splits:
        raise ValueError(
            f"Manifest is missing required splits: {', '.join(sorted(missing_splits))}"
        )


def split_summary(entries: list[ManifestEntry]) -> dict[str, dict[str, int]]:
    """Return counts by split and source."""

    summary: dict[str, dict[str, int]] = {}
    for entry in entries:
        split_counts = summary.setdefault(entry.split, {})
        split_counts[entry.source] = split_counts.get(entry.source, 0) + 1
    return summary


def build_manifest_from_directories(
    images_dir: str | Path,
    masks_dir: str | Path,
    output_path: str | Path,
    source: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> list[ManifestEntry]:
    """Create a 70/15/15 manifest by matching image and mask filenames."""

    _validate_split_ratios(train_ratio, val_ratio)
    image_dir = resolve_path(images_dir)
    mask_dir = resolve_path(masks_dir)
    output = resolve_path(output_path)
    image_paths = sorted([path for path in image_dir.iterdir() if path.is_file()])
    mask_by_name = {path.name: path for path in mask_dir.iterdir() if path.is_file()}

    matched = [
        (image, mask_by_name[image.name])
        for image in image_paths
        if image.name in mask_by_name
    ]
    if not matched:
        raise ValueError(f"No image/mask filename pairs found in {image_dir} and {mask_dir}")

    random.Random(seed).shuffle(matched)
    n_train, n_val, _ = _split_counts(len(matched), train_ratio, val_ratio)

    rows: list[ManifestEntry] = []
    for index, (image, mask) in enumerate(matched):
        split = "train" if index < n_train else "val" if index < n_train + n_val else "test"
        rows.append(
            ManifestEntry(
                image_path=image.resolve(),
                mask_path=mask.resolve(),
                source=source,
                patient_id=image.stem,
                split=split,
            )
        )

    write_manifest(output, rows)
    return rows


def write_manifest(output_path: str | Path, entries: list[ManifestEntry]) -> None:
    """Write manifest entries to CSV with repo-relative paths when possible."""

    output = resolve_path(output_path)
    path_root = Path.cwd().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_MANIFEST_COLUMNS)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "image_path": _manifest_path(entry.image_path, root=path_root),
                    "mask_path": _manifest_path(entry.mask_path, root=path_root),
                    "source": entry.source,
                    "patient_id": entry.patient_id,
                    "split": entry.split,
                }
            )


def build_fastai_dataloaders(config: dict[str, Any]) -> Any:
    """Build fastai DataLoaders from a manifest-backed config."""

    from fastai.data.block import DataBlock
    from fastai.data.transforms import FuncSplitter, Normalize
    from fastai.vision.augment import aug_transforms
    from fastai.vision.data import ImageBlock, MaskBlock, imagenet_stats

    root = Path(config.get("_project_root", Path.cwd()))
    data_cfg = config.get("data", {})
    manifest_path = data_cfg.get("manifest_path", "data/manifest.csv")
    entries = load_manifest(manifest_path, root=root)
    validate_manifest(entries, require_files=True)

    train_val_entries = [entry for entry in entries if entry.split in {"train", "val"}]
    image_to_mask = {str(entry.image_path): entry.mask_path for entry in train_val_entries}
    image_to_split = {str(entry.image_path): entry.split for entry in train_val_entries}
    items = [entry.image_path for entry in train_val_entries]

    codes = data_cfg.get("codes", ["background", "lung"])
    image_size = int(data_cfg.get("image_size", 128))
    batch_size = int(data_cfg.get("batch_size", 4))
    num_workers = int(data_cfg.get("num_workers", 0))

    datablock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes)),
        get_items=lambda _: items,
        splitter=FuncSplitter(lambda item: image_to_split[str(item)] == "val"),
        get_y=lambda item: image_to_mask[str(item)],
        batch_tfms=[
            *aug_transforms(size=(image_size, image_size)),
            Normalize.from_stats(*imagenet_stats),
        ],
    )
    dataloaders = datablock.dataloaders(root, bs=batch_size, num_workers=num_workers)
    dataloaders.vocab = codes
    return dataloaders


def _entry_from_row(row: dict[str, str], root: Path) -> ManifestEntry:
    return ManifestEntry(
        image_path=resolve_path(row["image_path"], root).resolve(),
        mask_path=resolve_path(row["mask_path"], root).resolve(),
        source=(row.get("source") or "unknown").strip() or "unknown",
        patient_id=(row.get("patient_id") or "").strip(),
        split=(row.get("split") or "").strip().lower(),
    )


def _validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    if not 0 <= train_ratio <= 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio <= 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1 to leave a test split.")


def _split_counts(n_total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n_total < 3:
        raise ValueError(
            "At least 3 matched image/mask pairs are required for train/val/test splits."
        )

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    counts = {"train": n_train, "val": n_val, "test": n_test}

    for split in ("train", "val", "test"):
        if counts[split] == 0:
            donor = max(counts, key=counts.get)
            if counts[donor] <= 1:
                raise ValueError("Could not create non-empty train/val/test splits.")
            counts[donor] -= 1
            counts[split] = 1

    return counts["train"], counts["val"], counts["test"]


def _manifest_path(path: Path, root: Path) -> str:
    value = Path(path)
    if not value.is_absolute():
        return value.as_posix()

    try:
        relative = value.resolve().relative_to(root)
    except ValueError:
        try:
            relative = Path(os.path.relpath(value, root))
        except ValueError:
            return str(value)

    return relative.as_posix()
