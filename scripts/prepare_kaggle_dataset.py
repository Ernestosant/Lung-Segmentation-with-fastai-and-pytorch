"""Prepare a manifest from the public Kaggle chest X-ray mask dataset.

Expected Kaggle dataset:
https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lung_segmentation.data import ManifestEntry, split_summary, write_manifest

IMAGE_DIR_NAMES = {"cxr_png", "cxr-png", "image", "images"}
MASK_DIR_NAMES = {"mask", "masks", "lung masks", "lung_mask", "lung_masks"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build data/manifest.csv from Kaggle chest X-ray images and masks."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Extracted Kaggle dataset directory, e.g. /kaggle/input/chest-xray-masks-and-labels.",
    )
    parser.add_argument("--out", default="data/manifest.csv", help="Output manifest CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--preprocess-size",
        type=int,
        default=256,
        help="Write resized image copies and binary masks under data/processed. Use 0 to disable.",
    )
    return parser


def prepare_manifest(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    preprocess_size: int | None = None,
) -> list[ManifestEntry]:
    """Create manifest entries from the Kaggle folder and write them to CSV."""

    root = Path(input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Kaggle input directory not found: {root}")

    image_paths = _find_images(root)
    mask_paths = _find_masks(root)
    pairs = _match_image_mask_pairs(image_paths, mask_paths)
    if not pairs:
        raise ValueError(
            "No image/mask pairs were found. Expected folders like CXR_png/ and masks/."
        )

    if preprocess_size and preprocess_size > 0:
        pairs = _preprocess_pairs(pairs, Path(output_path), preprocess_size)

    random.Random(seed).shuffle(pairs)
    split_by_stem = _assign_splits([image.stem for image, _ in pairs], train_ratio, val_ratio)

    entries = [
        ManifestEntry(
            image_path=image,
            mask_path=mask,
            source=_source_from_stem(image.stem),
            patient_id=image.stem,
            split=split_by_stem[image.stem],
        )
        for image, mask in pairs
    ]
    write_manifest(output_path, entries)
    return entries


def _find_images(root: Path) -> list[Path]:
    image_dirs = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.lower() in IMAGE_DIR_NAMES
    ]
    if not image_dirs:
        image_dirs = [root]
    return sorted(
        path.resolve()
        for directory in image_dirs
        for path in directory.iterdir()
        if path.is_file() and _is_image_file(path) and "_mask" not in path.stem.lower()
    )


def _find_masks(root: Path) -> list[Path]:
    mask_dirs = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.lower() in MASK_DIR_NAMES
    ]
    if not mask_dirs:
        mask_dirs = [root]
    return sorted(
        path.resolve()
        for directory in mask_dirs
        for path in directory.iterdir()
        if path.is_file() and _is_image_file(path)
    )


def _match_image_mask_pairs(
    image_paths: list[Path],
    mask_paths: list[Path],
) -> list[tuple[Path, Path]]:
    images_by_stem = {path.stem: path for path in image_paths}
    pairs = []
    for mask in mask_paths:
        image_stem = _image_stem_from_mask(mask)
        image = images_by_stem.get(image_stem)
        if image is not None:
            pairs.append((image, mask))
    return sorted(pairs, key=lambda pair: pair[0].stem)


def _preprocess_pairs(
    pairs: list[tuple[Path, Path]],
    output_path: Path,
    size: int,
) -> list[tuple[Path, Path]]:
    processed_root = output_path.parent / "processed" / f"kaggle_chest_xray_masks_{size}"
    image_dir = processed_root / "images"
    mask_dir = processed_root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    processed_pairs = []
    for image_path, mask_path in pairs:
        processed_image = image_dir / image_path.name
        processed_mask = mask_dir / mask_path.name
        _write_resized_image(image_path, processed_image, size)
        _write_binary_mask(mask_path, processed_mask, size)
        processed_pairs.append((processed_image.resolve(), processed_mask.resolve()))
    return processed_pairs


def _write_resized_image(input_path: Path, output_path: Path, size: int) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), resized)


def _write_binary_mask(input_path: Path, output_path: Path, size: int) -> None:
    mask = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {input_path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask.astype("uint8"), 127, 1, cv2.THRESH_BINARY)
    binary = cv2.resize(binary, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(output_path), binary.astype("uint8"))


def _image_stem_from_mask(mask_path: Path) -> str:
    stem = mask_path.stem
    for suffix in ("_mask", "-mask", "_lung", "-lung"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _assign_splits(
    stems: list[str],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    if not 0 <= train_ratio <= 1 or not 0 <= val_ratio <= 1:
        raise ValueError("train_ratio and val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must leave room for a test split.")
    if len(stems) < 3:
        raise ValueError("At least 3 matched image/mask pairs are required.")

    n_train = int(len(stems) * train_ratio)
    n_val = int(len(stems) * val_ratio)
    n_test = len(stems) - n_train - n_val
    counts = {"train": n_train, "val": n_val, "test": n_test}

    for split in ("train", "val", "test"):
        if counts[split] == 0:
            donor = max(counts, key=counts.get)
            if counts[donor] <= 1:
                raise ValueError("Could not create non-empty train/val/test splits.")
            counts[donor] -= 1
            counts[split] = 1

    split_labels = (
        ["train"] * counts["train"]
        + ["val"] * counts["val"]
        + ["test"] * counts["test"]
    )
    return dict(zip(stems, split_labels, strict=True))


def _source_from_stem(stem: str) -> str:
    if stem.startswith("CHNCXR"):
        return "shenzhen"
    if stem.startswith("MCUCXR"):
        return "montgomery"
    return "kaggle_chest_xray_masks"


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    entries = prepare_manifest(
        args.input_dir,
        args.out,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        preprocess_size=args.preprocess_size,
    )
    print(f"Wrote {len(entries)} matched image/mask rows to {args.out}")
    print(split_summary(entries))


if __name__ == "__main__":
    main()
