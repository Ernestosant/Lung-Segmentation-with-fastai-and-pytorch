"""Build a CSV manifest by matching image and mask filenames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create data/manifest.csv from image and mask folders."
    )
    parser.add_argument("--images", required=True, help="Directory containing X-ray images.")
    parser.add_argument("--masks", required=True, help="Directory containing matching masks.")
    parser.add_argument("--source", required=True, help="Dataset/source label, e.g. montgomery.")
    parser.add_argument("--out", default="data/manifest.csv", help="Output manifest path.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from lung_segmentation.data import build_manifest_from_directories, split_summary

    entries = build_manifest_from_directories(
        images_dir=args.images,
        masks_dir=args.masks,
        output_path=args.out,
        source=args.source,
        seed=args.seed,
    )
    print(f"Wrote {len(entries)} rows to {args.out}")
    print(split_summary(entries))


if __name__ == "__main__":
    main()
