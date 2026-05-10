"""Prediction CLI."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lung segmentation inference on one image.")
    parser.add_argument("--checkpoint", required=True, help="Path to exported fastai .pkl model.")
    parser.add_argument("--image", required=True, help="Path to the input chest X-ray image.")
    parser.add_argument("--out", default="artifacts/predictions", help="Output directory.")
    parser.add_argument(
        "--no-equalize",
        action="store_true",
        help="Disable histogram equalization.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from lung_segmentation.infer import predict_image_file

    paths = predict_image_file(
        args.checkpoint,
        args.image,
        args.out,
        equalize=not args.no_equalize,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
