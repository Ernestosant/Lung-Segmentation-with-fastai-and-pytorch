"""Training CLI."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a lung segmentation model.")
    parser.add_argument("--config", default="configs/resnet34.yaml", help="Path to YAML config.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from lung_segmentation.config import load_config
    from lung_segmentation.training import train_from_config

    export_path = train_from_config(load_config(args.config))
    print(f"Exported model: {export_path}")


if __name__ == "__main__":
    main()
