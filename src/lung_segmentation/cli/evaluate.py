"""Evaluation CLI."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a lung segmentation checkpoint.")
    parser.add_argument("--config", default="configs/resnet34.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to exported fastai .pkl model.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from lung_segmentation.config import load_config
    from lung_segmentation.evaluation import evaluate_from_config

    summary = evaluate_from_config(load_config(args.config), args.checkpoint, split=args.split)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
