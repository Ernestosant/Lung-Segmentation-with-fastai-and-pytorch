"""Run Kaggle-oriented training, evaluation, and qualitative reporting."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from lung_segmentation.config import load_config, resolve_path
from lung_segmentation.data import load_manifest, split_summary, validate_manifest
from lung_segmentation.evaluation import evaluate_from_config
from lung_segmentation.infer import load_learner_cached, predict_mask_array
from lung_segmentation.preprocessing import overlay_mask, read_image, read_mask
from lung_segmentation.training import train_from_config
from prepare_kaggle_dataset import prepare_manifest

MODEL_CONFIGS = {
    "resnet18": "configs/resnet18.yaml",
    "resnet34": "configs/resnet34.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate configured U-Net models and write result artifacts."
    )
    parser.add_argument("--models", nargs="+", default=["resnet18", "resnet34"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--input-dir",
        help="Optional Kaggle dataset directory. If set, data/manifest.csv is rebuilt first.",
    )
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--publish-docs",
        action="store_true",
        help="Update segmentation.PNG and docs/results.md from generated outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    os.chdir(ROOT)
    args = build_parser().parse_args(argv)

    if args.input_dir:
        prepare_manifest(args.input_dir, args.manifest)

    manifest_entries = load_manifest(args.manifest, root=ROOT)
    validate_manifest(manifest_entries, require_files=True, require_test=args.split == "test")
    summaries: dict[str, dict[str, Any]] = {}
    figure_paths: dict[str, Path] = {}

    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            supported = ", ".join(sorted(MODEL_CONFIGS))
            raise ValueError(f"Unsupported model '{model_name}'. Choose from: {supported}")

        config = load_config(MODEL_CONFIGS[model_name])
        config["data"]["manifest_path"] = args.manifest
        checkpoint = _checkpoint_path(config)
        if not args.skip_training:
            checkpoint = train_from_config(config)
        elif not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for --skip-training: {checkpoint}. "
                "Run without --skip-training first or place the model in artifacts/models/."
            )

        summary = evaluate_from_config(config, checkpoint, split=args.split)
        summary = _copy_metric_outputs(model_name, summary)
        summaries[model_name] = summary

        if not args.skip_figures:
            figure_path = resolve_path(
                f"artifacts/predictions/{model_name}_{args.split}_examples.png"
            )
            _write_example_grid(
                config,
                checkpoint,
                split=args.split,
                output_path=figure_path,
                n_examples=args.examples,
            )
            figure_paths[model_name] = figure_path

    table_path = resolve_path("artifacts/metrics/results_table.md")
    _write_results_table(table_path, summaries)
    report_path = resolve_path("artifacts/metrics/model_results.md")
    _write_results_report(report_path, summaries, figure_paths, manifest_entries)

    if args.publish_docs:
        _publish_docs(summaries, figure_paths, manifest_entries)

    print(f"Results table: {table_path}")
    print(f"Results report: {report_path}")


def _checkpoint_path(config: dict[str, Any]) -> Path:
    artifacts_cfg = config.get("artifacts", {})
    output_dir = resolve_path(artifacts_cfg.get("output_dir", "artifacts/models"))
    return output_dir / artifacts_cfg.get("export_name", "model.pkl")


def _copy_metric_outputs(model_name: str, summary: dict[str, Any]) -> dict[str, Any]:
    metrics_dir = resolve_path("artifacts/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    split = summary["split"]

    summary_json = metrics_dir / f"{model_name}_{split}_summary.json"
    per_image_csv = metrics_dir / f"{model_name}_{split}_per_image.csv"

    source_json = Path(summary["summary_json"])
    source_csv = Path(summary["per_image_csv"])
    if source_json.resolve() != summary_json.resolve():
        shutil.copyfile(source_json, summary_json)
    if source_csv.resolve() != per_image_csv.resolve():
        shutil.copyfile(source_csv, per_image_csv)

    summary["summary_json"] = str(summary_json)
    summary["per_image_csv"] = str(per_image_csv)
    return summary


def _write_example_grid(
    config: dict[str, Any],
    checkpoint: str | Path,
    *,
    split: str,
    output_path: Path,
    n_examples: int,
) -> None:
    entries = [
        entry
        for entry in load_manifest(config["data"]["manifest_path"], root=ROOT)
        if entry.split == split
    ]
    if not entries:
        raise ValueError(f"No rows found for split '{split}'.")

    selected = sorted(entries, key=lambda entry: entry.patient_id)[:n_examples]
    learner = load_learner_cached(str(resolve_path(checkpoint)))
    mask_threshold = int(config.get("preprocessing", {}).get("mask_threshold", 127))
    equalize = bool(config.get("preprocessing", {}).get("equalize_histogram", True))

    rows = []
    for entry in selected:
        image = read_image(entry.image_path)
        target = read_mask(entry.mask_path, threshold=mask_threshold) * 255
        prediction = predict_mask_array(learner, image, equalize=equalize)
        rows.append(
            [
                _panel(image, "X-ray"),
                _panel(_mask_to_rgb(target), "Ground Truth"),
                _panel(_mask_to_rgb(prediction), "Prediction"),
                _panel(overlay_mask(image, prediction), "Overlay"),
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_grid(rows, output_path)


def _panel(array: np.ndarray, title: str, size: int = 256, label_height: int = 34) -> Image.Image:
    image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + label_height), "white")
    x = (size - image.width) // 2
    y = label_height + (size - image.height) // 2
    canvas.paste(image, (x, y))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    text_width = draw.textlength(title, font=font)
    draw.text(((size - text_width) / 2, 10), title, fill=(20, 20, 20), font=font)
    return canvas


def _save_grid(rows: list[list[Image.Image]], output_path: Path, gutter: int = 14) -> None:
    panel_width, panel_height = rows[0][0].size
    width = len(rows[0]) * panel_width + (len(rows[0]) - 1) * gutter
    height = len(rows) * panel_height + (len(rows) - 1) * gutter
    canvas = Image.new("RGB", (width, height), (248, 250, 252))
    for row_index, row in enumerate(rows):
        for col_index, panel in enumerate(row):
            x = col_index * (panel_width + gutter)
            y = row_index * (panel_height + gutter)
            canvas.paste(panel, (x, y))
    canvas.save(output_path)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    clean = np.asarray(mask)
    if clean.max(initial=0) <= 1:
        clean = clean * 255
    clean = np.clip(clean, 0, 255).astype(np.uint8)
    return np.stack([clean, clean, clean], axis=-1)


def _write_results_table(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    lines = [
        "| Model | Split | Dice | IoU | Pixel Accuracy |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model_name, summary in summaries.items():
        metrics = summary["global"]
        lines.append(
            f"| U-Net {model_name.title()} | {summary['split']} | "
            f"{metrics['dice']:.4f} | {metrics['iou']:.4f} | "
            f"{metrics['pixel_accuracy']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_results_report(
    path: Path,
    summaries: dict[str, dict[str, Any]],
    figure_paths: dict[str, Path],
    entries: list[Any],
) -> None:
    lines = [
        "# Model Results",
        "",
        "Generated by `scripts/kaggle_reproduce.py`.",
        "",
        "## Dataset",
        "",
        "- Source: Kaggle `nikhilpandey360/chest-xray-masks-and-labels`.",
        f"- Split counts: `{split_summary(entries)}`.",
        "",
        "## Metrics",
        "",
    ]
    table_path = resolve_path("artifacts/metrics/results_table.md")
    lines.append(table_path.read_text(encoding="utf-8").strip())
    if figure_paths:
        lines.extend(["", "## Qualitative Examples", ""])
        for model_name, figure_path in figure_paths.items():
            lines.append(f"- {model_name}: `{_display_path(figure_path)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _publish_docs(
    summaries: dict[str, dict[str, Any]],
    figure_paths: dict[str, Path],
    entries: list[Any],
) -> None:
    docs_results = resolve_path("docs/results.md")
    _write_results_report(docs_results, summaries, figure_paths, entries)

    if figure_paths:
        preferred = figure_paths.get("resnet34") or next(iter(figure_paths.values()))
        shutil.copyfile(preferred, resolve_path("segmentation.PNG"))


def _display_path(path: str | Path) -> str:
    value = Path(path).resolve()
    try:
        return value.relative_to(ROOT).as_posix()
    except ValueError:
        return str(value)


if __name__ == "__main__":
    main()
