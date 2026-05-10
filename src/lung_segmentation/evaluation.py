"""Evaluation utilities for manifest-backed test sets."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .config import resolve_path
from .data import load_manifest, validate_manifest
from .infer import load_learner_cached, predict_mask_array
from .metrics import dice_score, iou_score, pixel_accuracy
from .preprocessing import read_image, read_mask


def evaluate_from_config(
    config: dict[str, Any],
    checkpoint: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate a model checkpoint on one manifest split."""

    root = Path(config.get("_project_root", Path.cwd()))
    data_cfg = config.get("data", {})
    artifacts_cfg = config.get("artifacts", {})
    manifest_path = data_cfg.get("manifest_path", "data/manifest.csv")

    entries = load_manifest(manifest_path, root=root)
    validate_manifest(entries, require_files=True, require_test=(split == "test"))
    selected = [entry for entry in entries if entry.split == split]
    if not selected:
        raise ValueError(f"No rows found for split '{split}' in {manifest_path}")

    learner = load_learner_cached(str(resolve_path(checkpoint)))
    rows = []
    mask_threshold = int(config.get("preprocessing", {}).get("mask_threshold", 127))
    equalize = bool(config.get("preprocessing", {}).get("equalize_histogram", True))

    for entry in selected:
        image = read_image(entry.image_path)
        target = read_mask(entry.mask_path, threshold=mask_threshold) * 255
        prediction = predict_mask_array(learner, image, equalize=equalize)
        rows.append(
            {
                "image_path": str(entry.image_path),
                "mask_path": str(entry.mask_path),
                "source": entry.source,
                "patient_id": entry.patient_id,
                "split": entry.split,
                "dice": dice_score(prediction, target),
                "iou": iou_score(prediction, target),
                "pixel_accuracy": pixel_accuracy(prediction, target),
            }
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(resolve_path(checkpoint)),
        "split": split,
        "n_images": len(rows),
        "global": _aggregate(rows),
        "by_source": {
            source: _aggregate([row for row in rows if row["source"] == source])
            for source in sorted({row["source"] for row in rows})
        },
    }

    metrics_dir = resolve_path(artifacts_cfg.get("metrics_dir", "artifacts/metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    name = f"{Path(checkpoint).stem}_{split}"
    csv_path = metrics_dir / f"{name}_per_image.csv"
    json_path = metrics_dir / f"{name}_summary.json"
    _write_rows(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["per_image_csv"] = str(csv_path)
    summary["summary_json"] = str(json_path)
    return summary


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "dice": mean(row["dice"] for row in rows),
        "iou": mean(row["iou"] for row in rows),
        "pixel_accuracy": mean(row["pixel_accuracy"] for row in rows),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
