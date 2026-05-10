"""Training orchestration for fastai U-Net models."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_path
from .data import build_fastai_dataloaders
from .metrics import make_fastai_metrics


def train_from_config(config: dict[str, Any]) -> Path:
    """Train and export a fastai learner from a project config."""

    import torch
    from fastai.layers import Mish
    from fastai.optimizer import ranger
    from fastai.vision.learner import unet_learner

    seed = int(config.get("project", {}).get("seed", 42))
    _set_seed(seed)

    dataloaders = build_fastai_dataloaders(config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    artifacts_cfg = config.get("artifacts", {})

    learner = unet_learner(
        dataloaders,
        _encoder(model_cfg.get("encoder", "resnet34")),
        metrics=make_fastai_metrics(data_cfg.get("ignore_index")),
        self_attention=bool(model_cfg.get("self_attention", True)),
        act_cls=Mish if model_cfg.get("activation", "mish").lower() == "mish" else None,
        opt_func=ranger,
    )

    if bool(model_cfg.get("mixed_precision", False)) and torch.cuda.is_available():
        learner = learner.to_fp16()

    frozen_epochs = int(training_cfg.get("frozen_epochs", 0))
    if frozen_epochs > 0:
        learner.fit_one_cycle(frozen_epochs, float(training_cfg.get("frozen_lr", 0.003)))

    unfrozen_epochs = int(training_cfg.get("unfrozen_epochs", 0))
    if unfrozen_epochs > 0:
        learner.unfreeze()
        lr_low = float(training_cfg.get("unfrozen_lr_low", 1e-5))
        lr_high = float(training_cfg.get("unfrozen_lr_high", 1e-3))
        learner.fit_one_cycle(unfrozen_epochs, lr_max=slice(lr_low, lr_high))

    final_epochs = int(training_cfg.get("final_epochs", 0))
    if final_epochs > 0:
        learner.fit_one_cycle(final_epochs, float(training_cfg.get("final_lr", 4e-4)))

    output_dir = resolve_path(artifacts_cfg.get("output_dir", "artifacts/models"))
    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = output_dir / artifacts_cfg.get("export_name", "model.pkl")
    learner.export(str(export_path))
    return export_path


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _encoder(name: str) -> Any:
    from fastai.vision.all import resnet18, resnet34, resnet50

    encoders = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
    }
    key = name.lower()
    if key not in encoders:
        raise ValueError(f"Unsupported encoder '{name}'. Choose one of: {', '.join(encoders)}")
    return encoders[key]
