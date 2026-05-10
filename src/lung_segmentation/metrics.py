"""Segmentation metrics for binary lung masks."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_binary_array(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim > 2:
        array = np.squeeze(array)
    if array.max(initial=0) > 1:
        array = array > 127
    return array.astype(bool)


def dice_score(prediction: Any, target: Any, eps: float = 1e-7) -> float:
    """Compute Dice score for binary masks."""

    pred = _as_binary_array(prediction)
    truth = _as_binary_array(target)
    intersection = np.logical_and(pred, truth).sum()
    denominator = pred.sum() + truth.sum()
    if denominator == 0:
        return 1.0
    return float((2 * intersection + eps) / (denominator + eps))


def iou_score(prediction: Any, target: Any, eps: float = 1e-7) -> float:
    """Compute intersection over union for binary masks."""

    pred = _as_binary_array(prediction)
    truth = _as_binary_array(target)
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred, truth).sum()
    return float((intersection + eps) / (union + eps))


def pixel_accuracy(prediction: Any, target: Any) -> float:
    """Compute pixel accuracy for binary masks."""

    pred = _as_binary_array(prediction)
    truth = _as_binary_array(target)
    return float((pred == truth).mean())


def fastai_dice(inp: Any, targ: Any) -> Any:
    """Fastai-compatible foreground Dice metric."""

    return _fastai_dice_impl(inp, targ)


def fastai_iou(inp: Any, targ: Any) -> Any:
    """Fastai-compatible foreground IoU metric."""

    return _fastai_iou_impl(inp, targ)


def fastai_pixel_accuracy(inp: Any, targ: Any) -> Any:
    """Fastai-compatible pixel accuracy metric."""

    return _fastai_pixel_accuracy_impl(inp, targ)


def make_fastai_metrics(ignore_index: int | None = None) -> list[Any]:
    """Create fastai metrics that optionally ignore one class index."""

    def dice(inp: Any, targ: Any) -> Any:
        return _fastai_dice_impl(inp, targ, ignore_index=ignore_index)

    def iou(inp: Any, targ: Any) -> Any:
        return _fastai_iou_impl(inp, targ, ignore_index=ignore_index)

    def acc(inp: Any, targ: Any) -> Any:
        return _fastai_pixel_accuracy_impl(inp, targ, ignore_index=ignore_index)

    dice.__name__ = "dice"
    iou.__name__ = "iou"
    acc.__name__ = "pixel_accuracy"
    return [dice, iou, acc]


def _fastai_dice_impl(inp: Any, targ: Any, ignore_index: int | None = None) -> Any:
    import torch

    pred, target = _torch_prediction_and_target(inp, targ, ignore_index=ignore_index)
    pred = pred == 1
    target = target == 1
    intersection = (pred & target).float().sum()
    denominator = pred.float().sum() + target.float().sum()
    if denominator == 0:
        return torch.tensor(1.0, device=inp.device)
    return (2 * intersection + 1e-7) / (denominator + 1e-7)


def _fastai_iou_impl(inp: Any, targ: Any, ignore_index: int | None = None) -> Any:
    import torch

    pred, target = _torch_prediction_and_target(inp, targ, ignore_index=ignore_index)
    pred = pred == 1
    target = target == 1
    union = (pred | target).float().sum()
    if union == 0:
        return torch.tensor(1.0, device=inp.device)
    intersection = (pred & target).float().sum()
    return (intersection + 1e-7) / (union + 1e-7)


def _fastai_pixel_accuracy_impl(inp: Any, targ: Any, ignore_index: int | None = None) -> Any:
    import torch

    pred, target = _torch_prediction_and_target(inp, targ, ignore_index=ignore_index)
    if target.numel() == 0:
        return torch.tensor(1.0, device=inp.device)
    return (pred == target).float().mean()


def _torch_prediction_and_target(
    inp: Any, targ: Any, ignore_index: int | None = None
) -> tuple[Any, Any]:
    pred = inp.argmax(dim=1) if getattr(inp, "ndim", 0) == 4 else inp
    target = targ.squeeze(1) if getattr(targ, "ndim", 0) == 4 else targ
    if ignore_index is not None:
        valid = target != ignore_index
        pred = pred[valid]
        target = target[valid]
    return pred, target
