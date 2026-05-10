"""Model loading and inference utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_path
from .metrics import fastai_pixel_accuracy
from .preprocessing import (
    apply_mask,
    ensure_uint8_rgb,
    equalize_histogram_rgb,
    overlay_mask,
    postprocess_mask,
    read_image,
    write_mask,
    write_rgb,
)


def register_legacy_fastai_symbols() -> None:
    """Register symbols required by legacy exported fastai learners."""

    import __main__

    def get_msk(obj: Any | None = None) -> Any:
        return True if obj is None else obj

    def acc_camvid(inp: Any | None = None, targ: Any | None = None) -> Any:
        if inp is None or targ is None:
            return True
        return fastai_pixel_accuracy(inp, targ)

    __main__.get_msk = get_msk
    __main__.acc_camvid = acc_camvid


@lru_cache(maxsize=4)
def load_learner_cached(checkpoint: str, cpu: bool = True) -> Any:
    """Load and cache a fastai exported learner."""

    register_legacy_fastai_symbols()
    from fastai.learner import load_learner

    checkpoint_path = resolve_path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    return load_learner(checkpoint_path, cpu=cpu)


def list_model_checkpoints(model_dir: str | Path) -> list[Path]:
    """List available `.pkl` model checkpoints."""

    directory = resolve_path(model_dir)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.pkl"))


def predict_mask_array(
    learner: Any,
    image: np.ndarray,
    *,
    equalize: bool = True,
    clean: bool = True,
) -> np.ndarray:
    """Predict a binary 0/255 lung mask for an RGB image array."""

    from fastai.vision.core import PILImage

    rgb = ensure_uint8_rgb(image)
    model_input = equalize_histogram_rgb(rgb) if equalize else rgb
    dataloader = learner.dls.test_dl([PILImage.create(model_input)])
    predictions = learner.get_preds(dl=dataloader)
    raw_mask = prediction_to_mask(predictions[0][0])
    if clean:
        return postprocess_mask(raw_mask, output_shape=rgb.shape[:2])
    return raw_mask


def prediction_to_mask(prediction: Any) -> np.ndarray:
    """Convert model output into a 0/255 uint8 mask."""

    if hasattr(prediction, "detach"):
        prediction = prediction.detach().cpu().numpy()
    array = np.asarray(prediction)

    if array.ndim == 3:
        array = np.argmax(array, axis=0)
    elif array.ndim == 4:
        array = np.argmax(array[0], axis=0)
    elif array.ndim != 2:
        raise ValueError(f"Unsupported prediction shape: {array.shape}")

    if array.max(initial=0) <= 1:
        array = array * 255
    return np.clip(array, 0, 255).astype(np.uint8)


def predict_image(
    checkpoint: str | Path,
    image: np.ndarray,
    *,
    equalize: bool = True,
) -> dict[str, np.ndarray]:
    """Run model inference and return mask, overlay, and segmented image."""

    learner = load_learner_cached(str(resolve_path(checkpoint)))
    rgb = ensure_uint8_rgb(image)
    mask = predict_mask_array(learner, rgb, equalize=equalize)
    return {
        "mask": mask,
        "overlay": overlay_mask(rgb, mask),
        "segmented": apply_mask(rgb, mask),
    }


def predict_image_file(
    checkpoint: str | Path,
    image_path: str | Path,
    output_dir: str | Path,
    *,
    equalize: bool = True,
) -> dict[str, Path]:
    """Predict and write mask, overlay, and segmented image outputs."""

    image = read_image(image_path)
    outputs = predict_image(checkpoint, image, equalize=equalize)
    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    mask_path = out_dir / f"{stem}_mask.png"
    overlay_path = out_dir / f"{stem}_overlay.png"
    segmented_path = out_dir / f"{stem}_segmented.png"

    write_mask(mask_path, outputs["mask"])
    write_rgb(overlay_path, outputs["overlay"])
    write_rgb(segmented_path, outputs["segmented"])
    return {"mask": mask_path, "overlay": overlay_path, "segmented": segmented_path}
