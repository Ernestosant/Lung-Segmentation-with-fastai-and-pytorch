"""Image and mask preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def ensure_uint8_rgb(image: np.ndarray | Image.Image) -> np.ndarray:
    """Return an RGB uint8 image from a PIL image or numpy array."""

    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    else:
        image = np.asarray(image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a grayscale, RGB, or RGBA image; got shape {image.shape}")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def read_image(path: str | Path) -> np.ndarray:
    """Read an image from disk as RGB uint8."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    """Read a segmentation mask as binary uint8 values in {0, 1}."""

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return binarize_mask(mask, threshold=threshold, channel_order="BGR")


def write_rgb(path: str | Path, image: np.ndarray) -> None:
    """Write an RGB image to disk using OpenCV."""

    output = ensure_uint8_rgb(image)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    """Write a binary mask to disk as 0/255 uint8."""

    output = np.asarray(mask)
    if output.max(initial=0) <= 1:
        output = output * 255
    output = np.clip(output, 0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), output)


def equalize_histogram_rgb(image: np.ndarray | Image.Image) -> np.ndarray:
    """Apply grayscale histogram equalization and return an RGB image."""

    rgb = ensure_uint8_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)


def binarize_mask(
    mask: np.ndarray | Image.Image,
    threshold: int = 127,
    channel_order: str = "RGB",
) -> np.ndarray:
    """Convert a grayscale or color mask to binary class ids in {0, 1}.

    PIL inputs are treated as RGB/RGBA. Arrays from OpenCV should pass
    ``channel_order="BGR"`` because ``cv2.imread`` returns BGR/BGRA channels.
    """

    if isinstance(mask, Image.Image):
        mask = np.asarray(mask)
        channel_order = "RGB"
    else:
        mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = _color_mask_to_gray(mask, channel_order=channel_order)

    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)


def _color_mask_to_gray(mask: np.ndarray, channel_order: str) -> np.ndarray:
    order = channel_order.upper()
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    if mask.shape[2] == 3 and order == "RGB":
        return cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    if mask.shape[2] == 3 and order == "BGR":
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[2] == 4 and order == "RGB":
        return cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
    if mask.shape[2] == 4 and order == "BGR":
        return cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
    raise ValueError(
        "Expected a grayscale, RGB/BGR, or RGBA/BGRA mask; "
        f"got shape {mask.shape} with channel_order={channel_order!r}"
    )


def postprocess_mask(
    mask: np.ndarray,
    output_shape: tuple[int, int] | None = None,
    close_kernel: int = 15,
    blur_kernel: int = 5,
    threshold: int = 127,
) -> np.ndarray:
    """Resize and clean a predicted mask, returning 0/255 uint8."""

    output = np.asarray(mask)
    if output.max(initial=0) <= 1:
        output = output * 255
    output = np.clip(output, 0, 255).astype(np.uint8)

    if output_shape is not None:
        height, width = output_shape
        output = cv2.resize(output, (width, height), interpolation=cv2.INTER_NEAREST)

    if close_kernel and close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)

    if blur_kernel and blur_kernel > 1:
        output = cv2.blur(output, (blur_kernel, blur_kernel))

    _, output = cv2.threshold(output, threshold, 255, cv2.THRESH_BINARY)
    return output.astype(np.uint8)


def apply_mask(image: np.ndarray | Image.Image, mask: np.ndarray) -> np.ndarray:
    """Return the RGB image with pixels outside the mask set to zero."""

    rgb = ensure_uint8_rgb(image)
    clean_mask = postprocess_mask(mask, output_shape=rgb.shape[:2], close_kernel=0, blur_kernel=0)
    return cv2.bitwise_and(rgb, rgb, mask=clean_mask)


def overlay_mask(
    image: np.ndarray | Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 220, 120),
    alpha: float = 0.35,
) -> np.ndarray:
    """Overlay a binary mask on an RGB image."""

    rgb = ensure_uint8_rgb(image)
    clean_mask = postprocess_mask(mask, output_shape=rgb.shape[:2], close_kernel=0, blur_kernel=0)
    overlay = rgb.copy()
    color_layer = np.zeros_like(rgb)
    color_layer[:, :] = np.asarray(color, dtype=np.uint8)
    mask_bool = clean_mask > 0
    overlay[mask_bool] = cv2.addWeighted(rgb, 1 - alpha, color_layer, alpha, 0)[mask_bool]
    return overlay
