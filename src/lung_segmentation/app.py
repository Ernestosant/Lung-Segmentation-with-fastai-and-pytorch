"""Gradio application for lung segmentation inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import resolve_path
from .infer import list_model_checkpoints, predict_image


def create_demo(model_dir: str | Path = "artifacts/models"):
    """Create the Gradio Blocks demo."""

    import gradio as gr

    directory = resolve_path(model_dir)
    checkpoints = list_model_checkpoints(directory)
    choices = [path.name for path in checkpoints]

    def run(image: np.ndarray, model_name: str):
        if image is None:
            raise gr.Error("Upload a chest X-ray image first.")
        if not model_name:
            raise gr.Error(f"No .pkl models found in {directory}.")
        checkpoint = directory / model_name
        outputs = predict_image(checkpoint, image)
        return outputs["mask"], outputs["overlay"], outputs["segmented"]

    with gr.Blocks(title="Lung Segmentation") as demo:
        with gr.Row():
            model = gr.Dropdown(
                choices=choices,
                value=choices[0] if choices else None,
                label="Model",
                interactive=True,
            )
        with gr.Row():
            image = gr.Image(label="Input", type="numpy")
            mask = gr.Image(label="Mask", type="numpy")
            overlay = gr.Image(label="Overlay", type="numpy")
            segmented = gr.Image(label="Segmented", type="numpy")
        image.change(run, inputs=[image, model], outputs=[mask, overlay, segmented])
        model.change(run, inputs=[image, model], outputs=[mask, overlay, segmented])
    return demo
