# Experiment Protocol

## Objective

Train and evaluate U-Net models for lung field segmentation in chest X-ray
images using a reproducible fastai/PyTorch workflow.

## Data Preparation

1. Place local images and masks under ignored folders such as `data/raw/`.
2. Build `data/manifest.csv` with `scripts/build_manifest.py` or an equivalent
   audited process.
3. Confirm the fixed 70/15/15 split and source labels.
4. Confirm masks are binary or can be binarized with the configured threshold.

## Preprocessing

- Convert input images to RGB.
- Apply grayscale histogram equalization when `equalize_histogram: true`.
- Resize training batches to the configured image size.
- Binarize masks using `mask_threshold`.

## Models

The configured baseline models are:

- `configs/resnet18.yaml`: U-Net with ResNet18 encoder.
- `configs/resnet34.yaml`: U-Net with ResNet34 encoder.

Both use self-attention, Mish activation, and the ranger optimizer to preserve
the core behavior of the original notebook.

## Evaluation

Evaluate the exported `.pkl` learner on the test split:

```powershell
lungseg-evaluate --config configs/resnet34.yaml --checkpoint artifacts/models/resnet34_Dlr.pkl
```

Report:

- global Dice, IoU, and pixel accuracy
- per-source Dice, IoU, and pixel accuracy
- number of images in each split and source
- representative qualitative examples from `artifacts/predictions/`

## Acceptance Criteria

- The manifest validates with train, validation, and test rows.
- The training command exports a `.pkl` model to `artifacts/models/`.
- The evaluation command writes per-image CSV and summary JSON files.
- The app launches and returns mask, overlay, and segmented outputs.
