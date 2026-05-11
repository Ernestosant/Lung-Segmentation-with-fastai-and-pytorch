# Model Card

## Model Details

- Task: lung field segmentation in chest X-ray images.
- Input: chest X-ray image.
- Output: binary lung mask, overlay, and segmented image.
- Architecture: fastai/PyTorch U-Net with ImageNet-pretrained ResNet encoder.
- Candidate models: U-Net ResNet18 and U-Net ResNet34.
- Project status: research and education repository, not a clinical product.

## Intended Use

This model is intended for reproducible biomedical image segmentation
experiments, educational demonstrations, and scholarship portfolio review. It
may be useful as a preprocessing example for computer-aided chest radiography
analysis.

It must not be used for diagnosis, treatment planning, triage, patient
monitoring, or autonomous clinical decision making.

## Training And Evaluation Data

Training and evaluation data are defined by `data/manifest.csv`.

The reproducible public-data workflow targets Kaggle
`nikhilpandey360/chest-xray-masks-and-labels`, which contains chest X-ray images
and corresponding lung masks derived from public Montgomery and Shenzhen
sources. Kaggle reports this dataset under the CC0-1.0 license; confirm the
dataset page before redistributing derived visual examples.

Private medical images, masks, and model weights are intentionally excluded from
Git history.

## Metrics

Primary metrics:

- Dice
- IoU

Secondary metric:

- pixel accuracy

Current results status:

| Model | Split | Dice | IoU | Pixel Accuracy |
| --- | --- | ---: | ---: | ---: |
| U-Net ResNet18 | Test | 0.9553 | 0.9157 | 0.9772 |
| U-Net ResNet34 | Test | 0.9559 | 0.9170 | 0.9776 |

Metrics were regenerated on May 11, 2026 in a private Kaggle Kernel using a
Tesla P100-PCIE-16GB and 386 test images. Model weights are local-only artifacts
under `artifacts/models/` and are ignored by Git.

## Limitations

- Performance may not generalize to scanners, acquisition protocols, patient
  populations, disease presentations, or image qualities absent from the
  training data.
- Pixel accuracy can be inflated by large background regions, so Dice and IoU
  should drive interpretation.
- The current split policy is deterministic and manifest-based, but patient-level
  leakage prevention depends on the availability and quality of patient
  identifiers.
- Exported fastai `.pkl` learners may depend on preprocessing choices and
  compatibility symbols registered by this repository.
- This project segments lung fields only; it does not detect disease, assess
  severity, or make clinical recommendations.

## Ethical And Safety Notes

Use only de-identified data with appropriate permissions. Do not upload private
medical images, masks, model weights, or patient metadata to Git. Public figures
in this repository should come only from datasets whose license permits
redistribution or should be kept as local artifacts.

Clinical disclaimer: this repository is research software and must not be used
as a medical device.
