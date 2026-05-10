# Model Card

## Model Details

- Task: binary lung field segmentation in chest X-ray images.
- Architecture: fastai U-Net with ImageNet-pretrained ResNet encoder.
- Candidate encoders: ResNet18 and ResNet34.
- Output: binary lung mask, overlay, and segmented image.

## Intended Use

This model is for research, education, and reproducibility demonstrations. It is
not intended for clinical diagnosis, treatment planning, triage, or autonomous
medical decision making.

## Training Data

Training data is defined by `data/manifest.csv`. Report source-level composition
and patient-level split availability before presenting results.

## Evaluation

Primary metrics:

- Dice
- IoU

Secondary metric:

- pixel accuracy

Results should be reported globally and by source:

| Model | Split | Source | Dice | IoU | Pixel accuracy |
| --- | --- | --- | --- | --- | --- |
| resnet34_Dlr | test | all | TBD | TBD | TBD |

## Limitations

- Performance may not generalize to scanners, acquisition protocols, disease
  presentations, or demographics absent from the training data.
- Pixel accuracy can look high on segmentation tasks with large background
  regions, so Dice and IoU should drive interpretation.
- Exported fastai `.pkl` files may depend on preprocessing and custom metric
  symbols; this repository registers compatibility symbols for legacy models.

## Ethical And Safety Notes

Use only de-identified data with appropriate permissions. This project should be
presented as research software, not as a deployable clinical product.
