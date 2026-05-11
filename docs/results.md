# Model Results

Final test metrics were regenerated on May 11, 2026 in a private Kaggle Kernel
using a Tesla P100-PCIE-16GB.

## Dataset And Split

- Dataset: Kaggle `nikhilpandey360/chest-xray-masks-and-labels`.
- License reported by Kaggle metadata: CC0-1.0.
- Sources represented by filename prefix: Shenzhen (`CHNCXR`) and Montgomery
  (`MCUCXR`).
- Matched image/mask rows: 1,408.
- Split policy: deterministic 70/15/15 train/validation/test split with seed
  `42`.
- Split summary: train 702, validation 320, test 386.
- Working copies for the Kaggle run were resized once to `256x256`; configured
  model input size remained `128x128`.
- The P100 run pinned `torch==2.5.1+cu121` and `torchvision==0.20.1+cu121`
  because the default Kaggle PyTorch build did not support Tesla P100 `sm_60`.
- Data privacy: private medical images, masks, and weights are not tracked in
  Git.

Prepare the manifest on Kaggle:

```powershell
python -m pip install --force-reinstall --no-cache-dir `
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 `
  --index-url https://download.pytorch.org/whl/cu121

python scripts/prepare_kaggle_dataset.py `
  --input-dir /kaggle/input/chest-xray-masks-and-labels `
  --out data/manifest.csv `
  --preprocess-size 256
```

## Metrics

| Model | Split | Test Images | Dice | IoU | Pixel Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| U-Net ResNet18 | test | 386 | 0.9553 | 0.9157 | 0.9772 |
| U-Net ResNet34 | test | 386 | 0.9559 | 0.9170 | 0.9776 |

Per-source test metrics:

| Model | Source | Dice | IoU | Pixel Accuracy |
| --- | --- | ---: | ---: | ---: |
| U-Net ResNet18 | Montgomery | 0.9683 | 0.9391 | 0.9843 |
| U-Net ResNet18 | Shenzhen | 0.9514 | 0.9086 | 0.9751 |
| U-Net ResNet34 | Montgomery | 0.9685 | 0.9395 | 0.9842 |
| U-Net ResNet34 | Shenzhen | 0.9521 | 0.9102 | 0.9756 |

Downloaded local metric artifacts:

- `artifacts/metrics/resnet18_test_summary.json`
- `artifacts/metrics/resnet18_test_per_image.csv`
- `artifacts/metrics/resnet34_test_summary.json`
- `artifacts/metrics/resnet34_test_per_image.csv`
- `artifacts/metrics/results_table.md`
- `artifacts/metrics/model_artifacts.json`

## Local Model Artifacts

Model weights are intentionally local-only and ignored by Git.

| Model | Local File | Size | SHA256 |
| --- | --- | ---: | --- |
| U-Net ResNet18 | `artifacts/models/unet_resnet18_epoch10.pkl` | 132,353,078 bytes | `31b5a2e120d9e44f577d4832be3e9cf3e20b9fb2b0801a5365a2de3087a478ec` |
| U-Net ResNet34 | `artifacts/models/resnet34_Dlr.pkl` | 172,803,609 bytes | `d36749df8827c6f6f35568d869b0a5567ea7816e4cee95e3539d382f873c04ce` |

## Qualitative Examples

The README figure `segmentation.PNG` was refreshed from
`artifacts/predictions/resnet34_test_examples.png` and contains three public
test examples with four columns: X-ray, ground-truth mask, predicted mask, and
overlay.

Additional local figure:

- `artifacts/predictions/resnet18_test_examples.png`

## Limitations

- Results should be interpreted as dataset-specific segmentation performance,
  not clinical validation.
- Public Montgomery/Shenzhen data may not represent modern scanners, acquisition
  protocols, severe disease patterns, or diverse populations.
- Pixel accuracy is reported for completeness but can overstate segmentation
  quality when background dominates the image.
- The Kaggle run used resized working copies for efficient P100 training; future
  reports should keep preprocessing details attached to metric tables.
