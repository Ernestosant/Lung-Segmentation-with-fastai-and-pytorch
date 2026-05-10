# Reproducibility Guide

## Environment

Use Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Record the exact environment for each final result:

```powershell
python -m pip freeze > artifacts/metrics/environment.txt
```

## Randomness

Configs include a project seed. Training still depends on hardware, CUDA
kernels, and fastai/PyTorch implementation details, so final reports should
include the package versions and GPU used.

## Artifacts

Keep these files local:

- `artifacts/models/*.pkl`
- `artifacts/metrics/*.json`
- `artifacts/metrics/*.csv`
- `artifacts/predictions/*.png`

For presentation, report file hashes for model checkpoints:

```powershell
Get-FileHash artifacts/models/resnet34_Dlr.pkl -Algorithm SHA256
```

## Hugging Face Space

If publishing the demo to Hugging Face Spaces, keep the repository license
consistent with Apache-2.0 or clearly document why a different model license is
required. Avoid committing private medical images. Large `.pkl` files should be
managed intentionally with Git LFS or an external release artifact.

## Reproduction Checklist

- `data/manifest.csv` is versioned without private data.
- Local raw data paths are documented.
- Config file is saved with the result.
- Model checkpoint hash is recorded.
- Evaluation JSON and per-image CSV are saved.
- Qualitative examples are generated from the same checkpoint and split.
