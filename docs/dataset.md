# Dataset Documentation

## Scope

The intended dataset contains chest X-ray images and binary lung masks. The
original prototype described a mixture of Montgomery data and manually
segmented COVID-19 cases. The repository does not track medical images or masks.

## Manifest Schema

`data/manifest.csv` is the source of truth:

```csv
image_path,mask_path,source,patient_id,split
```

- `image_path`: absolute path or path relative to the repository root.
- `mask_path`: matching binary lung mask path.
- `source`: dataset label, for example `montgomery` or `covid_manual`.
- `patient_id`: patient identifier when available; otherwise use the image stem
  and document that image-level splitting was used.
- `split`: one of `train`, `val`, or `test`.

## Split Policy

Use a fixed 70/15/15 train/validation/test split. Prefer patient-level splitting
to prevent leakage across splits. When patient identifiers are unavailable, use a
seeded image-level split and report it as a limitation.

## Data Handling

Do not commit medical images, masks, or exported model weights. Keep local files
under ignored folders such as `data/raw/`, `data/processed/`, and
`artifacts/models/`.

## Quality Checks

Before training, confirm:

- every image has exactly one mask
- masks are binary class-id images or can be binarized with the documented
  threshold
- sources are balanced enough to support per-source reporting
- no patient appears in more than one split when patient identifiers are known
