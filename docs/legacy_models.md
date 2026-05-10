# Legacy Model Notes

The original repository described two exported learners:

- `unet_resnet18_epoch10.pkl`
- `resnet34_Dlr.pkl`

Those weights should be stored locally in `artifacts/models/` and evaluated
with the current metric pipeline before reporting results. The old notes used
pixel accuracy-like wording; this project reports Dice and IoU as primary
segmentation metrics.
