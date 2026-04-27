# Satlas / SatlasPretrain Notes

## What was downloaded

Downloaded checkpoints:

- `artifacts/models/satlaspretrain/aerial_swinb_mi.pth`
- `artifacts/models/satlaspretrain/sentinel2_swinb_mi_ms.pth`

Observed file sizes on disk:

- `aerial_swinb_mi.pth`: about `480 MB`
- `sentinel2_swinb_mi_ms.pth`: about `413 MB`

## Which one is the largest

Among the officially listed `satlaspretrain_models` checkpoints, the largest file was:

- `Aerial_SwinB_MI`: `503,495,370` bytes from the Hub response

For the target domain of satellite land-change analysis, the most relevant large checkpoint is:

- `Sentinel2_SwinB_MI_MS`: `433,151,730` bytes from the Hub response

## Architecture

Both large checkpoints use the same high-level design:

1. `Swin v2 Base` backbone
2. multi-image aggregation backbone with temporal `max` pooling
3. `FPN`
4. `upsample`
5. task-specific heads

The local checkpoint structure confirms this:

- `backbone.*`
- `intermediates.*`
- `heads.*`

Local tensor counts:

- `sentinel2_swinb_mi_ms.pth`: `529` tensors
- `aerial_swinb_mi.pth`: `569` tensors

Local parameter counts after loading backbone + FPN:

- `sentinel2_swinb_mi_ms.pth`: `89,599,456`
- `aerial_swinb_mi.pth`: `89,587,168`

## Official config details

### Sentinel2_SwinB_MI_MS

From `configs/sentinel2/swinb_mi_ms.txt`:

- input channels: `9`
- channel order: `tci, fake, fake, b05, b06, b07, b08, b11, b12`
- number of images in a temporal group: `8`
- aggregation: `max`
- tasks:
  - `polygon`
  - `point`
  - `land_cover`
  - `dem`
  - `crop_type`
  - `tree_cover`

### Aerial_SwinB_MI

From `configs/aerial/swinb_mi.txt`:

- input channels: `3`
- number of images in a temporal group: `4`
- aggregation: `max`
- tasks:
  - `polygon`
  - `point`
  - `land_cover`
  - `dem`
  - `crop_type`
  - `tree_cover`
  - `rooftop_solar_panel`
  - `building`

## Important limitation

`satlaspretrain_models` is primarily a convenient loader for pretrained backbone and FPN weights.
The repository explicitly states that prediction heads are task-specific.

So:

- loading `head=SEGMENT` through `satlaspretrain_models` does **not** mean a ready-made semantic land-cover model was loaded
- in that path, the head is randomly initialized for fine-tuning convenience

If you want a true ready-made task model, use the full `satlas` codebase and the published task checkpoints such as:

- `solar_farm`
- `tree_cover`
- `wind_turbine`
- `marine_infrastructure`

## Change-detection implication

These checkpoints are not turnkey change-detection models.
They are best used in one of two ways:

1. as feature extractors inside a Siamese change-detection network
2. as semantic backbones for `T1` and `T2` followed by class-transition analysis

For your government project, the second path is more interpretable:

- predict semantic outputs on `T1`
- predict semantic outputs on `T2`
- compute transition matrix
- generate textual explanation only from observed transitions
