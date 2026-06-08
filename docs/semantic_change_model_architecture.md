# Semantic Change Model Architecture

Date: 2026-05-30

## Decision

The project model path is semantic-first:

```text
T1 semantic segmentation -> T2 semantic segmentation -> transition matrix -> interpretable report
```

The implemented model code is in:

- `src/land_change_detection/models/semantic_change.py`
- `src/land_change_detection/training/losses.py`

The architecture is a Siamese semantic-transition network:

1. A shared encoder-decoder segmenter processes T1 and T2 independently.
2. The primary outputs are T1 and T2 land-cover class logits.
3. A small auxiliary change head consumes T1 features, T2 features, and absolute feature difference.
4. The transition map is derived deterministically from semantic class IDs:

```text
transition_id = before_class_id * num_classes + after_class_id
```

This keeps the answer to "what changed into what?" separate from the auxiliary "where did anything change?" mask.

## Default Input

The default model input is a six-band EO tensor ordered as:

```text
blue, green, red, narrow_nir, swir1, swir2
```

That matches the local `select_prithvi_bands` mapping from OSCD/Sentinel-2-style 13-band stacks and the Prithvi-EO-2.0 HLS band order.

## Why This Architecture

Confirmed local facts:

- The current runnable UI already supports T1/T2 semantic segmentation, transition summaries, grid evidence, and VLM explanation.
- `mfaytin/mask2former-satellite` remains the RGB prototype/fallback.
- `prithvi_terratorch` is intentionally separate from RGB inference because Prithvi expects EO bands and a TerraTorch task head.

External model evidence:

- Prithvi-EO-2.0 is a ViT/MAE Earth-observation foundation model family trained on HLS time-series data and supports TerraTorch fine-tuning. Source: Prithvi-EO-2.0 paper, 2024, https://arxiv.org/abs/2412.02732
- The NASA-IMPACT Prithvi-EO-2.0 repository documents TerraTorch fine-tuning examples for downstream segmentation tasks. Source: NASA-IMPACT GitHub, 2024, https://github.com/NASA-IMPACT/Prithvi-EO-2.0
- TerraTorch config examples use `prithvi_eo_v2_300`, six EO bands, a UNet-style decoder, and semantic segmentation tasks. Source: Hugging Face config for `Prithvi-EO-2.0-300M-BurnScars`, 2025, https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/blob/main/config.json
- `mfaytin/mask2former-satellite` is a Mask2Former/Swin semantic segmentation checkpoint trained on OpenEarthMap classes, suitable as the current RGB fallback. Source: Hugging Face model card, 2026, https://huggingface.co/mfaytin/mask2former-satellite
- CDMamba is a strong binary remote-sensing change-detection model, but it predicts changed/unchanged masks rather than semantic transitions. Source: CDMamba paper, 2024, https://arxiv.org/abs/2406.04207

## Implementation Status

Implemented now:

- `SemanticChangeModelConfig`
- `SemanticChangeModel`
- `SemanticChangeOutput`
- deterministic transition-map encoding
- composite semantic/change training loss
- unit tests for output shapes, validation, transition encoding, and differentiability

Still open:

- Replace the lightweight local encoder with a TerraTorch Prithvi-EO-2.0-300M-TL encoder when TerraTorch and compatible checkpoints are available.
- Add a dataset class for paired six-band Sentinel-2/OSCD chips with T1/T2 semantic labels.
- Add a training script that saves checkpoints compatible with the model module.
- Add CDMamba only as a binary benchmark once semantic maps are stable.
