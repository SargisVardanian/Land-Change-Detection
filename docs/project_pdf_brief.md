# Project Proposal PDF Brief

Source PDF: `25DD-1B035_LandCoverChangeDetection2025_am_250723_202701.pdf`

Extraction date: 2026-04-15

## Core Aim

The proposal defines an AI-powered system for detecting and analyzing land cover change from satellite and drone imagery. The intended system combines foundation vision models, temporal change detection algorithms, multi-source Earth observation data, and an interactive interface that returns interpretable change maps and explanations.

The local project should therefore be treated as more than a binary change mask demo. Its durable goal is land-cover transition analysis for environmental and agricultural decision-making, especially in Armenia.

## Armenia-Specific Use Cases

- cultivated vs non-cultivated zones
- irrigated vs non-irrigated fields
- dry vs wet regions
- deforestation and forest-cover dynamics
- urban expansion around Yerevan and other settlements
- water-body shrinkage
- agricultural land-use dynamics
- land-use planning, climate adaptation, and resource optimization

## Data Requirements

The proposal expects multi-temporal and multi-resolution imagery:

- Sentinel-2 optical imagery for regional 10 m monitoring;
- Landsat for longer historical context;
- UAV feeds or aerial orthophotos for fine-grained changes;
- possible commercial high-resolution sources such as PlanetScope or Maxar if available;
- possible Sentinel-1/SAR or DEM integration for challenging cases such as floods or terrain-related changes.

The proposal also stresses co-registration, common coordinate systems, radiometric normalization, and careful ground-truth creation from land-cover maps, local experts, OpenStreetMap updates, and government reports.

## Model Requirements

The proposal explicitly motivates foundation vision and remote-sensing foundation models:

- SAM and zero/few-shot segmentation ideas;
- DINOv2 and self-supervised visual features;
- Vision Transformers and Swin-style hierarchical transformers;
- domain-specific remote-sensing foundation models such as SatlasPretrain-like systems;
- Siamese or two-stream before/after feature extraction;
- late fusion, absolute feature difference, multiscale feature fusion, and UNet-like decoders;
- parameter-efficient tuning such as adapters or LoRA-style training;
- BCE plus Dice/IoU losses for binary masks where binary change detection is used.

## Architecture Implication For This Repo

The current repo direction should remain semantic-first:

`T1 semantic segmentation -> T2 semantic segmentation -> transition matrix -> explanation`

This satisfies the proposal's requirement for interpretable land-cover transitions better than a pure binary change detector. Binary change detection should be added as an auxiliary baseline and quality cross-check.

## Evaluation Expectations

The proposal expects:

- local validation against known Armenian changes;
- public benchmarking on datasets such as LEVIR-CD and WHU-CD;
- precision, recall, F1, IoU, and spatial accuracy;
- prototype outputs such as GeoTIFF change masks and vector polygons;
- an interactive tool or web/GIS interface usable by stakeholders.

## Practical Project Interpretation

For this codebase, the proposal points to this model stack:

1. `Prithvi-EO-2.0` for multispectral EO segmentation and/or fine-tuned semantic mapping.
2. Current `mask2former-satellite` only as RGB prototype/fallback.
3. `CDMamba` as a binary change-detection baseline.
4. VLMs such as Qwen/EarthDial/Gemma only after deterministic visual evidence exists, to explain measured segmentation/change outputs.
