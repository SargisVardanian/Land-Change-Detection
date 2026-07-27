# Dataset Curriculum

No single public dataset gives all of the following at once:

- segmentation
- text descriptions
- old/new pairs
- seasonal diversity
- multispectral EO coverage

The platform should therefore use a staged curriculum instead of searching for one perfect dataset.

## Stage 1: Clean Text-To-Pair Retrieval

- LEVIR-CC

Use this stage for the first caption-to-pair benchmark.

## Stage 2: Grounded Retrieval With Masks

- LEVIR-MCI

Use this stage for parser validation, sample-grid QA, binary change segmentation plus retrieval, and overfit-100 debugging.

## Stage 3: Transition-Aware Pair Retrieval

- SECOND-CC
- Hi-UCD
- TERRA-CD
- ChangeNet
- HRSCD
- HZNU-FCD
- S2Looking
- xBD

Use this stage to retrieve by semantic transition direction, not only by final scene similarity.

## Stage 4: Scale-Up Caption And Reasoning Data

- RSCC
- RSRCC
- ChangeIMTI
- CC-Foundation
- UCCD

Use this stage only after the first grounded and semantic benchmarks are stable.

## Stage 5: Longer-Horizon Temporal Retrieval

- DynamicEarthNet
- SpaceNet 7

Use this stage later for trend retrieval, temporal prediction, and multi-step change behavior. These are not default bootstrap datasets.

## Segmentation / Taxonomy Support

- OpenEarthMap
- Dynamic World
- ESA WorldCover

These support class vocabulary alignment and semantic interpretation rather than first-pass caption retrieval.

## Project-Specific Extension

Custom Sentinel manifests should be used later to add:

- Armenia-specific seasonality
- irrigated vs non-irrigated patterns
- dry vs wet regional dynamics
- local validation events
