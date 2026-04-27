# SOTA / Hugging Face Review For Satellite Land-Change Models

Date: 2026-04-15

## 2026-04-27 Supervisor-Facing Clarification

The architecture statement in the Google Doc is a project recommendation prepared with GPT assistance, but it is not an unsupported "GPT opinion." It is grounded in the current repository structure and in external remote-sensing literature.

The practical meaning is:

- `Mask2Former` is the current implemented semantic baseline because it already produces per-pixel land-cover classes from RGB imagery.
- `Prithvi-EO-2.0` is the next semantic backbone target, but it must be integrated through a multispectral/TerraTorch path. It should not be forced into the current RGB-only interface.
- `CDMamba` is a binary changed/unchanged validation model. It can help check whether the semantic pipeline localizes changed regions, but it cannot explain `cropland -> built_up`, `water -> dry land`, or similar semantic transitions by itself.
- `Qwen`, `EarthDial`, and `Gemma` belong in the explanation layer only. They may summarize measured masks, top transitions, uncertainty, and visual evidence, but they must not generate the primary masks or transition matrix.

So the answer to the supervisor is: this is our proposed technical architecture, written with AI help, and the core idea is to keep the evidence source deterministic and pixel-based. The VLM is only a reporting assistant after the segmentation/change computation is finished.

Current implementation status:

- Implemented: RGB `Mask2Former` semantic segmentation for T1/T2, grid-based cell packs, transition summaries, deterministic reporting, and constrained VLM explanation paths.
- Planned explicitly: `prithvi_terratorch` multispectral backend using six EO channels from OSCD/Sentinel-2-style stacks.
- Not implemented yet: TerraTorch task head/config for Prithvi segmentation, Prithvi fine-tuning/evaluation, CDMamba benchmark integration, Google Earth Engine production data ingestion.

Recommended next implementation order:

1. Keep the current Mask2Former flow as the UI/prototype baseline.
2. Add a TerraTorch Prithvi-EO-2.0-300M-TL segmentation experiment on six-band inputs.
3. Feed Prithvi T1/T2 semantic maps into the existing transition summary/reporting code.
4. Add CDMamba as a binary benchmark panel or script.
5. Use VLM reports only as natural-language summaries of deterministic evidence.

## 2026-04-16 Implementation Addendum

The current app should not replace `mfaytin/mask2former-satellite` blindly with a heavier Hugging Face model. For the running RGB crop UI, `mask2former-satellite` remains the safest drop-in segmentation/classification model because it is already a Transformers `Mask2FormerForUniversalSegmentation` checkpoint trained for OpenEarthMap land-cover classes. The output should remain diagnostic only: these semantic maps are useful for visualization and weak cross-checking, but they should not drive the final physical interpretation unless the user explicitly enables semantic hints.

The better scientific path is still Prithvi, but not as a direct RGB replacement. `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL` and `600M-TL` are TerraTorch Earth-observation foundation models for multispectral/multi-temporal EO data. They should be integrated as a separate production semantic backend only after the app can feed the correct HLS-style channels and task head. Until then, forcing Prithvi through the current RGB-only `Mask2FormerSatelliteSegmenter` interface would be technically wrong and would likely make the output less stable.

The VLM path was changed accordingly: Qwen/Gemma no longer generate the full 16-cell table. The app computes the full grid deterministically, sends only BEFORE, AFTER, and one change-guide image to the model, and asks the model only for a short optional `scene_overview`. This reduces autoregressive token work and prevents the model from rewriting measured cell evidence into overconfident `building/pad/roof` language.

Canonical Google Doc: `https://docs.google.com/document/d/1R1sjOH4ZEfWCFuIVV7QiRqnVzlhtyEDksJ5ntEisTo4/edit?usp=sharing`

Document ID: `1R1sjOH4ZEfWCFuIVV7QiRqnVzlhtyEDksJ5ntEisTo4`

## Project Goal From Proposal PDF

The proposal defines an AI-powered land cover change detection and analysis system for satellite, aerial, and UAV imagery. The system is meant to use foundation vision models, label-efficient learning, multi-temporal data, and an interactive interface that returns interpretable change maps and model explanations.

The Armenia-specific target is practical land monitoring: cultivated vs non-cultivated zones, irrigated vs non-irrigated fields, dry vs wet regions, deforestation, urban expansion, water-body shrinkage, and agricultural dynamics. The proposal explicitly mentions Sentinel-2, UAV feeds, aerial orthophotos, possible SAR/DEM integration, public benchmark comparison, and stakeholder-facing outputs.

The project should therefore stay semantic-first:

`T1 semantic segmentation -> T2 semantic segmentation -> land-cover transition matrix -> interpretable report`

Binary change detection remains useful, but only as an auxiliary answer to "where did anything change?" The project also needs semantic evidence for "what changed into what?"

## Current Local Implementation

Confirmed local implementation facts:

- The README states that the project moved from binary change detection to `semantic-first` land surface mapping.
- The current UI uses `mfaytin/mask2former-satellite` for RGB crop segmentation on T1/T2, then computes transition diffs between semantic maps.
- The repo downloads `mfaytin/mask2former-satellite` and `ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL`.
- `src/land_change_detection/semantic_surface.py` already contains the important OSCD 13-band to Prithvi six-band mapping: Blue, Green, Red, narrow NIR, SWIR1, SWIR2.
- Qwen/EarthDial/Gemma paths are present as explanation/reporting models, not segmentation models.

Current local model interpretation:

1. `mfaytin/mask2former-satellite` is the running prototype segmenter.
2. `Prithvi-EO-2.0-600M-TL` is downloaded but not yet integrated as the core multispectral segmenter.
3. VLMs enrich explanations after visual evidence exists.

## Hugging Face Model Landscape

### Segmentation And Land Cover

`mfaytin/mask2former-satellite`

- Hugging Face facts: Mask2Former with Swin Transformer backbone, trained on OpenEarthMap, 9 land-cover classes, MIT license, best reported mIoU 0.5202.
- Classes: background, bareland, grass, pavement, road, tree, water, cropland, building.
- Limitations from the model card: geographic bias, resolution sensitivity, sensor/platform mismatch, and seasonal variation.
- Project judgment: keep as an RGB prototype and fallback. It is not the final SOTA path for Sentinel-2 multispectral land-cover transition analysis.

`ibm-nasa-geospatial/Prithvi-EO-2.0` family

- Hugging Face facts: NASA/IBM/Juelich Earth Observation foundation model family, TerraTorch integration, Apache-2.0 license, paper `arXiv:2412.02732` (2024).
- Relevant models: `Prithvi-EO-2.0-300M-TL`, `Prithvi-EO-2.0-600M-TL`, plus tiny/100M/300M/600M variants.
- Model-card facts: pre-trained on NASA HLS V2 using six bands in order: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2; fine-tuning is supported through TerraTorch.
- Project judgment: this is the best main path for the proposal because it is EO-native, multispectral, multi-temporal, and already matches the repo's six-band mapping.

`BiliSakura/SegEarth-OV` / `Dingyi111/SegEarth-OV`

- Hugging Face facts: open-vocabulary remote-sensing segmentation, CLIP/SAR/SAM3 variants, MIT license, associated with SegEarth-OV CVPR 2025 and follow-up arXiv work.
- Project judgment: promising research baseline for open-vocabulary segmentation and missing classes, but not the first production path because it is recent, less battle-tested, and not clearly aligned to OSCD/Sentinel-2 multispectral data.

`ratnaonline1/segFormer-b4-city-satellite-segmentation-1024x1024`

- Hugging Face facts: SegFormer B4 city satellite segmentation model.
- Project judgment: useful for city segmentation demos, but lower priority than Prithvi because it is narrower and not tailored to multispectral land-cover transition work.

`allenai/satlas-pretrain`

- Hugging Face facts: SatlasPretrain model card links to `arXiv:2211.15660`, Apache-2.0 license.
- Local project note is correct: SatlasPretrain is better treated as a backbone/feature source unless task-specific heads and checkpoints are used.
- Project judgment: not the first model to implement now.

### Binary Change Detection

`CDMamba`, `arXiv:2406.04207`

- Task: binary remote-sensing image change detection.
- Core idea: combine Mamba global modeling with local convolutional detail through Scaled Residual ConvMamba (SRCM), then use Adaptive Global Local Guided Fusion (AGLGF) for bi-temporal interaction.
- Reported paper results: WHU-CD / LEVIR-CD / LEVIR+-CD F1 scores of 93.76 / 90.75 / 83.01 and IoU scores of 88.26 / 83.07 / 70.95.
- Efficiency facts from the paper: 11.90M parameters and 13.58 min per LEVIR+-CD training epoch; smaller than ChangeFormer and RS-Mamba, but not the fastest model.
- Project judgment: add it as a strong binary mask baseline. Do not use it as the main answer for semantic land-cover transitions because it does not classify transition type.

ChangeFormer / BIT / RS-Mamba / ChangeMamba

- These remain important baselines in the literature.
- CDMamba directly compares with them and reports stronger F1/IoU on the three benchmark datasets under the paper setup.
- Project judgment: if only one binary baseline is added first, add CDMamba before restoring older ChangeFormer integration.

### Classification And Feature Extraction

DINOv2 remote-sensing variants and ViT/Swin remote-sensing classifiers found on Hugging Face are useful for embeddings, classification probes, or pretraining comparisons. They do not solve pixel-level transition mapping by themselves, so they are secondary for the current implementation.

### VLM / Explanation Layer

Project VLM candidates include:

- `Qwen/Qwen3-VL-4B-Thinking`
- `mlx-community/Qwen3-VL-4B-Thinking-3bit`
- `mlx-community/Qwen3.5-0.8B-4bit`
- `AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct`
- `AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct`
- `akshaydudhane/EarthDial_4B_RGB`
- `gemma4:e4b` through Ollama

Project judgment: keep VLMs as explainers. They should produce text from measured segmentation and change evidence; they should not be trusted as the primary pixel-mask source.

## Recommended Model Stack

1. Main production path: `Prithvi-EO-2.0-300M-TL` first, then `Prithvi-EO-2.0-600M-TL`.
   - Implement through TerraTorch.
   - Use the existing OSCD 13-band to six-band mapping.
   - Generate T1/T2 semantic maps and transition matrices.
   - Use 300M-TL for practical iteration, 600M-TL for heavier benchmark runs.

2. Prototype/fallback path: `mfaytin/mask2former-satellite`.
   - Keep it because it is already integrated.
   - Use it for RGB-only demos and UI continuity.
   - Do not present it as final SOTA.

3. Binary baseline path: `CDMamba`.
   - Add as an optional panel or benchmark script.
   - Compare CDMamba binary masks with semantic transition masks.
   - Use it to validate changed-region localization, not transition labels.

4. Research path: `SegEarth-OV`.
   - Evaluate only after the Prithvi path is stable.
   - Use it to explore open-vocabulary classes not covered by fixed land-cover schemas.

5. Explanation path: Qwen/EarthDial/Gemma.
   - Use after deterministic segmentation/change computation.
   - Feed measured evidence into the VLM: top transitions, changed area percentage, sensor/source metadata, confidence notes.

## CDMamba Paper Review

CDMamba is relevant because it is a strong 2024/2025 binary change-detection paper for high-resolution optical remote-sensing imagery. It directly addresses weaknesses in pure Transformer and pure Mamba approaches by adding local convolutional clues for dense prediction.

The most important architectural points:

- SRCM fuses Mamba-style global context with convolutional local detail.
- AGLGF dynamically guides bi-temporal feature interaction using global and local features from the other timestamp.
- The paper evaluates on WHU-CD, LEVIR-CD, and LEVIR+-CD.
- Metrics are standard binary CD metrics: precision, recall, F1, IoU, and OA.

The most important limitation for this project:

CDMamba produces binary changed/unchanged masks. The proposal and current repo need semantic land-cover transition evidence: forest to bare ground, cropland to built-up, water to dry land, irrigated to non-irrigated, and related Armenia-specific categories. Therefore CDMamba should be integrated as a baseline and cross-check, not as the main semantic system.

## Implementation Order

1. Keep `mask2former-satellite` as the current RGB prototype.
2. Add a TerraTorch/Prithvi segmenter beside `Mask2FormerSatelliteSegmenter`.
3. Use `select_prithvi_bands` from `semantic_surface.py` for OSCD 13-band to six-band input.
4. Run Prithvi T1/T2 segmentation and feed outputs into the existing transition summary UI.
5. Add CDMamba as an optional binary baseline after the semantic path works.
6. Add comparison outputs: Mask2Former RGB vs Prithvi multispectral vs CDMamba binary mask.
7. Keep VLM reports constrained to deterministic evidence from segmentation and change maps.

## Sources

- Project proposal PDF, 2026: `25DD-1B035_LandCoverChangeDetection2025_am_250723_202701.pdf`
- Hugging Face model card, `mfaytin/mask2former-satellite`, updated 2026: https://huggingface.co/mfaytin/mask2former-satellite
- Hugging Face model card, `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL`, updated 2025: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL
- Hugging Face model card, `ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL`, updated 2025: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL
- Prithvi-EO-2.0 paper, 2024, `arXiv:2412.02732`: https://arxiv.org/abs/2412.02732
- Hugging Face model card, `BiliSakura/SegEarth-OV`, 2026: https://huggingface.co/BiliSakura/SegEarth-OV
- SegEarth-OV paper, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/papers/Li_SegEarth-OV_Towards_Training-Free_Open-Vocabulary_Segmentation_for_Remote_Sensing_Images_CVPR_2025_paper.pdf
- Hugging Face model card, `allenai/satlas-pretrain`, updated 2024: https://huggingface.co/allenai/satlas-pretrain
- SatlasPretrain paper, ICCV 2023 / arXiv 2022, `arXiv:2211.15660`: https://arxiv.org/abs/2211.15660
- CDMamba paper, 2024/2025, `arXiv:2406.04207`: https://arxiv.org/pdf/2406.04207
- CDMamba Hugging Face paper page, 2024: https://huggingface.co/papers/2406.04207
- CDMamba GitHub: https://github.com/zmoka-zht/CDMamba
