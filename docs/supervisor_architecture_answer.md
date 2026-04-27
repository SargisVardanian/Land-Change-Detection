# Supervisor Architecture Answer

Date: 2026-04-27

## Short Answer

The architecture in the Google Doc is our project recommendation, drafted with GPT assistance but grounded in the current code and remote-sensing literature.

The key point is not "GPT says so." The key point is that the project goal requires semantic evidence:

`T1 semantic segmentation -> T2 semantic segmentation -> transition matrix -> explanation`

That lets the system answer what changed into what, for example `cropland -> built_up` or `vegetation -> bare soil`. A binary change detector can only answer where something changed.

## Why Semantic Segmentation Is The Source Of Truth

The current code already follows this structure:

- `Mask2FormerOpenEarthMapBackend` produces a semantic map for each timestamp.
- `LandChangePipelineV2` runs T1/T2 segmentation and builds aligned cell packs.
- deterministic reporting computes cell-level semantic differences before optional VLM explanation.
- VLM models are constrained to explanation/reporting after measured evidence exists.

So the intended evidence chain is pixel-first and reproducible. The VLM should explain measured outputs; it should not invent masks, classes, or transitions.

## Role Of Each Model

| Model family | Role | Why |
| --- | --- | --- |
| `mfaytin/mask2former-satellite` | Current baseline | Already integrated semantic segmentation checkpoint trained on OpenEarthMap-style land-cover classes. |
| `Prithvi-EO-2.0-300M-TL` | First main target | EO-native foundation model for multispectral/multitemporal data; practical first TerraTorch integration target. |
| `Prithvi-EO-2.0-600M-TL` | Larger benchmark target | Stronger/heavier model to test after 300M path is stable. |
| `CDMamba` | Binary validation baseline | Good for changed/unchanged localization, but not semantic transition labels. |
| Qwen / EarthDial / Gemma | Explanation layer | Useful for human-readable summaries after deterministic masks and transition tables are computed. |

## What To Improve Next

1. Keep `Mask2Former` as the running UI/prototype baseline.
2. Implement a separate `prithvi_terratorch` backend instead of pretending Prithvi is RGB-compatible.
3. Use the existing six-band mapping from OSCD/Sentinel-2-style stacks: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2.
4. Add transition-matrix metrics and per-class change reports.
5. Add `CDMamba` only after semantic maps work, as a binary comparison layer.
6. Keep VLM output explicitly labelled as explanation, not evidence.

## Sources

- Mask2Former satellite model card, Hugging Face, accessed 2026: https://huggingface.co/mfaytin/mask2former-satellite
- Prithvi-EO-2.0 paper, 2024: https://arxiv.org/abs/2412.02732
- Prithvi-EO-2.0 model card, Hugging Face, accessed 2026: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL
- CDMamba paper, 2024: https://arxiv.org/abs/2406.04207
- OpenMapCD dataset and paper, 2024: https://zenodo.org/records/14028095
