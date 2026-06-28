# Project Agent Instructions

## Canonical Project Goal

This project is an AI-powered land cover change detection and analysis system for satellite, aerial, and UAV imagery, with a practical Armenia-focused environmental and agricultural use case.

The main technical direction is semantic-first land-change analysis:

`T1 semantic segmentation -> T2 semantic segmentation -> land-cover transition matrix -> interpretable change report`

Binary change detection is useful as an auxiliary baseline for "where did anything change?", but it must not replace semantic transition analysis for "what changed into what?".

## Canonical SOTA Google Doc

All final research, model-selection, Hugging Face, and SOTA summaries for this project must be written to this Google Doc when Google Docs write access is available:

- URL: `https://docs.google.com/document/d/1R1sjOH4ZEfWCFuIVV7QiRqnVzlhtyEDksJ5ntEisTo4/edit?usp=sharing`
- Document ID: `1R1sjOH4ZEfWCFuIVV7QiRqnVzlhtyEDksJ5ntEisTo4`
- Title observed through connector: `State_of_the_Art_Model`

Before writing, verify the target document ID. Do not create a replacement Google Doc. Do not claim content was written to Google Docs unless the Google Docs batch update succeeds.

If Google Docs write fails with OAuth or `401 Reauthentication required`, report that the target Doc was correct, explain that Google Drive/Docs reauthentication is required, and preserve the exact final content in:

`docs/sota_hf_satellite_models_review.md`

## Proposal PDF Context

Treat this proposal PDF as project-level context:

`25DD-1B035_LandCoverChangeDetection2025_am_250723_202701.pdf`

The same file may also exist in `~/Downloads`; the repo-root copy is the canonical project copy when hashes match.

Core proposal requirements:

- use foundation vision models such as SAM, retired visual baseline, Vision Transformers, and remote-sensing foundation models;
- support multi-temporal and multi-resolution imagery including Sentinel-2, UAV feeds, aerial orthophotos, and possibly Sentinel-1/SAR or DEMs;
- address Armenia-specific needs: cultivated vs non-cultivated zones, irrigated vs non-irrigated fields, dry vs wet regions, deforestation, urban expansion, water-body shrinkage, and agricultural dynamics;
- prefer label-efficient, few-shot, self-supervised, transfer-learning, and parameter-efficient fine-tuning approaches;
- produce interpretable change maps and explanations for policy and practical decision-making;
- benchmark against local Armenian validation events and public datasets such as LEVIR-CD and WHU-CD.

## Current Model Direction

Confirmed local/project model posture:

- `mfaytin/mask2former-satellite`: current RGB-only semantic segmentation prototype/fallback.
- `ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL`: downloaded foundation-model candidate for multispectral EO work.
- `Prithvi-EO-2.0-300M-TL`: preferred first implementation target for practical TerraTorch iteration.
- `CDMamba` (`arXiv:2406.04207`): strong binary change-detection baseline, not a semantic transition model.
- Qwen/EarthDial/Gemma VLMs: explanation/reporting layer only, not primary pixel-mask evidence.

## Evidence Policy

For scientific or model-quality claims:

- inspect local code and artifacts first;
- use Hugging Face plugin data for model availability and model-card facts;
- run an internet literature check against primary sources, peer-reviewed papers, official model cards, or arXiv papers with clear metadata;
- include source links and publication years in final reports;
- separate confirmed local facts, external claims, recommendations, and open checks.

## Verification Expectations

Substantial work is complete only when:

1. edits or Docs updates are applied;
2. verification commands are run where possible;
3. Google Doc write success or OAuth blocker is explicitly reported;
4. a reusable lesson is ready for project/global memory.
