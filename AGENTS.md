# Project Agent Instructions

## Canonical Project Goal

This project is an AI-powered land cover change detection and analysis system for satellite, aerial, and UAV imagery, with a practical Armenia-focused environmental and agricultural use case.

The main technical direction is UniChange v2: a multimodal temporal change
model with one shared temporal representation for text-to-pair retrieval,
pair-to-pair retrieval, captioning, segmentation, text-conditioned grounding,
and event-level descriptions:

`images [B,T,C,H,W] -> UniverSat per timestamp -> TemporalChangeEncoder -> pair_embedding + change_tokens + event_tokens`

Semantic transition analysis remains mandatory as a supervised baseline and as
segmentation/grounding supervision:

`T1 semantic segmentation -> T2 semantic segmentation -> land-cover transition matrix -> interpretable change report`

Binary change detection is useful for "where did anything change?", but it must
not replace semantic transition analysis or the multimodal retrieval/captioning
objectives.

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

## UniChange v2 Stage-1 cluster context

Read `docs/UNICHANGE_V2_STAGE1_HANDOFF.md` before changing Stage-1 retrieval code.

- Baseline branch: `codex/unichange-universat-jina-v5`.
- Baseline full-run commit: `ab10cb3d3995778e2d1810ed06a15dc0f08620bf`.
- Next-development branch: `codex/unichange-v2-stage1-next`.
- Cluster account/partition/QoS: `research / research / researcher`.
- Project root: `/mnt/weka/svardanyan/rs_change_project`.
- Baseline code checkout: `/mnt/weka/svardanyan/rs_change_project/code/project`.
- Python: `/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python`.
- Environment: `source "$HOME/rschange_env.sh"`.
- UniverSat source/checkpoint: `$RS_PROJECT_ROOT/external/UniverSat`, `$RS_PROJECT_ROOT/models/universat-base`.
- Jina checkpoint: `$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval`.

Do not modify the checkout used by an active Slurm job. Use a separate worktree for the next branch. The first baseline run was submitted as training job `83161`, dependent evaluation job `83162`, with run directory:

`/mnt/weka/svardanyan/rs_change_project/runs/unichange_v2_retrieval_1gpu_20260702-125746`

Those IDs and the metrics below are historical context, not proof that the jobs eventually completed. Inspect `sacct`, JSON reports and artifacts before claiming completion.

### Dataset audit

- 10,077 image pairs: train 6,815, validation 1,333, test 1,929.
- Five captions per pair; 50,385 caption rows.
- 19,266 unique normalized captions; about 65.5% of rows are duplicates.
- No missing files and no split overlap were found.
- `changeflag`: 5,039 no-change / 5,038 change.
- Binary masks: 4,978 empty / 5,099 changed.
- 511 changeflag/mask disagreements were found. Captions and masks can also disagree; do not blindly filter the official benchmark.
- Stage-1 uses T1, T2 and captions. Masks are diagnostics unless an experiment explicitly adds auxiliary supervision.

### Baseline gates

The baseline smoke and memory gates passed on an H100 80 GB at commit `ab10cb3...`:

- smoke job `83140`: `COMPLETED 0:0`, 10 finite BF16 steps, correct frozen/trainable gradients and exact checkpoint roundtrip;
- memory job `83141`: `COMPLETED 0:0`, physical batch 32 passed, about 55.7 GB allocated and 61.9 GB reserved.

Any code change makes those reports stale. Run Stage-1-next smoke and memory probes at the exact new commit before submitting full training.

### Preliminary baseline results at epoch 8

| epoch | train loss | dup-aware R@1 | R@5 | R@10 | MRR | mean rank | exact-pair R@1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.9135 | 0.4029 | 0.5209 | 0.5377 | 0.4668 | 125.35 | 0.0033 |
| 2 | 3.6123 | 0.5047 | 0.5320 | 0.5592 | 0.5249 | 90.60 | 0.0042 |
| 3 | 3.5011 | 0.4077 | 0.5431 | 0.5740 | 0.4812 | 84.98 | 0.0060 |
| 4 | 3.4434 | 0.5091 | 0.5419 | 0.5668 | 0.5316 | 74.66 | 0.0071 |
| 5 | 3.3848 | 0.5092 | 0.5479 | 0.5730 | 0.5347 | 70.78 | 0.0078 |
| 6 | 3.3455 | 0.5158 | 0.5616 | 0.5977 | 0.5443 | 70.33 | 0.0108 |
| 7 | 3.3277 | 0.5148 | 0.5547 | 0.5877 | 0.5413 | 68.99 | 0.0104 |
| 8 | 3.2898 | 0.5163 | 0.5565 | 0.5905 | 0.5422 | 64.78 | 0.0111 |

Interpret exact-pair metrics strictly as diagnostics. Stage-1-next now targets semantic text-to-pair process retrieval: a query should retrieve all scenes with matching change processes, while detailed queries should prefer object type, direction, count, scale and location details.

### Stage-1-next requirements

The new path must preserve the baseline scripts and add a separate feature-gated experiment with:

1. before/after direction embeddings;
2. explicit change fusion using `F1`, `F2`, `F2-F1`, `abs(F2-F1)` and `F1*F2`;
3. stable normalized caption groups during training and evaluation;
4. set-mass multi-positive InfoNCE blended with detached semantic soft targets from frozen pre-adapter Jina text embeddings;
5. text-to-pair weighted more strongly than pair-to-text, with pair-to-text kept only as the existing 0.25 regularizer;
6. detail-aware sampling of at most two captions per pair during training: one slot prefers the most detailed caption and one slot uses deterministic rare-caption/rotation logic, with all captions retained for validation;
7. bounded trainable CLIP-style temperature;
8. no weight decay for biases, normalization parameters, direction/global tokens and logit scale;
9. logged pre-clip gradient norms and clipping frequency;
10. semantic best-checkpoint criteria only;
11. semantic recall/nDCG, detailed/directional/location/count query metrics, exact diagnostics, caption-frequency and mask-size-stratified metrics;
12. a periodic fixed train-subset evaluation;
13. configurable Jina `text_max_length`, default `256` for semantic Stage-1-next experiments.

Do not add a pair-to-pair projection head, loss, sampler or checkpoint metric in Stage-1-next. Exact-pair metrics remain diagnostics only: `exact_pair_R@1/R@5/R@10`, `exact_pair_MRR` and exact ranks. They must not drive best scores, checkpoint filenames, composite score or readiness success criteria.

Stage-1-next checkpoint selection is:

- `best_semantic_r1.pt`
- `best_semantic_r5.pt`
- `best_semantic_ndcg10.pt`
- `best_detailed_r5.pt`
- `best_macro_semantic.pt`
- `best_composite.pt`
- `best_retrieval.pt` as the `best_composite.pt` alias
- `last_retrieval.pt`

Composite score:

```text
0.30 * macro semantic R@1
+ 0.25 * macro semantic R@5
+ 0.20 * macro semantic R@10
+ 0.15 * macro semantic nDCG@10
+ 0.10 * detailed-query R@5
```

Early stopping monitors `composite` with default patience `4` validation epochs and minimum improvement `0.002`. Always preserve `last_retrieval.pt`.

Do not add encoder depth merely because aggregate R@1 plateaus. First diagnose directionality, objective mismatch, caption duplication, label noise and the train/validation gap.

### Multi-dataset Stage-1-next manifests

Stage-1-next supports one canonical JSONL temporal-caption manifest for LEVIR-MCI, SECOND-CC and optional RSCC adapters. Each row must contain:

`schema_version`, `dataset_name`, namespaced `pair_id`, `original_id`, `split`, `t1_path`, `t2_path`, `captions`, `normalized_caption_groups`, `caption_source`, `time_order`, nullable `mask_path`, nullable `semantic_t1_path`, nullable `semantic_t2_path`, nullable `sensor`, nullable `spatial_resolution`, `image_width`, `image_height`, `source_metadata` and `preprocessing_fingerprint`.

Pair IDs are namespaced as `levir_mci:<split>:<original_id>`, `second_cc:<split>:<original_id>` and `rscc:<split>:<original_id>`. Official train/val/test splits must be preserved; do not randomly re-split these datasets. Manifest building must not resize, recompress or otherwise destructively preprocess imagery. Runtime transforms own resizing.

Preprocessing commands:

```bash
PYTHONPATH="$PWD/src:$PWD/scripts:$PWD" python scripts/prepare_levir_mci_manifest.py \
  --root /path/to/LEVIR-MCI --output manifests/levir_mci.jsonl \
  --audit-report reports/levir_mci_manifest_audit.json

PYTHONPATH="$PWD/src:$PWD/scripts:$PWD" python scripts/prepare_second_cc_manifest.py \
  --root /path/to/SECOND-CC --annotations /path/to/SECOND-CC-AUG.json \
  --expected-pairs 6041 --expected-captions 30205 \
  --output manifests/second_cc.jsonl \
  --audit-report reports/second_cc_manifest_audit.json

PYTHONPATH="$PWD/src:$PWD/scripts:$PWD" python scripts/merge_temporal_caption_manifests.py \
  --manifest manifests/levir_mci.jsonl --manifest manifests/second_cc.jsonl \
  --output manifests/stage1_next_mixed.jsonl \
  --audit-report reports/stage1_next_mixed_manifest_audit.json
```

SECOND-CC must be read from the official Karpathy JSON and raw paths `<root>/<split>/rgb/A`, `<root>/<split>/rgb/B`, `<root>/<split>/sem/A`, and `<root>/<split>/sem/B` through this manifest path, not from HDF5. The semantic maps are semantic supervision metadata, not binary masks.

RSCC is optional and disabled by default. Use `scripts/prepare_rscc_manifest.py` with `--caption-policy qvq_ground_truth_only|model_generated_only|all`; the default is `qvq_ground_truth_only`. The official 988-pair RSCC/xBD QvQ-Max subset is benchmark ground truth, but its captions are still `caption_source=model_generated`, not human annotation. Preserve `source_metadata.caption_generator="QvQ-Max"`, `benchmark_ground_truth=true`, `benchmark_subset="rscc_xbd_988"`, `training_default_enabled=false`, `license_family="xBD"` and `research_only=true`. QvQ rows default to `split="test"` unless an explicit authoritative split is present. Do not classify generated full-RSCC captions as human, and keep them distinguishable from QvQ-Max benchmark rows.

ChangeChat is optional and disabled by default. `scripts/prepare_changechat_retrieval_manifest.py` may extract declarative retrieval descriptions from captioning, localization, quantification and GPT-assisted description fields, but must not use bare questions as retrieval captions.

Mixed Stage-1-next training may use repeated manifest arguments and explicit weights:

```bash
python scripts/train_unichange_v2_stage1_next.py DATA_ROOT OUT UNIVERSAT CHECKPOINT JINA 32 25 8 \
  --temporal-depth 6 \
  --train-manifest manifests/stage1_next_mixed.jsonl \
  --val-manifest manifests/stage1_next_mixed.jsonl \
  --dataset-weight levir_mci=0.55 \
  --dataset-weight second_cc=0.45
```

Validation metrics must include combined metrics and per-dataset query subsets for LEVIR-MCI and SECOND-CC.

On Slurm, mixed training uses colon-separated environment variables:

```bash
TRAIN_MANIFESTS=/path/levir.jsonl:/path/second_cc.jsonl
VAL_MANIFESTS=/path/levir.jsonl:/path/second_cc.jsonl
DATASET_WEIGHTS=levir_mci=0.55:second_cc=0.45
```

Alternatively set `DATASET_CONFIG` to a JSON config containing `train_manifests`, `val_manifests`, `dataset_sampling_weights`, `allowed_caption_sources`, `semantic_soft_target_weight`, `semantic_teacher_top_k` and `semantic_teacher_temperature`.

Mixed evaluation uses the same validation manifest contract:

```bash
VAL_MANIFESTS=/path/levir.jsonl:/path/second_cc.jsonl
DATASET_WEIGHTS=levir_mci=0.55:second_cc=0.45
```

`evaluate_unichange_v2_stage1_next.sbatch` keeps LEVIR-only compatibility when neither `VAL_MANIFESTS` nor `DATASET_CONFIG` is set. Mixed training readiness must use a mixed smoke report; LEVIR-only smoke is not acceptable for mixed manifest training.

### Required validation

```bash
source "$HOME/rschange_env.sh"
cd "$CODE_ROOT"
PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts:$CODE_ROOT" \
  "$RSCHANGE_PYTHON" -m compileall -q src scripts tests
PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts:$CODE_ROOT" \
  "$RSCHANGE_PYTHON" -m pytest -q
bash -n cluster/ysu/*.sbatch
```

Then run exact-commit Stage-1-next smoke and memory probes. Require Slurm `COMPLETED 0:0`, valid reports, real H100/BF16, 10 finite steps, correct gradients, checkpoint roundtrip, `text_max_length=256` for the semantic experiment and a physical local batch of at least 32.

Never claim training, evaluation or a cluster gate succeeded without user-provided Slurm state and report contents.
