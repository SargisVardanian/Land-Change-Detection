# UniChange v2 Stage-1 handoff and next experiment

## Purpose

Stage-1 learns text-to-bi-temporal-pair retrieval. It is not yet captioning, grounding, segmentation or pair-to-pair retrieval. The long-term project should reuse the learned change representation in later tasks, so the representation must preserve direction, location and semantics of change rather than only separate generic no-change from changed scenes.

## Why the baseline remains valuable

The first full run is a controlled baseline: frozen UniverSat and Jina, four trainable temporal/spatial blocks, simple visual projection, symmetric multi-positive contrastive learning, physical batch 32. Its early validation improved strongly, then slowed around epochs 6–8. This is not enough to conclude that the encoder is too small.

The baseline exposed several structural issues:

- Without temporal position/direction embeddings and with temporal averaging, T1/T2 reversal can be nearly invisible to the model.
- The original positive objective averages `-log p` across every positive. With five captions and duplicated text, this imposes a positive-count-dependent loss floor.
- Training inferred duplicates from nearly identical embeddings, while evaluation used normalized text hashes.
- All five captions were used every step, amplifying generic repeated templates.
- Aggregate duplicate-aware R@1 can be dominated by frequent no-change language and hide poor exact or rare-caption retrieval.
- Best-checkpoint selection used only duplicate-aware R@1.
- Gradient clipping was applied without logging how often it was active.
- Weight decay was applied indiscriminately.

## Stage-1-next architecture

The next path keeps the frozen backbones and the 512-dimensional shared space. The change encoder adds:

```text
before = proj(F1) + e_before
after  = proj(F2) + e_after

explicit = MLP(concat(
    before,
    after,
    after - before,
    abs(after - before),
    before * after
))
```

The explicit change context is injected into both temporal streams and added to the final change tokens. This makes appearance/disappearance direction representable without depending on the network to discover subtraction from a small dataset.

Depth remains four for the first next experiment. A six-block ablation should be run only after the objective/data fixes are measured.

## Stage-1-next objective

For a query with a positive set `P`, use:

```text
L_set = logsumexp(all logits) - logsumexp(positive logits)
```

This maximizes probability mass assigned to the valid set. It aligns with duplicate-aware retrieval and avoids requiring uniform high probability for every positive.

Stage-1-next optimizes semantic text-to-pair process retrieval, not exact-pair retrieval and not pair-to-pair retrieval. Exact normalized-caption groups remain hard positives. In addition, each batch builds a detached teacher relevance matrix from frozen pre-adapter Jina text embeddings:

- compare each text query to every caption teacher embedding for each candidate pair;
- aggregate caption similarities to pair relevance with max;
- force original caption-to-pair and exact normalized-caption-group associations to relevance `1.0`;
- block contradictory temporal directions, including no-change versus changed, appeared/constructed/added versus disappeared/demolished/removed, and increased/expanded versus decreased/reduced;
- keep top-k semantic candidates per query and normalize to a teacher target distribution.

The text-to-pair loss is:

```text
L_text_to_pair =
  (1 - semantic_soft_target_weight) * L_set
  + semantic_soft_target_weight * L_semantic_teacher
```

Defaults are `semantic_soft_target_weight=0.25`, `semantic_teacher_top_k=8`, `semantic_teacher_temperature=0.05` and Jina `text_max_length=256`.

The default directional weighting is:

```text
0.75 * L(text -> pair) + 0.25 * L(pair -> text)
```

Text-to-pair is the deployed direction. Pair-to-text remains as a regularizer but is intrinsically more ambiguous because one pair has several valid captions.

Use stable normalized caption hashes in both training and evaluation. Use a bounded trainable logit scale initialized at temperature 0.07.

Do not add a pair-to-pair projection head, loss, sampler or checkpoint metric for Stage-1-next.

## Caption sampling

Training samples at most two captions from each pair per epoch. One slot prefers the most detailed caption, using a bounded score for token count plus directional, object, count, location and size/severity indicators. The other slot keeps the deterministic rare-caption/rotation logic for `(seed, epoch, pair_id)`. Validation and final evaluation use all captions.

This reduces domination by generic templates while retaining full coverage across epochs. It does not delete official dataset rows.

## Diagnostics

Every epoch should record:

- semantic recall@1/@5/@10 and semantic nDCG@5/@10;
- detailed-query, directional-query, location-query and count-query R@5/R@10;
- per-dataset cross-corpus and within-dataset semantic metrics plus macro validation-dataset averages;
- duplicate-aware R@1/R@5/R@10, MRR, median and mean rank as continuity diagnostics;
- exact-pair R@1/R@5/R@10, exact MRR and ranks as diagnostics only;
- unique-caption, rare-caption (2–5 relevant pairs) and frequent-caption metrics;
- no-change/small/medium/large mask strata when masks are available;
- fixed train-subset retrieval metrics every two epochs;
- pre-clip gradient mean/p90 and clipping fraction;
- learned logit scale/effective temperature;
- throughput and peak VRAM.

A train/validation gap is required to diagnose capacity:

- train and validation both weak: objective, capacity or frozen-backbone bottleneck;
- train strong and validation weak: overfitting, ambiguity or label noise;
- aggregate strong but unique/exact weak: generic-caption domination;
- direction-specific queries weak: temporal-direction bottleneck.

## Checkpoint policy

Save independent best checkpoints for:

- `best_semantic_r1.pt`
- `best_semantic_r5.pt`
- `best_semantic_ndcg10.pt`
- `best_detailed_r5.pt`
- `best_macro_semantic.pt`
- `best_composite.pt`
- `best_retrieval.pt` as an alias of the composite winner
- `last_retrieval.pt`

The composite score is:

```text
0.30 * macro semantic R@1
+ 0.25 * macro semantic R@5
+ 0.20 * macro semantic R@10
+ 0.15 * macro semantic nDCG@10
+ 0.10 * detailed-query R@5
```

Exact metrics must not affect best scores, filenames, composite scoring or readiness success criteria. Do not save `best_exact_r10.pt`. Early stopping monitors `composite` with default patience `4` validation epochs and minimum improvement `0.002`; always keep `last_retrieval.pt`.

## Dataset cleaning policy

Do not mutate the official LEVIR-MCI split for the main comparable result. Build an optional clean-conflict ablation that flags, rather than silently deletes:

- all no-change captions with mask fraction above 0.05;
- changed-caption consensus with an empty mask;
- appeared/disappeared contradictions within one pair;
- simultaneous disagreement among caption consensus, changeflag and mask.

Compare official-full and conflict-filtered training. Masks are coarse and can be wrong, so disagreement is not automatically proof that captions are wrong.

## Multi-dataset manifest pipeline

Stage-1-next now has a canonical temporal-caption manifest schema for LEVIR-MCI, SECOND-CC and optional RSCC. Every JSONL row carries:

```text
schema_version
dataset_name
pair_id                 # levir_mci:<split>:<original_id>, second_cc:<split>:<original_id>, rscc:<split>:<original_id>
original_id
split
t1_path
t2_path
captions
normalized_caption_groups
caption_source          # human | model_generated | semantic_template
time_order              # [before, after]
mask_path
semantic_t1_path
semantic_t2_path
sensor
spatial_resolution
image_width
image_height
source_metadata
preprocessing_fingerprint
```

The manifest scripts preserve official train/val/test splits and never resize or recompress images. SECOND-CC is intentionally loaded from raw files or JSON/JSONL/CSV annotations, not from HDF5. RSCC support is adapter-only and not part of the default training mix; default RSCC policy is `human_subset_only`, model-generated RSCC captions require explicit opt-in and remain identified by `caption_source`.

Commands:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD"

python scripts/prepare_levir_mci_manifest.py \
  --root /path/to/LEVIR-MCI \
  --output manifests/levir_mci.jsonl \
  --audit-report reports/levir_mci_manifest_audit.json

python scripts/prepare_second_cc_manifest.py \
  --root /path/to/SECOND-CC \
  --annotations /path/to/SECOND-CC-AUG.json \
  --expected-pairs 6041 \
  --expected-captions 30205 \
  --output manifests/second_cc.jsonl \
  --audit-report reports/second_cc_manifest_audit.json

python scripts/prepare_rscc_manifest.py \
  --root /path/to/RSCC \
  --annotations /path/to/RSCC_qvq.jsonl \
  --caption-policy human_subset_only \
  --expected-pairs EXPECTED \
  --output manifests/rscc_human.jsonl \
  --audit-report reports/rscc_manifest_audit.json

python scripts/audit_temporal_caption_manifest.py \
  manifests/levir_mci.jsonl manifests/second_cc.jsonl \
  --output reports/mixed_manifest_audit.json

python scripts/merge_temporal_caption_manifests.py \
  --manifest manifests/levir_mci.jsonl \
  --manifest manifests/second_cc.jsonl \
  --output manifests/stage1_next_mixed.jsonl \
  --audit-report reports/stage1_next_mixed_manifest_audit.json
```

The initial recommended training mix is LEVIR-MCI weight `0.55` and SECOND-CC weight `0.45`:

```bash
python scripts/train_unichange_v2_stage1_next.py DATA_ROOT OUT UNIVERSAT CHECKPOINT JINA 32 25 8 \
  --temporal-depth 6 \
  --train-manifest manifests/stage1_next_mixed.jsonl \
  --val-manifest manifests/stage1_next_mixed.jsonl \
  --dataset-weight levir_mci=0.55 \
  --dataset-weight second_cc=0.45
```

SECOND-CC uses the official raw layout `<root>/<split>/rgb/A/<filename>`, `<root>/<split>/rgb/B/<filename>`, `<root>/<split>/sem/A/<filename>`, and `<root>/<split>/sem/B/<filename>`. `sem/A` and `sem/B` are stored as `semantic_t1_path` and `semantic_t2_path`; they are not binary masks.

The dataloader propagates `dataset_name` into every batch. Validation reports combined metrics plus two per-dataset modes from the same deterministic rank tensor: cross-corpus query-slice metrics and within-dataset query-and-candidate metrics.

Slurm mixed training accepts:

```bash
TRAIN_MANIFESTS=/path/levir.jsonl:/path/second_cc.jsonl
VAL_MANIFESTS=/path/levir.jsonl:/path/second_cc.jsonl
DATASET_WEIGHTS=levir_mci=0.55:second_cc=0.45
```

or a `DATASET_CONFIG` JSON with `train_manifests`, `val_manifests`, `dataset_sampling_weights`, `allowed_caption_sources`, `semantic_soft_target_weight`, `semantic_teacher_top_k` and `semantic_teacher_temperature`.

Mixed evaluation accepts `VAL_MANIFESTS` plus `DATASET_WEIGHTS`, or the same `DATASET_CONFIG`. The evaluation report must include `data_mode`, validation manifest fingerprints, dataset names, validation row counts, combined metrics, and both cross-corpus and within-dataset per-dataset metrics. LEVIR-only evaluation remains the fallback only when no manifest configuration is supplied.

The Stage-1-next readiness gate is data-aware: mixed training requires a mixed smoke report whose manifest fingerprints, dataset names, row counts and weights match the requested training manifests. The smoke and memory reports also gate `text_max_length`, default `256`. The memory probe is only an architecture/batch shape validation and reports `memory_data_mode=shape_probe`; it is not evidence that SECOND-CC ingestion was validated.

## Cluster workflow

Develop in a separate worktree. New code invalidates old smoke/memory reports. Run:

1. full tests and shell syntax checks;
2. `smoke_unichange_v2_stage1_next.sbatch`;
3. dependent `probe_unichange_v2_stage1_next_memory.sbatch`;
4. inspect both reports and Slurm `COMPLETED 0:0`;
5. submit `train_unichange_v2_stage1_next.sbatch` only if physical batch 32 fits;
6. chain `evaluate_unichange_v2_stage1_next.sbatch` after successful training, using the clean 48-query qualitative manifest and full validation corpus metrics.

The existing baseline scripts must remain usable for reproducibility.
