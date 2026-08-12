# QCPR r20 exact-supervision report

## Status

`HOLD_HUMAN_REVIEW_REQUIRED`

The immutable r19g release remains unchanged. The 1,500-row independent review
package is ready, but both reviewer decision sheets are empty and adjudication
has not started. Agent diagnostics and automated attribute heuristics are not
used as human labels.

## Required calibration

| stratum | required | completed |
|---|---:|---:|
| exact_discriminative | 300 | 0 |
| semantic_multi_positive | 300 | 0 |
| generic_no_change | 300 | 0 |
| stable_scene_specific | 300 | 0 |
| localized_direction | 300 | 0 |
| **total** | **1500** | **0** |

Valid human estimates are therefore unavailable:

- exact-identifiability precision: `null`
- weak-caption rate: `null`
- semantic-alternative rate: `null`
- source-conditioned values: `null`

The exact-discriminative stratum is source-balanced by design: 150 unique
LEVIR physical pairs and 150 unique SECOND physical pairs. This makes the
future source-conditioned estimate identifiable, but does not create a human
estimate before review.

## r19g preservation

- `R19G_IMMUTABLE = true`
- train: 23124 queries / 7291 pairs
- development: 3181 queries / 995 pairs
- test: 779 queries / 244 pairs
- legacy development remains frozen exact evaluation

## r20 candidate

- candidate lineage: `CANDIDATE_HOLD`
- r20 exact-train manifest: **not materialized** until review/adjudication
- `R20_TRAIN_QUERY_COUNT = null`
- `R20_TRAIN_PAIR_COUNT = null`
- `R20_MATCHED_EXACT_TRAINING_READY = false`
- `DATASET_AGENT_RECOMMENDATION = DO_NOT_TRAIN_YET`

The correct next action is to have two independent human reviewers complete the
package at `/mnt/weka/svardanyan/rs_change_project/manifests/qcpr_r20_candidate_exact_supervision_20260812/human_review_package`, adjudicate disagreements, and rerun the
promotion builder. No model training is authorized by this report.
