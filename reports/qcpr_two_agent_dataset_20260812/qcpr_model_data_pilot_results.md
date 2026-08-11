# QCPR MODEL×DATA diagnostic pilot

- Status: **PASS_AGENT_DIAGNOSTIC_REVIEW**
- Review type: **AGENT_DIAGNOSTIC_REVIEW**
- Dataset release: `qcpr_bitemporal_v2_train_20260808_final_r19g` / `6d90faeded7a0cadfd9e0a46b48c9d6b9d623cea2d9cbd7439b07bedc62ab871`
- Canonical model: `B20_step_1140` (20 presentations per physical pair)
- Sample: `200` cases, seed `20260812`, sample SHA `a4c097c994d7692bd9862ceb817d6e10cbdc217f447787d1a477ffaa9663ed0c`
- Primary ranking: `B20.global_stage1` only; reranked ranks were not mixed into the review.

## Verdict

`DATASET_AGENT_VERDICT = SUPERVISION_DATA_DOMINANT`

Non-control cases: `140`; supervision-ambiguity evidence: `122`; model-family evidence: `12`; unresolved: `6`.

## Category counts

| Category | Count |
|---|---:|
| CAPTION_COLLISION | 2 |
| MULTI_CHANGE_COMPOSITION_ERROR | 8 |
| SEMANTICALLY_VALID_NONEXACT_RESULT | 63 |
| TEMPORAL_DIRECTION_ERROR | 4 |
| TOP1_CONTROL | 60 |
| UNKNOWN | 6 |
| WEAK_EXACT_CAPTION | 57 |

## Failure-family enrichment vs rank-1 controls

| Family | Failure n | Failure rate | Control n | Control rate | Enriched |
|---|---:|---:|---:|---:|:---:|
| SPATIAL_COMPOSITION_ERROR | 0 | 0.000 | 0 | 0.000 | false |
| SMALL_OBJECT_RESOLUTION_ERROR | 0 | 0.000 | 0 | 0.000 | false |
| MULTI_CHANGE_COMPOSITION_ERROR | 8 | 0.057 | 0 | 0.000 | false |
| TEMPORAL_DIRECTION_ERROR | 4 | 0.029 | 0 | 0.000 | false |
| SOURCE_STYLE_SHORTCUT | 0 | 0.000 | 0 | 0.000 | false |
| REGISTRATION_VIEWPOINT_ERROR | 0 | 0.000 | 0 | 0.000 | false |
| UNKNOWN | 6 | 0.043 | 0 | 0.000 | false |

TOP 3 FAILURE CATEGORIES = SEMANTICALLY_VALID_NONEXACT_RESULT, WEAK_EXACT_CAPTION, MULTI_CHANGE_COMPOSITION_ERROR
MODEL-SPECIFIC FAILURE FAMILY SUPPORTED = NO (multi-change n=8 and temporal-direction n=4 are small-n diagnostic signals only)
SUPERVISION AMBIGUITY MATERIAL = YES
RECOMMENDED NEXT MODEL ACTION = KEEP_BASE

## Guardrails

- No frozen relevance labels were changed.
- No hard negatives were created.
- Direction train/eval readiness remains false.
- Route-related failure was not linked because route fields are absent from the exported Top-20.
- Small-n family counts are reported as diagnostic signals and are not promoted as clear enrichment.
- This diagnostic does not authorize vNext or expanded training.
