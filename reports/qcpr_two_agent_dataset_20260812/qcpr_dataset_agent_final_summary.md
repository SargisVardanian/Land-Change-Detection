# QCPR Dataset Agent final reviewer summary

- Status: `PASS_AGENT_DIAGNOSTIC_REVIEW`
- Producer: `DATASET_AGENT`
- Producer git SHA: `99bf9d32583fa4de4764662c3e040f950bfb5698`
- Dataset release: `qcpr_bitemporal_v2_train_20260808_final_r19g`
- Dataset release SHA: `6d90faeded7a0cadfd9e0a46b48c9d6b9d623cea2d9cbd7439b07bedc62ab871`
- Updated at: `2026-08-11T23:16:40Z`
- Model: `B20_step_1140`, canonical budget `20`

## Dataset-side verdict

`DATASET_AGENT_VERDICT = SUPERVISION_DATA_DOMINANT`

The frozen pilot contains 200 cases: 100 with B20 exact rank >10, 40 with rank 2–10, and 60 rank-1 controls. It is balanced by source (102 LEVIR-MCI / 98 SECOND) and includes 87 collision-group and 113 non-collision cases. Review type is `AGENT_DIAGNOSTIC_REVIEW`, not human verification.

Failure counts, excluding the 60 controls:

| Category | Count |
|---|---:|
| `SEMANTICALLY_VALID_NONEXACT_RESULT` | 63 |
| `WEAK_EXACT_CAPTION` | 57 |
| `MULTI_CHANGE_COMPOSITION_ERROR` | 8 |
| `TEMPORAL_DIRECTION_ERROR` | 4 |
| `CAPTION_COLLISION` | 2 |
| `UNKNOWN` | 6 |

`SUPERVISION AMBIGUITY MATERIAL = YES`. Multi-change (8) and temporal-direction (4) are small-n diagnostic signals only; no model-specific family passes the clear-enrichment gate. `RECOMMENDED NEXT MODEL ACTION = KEEP_BASE`.

## Contract and safety decision

- Primary review ranking was `B20_global_stage1`; reranked scores were not mixed.
- No relevance labels or manifests were changed.
- No hard negatives were created.
- `DIRECTION_TRAIN_READY = false` and `DIRECTION_EVAL_READY = false` remain unchanged.
- No vNext architecture or expanded training is authorized by this handoff.

## Published artifacts

- Sample: `qcpr_model_data_pilot_sample.jsonl`, SHA256 `a4c097c994d7692bd9862ceb817d6e10cbdc217f447787d1a477ffaa9663ed0c`
- Results: `qcpr_model_data_pilot_results.json`, canonical artifact SHA256 `fc15a9e04af7a20f34ecba71c9788073c4ebbafc3ae9366892d8d188b122ce5a`
- Dataset→Model handoff: `handoff_dataset_to_model.json`, canonical artifact SHA256 `1982514aed12fc6d19c9a7c2eff2b6990634f8b4e542e2350bddd1cbff740a40`

All artifacts are tied to the same r19g release SHA and the model handoff was validated against the same 3,181-query / 995-gallery development contract.
