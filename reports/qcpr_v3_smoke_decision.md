# QCPR v3 bounded smoke decision

## Synthetic smoke

- Code SHA: 5eab23719aa78515383befd97145c49ffe5fc714
- Job: 207556, COMPLETED, exit 0:0, node gpu02
- Contract: 16 fixed optimizer steps; synthetic matrix 4 x 4; no masks
- Checkpoint SHA256: a151c1f22237dabe0389109fd9bd922850f16eabacf411d4e26a4b632a0b53ee

This is MODEL_V3_SYNTHETIC_SMOKE = PASS, not a real retrieval result.

## Real integration smoke

- Code SHA: 4aaa7649daa0079689b577cf63184ca01c05f4e7
- Job: 207560, FAILED, exit 1:0, node gpu04
- Run root: /mnt/weka/svardanyan/rs_change_project/runs/qcpr_v3_real_integration_smoke_4aaa764_20260805
- Contract reached: 8 optimizer steps; 8 physical pairs x 2 captions = 16 queries; score matrix 16 x 8; native target [B,2,1024,768]
- Real UniSat and Jina paths loaded.
- Checkpoint global step: 8.
- Fresh-process checkpoint roundtrip: PASS; maximum score difference 0.
- Failure: ENGINEERING_RUNTIME_DIAGNOSTIC_BUG in evidence deletion indexing. The helper used the hidden dimension instead of the spatial-token axis.
- Peak CUDA and final batch-contract files were not written because the job failed in post-run diagnostics.

A safe deletion helper and regression test were added after the job. No second H100 job was submitted. The real smoke is therefore FAIL, not a scientific retrieval-quality result.

Readiness:

- MODEL_V3_SYNTHETIC_SMOKE = PASS
- MODEL_V3_REAL_INTEGRATION_SMOKE = FAIL
- MODEL_V3_MECHANISM = BLOCKED
- MODEL_V3_MAIN = NOT_AUTHORIZED
- P2_REAL/P2_SEMANTIC/P2_LONG_SERIES = BLOCKED_DATASET_CONTRACT

Job 207555 is classified as ENGINEERING_RUNTIME_DRIVER_BUG: the driver reused one autograd graph across optimizer steps. It remains historical evidence and was not rewritten.

No model improvement is claimed.
