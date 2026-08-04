# QCPR Model-v3 real integration smoke decision

Status: MODEL_V3_REAL_INTEGRATION_SMOKE = FAIL (post-run engineering diagnostic)

- Job: 207560
- Run root: /mnt/weka/svardanyan/rs_change_project/runs/qcpr_v3_real_integration_smoke_4aaa764_20260805
- Executed code SHA: 4aaa7649daa0079689b577cf63184ca01c05f4e7
- Node: gpu04, H100, exit 1:0
- Requested/completed optimizer steps: 8/8
- Real contract: 8 physical pairs x 2 captions = 16 queries; score matrix 16x8; native target [B,2,1024,768].

The real UniSat and Jina paths loaded, the training loop reached checkpoint global step 8, and the fresh-process checkpoint roundtrip passed with zero score difference. The job failed afterward in the evidence deletion diagnostic because the helper divided flattened temporal-spatial indices by the hidden dimension rather than the number of spatial tokens per frame. This is classified as ENGINEERING_RUNTIME_DIAGNOSTIC_BUG, not a scientific retrieval result.

A safe deletion helper and regression test were added after the job. No second H100 job was submitted. Therefore:

- MODEL_V3_SYNTHETIC_SMOKE = PASS
- MODEL_V3_REAL_INTEGRATION_SMOKE = FAIL
- MODEL_V3_MECHANISM = BLOCKED
- MODEL_V3_MAIN = NOT_AUTHORIZED
- P2_REAL/P2_SEMANTIC/P2_LONG_SERIES = BLOCKED_DATASET_CONTRACT

No retrieval improvement is claimed.
