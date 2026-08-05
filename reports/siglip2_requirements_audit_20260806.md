# SigLIP-2 track requirements audit

This audit deliberately separates technical integration from scientific
retrieval evidence. It is a model-side report only: Dataset-v2 manifests and
historical R1/C0/C1 artifacts were not modified.

## Verified

- Branch code was published and cluster-synchronized.
- Official SigLIP-2 and GeoRSCLIP source audits exist with pinned revisions and
  weight hashes.
- The minimal SigLIP-2 model has two temporal Transformer blocks, one pair
  token, one blockwise query-conditioned evidence path, and one listwise loss.
- The real H100 smoke `207780` completed 16 steps with finite gradients,
  frozen-backbone isolation, checkpoint roundtrip, and checksum validation.
- Chunked evidence values and gradients match the unchunked reference in CPU
  regression tests.

## Explicitly not claimed

- The smoke is not a retrieval-quality result.
- Full-gallery rankings, frozen step-0 metrics, GeoRSCLIP's 256-step baseline,
  SigLIP-2 Phase A, and Phase B have not been run.
- The selected smoke did not exercise a multi-positive query.
- The selected smoke failed the top-evidence deletion criterion; mechanism
  readiness remains blocked.
- Time-reversal instrumentation was added after `207780`; it is available for
  the next bounded smoke but is not retroactively attributed to that run.

## Current readiness

```text
SIGLIP2_DOWNLOAD = VALID
SIGLIP2_SMOKE = PASS
SIGLIP2_MECHANISM = BLOCKED_EVIDENCE_DELETION
SIGLIP2_PHASE_A = NOT_AUTHORIZED
SIGLIP2_PHASE_B = NOT_AUTHORIZED
SIGLIP2_MAIN_MODEL = NOT_AUTHORIZED
GEORSCLIP_BASELINE = WRAPPER_VALID_BASELINE_NOT_RUN
P2_REAL/P2_SEMANTIC/P2_LONG_SERIES = BLOCKED_DATASET_CONTRACT
```

The current task intentionally stops before any >32-step job. A later
authorization is required for H100 calibration, 256-step comparison, and the
Phase-B continuation.
