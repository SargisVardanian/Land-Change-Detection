# SigLIP-2 track requirements audit

This is a model-side audit. Dataset-v2 manifests and historical R1/C0/C1
artifacts were not modified.

## Verified

- The official SigLIP-2 source is pinned to revision
  `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`, with local safe-tensor hash
  `6125cacc01fa93bdc98a0c5101cefcd69b2ed1f8ab4f38d86f4ad5984f5dc863`.
- The GeoRSCLIP source is pinned to revision
  `4920188e6eba4e711ef9848cfd7cb77e874ee33f`, with checkpoint hash
  `129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01`.
- The corrected real H100 smoke `208013` completed 8 steps with the real
  loader, real tokenizer, frozen towers, finite gradients, checkpoint
  round-trip and evidence-path diagnostics.
- The corrected SigLIP-2 frozen evaluation `208021` uses 6,310 exact queries
  against the complete 1,928-pair gallery. The earlier 1,262-pair diagnostic
  is not used for the current benchmark.
- The GeoRSCLIP frozen step-zero evaluation `208024` uses the same full
  6,310 x 1,928 gallery/query contract and has zero optimizer steps.
- Code-quality audit: 28 package modules, no import cycles, no duplicate
  public symbols, no tracked model binaries, compileall/Pyright/shell/diff
  checks pass. Ruff is unavailable in the cluster image.

## Explicitly not claimed

- No retrieval-quality improvement is claimed from a smoke or frozen step-zero
  evaluation.
- The 256-step GeoRSCLIP head-only baseline was not run.
- SigLIP-2 Phase A and Phase B were not run.
- The selected smoke did not exercise a multi-positive query.
- P2, RSCC text and any generated-unverified text were not used.
- No pixel segmentation or mask-supervised training was run.

## Current bounded-task readiness

```text
SIGLIP2_DOWNLOAD = VALID
SIGLIP2_REAL_INTEGRATION_SMOKE = PASS_TECHNICAL_SMOKE
SIGLIP2_COMMON_GALLERY_STEP0 = PASS
GEORSCLIP_STEP0 = PASS_FROZEN_EVALUATION
SIGLIP2_MECHANISM = READY_FOR_AUTHORIZATION
SIGLIP2_PHASE_A = NOT_AUTHORIZED
SIGLIP2_PHASE_B = NOT_AUTHORIZED
SIGLIP2_MAIN_MODEL = NOT_AUTHORIZED
P2_REAL/P2_SEMANTIC/P2_LONG_SERIES = BLOCKED_DATASET_CONTRACT
```

The current task intentionally stops before any job over 32 steps. A later
explicit authorization is required for H100 calibration, the 256-step
GeoRSCLIP comparison, and SigLIP-2 training phases.
