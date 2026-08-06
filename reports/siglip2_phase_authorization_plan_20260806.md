# SigLIP-2 phase authorization plan — 2026-08-06

The exact-core technical gates currently available are complete, but they do
not establish a trained retrieval improvement.

## Completed bounded evidence

1. Real SigLIP-2 integration smoke: job `208013`, 8 steps, physical batch 8,
   score matrix `[16,8]`, finite gradients, frozen-backbone isolation,
   evidence causality diagnostics and checkpoint round-trip PASS.
2. Corrected SigLIP-2 frozen common-gallery evaluation: job `208021`, 6,310
   exact queries and the complete 1,928-pair gallery.
3. Frozen GeoRSCLIP step-zero common-gallery evaluation: job `208024`, the
   same 6,310 x 1,928 contract, zero optimizer steps.

## Not launched

The following remain intentionally unlaunched because the current task
explicitly limits GPU jobs to at most 32 steps:

1. H100 batch calibration beyond the completed historical calibration.
2. Frozen GeoRSCLIP temporal-head baseline at 256 steps.
3. SigLIP-2 Phase A at 256 steps.
4. SigLIP-2 Phase B at 1,536 additional steps.

No mechanism pilot, main training run, P2 run, or unrestricted production
training was submitted.

## Authorization gates for a later task

Before a longer run, require the exact published SHA and clean worktree,
full current suite, frozen common-gallery artifacts, explicit approved
training budget, and the unchanged LEVIR-MCI + SECOND-CC exact-core contract.
The later report must compare full-gallery rankings and must not infer quality
from loss curves or step-zero embeddings.

## Dataset handoff

Verified semantic, localized, directional, stable-scene and long-series query
views remain Dataset-Agent requirements. Their absence blocks P2-real,
P2-semantic and P2-long-series, but does not invalidate the exact-core
technical smoke.

## Current status

```text
SIGLIP2_REAL_INTEGRATION_SMOKE = PASS_TECHNICAL_SMOKE
SIGLIP2_COMMON_GALLERY_STEP0 = PASS
GEORSCLIP_STEP0 = PASS_FROZEN_EVALUATION
GEORSCLIP_256_STEP_BASELINE = NOT_AUTHORIZED
SIGLIP2_PHASE_A = NOT_AUTHORIZED
SIGLIP2_PHASE_B = NOT_AUTHORIZED
P2 = BLOCKED_DATASET_CONTRACT
```
