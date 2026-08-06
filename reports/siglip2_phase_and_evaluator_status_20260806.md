# SigLIP-2 phase/evaluator status — 2026-08-06

Code is published at `229ebf149697c82b5d64375c2e7ee1c7214506ea` on
`codex/qcpr-siglip2-temporal-training`; PR #7 remains draft and mergeable.

## Implemented and validated

- real mask-free image/text runtime helpers for the approved LEVIR-MCI +
  SECOND-CC exact core;
- GradCache-style one-logical-matrix listwise training;
- explicit positive/ignored masks without caption-collision inference;
- per-module gradient and frozen-backbone audits;
- guarded Phase-A/B driver and Slurm launcher;
- corrected full-gallery evaluator with 6,310 queries and 1,928 physical
  gallery items;
- Top-K evidence reranking;
- frozen GeoRSCLIP step-zero evaluator;
- deterministic ranking, exposure and artifact hashes.

## Verified jobs

### Real SigLIP-2 integration smoke — job 208013

Completed `0:0` on H100 `gpu05`: 8 steps, physical batch 8, 16 queries,
score matrix `[16,8]`, native visual `[8,2,256,768]`, temporal `[8,512,768]`,
text `[16,64,768]`, peak allocated/reserved `2.001/2.031 GiB` and checkpoint
round-trip PASS. Query swap, evidence zeroing, top-vs-bottom deletion,
detach-gradient and time-reversal diagnostics passed. No explicit
multi-positive query was exercised.

### SigLIP-2 common-gallery step-zero — job 208021

Completed `0:0` with full score matrix `[6310,1928]`. Full-gallery MRR is
`0.0059978589` and Hit@10 is `0.0087163234`. This is frozen/step-zero
evaluation, not a trained-improvement claim.

### GeoRSCLIP step-zero — job 208024

Completed `0:0` with the same `[6310,1928]` contract, zero optimizer steps,
MRR `0.0036828369` and Hit@10 `0.0030110935`. The temporal head is untrained;
this is a frozen reference only.

## Validation

- full suite: **648 passed, 3 skipped, 16 warnings** in canonical cluster
  Python (`379.09 s`);
- focused SigLIP-2 contracts: **39 passed, 2 warnings**;
- Pyright: `0 errors, 0 warnings`;
- compileall, shell syntax and `git diff --check`: PASS;
- Ruff: unavailable in the cluster image and recorded as such.

## Not run

The following remain intentionally unlaunched:

- H100 batch calibration in this bounded task;
- 256-step GeoRSCLIP temporal-head baseline;
- SigLIP-2 Phase A at 256 steps;
- SigLIP-2 Phase B at 1,536 additional steps;
- mechanism pilot, main training and P2.

## Readiness

```text
B1_CONTROL = VERIFIED
SIGLIP2_SYNTHETIC_SMOKE = PASS
SIGLIP2_REAL_INTEGRATION_SMOKE = PASS_TECHNICAL_SMOKE
SIGLIP2_COMMON_GALLERY_STEP_ZERO = PASS
GEORSCLIP_STEP_ZERO = PASS_FROZEN_EVALUATION
SIGLIP2_MECHANISM = READY_FOR_AUTHORIZATION
SIGLIP2_PHASE_A = NOT_AUTHORIZED
SIGLIP2_PHASE_B = NOT_AUTHORIZED
SIGLIP2_MAIN = NOT_AUTHORIZED
P2_REAL/P2_SEMANTIC/P2_LONG_SERIES = BLOCKED_DATASET_CONTRACT
```
