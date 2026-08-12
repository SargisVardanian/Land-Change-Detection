# QCPR SigLIP-2 Phase-A decision package

Status: `PHASE_A_BOUNDED_PILOT_COMPLETE_PHASE_B_BLOCKED`

Code SHA: `ddc25d2bd21922544ee9ef217e5b3f4b62896e18`  
Branch: `codex/qcpr-siglip2-temporal-training`  
PR: [#7](https://github.com/SargisVardanian/Land-Change-Detection/pull/7), draft

The approved exact core and common development gallery were unchanged:

- 6,310 exact queries;
- 1,928 physical pairs;
- full score matrix `6310 × 1928`;
- mask-free integrity passed.

## Bounded comparison

| Track | Job | Steps | MRR step 0 | MRR step 128 | MRR step 256 | Hit@10 step 256 |
|---|---:|---:|---:|---:|---:|---:|
| GeoRSCLIP temporal head | 208085 | 256 | 0.003683 | 0.007764 | 0.008140 | 0.013471 |
| SigLIP2 Phase A | 208161 | 256 | 0.005998 | 0.013833 | 0.014156 | 0.023138 |

SigLIP2 Phase A step 256 also reduced mean/median rank to `407.57/278`,
compared with `892.89/860` at its step-zero checkpoint. This is bounded,
single-seed pilot evidence on one exact gallery, not a final model-selection
claim.

Per-source step-256 global metrics for SigLIP2:

| Source | MRR | Hit@10 | Mean rank | Median rank |
|---|---:|---:|---:|---:|
| LEVIR-MCI | 0.010219 | 0.014393 | 474.25 | 387 |
| SECOND-CC | 0.018570 | 0.032941 | 332.82 | 177 |

## Evidence-path result

The evidence path is technically present and receives gradients during
training, but Phase-A did not make it a useful spatial reranker:

- map shape: `[6310, 2, 16, 16]`;
- effective token count at step 256: approximately `493/512`;
- 100% of maps exceed the 95%-uniform entropy threshold;
- Top-20/50/100 reranking changed MRR only at floating-point noise level.

Therefore the result supports a global temporal retrieval signal, while the
query-conditioned evidence mechanism remains scientifically unvalidated.

## Engineering disposition

- Geo jobs `208051` and `208083` are preserved engineering-only failures;
  valid replacement: `208085`.
- SigLIP2 milestone evaluator `208211` is preserved as an evaluation-only OOM
  at rerank query batch 4. Retry `208232` passed all milestones with batch 1.
- The active evaluator default is now rerank query batch 1.
- No Dataset-v2 manifest, historical R1/C0/C1 artifact, P2 job or main run was
  modified/launched.

## Readiness

`B1_CONTROL = VERIFIED`  
`GEORSCLIP_TEMPORAL_256 = PASS_BOUNDED_COMMON_GALLERY_PILOT`  
`SIGLIP2_PHASE_A = PASS_BOUNDED_COMMON_GALLERY_PILOT`  
`SIGLIP2_PHASE_B = NOT_AUTHORIZED_REQUIRES_DECISION_PACKAGE`  
`MODEL_V3_MAIN = NOT_AUTHORIZED`  
`P2_REAL/P2_SEMANTIC/P2_LONG_SERIES = BLOCKED_DATASET_CONTRACT`

No final model-improvement or state-of-the-art claim is made. Phase B (1536
additional steps) requires a separate authorization after review of this
package.
