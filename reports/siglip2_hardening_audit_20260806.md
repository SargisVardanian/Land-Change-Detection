# SigLIP-2 hardening audit — 2026-08-06

## Decision

The hardened real integration smoke completed successfully. Job `208013` on
`gpu05` ran the real SigLIP-2 towers, real image loader, real tokenizer and
eight BF16 optimizer steps on the exact LEVIR-MCI + SECOND-CC core. It is a
technical integration and mechanism smoke, not a retrieval-quality result.

The corrected frozen common-gallery evaluation also completed: job `208021`
uses the full 1,928-pair gallery and 6,310 exact queries. The frozen
GeoRSCLIP step-zero comparison completed separately as job `208024` on the
same 6,310 x 1,928 contract. Neither result authorizes a claim of trained
improvement.

Historical failure `207778` remains preserved as an engineering-only
diagnostic failure. No historical artifact was rewritten.

## Verified gates

- Full suite on the current code: **645 passed, 3 skipped, 16 warnings** in
  351.47 seconds.
- `compileall`: PASS.
- Pyright for `src/qcpr_siglip2`: 0 errors, 0 warnings, 0 informations.
- Shell syntax and `git diff --check`: PASS.
- Ruff is unavailable in the cluster image; this is recorded rather than
  reported as a false PASS.
- Real CPU forward: native visual `[1,2,256,768]`, text `[1,64,768]`, score
  `[1,1]`, finite.
- Real CPU backward: score `[4,2]`, finite loss, 35 trainable modules with
  gradients.
- GeoRSCLIP wrapper: weights-only load, exact state-dict match, frozen image
  `[1,2,49,512]` and text `[1,77,512]` contract PASS.

### Bounded H100 evidence

- Job `208013`: 8 steps, physical batch 8, 16 queries, score matrix `[16,8]`.
- Native visual tokens `[8,2,256,768]`; temporal tokens `[8,512,768]`;
  text tokens `[16,64,768]`.
- Peak allocated/reserved: `2.0012 / 2.03125 GiB`.
- Checkpoint round-trip: PASS; checkpoint SHA256 is recorded in the run
  manifest.
- Evidence query swap, zeroing, deletion, detach-gradient and time-reversal
  diagnostics: PASS. Explicit multi-positive runtime: not exercised.

### Frozen common-gallery evidence

- SigLIP-2 job `208021`: full gallery `[6310,1928]`; full MRR
  `0.0059978589`; Hit@10 `0.0087163234`.
- GeoRSCLIP job `208024`: full gallery `[6310,1928]`; full MRR
  `0.0036828369`; Hit@10 `0.0030110935`; optimizer steps `0`.

## Historical separation

Job `207778` remains preserved as an infrastructure/engineering-only
failure. Job `207706` remains a historical smoke with evidence-zeroing
causality but a failed top-evidence deletion diagnostic. Neither historical
run is rewritten or used as a retrieval-quality claim.

## Readiness

```text
MODEL_V3_SYNTHETIC_SMOKE = PASS_HISTORICAL
SIGLIP2_REAL_INTEGRATION_SMOKE = PASS_TECHNICAL_SMOKE
SIGLIP2_COMMON_GALLERY_STEP0 = PASS
GEORSCLIP_STEP0 = PASS_FROZEN_EVALUATION
SIGLIP2_MECHANISM = READY_FOR_AUTHORIZATION
SIGLIP2_PHASE_A = NOT_AUTHORIZED
SIGLIP2_PHASE_B = NOT_AUTHORIZED
MODEL_V3_MAIN = NOT_AUTHORIZED
P2_REAL/P2_SEMANTIC/P2_LONG_SERIES = BLOCKED_DATASET_CONTRACT
```

The current bounded task stops here. A separate explicit authorization is
required for any mechanism or main training run above the 32-step limit.
