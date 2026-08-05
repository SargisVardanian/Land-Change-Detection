# SigLIP-2 hardening audit — 2026-08-06

## Decision

The CPU/code contract is green, but the single bounded H100 smoke for the
hardened commit did not complete. Its exit was `1:0` after 32 seconds. This is
not a scientific retrieval result and does not authorize a mechanism or main
training run.

The failed job was `207778` on `gpu02`, run root
`/mnt/weka/svardanyan/rs_change_project/runs/qcpr_siglip2_real_smoke_b5873c0_20260806`.
The run wrote the batch/model/parameter contracts and `roundtrip_input.json`,
but no step metrics, checkpoint, CUDA profile, or gradient diagnostics. The
launcher sent Python stderr to node-local `/tmp`; that file was not available
from the login node after completion, so the exact Python traceback could not
be recovered. The failure is recorded as
`ENGINEERING_RUNTIME_DIAGNOSTIC_UNRESOLVED`, not as a model-quality failure.

No automatic second GPU submission was made.

## Verified gates

- Full suite: **631 passed, 3 skipped, 16 warnings** in 378.30 seconds.
- `compileall`: PASS.
- Pyright for `src/qcpr_siglip2`: 0 errors, 0 warnings, 0 informations.
- Shell syntax and `git diff --check`: PASS.
- Local Ruff: PASS. The cluster login PATH does not expose `ruff`; this is
  recorded rather than hidden.
- Real CPU forward: native visual `[1,2,256,768]`, text `[1,64,768]`, score
  `[1,1]`, finite.
- Real CPU backward: score `[4,2]`, finite loss, 35 trainable modules with
  gradients.
- GeoRSCLIP wrapper: weights-only load, exact state-dict match, frozen image
  `[1,2,49,512]` and text `[1,77,512]` contract PASS.

## Historical separation

Job `207706` remains a historical real-smoke PASS for code
`80d17d50845ac2a4086e32b68dd58c6371e92e27`, with 1.370 GiB peak allocated and
1.477 GiB peak reserved. It showed evidence-zeroing causality but failed the
top-evidence deletion diagnostic, so mechanism status stayed blocked. It does
not certify the hardened `b5873c01e83fc6adc637cb15fefcee5097e8efbf` code.

## Readiness

```text
MODEL_V3_SYNTHETIC_SMOKE = PASS_HISTORICAL
SIGLIP2_REAL_INTEGRATION_SMOKE = FAIL_UNDIAGNOSED_GPU_EXIT
SIGLIP2_MECHANISM = BLOCKED_REAL_SMOKE
SIGLIP2_PHASE_A = NOT_AUTHORIZED
SIGLIP2_PHASE_B = NOT_AUTHORIZED
MODEL_V3_MAIN = NOT_AUTHORIZED
P2_REAL/P2_SEMANTIC/P2_LONG_SERIES = BLOCKED_DATASET_CONTRACT
```

The next action is to make Slurm stdout/stderr land under the shared run root
and capture the traceback. That follow-up requires explicit authorization for
another GPU retry.
