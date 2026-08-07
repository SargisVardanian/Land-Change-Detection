# SigLIP-2 contract repair status

Status: `PHASE_A_REPAIR_REQUIRES_RERUN`

The repair is published on `codex/qcpr-siglip2-temporal-training` at final
commit `c146d74052ca8caa6c15aba5e1c2d54a6c67b420` (the code repair itself is
`16ccd12a5ec04d9c9b10daf720c974ff4adef0ce`; PR #7 remains draft).

## Changes

- The temporal adapter now builds `[PAIR_TOKEN, T1_PATCHES, T2_PATCHES, ...]`.
  It no longer inserts pooled per-frame tokens, which were hidden FRAME_CLS
  tokens and violated the minimal architecture.
- Text evidence excludes padding and tokenizer special IDs while the native
  tower still receives its full attention mask.
- Full-gallery and milestone reranking default to query batch size 1 after the
  preserved evaluation-only OOM at batch size 4.

Dataset-v2, historical R1/C0/C1 artifacts, and the completed Geo/SigLIP2 run
directories were not modified.

## Consequence

The Phase-A checkpoint from job `208161` remains immutable historical evidence,
but it belongs to the pre-repair architecture and cannot initialize Phase B
with strict state-dict compatibility. A fresh 256-step Phase-A run is required.
Phase B remains blocked until that rerun passes the technical gates and the
evidence maps are not near-uniform. No Phase-B or P2 job was submitted.

## Verification

- `git diff --check`: PASS
- local Python 3 `compileall`: PASS
- post-repair pytest: not run locally because the local environment lacks both
  PyTorch and pytest; canonical cluster revalidation is required.
- remote branch equals the local SHA; worktree is clean.

The prior common-gallery Phase-A result is preserved separately: MRR
`0.005998 → 0.014156` from step 0 to 256 on the 6,310-query/1,928-pair
gallery, but it must not be attributed to the repaired code until rerun.
