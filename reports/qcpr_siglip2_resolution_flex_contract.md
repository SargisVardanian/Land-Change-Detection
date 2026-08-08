# QCPR resolution-flexible TemporalSigLIP contract

Status: `MODEL_V3_RESOLUTION_FLEX_CPU_CONTRACT_PASS_REAL_NAFLEX_NOT_RUN`

The isolated branch implements the variable-token contract for
`google/siglip2-base-patch16-naflex` without introducing multi-vector ANN
retrieval. Each physical item still produces one normalized pair vector. The
adapter accepts `[B,T,N,D]`, keeps patch validity and rectangular grid
metadata, uses synchronized temporal geometry, and routes scenes above the
direct budget through a bounded query-independent reducer.

## CPU contract result

- Direct budgets 256, 576 and 1024: pass on synthetic contract inputs.
- Rectangular spatial shapes and padding masks: pass.
- Padded-content invariance: pass.
- Large-scene bounded reducer: pass.
- Explicit NaFlex processor budget: pass; omitted budget is rejected.
- Temporal reversal/double reversal: pass.
- Evidence gradients and checkpoint roundtrip: pass.
- Deterministic one-vector ANN interface: pass.
- Focused suite: `54 passed` using the repository's temporary local test stub
  because `transformers` is not installed in the Mac environment.
- `compileall`, `git diff --check`, Ruff and shell syntax: pass.

## Not yet demonstrated

The following are intentionally not claimed:

- real SigLIP2/NaFlex processor execution;
- real configured visual-backbone and Jina loading;
- real Dataset-v2 image loader execution;
- native-resolution 1024-patch H100 memory/timing;
- mechanism or main training improvement.

The cluster is not reachable from this environment, and the Dataset Agent
handoff is not independently readable here. No training job was submitted.

## Publication lineage

The implementation was published to PR #6 through the fast-forward lineage
`9d95181c…` → `cf88efd…` → `7277f6b…` on
`codex/qcpr-model-v3-temporal-retrieval`. The final metadata commit is checked
again in `reports/final_lineage.json` after this report is committed.

## Dataset boundary

No Dataset-v2 manifest or historical experiment artifact was modified. The
resolution-flexible model remains blocked from training until the current
Dataset Agent handoff and its independent authorization state are available.
