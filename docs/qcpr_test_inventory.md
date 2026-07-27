# QCPR test inventory

Baseline: 347 tests collected on the YSU cluster at commit 98e051.

## Classification

| Test area | Classification | Notes |
|---|---|---|
| `test_qcpr_patch_reranker.py` | active production regression + v1 compatibility | Preserve v1 state keys, scoring and losses. |
| `test_qcpr_candidate_explanations.py` | active regression; partly obsolete detail | Source-string renderer test must be replaced functionally. |
| `test_mixed_supervision_contracts.py` | active production; fake/synthetic-only | Target kinds, weights and readiness contracts. |
| `test_s2looking_qcpr_manifest.py` | active data regression; synthetic-only | Expand to direction target behavior. |
| `test_ucv2_stage1_next_*` | active production regressions | Config, optimizer, loss, metrics, wrappers and temporal behavior. |
| `test_temporal_caption_manifest_pipeline.py` | active integration; mostly synthetic | Mixed manifest/evaluator contracts. |
| `test_unichange_v2_retrieval_smoke.py` | active legacy compatibility | Keep legacy checkpoint/wrapper behavior. |
| real UniverSat/Jina forward/backward | cluster/H100-only | Run through Slurm at exact commit. |
| E0 strict load/full gallery | real integration + cluster/H100-only | Immutable checkpoint is read-only. |

## Removal/replacement ledger

| Removed test | Why obsolete | Replacement | Behavior preserved |
|---|---|---|---|
| `test_renderer_contract_uses_candidate_tokens_and_shows_both_times` | Searched source strings without executing behavior | `test_candidate_renderer_uses_candidate_logits_functionally` | Retrieved candidate tokens produce distinct logits and finite masks. |

No test is removed merely because it is old. QCPR v1 compatibility tests remain. Duplicate shape tests are consolidated only after replacement coverage exists.

## Counts

- Before cleanup: **347 collected**.
- Removed/replaced: **1** obsolete source-string test.
- Final count: **351**; verification result: **350 passed, 1 skipped**.
