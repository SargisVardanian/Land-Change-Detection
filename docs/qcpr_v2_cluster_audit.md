# QCPR v2 cluster audit

Baseline audited on YSU cluster from `98e05118c8e232271fb30beb8884eba675032980`.

## Reachability matrix

| Capability | State at 98e051 | Evidence |
|---|---|---|
| Candidate-specific retrieval masks | Implemented and reachable in evaluator | renderer indexes `corpus.patch_tokens[int(retrieved_index)]` |
| Shared mask weights for displayed mask and local pooling | Implemented and reachable | `masked_local_embedding` consumes `query_mask_logits.sigmoid()` |
| Temporal descriptor `[V1,V2,diff,absdiff,xy]` | Implemented but unreachable from production Stage1-next | only used by optional head created from an absent config field |
| Appeared/disappeared/changed head | Reachable only with an ad-hoc config attribute; untrained | no Stage1NextConfig/CLI/wrapper/loss wiring |
| Model-derived structured explanations | Not implemented | `_query_evidence` parses query keywords and copies query text |
| Initialization provenance | Legacy trainer only | written in `train_unichange_v2_retrieval.py`, not production core |
| QCPR training | Production Stage1-next path | loss and optimizer wiring live in `ucv2_stage1_next_core.py` |

## Confirmed gaps

1. `enable_temporal_explanation_channels` is not wired through `Stage1NextConfig`, the Stage1-next CLI, or Slurm wrappers.
2. `temporal_explanation_head` is absent from Stage1-next optimizer groups, gradient audit, loss, and checkpoint-selection metrics.
3. Core QCPR uses a transformed global query dotted with projected change patches. It is not the required query-conditioned MLP interaction and does not use Jina token embeddings locally.
4. Candidate patches are selected correctly, but the renderer overlays one shared query mask over both T1 and T2 rather than independently learned before/after evidence.
5. `_query_evidence` is keyword parsing, not model-derived evidence; `object_evidence` is misleadingly the original query.
6. The renderer calls `compute_retrieval_metrics` and then `compute_retrieval_ranks`, recomputing scores/ranks.
7. Initialization provenance was added to legacy `train_unichange_v2_retrieval.py`, not fully to production `ucv2_stage1_next_core.py`.

## Path classification

- `models/qcpr.py`: active model code, but the temporal descriptor is only partially reachable.
- `models/unichange_v2_retrieval.py`: active shared model path; optional temporal output is reachable but untrained.
- `models/temporal_change_encoder.py`: active production Stage1-next temporal representation.
- `train_unichange_v2_stage1_next.py` and `ucv2_stage1_next_core.py`: production trainer.
- `train_unichange_v2_retrieval.py`: legacy trainer; its provenance additions do not satisfy production requirements.
- `ucv2_cluster_common.py`: active model factory; currently accepts an undeclared temporal flag.
- `ucv2_retrieval_metrics.py`: active evaluator/scoring path.
- `render_unichange_v2_retrieval.py`: evaluator-only artifact generation.
- wrappers: active cluster path, missing v2/direction wiring at baseline.

## Compatibility risk

Commit 98e051 silently changed v1 `S_local` from max mask probability to mask-weighted cosine pooling while retaining the same checkpoint module names. Strict state loading can succeed while rankings change. Architecture versioning is required.

## Baseline test collection

With cluster repository-first `PYTHONPATH`, collection succeeded with **347 tests**. A plain environment-only invocation imported an older installed package and is not a valid baseline.
