# SigLIP-2 frozen common-gallery evaluation

Status: `PASS_FROZEN_EVALUATION`.

This is a step-zero initialization benchmark, not a training result and not
evidence of retrieval improvement.

## Contract

- Job: `208021`, node `gpu01`, exit `0:0`.
- Code: `35fb641b3fa9b40873bd604d76e44fa4363c7e43`.
- Checkpoint: `b8251b0c0c769db288a802c41bba6238c2469f3b29741b42136a2b6a0ab50fcb`.
- Queries: `6,310` exact queries.
- Gallery: `1,928` physical pairs.
- Score matrix: `6,310 × 1,928`.
- Generic no-change queries are excluded from the primary exact query set, but
  their 666 physical pairs remain in the gallery as valid candidates.
- No masks or dense-label sidecars were opened.

## Global results

| Metric | Value |
|---|---:|
| Hit@1 | 0.001426 |
| Hit@5 | 0.004754 |
| Hit@10 | 0.008716 |
| Hit@50 | 0.031696 |
| Hit@100 | 0.062599 |
| Hit@500 | 0.302219 |
| Full-gallery MRR | 0.005998 |
| Mean rank | 892.890 |
| Median rank | 860 |
| mAP@10 | 0.002987 |
| mAP@100 | 0.004339 |

The evaluator also records explicit multi-positive Recall@K, precision and
truncated MRR fields. Here exact queries have one positive, so Hit@K and
multi-positive Recall@K coincide numerically; their names remain distinct.

## Source split

| Source | Queries | Full MRR | Hit@10 | Hit@100 | Mean rank | Median rank |
|---|---:|---:|---:|---:|---:|---:|
| LEVIR-MCI | 4,449 | 0.004236 | 0.004498 | 0.047676 | 825.002 | 773 |
| SECOND-CC | 1,861 | 0.007973 | 0.013445 | 0.079328 | 968.994 | 985 |

## Top-K evidence reranking

Reranking was evaluated only inside global Top-20/50/100. Full-gallery MRR
changed only from `0.005997859` to `0.005998149` at Top-100, which is expected
for an untrained step-zero evidence pathway. This is not a claim that the
reranker improves retrieval.

## Integrity and resources

- Query IDs unique: PASS.
- Gallery IDs unique: PASS.
- Scores finite: PASS.
- Mask-free: PASS.
- Data-release SHA256:
  `ff4e71f0733c7c4c69d39b557f4f925bc8a07a1657430489f405b6fc837b1cf1`.
- Development-manifest SHA256:
  `89a69e144033957ce9c0756502b12c6e56a90f62801e7152d0001bfb339c30a9`.
- Peak CUDA allocated/reserved: `9.647 / 27.838 GiB`.
- Total wall time: `141.56 s`.

The preceding job `208019` is retained as a diagnostic-only artifact because
it evaluated a 1,262-pair incomplete gallery. Its numbers must not be used as
the common benchmark.

Full file hashes and the complete rankings are in the immutable run root:

`/mnt/weka/svardanyan/rs_change_project/runs/qcpr_siglip2_common_eval_step0_35fb641_20260806`
