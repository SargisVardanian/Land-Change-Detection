# QCPR Model-v3 hardening decision

## Decision

The branch is technically ready for a future mechanism authorization, but it
is not authorized for main training or P2. The final dataset handoff is
missing, and source-classification separability has not yet been measured.

```text
B1_CONTROL                    = VERIFIED
MODEL_V3_SYNTHETIC_SMOKE      = PASS
MODEL_V3_REAL_INTEGRATION     = PASS_TECHNICAL_SMOKE (job 209877)
MODEL_V3_MECHANISM            = READY_FOR_AUTHORIZATION_ONLY
MODEL_V3_MAIN                 = NOT_AUTHORIZED
P2_REAL/SEMANTIC/LONG_SERIES  = BLOCKED_DATASET_CONTRACT
FINAL_TRAINING_STATE          = WAIT_DATASET_FINAL_HANDOFF
```

## What is actually implemented

The active track is a clean direct `TemporalSigLIP`, not the older
UniverSat+Jina/evidence-reranker implementation. It uses the pinned
`google/siglip2-base-patch16-256` checkpoint for both image and text towers.
The same vision tower encodes each frame, producing native `256 x 768` patch
tokens for a 256-pixel input. Two learned temporal transformer blocks pool the
`[B,T,256,768]` sequence into one query-independent normalized 768-D pair
embedding. The text tower produces 64 token embeddings and one normalized
768-D query embedding.

The primary score is exactly:

```text
exp(clamped_logit_scale) * cosine(text_embedding, pair_embedding)
```

The loss is one symmetric multi-positive listwise retrieval scalar. There is
no reranker, no segmentation loss, no direction loss, no object/count loss,
and no mask access in training. Current trainable modules are the temporal
pair adapter and logit scale in Stage A; Stage B additionally exposes only
the last two SigLIP2 transformer blocks and their final projections at a much
lower learning rate. The full SigLIP2 towers remain frozen in Stage A.

The current post-retrieval soft map is diagnostic only. It is not part of the
training score and therefore is not yet a query-conditioned soft-segmentation
mechanism. A localized/evidence reranking phase needs a separate authorization
and verified localized query contract.

## Real integration smoke

Job `209877` completed `0:0` on an H100 using real LEVIR-MCI/SECOND-CC image
files, the real processor/tokenizer, and the pinned model:

| field | value |
|---|---:|
| steps | 8 |
| physical pairs | 8 |
| captions per pair | 2 |
| score matrix | 16 x 8 |
| visual tensor | 8 x 2 x 256 x 768 |
| text tensor | 16 x 64 x 768 |
| peak allocated | 1.456 GiB |
| peak reserved | 1.699 GiB |
| checkpoint roundtrip | PASS |
| mask access | false |
| multi-positive runtime | not exercised |

The checkpoint SHA is
`34ed77e4f34fb001953579756349b5760453f3f4f40ff7d34b238f364813504f`.

## Tokenizer hardening

The local checkpoint resolves to runtime class `SiglipModel` and uses a
`GemmaTokenizer` with vocabulary 256,000 and IDs `BOS=2, EOS=1, PAD=0`.
The inherited SigLIP config has the old `49406/49407/1` defaults during
Transformers construction; that upstream warning is recorded rather than
suppressed. The runtime contract validates tokenizer IDs against the actual
embedding table and synchronizes the text-config metadata. The weights are
unchanged. The audit is `PASS` in
`reports/siglip2_tokenizer_compatibility.json`.

## Common-gallery pilot result

All numbers below use the same 1,928-pair gallery and 6,310 exact queries.
`Hit@K` is named explicitly; it is not multi-positive Recall@K.

| arm | MRR | Hit@1 | Hit@5 | Hit@10 | Hit@100 | median rank | mean rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage A, step 512 | 0.04617 | 0.01569 | 0.05309 | 0.09398 | 0.45674 | 117 | 242.70 |
| Stage B, step 1024 | 0.05771 | 0.02076 | 0.06941 | 0.11696 | 0.52060 | 93 | 213.04 |
| Stage B, step 2048 | 0.05669 | 0.01870 | 0.07322 | 0.12203 | 0.49810 | 101 | 239.61 |

Relative to Stage A, Stage B step 1024 improves MRR by about 25.0% and Hit@10
by about 24.5%; step 2048 improves MRR by about 22.8% and Hit@10 by about
29.8%. This is bounded single-seed pilot evidence, not a final claim: the
evaluation code SHA is the earlier pilot SHA, and the final dataset handoff is
missing.

The Stage-B step-2048 source split is:

| source | queries | MRR | Hit@1 | Hit@5 | Hit@10 | median |
|---|---:|---:|---:|---:|---:|---:|
| LEVIR-MCI | 3,335 | 0.06333 | 0.02129 | 0.08306 | 0.13583 | 95 |
| SECOND-CC | 2,975 | 0.04924 | 0.01580 | 0.06218 | 0.10655 | 106 |

Embedding effective ranks are full for pair embeddings (768); text ranks are
655 at Stage A, 726 at Stage B step 1024 and 731 at step 2048. Norms remain
approximately one. No embedding collapse is observed. A source-classifier
separability test is still `NOT_RUN` because the old evaluator saved rankings
and rank diagnostics, but not the embedding matrices; it must be measured
before any source-shortcut conclusion.

## Important limitations and next scientific work

1. The current 256-patch input is fixed by the pinned 256px checkpoint. Larger
   native images are resized by the processor, so small-object/scale failure
   is not ruled out. Multi-scale or footprint-normalized views must be a
   controlled ablation, not a silent preprocessing change.
2. The feature path supports variable `T` at the temporal-adapter contract,
   but the real smoke and training data exercised only `T=2`; long-series
   claims require verified timestamps and sequence manifests.
3. The post-retrieval map is not causally trained. It cannot currently be
   advertised as query-conditioned soft segmentation. The next localized
   experiment should rerank only Top-K candidates with dense temporal tokens,
   use one unified relevance score, and use masks only for evaluation.
4. Exact retrieval is still ambiguous for generic/no-change and semantically
   overlapping captions. Those views remain separate until the Dataset Agent
   provides the final handoff.
5. Stage-B full run was an architecture pilot on the current hold release,
   not a final three-seed release experiment.

Historical job `207555` is classified as an engineering runtime driver bug
(reused autograd graph), not a scientific failure. Historical job `207560` is
an engineering diagnostic bug. Neither is deleted or rewritten.

The detailed machine-readable artifacts are:

- `reports/qcpr_model_v3_hardening_decision.json`
- `reports/temporal_siglip_pilot_analysis.json`
- `reports/siglip2_tokenizer_compatibility.json`
- `/mnt/weka/svardanyan/rs_change_project/runs/qcpr_siglip2_real_integration_smoke_7f962c2_20260808`
