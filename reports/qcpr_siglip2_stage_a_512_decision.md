# TemporalSigLIP Stage A — bounded decision package

Status: `PASS_BOUNDED_SINGLE_SEED_EXACT_CORE`

The run used the frozen SigLIP2 towers and trained only the temporal pair
adapter and logit scale for 512 fixed optimizer steps. It did not use RSCC
text, semantic multi-positive records, masks, hard-negative mining, or a
query-conditioned training score.

The final-code real integration smoke also passed at SHA
`ed96db2d2bf357c21c32657df4b4590f9ea51d54` (job `209868`, 8 steps, H100,
`256 × 128` logical score matrix, 8.838 GiB allocated / 10.447 GiB reserved,
checkpoint round-trip PASS).

## Immutable contracts

- Code used for training: `301de15b095a5816e637d884b1e4c3e3675b8b49`
- Current evaluation code: `1c201990eae777a41c370194d73099cb54481eed`
- Training job: `209853`
- Checkpoint SHA256: `7d634f18e8bac645a302cadbe7279429f29dfe7effa01dc406b8d48c737697db`
- Common gallery: 1,928 physical pairs and 6,310 exact queries
- All-diagnostic gallery: 1,928 physical pairs and 9,640 queries
- Common manifest SHA256: `89a69e144033957ce9c0756502b12c6e56a90f62801e7152d0001bfb339c30a9`
- Logical batch: 128 physical pairs, 256 text queries, score matrix `256 × 128`
- Physical microbatch: 32; captions per pair: 2
- Peak H100 memory: 8.840 GiB allocated / 10.006 GiB reserved
- Trainable parameters: 14,184,193; frozen SigLIP2 parameters: 375,234,050

## Exact primary comparison

All queries have one exact positive, so Hit@K and ordinary Recall@K coincide in
this table. The B1 numbers are the frozen common-gallery screen; Stage A is a
single-seed bounded pilot.

| Model | MRR | R@1 | R@5 | R@10 | R@50 | R@100 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 frozen | 0.00675 | 0.00095 | 0.00602 | 0.01078 | 0.04358 | 0.07464 | 872.47 | 837 |
| Stage A step 0 | 0.00530 | 0.00095 | 0.00301 | 0.00650 | 0.03138 | 0.06117 | 838.76 | 821 |
| Stage A step 256 | 0.04125 | 0.01236 | 0.04643 | 0.08526 | 0.27655 | 0.43867 | 243.27 | 123 |
| Stage A step 512 | **0.04617** | **0.01569** | **0.05309** | **0.09398** | **0.30063** | **0.45674** | **242.70** | **117** |

The exact gain is present in both physical sources at step 512:

| Source | Queries | MRR | R@10 | R@100 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|---:|
| LEVIR-MCI | 3,335 | 0.04910 | 0.10135 | 0.42489 | 258.46 | 138 |
| SECOND-CC | 2,975 | 0.04290 | 0.08571 | 0.49244 | 225.04 | 103 |

## No-change diagnostic

Generic no-change is not an exact-pair scientific target. On the diagnostic
subset, B1 versus Stage A step 512 was:

| Model | MRR | R@10 | R@100 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|
| B1 frozen | 0.00635 | 0.00931 | 0.08468 | 777.56 | 704 |
| Stage A step 512 | 0.00632 | 0.00901 | 0.06066 | 651.33 | 616 |

This is not an improvement in the primary no-change retrieval metrics. It
confirms that the new temporal adapter improves changed/exact retrieval while
the generic no-change language contract remains a bottleneck.

## Engineering and scientific decision

- Training completed at exactly 512/512 steps with finite loss and gradients.
- SigLIP2 towers remained frozen; checkpoint round-trip difference was zero.
- Full suite after the final evaluator fixes: `661 passed, 3 skipped, 16 warnings`.
- The evaluator was repaired twice during audit: the logit scale is moved to
  the embedding device, and the full physical gallery is retained while
  generic queries are filtered only from the exact query set.
- Semantic mAP/nDCG, localized retrieval and long-series retrieval were not
  exercised because those Dataset-v2 views are unavailable or not training
  enabled.
- The post-retrieval soft map remains diagnostic only; it is not in the
  training score and no causal localization claim is made.

Decision: `STAGE_A_EXACT_GAIN_BUT_NOCHANGE_LIMIT`. Stage B and any main or P2
training require a separate authorization and decision package.
