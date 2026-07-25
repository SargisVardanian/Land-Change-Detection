# QCPR residual PAIR retrieval and mask-free grounding

## Audited backbone contract

A real canonical training sample was passed through the frozen public UniverSat joint API on the cluster:

- input temporal series: [1, 2, 3, 256, 256];
- output native dense field: [1, 1024, 768];
- native grid: 32 x 32;
- all values finite.

The encoder API accepts [B,T,C,H,W] and [B,T] dates, so the model is not architecturally restricted to two timestamps. The current datasets happen to provide two timestamps.

## Retrieval path

The frozen joint UniverSat output is detached before the trainable adapter.

1. Two per-token residual bottleneck adapters preserve dimension 768. Each final up-projection is exactly zero initialized, making the initial adapted field exactly equal to the native field.
2. A learned PAIR token attends to all adapted tokens in three cross-attention blocks. Dense tokens never self-attend in this adapter, so attention complexity is linear in token count.
3. Every PAIR block uses pre-normalization, 12-head cross-attention, residual updates, FFN ratio 4, and independent LayerScale vectors initialized to 1e-3.
4. The baseline is a projected mean of adapted dense tokens. The final PAIR token produces a residual delta through an exactly zero-initialized projection. The initial retrieval embedding is therefore exactly the normalized baseline.
5. Only the final representation is projected from 768 to 512 dimensions.

Text uses frozen Jina plus its permitted external token projection, a two-layer small text adapter, and a normalized 512-D projection.

Training uses one balanced multi-positive SigLIP objective:

0.5 * mean(softplus(-positive_logits)) + 0.5 * mean(softplus(valid_negative_logits)).

Known caption collisions are excluded from valid negatives. No spatial, direction, object, count, mask, or counterfactual loss exists.

## Grounding path

After retrieval selection, the complete retrieval encoder is frozen. Hard candidates are mined only from frozen retrieval scores, physical identities, and collision metadata.

The grounder consumes contextual Jina tokens and adapted 768-D dense tokens before PAIR pooling. Query-to-token cross-attention produces one probability per native token. The grounded visual embedding is only the normalized probability-weighted sum of projected dense tokens. Neither global PAIR embedding nor global retrieval score is available to the grounder.

Different normalized caption groups from the same physical item are retained together in grounding batches. Their own query-conditioned regional embeddings form diagonal positives; cross-caption regional matches within the same item are generic valid negatives.

The only optimized grounding objective is the balanced pairwise SigLIP loss. Entropy, concentration, query sensitivity, pair sensitivity, deletion, insertion, and fixed-location collapse are diagnostics only. Held-out masks are opened only by the frozen evaluator.

## Primary evidence

- ViT: https://arxiv.org/abs/2010.11929
- SigLIP: https://arxiv.org/abs/2303.15343
- FILIP: https://arxiv.org/abs/2111.07783
- TCL: https://arxiv.org/abs/2212.00785
- TSEG: https://arxiv.org/abs/2205.04725
- GroupViT: https://arxiv.org/abs/2202.11094
- BIT: https://arxiv.org/abs/2103.00208
- RemoteCLIP: https://arxiv.org/abs/2306.11029
