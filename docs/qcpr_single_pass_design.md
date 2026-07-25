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

Text uses frozen Jina plus two pre-normalized ReZero attention/FFN blocks. Every residual gate starts at exactly zero, the base pooled projection starts as identity, and the learned pooled-token delta projection starts at exactly zero. Thus initial contextual tokens equal Jina tokens and the initial normalized query vector equals the frozen Jina pooled geometry.

Every physical item contributes exactly two captions per epoch, rotated deterministically across epochs. A micro-batch of 32 physical items therefore always forms a 64 x 32 contrastive matrix. The measured first real H100 forward/backward writes the actual batch, query count, matrix shape, accumulation, and CUDA peaks to `batch_contract.json`.

Training uses one balanced multi-positive SigLIP objective:

0.5 * mean(softplus(-positive_logits)) + 0.5 * mean(softplus(valid_negative_logits)).

Known caption collisions are excluded from valid negatives. No spatial, direction, object, count, mask, or counterfactual loss exists.

Before grounding, `retrieval_acceptance.json` requires development MRR above initialization by a positive numerical tolerance, finite metrics, Recall@5/10 no more than 0.002 below baseline, and the exact SHA256 of `best_retrieval.pt`. Median/mean rank and Recall@1 remain diagnostics and tie-breakers rather than mandatory gates. Slurm `afterok` alone is not treated as scientific acceptance.

## Grounding path

After retrieval selection, the complete retrieval encoder is frozen. Hard candidates are mined only from frozen retrieval scores, physical identities, and collision metadata.

The grounder consumes contextual Jina tokens and adapted 768-D dense tokens before PAIR pooling. Query-to-token cross-attention produces one probability per native token. The grounded visual embedding is only the normalized probability-weighted sum of projected dense tokens. Neither global PAIR embedding nor global retrieval score is available to the grounder.

Different normalized caption groups from the same physical item are retained together in grounding batches. Their own query-conditioned regional embeddings form diagonal positives; cross-caption regional matches within the same item are generic valid negatives.

Training samples 1 positive plus 3 negatives from a frozen 32-candidate cache. Checkpoint validation uses a fixed positive plus 15 hard negatives.

The only optimized grounding objective is the balanced pairwise SigLIP loss. Entropy, effective patch count, concentration, query sensitivity, pair sensitivity, learned-versus-uniform score, deletion, insertion, and fixed-location collapse are diagnostics only. `grounding_acceptance.json` records whether learned-map score beats uniform pooling, top-evidence deletion lowers score, and non-equivalent same-pair queries change the map.

Softmax weights are used only for grounded pooling. A separate sigmoid of the same logits is saved as an uncalibrated query-conditioned relevance map; it is not claimed to be a calibrated segmentation probability. Held-out masks are opened only by the frozen evaluator.

## Primary evidence

- ViT: https://arxiv.org/abs/2010.11929
- SigLIP: https://arxiv.org/abs/2303.15343
- FILIP: https://arxiv.org/abs/2111.07783
- TCL: https://arxiv.org/abs/2212.00785
- TSEG: https://arxiv.org/abs/2205.04725
- GroupViT: https://arxiv.org/abs/2202.11094
- BIT: https://arxiv.org/abs/2103.00208
- RemoteCLIP: https://arxiv.org/abs/2306.11029
