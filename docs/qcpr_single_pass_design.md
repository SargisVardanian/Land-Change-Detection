# QCPR single-pass retriever and emergent localization

This branch implements one retrieval representation per physical co-registered pair and a later mask-free query localization module.

## Evidence-informed design

- ViT motivates a learned CLS token over position-marked patch tokens: https://arxiv.org/abs/2010.11929
- CLIP motivates independent normalized text/image vectors and cached gallery search: https://arxiv.org/abs/2103.00020
- SigLIP motivates a pairwise sigmoid objective without a batch-wide softmax requirement: https://arxiv.org/abs/2303.15343
- DINO documents emergent localization in self-supervised ViT features: https://arxiv.org/abs/2104.14294
- GroupViT learns grouping from image-text supervision without pixel labels: https://arxiv.org/abs/2202.11094
- TCL aligns a text-grounded weighted region with text using image-text pairs: https://arxiv.org/abs/2212.00785
- TSEG converts patch-text scores into an interpolated relevance map under image-level supervision: https://arxiv.org/abs/2205.04725
- ZegCLIP directly compares text and patch embeddings in a one-stage dense system: https://arxiv.org/abs/2212.03588
- RemoteCLIP validates image-text retrieval in remote sensing: https://arxiv.org/abs/2306.11029
- GeoRSCLIP reports remote-sensing retrieval and semantic localization: https://arxiv.org/abs/2306.11300
- BIT models bi-temporal spatial-temporal context with Transformers: https://arxiv.org/abs/2103.00208
- ChangeFormer validates Siamese Transformer processing for co-registered pairs: https://arxiv.org/abs/2201.01293

## Implemented contract

The canonical run uses UniverSat output grid 32 x 32. The implementation is grid-configurable and never forces 16 x 16. Every aligned cell produces one descriptor from T1, T2, signed delta, absolute delta, and elementwise product. A six-layer pre-LN Transformer contextualizes these tokens with learned 2D positions and CLS_PAIR. There is no global visual bypass.

Frozen Jina pooled output seeds CLS_TEXT and frozen Jina token features enter a two-layer pre-LN Transformer adapter. The external Jina token projection, text adapter, and search projection are trainable; the Jina base Transformer is frozen.

Retrieval uses multi-positive pairwise sigmoid loss. Caption collisions in the quality audit are excluded from negatives. Localization receives frozen contextual pair tokens and a text search vector, produces cross-attention relevance weights, and is trained only by pair matching. Masks are absent from retrieval/localization datasets and loaded only by the final evaluator.
