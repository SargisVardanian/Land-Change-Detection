# TemporalSigLIP active architecture

This document defines the active model path after the bounded SigLIP2 pilot.
The previous `qcpr_siglip2` evidence/reranking modules remain historical
experimental code and are not imported by this model.

```text
T1 ─┐
    ├─ shared frozen SigLIP2 vision tower ─┐
T2 ─┘                                      │
                                           ↓
                native patch tokens [B,T,256,768]
                                           ↓
       [PAIR, T1_PATCHES, T2_PATCHES, ...]
                                           ↓
                 exactly two Transformer blocks
                                           ↓
                 normalized pair_embedding [B,768]

text ── shared frozen SigLIP2 text tower ──→ normalized text_embedding [Q,768]

score(q,p) = exp(clamped_logit_scale) * cosine(text_embedding, pair_embedding)
                                           ↓
                           direct full-gallery ranking → Top-10

after retrieval only:
  LayerNorm(T2_temporal_tokens - T1_temporal_tokens)
  × normalized text embedding
  → soft temporal change map [B,1,16,16]
```

## Trainable scope

Stage A trains only the temporal pair module and logit scale.  Both SigLIP2
towers are frozen.  Stage B is a fresh-optimizer continuation that unfreezes
exactly the final two vision blocks, final two text blocks and official final
normalization/projection modules.

## Explicit exclusions

The active path has no evidence bottleneck, evidence gate, relevance MLP,
change-token bank, late-interaction branch, Top-K reranker, segmentation
decoder, or separate segmentation loss.

The localization map is diagnostic-only until verified localized query records
exist.  It is not fed back into the primary retrieval score.
