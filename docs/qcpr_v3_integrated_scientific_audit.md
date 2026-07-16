# QCPR v3 integrated scientific and implementation audit

Date: 2026-07-16

## Decision

The active research implementation is QCPR v3. Legacy v1/v2 code is retained
only for historical readability and a temporary dataset/collator compatibility
boundary. Clean v3 model construction directly instantiates frozen pretrained
Jina v5 and UniverSat, a trainable temporal/global path, and the generic v3
grounder. It does not instantiate `UniChangeV2RetrievalModel` and does not use
the deleted historical E0 checkpoint.

## Scientific basis and limits

- CLIP motivates a global dual-encoder retrieval baseline, but its single-vector
  bottleneck does not prove local compositional grounding
  ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)).
- FILIP and ColBERT motivate token/patch or token/document late interaction;
  their success does not guarantee remote-sensing change reasoning
  ([Yao et al., 2021](https://arxiv.org/abs/2111.07783),
  [Khattab and Zaharia, 2020](https://arxiv.org/abs/2004.12832)).
- LAVT supports language-conditioned dense visual interaction, while
  Mask2Former supports learned mask decoding rather than treating raw attention
  as a final segmentation product
  ([Yang et al., 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_LAVT_Language-Aware_Vision_Transformer_for_Referring_Image_Segmentation_CVPR_2022_paper.html),
  [Cheng et al., 2022](https://arxiv.org/abs/2112.01527)).
- Slot Attention supports optional class-agnostic region slots, but no count
  claim is allowed without a verified count benchmark
  ([Locatello et al., 2020](https://arxiv.org/abs/2006.15055)).
- Recent bi-temporal token-interaction change models are supporting context,
  not evidence for language grounding; TiBT-Net must be compared only after
  matching data and metrics ([2026, DOI 10.3390/rs18050805](https://doi.org/10.3390/rs18050805)).

## Corrected mathematical contracts

1. Global bootstrap uses a set-valued contrastive objective. Exact pairs and
   normalized-caption-equivalent pairs form the positive set. This removes
   contradictory duplicate-caption negatives without declaring parser tags to
   be neural classes.
2. The frozen pre-adapter Jina embedding is only a text reference for a low-LR
   adapter preservation term. It is not image ground truth and not the deleted
   historical global teacher.
3. Global bootstrap executes `forward_global`; it never allocates QxCxpatch
   grounding or full-resolution mask tensors.
4. Cross-modal raw mask evidence is decoded once. Patch weights for local
   pooling are sampled from that same displayed decoded logit field, so
   visualization, segmentation loss and reranking cannot use independent maps.
5. Token-patch relevance aggregates top-k patches for every valid content token
   and combines arithmetic and soft-min evidence. One easy token cannot supply
   the entire compositional score.
6. Object/direction/location/count/relation parsers remain outside the neural
   representation and are audit/mining/evaluation metadata only.

## Corrected data contracts

- Retrieval-only and localization-capable derived manifests are separate.
  Global bootstrap cannot receive S2Looking localization-only batches.
- The capped pair sampler consumes no-change rows throughout training, enforces
  the per-batch cap, terminates for small batches, samples without replacement
  inside a batch, and downweights duplicate-caption-heavy pairs.
- Natural validation is never reweighted. Parser-derived structured metrics are
  audit-only; human structured gate remains `NOT_EVALUATED` until reviewed.

## Corrected GPU/runtime contracts

- CPU metadata stays on CPU; every tensor participating in forward, loss or
  indexing is placed on the canonical resolved CUDA device.
- Global-only and grounding phases have different memory paths.
- Every v3 run writes atomic `progress.json` with current stage, step/chunk,
  metrics, elapsed time and ETA.
- Text adapter uses a separate lower learning rate. Gradient clipping starts
  from the measured clean-bootstrap probe threshold and is accepted only when
  the observed clipping fraction is at most 20%.

## Experiment sequence

1. One integrated v3 smoke: clean global bootstrap, late interaction and mask
   grounding, with strict checkpoint round-trip and canonical parity.
2. One production-realistic memory probe.
3. One 100-step global-bootstrap diagnostic, evaluated at initialization and
   at the final checkpoint on identical fingerprints.
4. A 500-step pilot is submitted only if the 100-step run has finite gradients,
   acceptable clipping, reported candidate recall, non-degenerate loss and no
   data/device failure.
5. Full training is not automatic. Stage B and C start only after the new global
   checkpoint is explicitly accepted as a versioned global reference.

## Remaining scientific risks

- Caption-derived semantic relevance is a pseudo-target and may inflate broad
  retrieval for repeated no-change captions.
- Generic masks cannot become query-specific when only generic change masks are
  available; metrics must be split by supervision provenance.
- Counts, geometry and relations remain unsupported claims until a human-reviewed
  compositional benchmark exists.
- A single 500-step seed is diagnostic, not publishable evidence; final claims
  require fixed multi-seed ablations and paired uncertainty estimates.
