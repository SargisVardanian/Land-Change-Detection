# QCPR v3.1 implementation contract

QCPR v3.1 preserves the accepted engineering global retriever as a frozen
top-N candidate generator. It does not call the lost historical E0 checkpoint
a teacher and does not infer continuity with that run.

The localization path consumes query-conditioned dense temporal features at
32x32, 16x16, and 8x8. At every scale the field contains information derived
from T1, T2, signed T2-T1, absolute difference, and continuous position. A
top-down FPN fuses these dense features and the token-patch logits. Its single
256x256 logit field is the canonical source for segmentation loss, masked
pooling, reranking, evaluation, and rendering.

S2Looking is represented at pair level. One record contains the same T1/T2,
the appeared and disappeared masks, and one audited deterministic caption for
each direction. Query swap changes text while retaining images. Temporal
reversal remains an atomic image, mask, caption, and direction transformation.
Opposite-direction masks are supervised targets, not artificial empty masks.

The derived dataset is immutable and versioned under
`manifests/qcpr_v31`. Deterministic captions use only mask-supported direction,
component-count bucket, and coarse centroid location. They do not assert
relations. Generated paraphrases are outside this manifest until provenance,
direction consistency, duplicate checks, and human audit are available.

RemoteCLIP and SigLIP2 remain encoder ablations. They cannot replace the
production candidate generator unless fixed-split candidate recall and R@K
pass the predeclared regression threshold. EarthDial/Falcon are caption audit
tools only and never provide pixel ground truth.

Full training is prohibited until preprocessing geometry, mask survival,
direction mapping/routing, query-swap sanity, mask microfit,
counterfactual faithfulness, and global retrieval preservation all pass.
Tiny full-scene tiled inference is currently `NOT_IMPLEMENTED` and stress rows
do not participate in core acceptance gates.
