# QCPR v3 grounding and training contract

## Evidence that motivated this revision

The historical `codex/unichange-v2-stage1-next` run was a global dual encoder,
not a trained query-segmentation model. Its committed evaluation reported broad
text-derived `R@1=0.4508`, but exact-pair `R@1=0.0136` and exact-pair median rank
145. The attractive overlays in that branch therefore cannot be treated as
validated query-specific masks. The clean v3 smoke produced a true-pair global
rank of 64/64 and mask probabilities in approximately `[0.039, 0.098]`; its
contrast-enhanced field exposed a right/bottom border prior.

Three implementation defects explained the first v3 collapse:

1. positive mask predictions were rolled and reused as empty mismatch targets;
2. appeared/disappeared samples were counted both as query-specific and as a
   second direction loss;
3. zero-padded refiners created artificial border evidence, while averaging
   scales erased small high-resolution foreground.

## Canonical representations

For query tokens `Q in R^(L x D)` and a temporal patch field
`V in R^(N x D)`, the generic decoder produces grounded patches `G(q,V)` and
one decoded mask-logit field:

`M(q,V) in R^(H x W)`.

There is no object, direction, location or count-specific model head. Those
attributes are permitted only for audit, sampling, negative construction and
evaluation.

The decoded field is canonical:

`M -> visualization, segmentation loss, thresholding, patch sampling, local pooling`.

Patch weights are sampled from the displayed field, never from an independent
attention visualization.

## Stage A: global bootstrap

`S_global(q,i) = cosine(z_q, z_i)`

`L_A = L_multi-positive(S_global) + 0.1 L_text-preservation`.

Jina v5 and UniverSat remain frozen. This stage never executes the grounding or
mask decoder. Exact-pair and duplicate-aware broad metrics are reported
separately.

## Stage B: generic late interaction

For each valid content token, compute cosine similarity against all grounded
temporal patches. Both sides are L2-normalized. The token score combines the
mean Top-K patch evidence with a smooth worst-token term so one easy noun cannot
dominate the query.

`S_B = S_global + softplus(w_token) S_token`.

Mask-local evidence is disabled at B initialization (`w_local ~= 0`). B trains
only token/patch interaction and is accepted only if candidate recall is
preserved and conditional reranking gain and changed-only structured margins
are positive. Parser-derived margins remain audit-only.

## Stage C: query-conditioned soft grounding

Stage C must load an accepted B checkpoint. It cannot restart from Stage A.

For a verified target mask `Y`, the primary objective is:

`L_mask = L_Dice(M,Y) + L_focal(M,Y)`.

Each sample contributes once. Appeared/disappeared are reporting strata, not
additional copies of the same objective. Initial provenance weights are
hypotheses to ablate, not hidden ground-truth equivalences:

- verified query-specific or localization-only directional mask: 1.0;
- class-specific transition mask: 0.75;
- generic change union: 0.25 and never used for query-specific Dice claims;
- audited synthetic query mask: at most 0.25;
- unverified mismatch: 0.0.

A mismatch loss is valid only for a recomputed `wrong query x candidate pair`
whose incompatibility is explicitly verified. Positive logits must never be
permuted and relabelled as negatives.

The masked local embedding is

`r_local = sum_p sigmoid(M_p) V_p / (sum_p sigmoid(M_p) + eps)`.

Its cosine score is multiplied by a mask-validity gate based on peak
probability. A mask with peak below 0.10 contributes exactly zero, preventing a
near-empty mask from becoming a confident normalized random vector.

Stage C optimizes mask loss, masked-local contrastive loss and rerank
contrastive loss. It logs foreground/background probability, mass, entropy,
effective patch count, near-empty fraction and edge/center ratio.

## Supervision provenance

- LEVIR generic masks supervise generic change only. They are not spatial
  ground truth for count or location language.
- S2Looking appeared/disappeared masks are localization-only and never enter
  retrieval supervision.
- SECOND transitions enter query-specific metrics only when caption-to-class
  correspondence is verified.
- Natural retrieval validation is never rebalanced or mixed with S2Looking.

## Diagnostic gate before scientific training

A bounded 10--20 step H100 diagnostic on genuine supervised masks must show:

- decreasing mask loss;
- foreground probability increasing relative to background;
- non-empty recall above zero;
- no systematic border (`edge/center` controlled);
- mask peak above the near-empty threshold for supervised positives;
- controlled empty false-positive area;
- finite gradients and exact canonical mask parity.

Failure stops the pipeline. It is not permissible to rescale the PNG, weaken
the gate, or proceed to 100/500-step experiments.
