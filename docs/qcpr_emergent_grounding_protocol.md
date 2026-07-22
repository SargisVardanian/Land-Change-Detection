# QCPR emergent-grounding protocol

## Scientific tracks

- **A0** learns global text-to-physical-pair retrieval from LEVIR-MCI and
  SECOND-CC. It never reads pixel masks.
- **B** reranks A0 Top-N candidates with direct content-token/temporal-patch
  late interaction. It never calls the supervised mask decoder and never
  reads pixel masks.
- **C0** is a deterministic spatial view of B evidence. It has no trainable
  segmentation decoder and no pixel-supervised loss.
- **S0** is the former M0 supervised S2Looking segmentation probe. It is a
  separate upper-bound ablation and cannot gate A0, B, or C0.

## B and C0 mathematics

For normalized projected content token `t` and normalized temporal patch `p`,

`a[t,p] = cosine(W_text(q_t), W_patch(d_p))`.

B computes `s_t = mean(top4_p(a[t,p]))`, then averages `s_t` over valid
content tokens. The production reranking score is
`S_B = S_global + 0.1 * S_local`.

C0 keeps the complete similarity tensor and computes, independently for every
query/candidate pair,

`e[p] = mean_valid_content_tokens(a[t,p])`.

The canonical emergent patch logit is the deterministic per-map standardization
`(e[p] - mean_p(e)) / max(std_p(e), 1e-6)`. Patch probability is its sigmoid.
The first (highest-resolution) native temporal scale is reshaped to its patch
grid and upsampled with fixed bilinear interpolation (`align_corners=False`).
No learned mask decoder participates in this path.

The model API exposes:

- `token_patch_similarity [Q,C,N,L]`;
- `emergent_patch_logits [Q,C,N]`;
- `emergent_patch_probability [Q,C,N]`;
- `emergent_soft_map [Q,C,H,W]`.

For a matched physical-pair evaluator, candidate indexing reduces the first
three outputs to `[Q,N,L]`, `[Q,N]`, and `[Q,N]` respectively.

## Evidence boundaries

S2Looking masks may be used only by S0 training or by the frozen, post-training
C0 evaluator. They are forbidden in A0/B optimization, checkpoint selection,
and target-aware cropping. C0 results must be called *weakly supervised
emergent soft localization* unless post-training evidence supports a stronger
claim.
