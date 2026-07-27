# QCPR v3 tensor contract

Default symbolic dimensions: batch `B`, candidates `C`, text tokens `L≤256`,
model width `D=512`, input `H=W=256`, scales `s∈{32,16,8}`, slots `K=8` when
explicitly enabled.

| Tensor | Shape | Contract |
|---|---:|---|
| images | `[B,2,3,H,W]` | ordered T1/T2 RGB pair |
| visual features | `[B,2,N_32,D_v]` | frozen UniverSat frame tokens |
| per-scale T1/T2 | `[B,2,N_s,D]` | projected/adaptively pooled tokens |
| temporal descriptor scale `s` | `[B,N_s,D]` | learned projection of T1,T2,signed/absolute delta, continuous position |
| temporal field | `[B,N,D]`, `N=32²+16²+8²=1344` | concatenated multi-scale descriptors |
| scale IDs | `[N]` | integer routing metadata, never semantic labels |
| normalized coordinates | `[N,2]` | continuous `(x,y)` positions |
| global text | `[Q,D]` | normalized query embedding |
| text tokens | `[Q,L,D]` | valid Jina tokens projected to model width |
| attention mask | `[Q,L]` | `true` only for valid tokens |
| grounded patches | `[Q,C,N,D]` | candidate-specific generic text-conditioned field (chunked in production) |
| patch mask logits | `[Q,C,N]` | canonical logits used by pooling and scoring |
| base-grid mask logits | `[Q,C,32,32]` | canonical highest-resolution patch field |
| decoded mask logits | `[Q,C,H,W]` | learned multi-scale output; displayed and supervised mask |
| local embedding | `[Q,C,D]` | canonical mask-weighted temporal pooling |
| `S_global,S_local,S_token_patch,S_reranked` | `[Q,C]` | score matrices |
| slots (optional) | `[Q,C,K,D]` | class-agnostic region slots |
| slot masks/activation | `[Q,C,K,N]`, `[Q,C,K]` | optional ablation outputs |

The canonical scoring API accepts query and candidate chunks and returns all
scores, mask logits, local embeddings, temporal descriptors, coordinates,
scale IDs, optional temporal maps and memory diagnostics. Trainer, evaluator,
renderer and mask checker are forbidden from reimplementing these equations.

The displayed soft mask is `sigmoid(decoded_mask_logits)`. The local score uses
the corresponding canonical patch logits before learned decoding. Decoder
lateral refinements are constrained by deep supervision against the same target;
they are not an unrelated attention visualization.
