# QCPR v3: generic temporal grounding — scientific hypotheses

Status: design for review; no v3 implementation or Slurm job has been started.

## Why v3

QCPR v1 is the immutable global retrieval baseline.  It maps a caption and a
bi-temporal image pair to one embedding each, then ranks by cosine similarity.
That is useful for broad semantic retrieval, but one pair vector is an
information bottleneck for compositional requests such as “two houses appeared
in the upper-right along the curved road”.  This is the same general limitation
that motivates fine-grained language–image late interaction in FILIP: words and
visual patches can be aligned without discarding all local correspondence into
one global vector ([Yao et al., 2021](https://arxiv.org/abs/2111.07783)).

V3 keeps v1-compatible global retrieval as a candidate generator.  It does
**not** retain the v2 internal `object/direction/location/count/relation`
scoring heads.  Those labels remain useful outside the model for data audit,
sampling, counterfactual-negative selection, and evaluation.

The proposal combines four established inductive biases, but their usefulness
for this particular remote-sensing task remains a testable hypothesis:

* global contrastive text–image retrieval ([Radford et al., 2021](https://arxiv.org/abs/2103.00020));
* patch/token late interaction ([Yao et al., 2021](https://arxiv.org/abs/2111.07783));
* language-conditioned dense segmentation ([Yang et al., 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_LAVT_Language-Aware_Vision_Transformer_for_Referring_Image_Segmentation_CVPR_2022_paper.pdf));
* optional class-agnostic region slots ([Locatello et al., 2020](https://arxiv.org/abs/2006.15055)) and a mask-transformer style multi-scale decoder ([Cheng et al., 2022](https://arxiv.org/abs/2112.01527)).

None of these sources proves that v3 will understand Armenian land change,
quantities, or spatial relations.  The experiment matrix defines the falsifiable
tests.

## Architecture, exact tensor contract

For a batch of `B` ordered image pairs, native RGB input resolution
`H0 × W0 = 256 × 256`, model width `D = 512`, text length `L ≤ 256`, and
`S = 3` visual scales:

| Symbol | Shape | Meaning |
|---|---:|---|
| `X` | `[B, 2, 3, H0, W0]` | ordered images `(T1, T2)` |
| `F¹_s, F²_s` | `[B, H_s, W_s, D_s]` | shared EO-backbone features at scale `s` |
| `P_s` | `[B, N_s, D]`, `N_s=H_sW_s` | projected temporal patches, with `H_s,W_s ∈ {(32,32),(16,16),(8,8)}` initially |
| `P` | `[B, N, D]`, `N=1344` | concatenation of all scales (`1024+256+64`) |
| `C` | `[B, N, D]` | continuous 2-D positional encoding projected to `D` |
| `Q` | `[B, L, D]` | Jina token embeddings and attention mask |
| `z_pair,z_text` | `[B,D]` | v1-compatible normalized global embeddings |
| `H` | `[B,N,D]` | generic query-conditioned temporal patch field |
| `m` | `[B,N]` | canonical low-resolution soft query mask logits/probabilities |
| `M` | `[B,H0,W0]` | multi-scale decoder mask, supervised at image resolution |
| `R` (optional) | `[B,K,D]`, `K=8` initially | class-agnostic learned region slots |
| `A` (optional) | `[B,K,N]` | slot-to-patch assignment, one soft mask per slot |

`K=8` is an ablation configuration, not a claim that a scene contains eight
objects.  Slots are exchangeable and have no “house”, “road”, or “tree” class
meaning.

### 1. Frozen/v1-compatible candidate generator

The global path remains separately usable:

\[
S_g(q,I)=\frac{z_{text}(q)^\top z_{pair}(I)}{\tau},\qquad
\mathcal C_N(q)=\operatorname{TopN}_{I\in\mathcal G} S_g(q,I).
\]

At first, v3 either uses immutable v1 embeddings or a strictly validated
v1-compatible student.  The expensive grounding decoder is evaluated only for
`I ∈ C_N(q)`.  Report `candidate recall@N` before judging reranking.

### 2. Generic temporal patch field

At each scale, the model builds a generic ordered descriptor, not an attribute
classifier:

\[
d_{s,p}=\operatorname{LN}\!\left(W_s[ f^1_{s,p}; f^2_{s,p};
 f^2_{s,p}-f^1_{s,p}; |f^2_{s,p}-f^1_{s,p}|; c_{s,p}]\right)\in\mathbb R^D.
\]

`c_{s,p}` is a learned continuous/Fourier encoding of normalized coordinates
`(x,y,x²,y²,xy)` and scale.  It permits the **learned** grounding function to
distinguish “upper-right” from “lower-left”; there is no `upper_right` head.
Neighbourhood/context is learned through multi-scale self-attention over `P`.

### 3. Generic cross-modal grounding decoder

For decoder layers `ℓ=1…L_g`, patches query text tokens and then contextualize
spatially:

\[
U^{(\ell)}=\operatorname{CrossAttn}(H^{(\ell-1)},Q,Q),\quad
H^{(\ell)}=\operatorname{SpatialBlock}(H^{(\ell-1)}+U^{(\ell)}),\quad H^{(0)}=P.
\]

The single learned interaction is responsible for object appearance, temporal
verbs, spatial language, relation language, and numerical words.  No parser
output is an input to `H` or to an internal attribute score.

### 4. One faithful mask, one local score

The same canonical patch logits drive all three uses:

\[
m_p=w_m^\top H_p,\quad a_p=\sigma(m_p),\quad
r_{q,I}=\frac{\sum_p a_p H_p}{\sum_p a_p+\epsilon},\quad
S_l(q,I)=\cos(W_q\bar Q, W_r r_{q,I}).
\]

`m` is decoded with learned multi-scale lateral features into `M`, rather than
bilinearly enlarging only a `32×32` field.  The renderer, local pooling, and
reranker must consume the same pre-decoder logits/mask weights via one model
API.  Any mismatch is a correctness bug, tested numerically.

For a candidate pool, late interaction is a generic learned decoder output
`S_l`; it is not a hand-written average of object/location/count scores.  A
small calibration layer may combine `S_g` and `S_l` only after the corresponding
ablation proves positive conditional gain.

### 5. Optional generic region slots

Slots are introduced only in ablation D:

\[
R^{(0)}\in\mathbb R^{K\times D},\quad
A_{k,p}=\operatorname{softmax}_{k}(R_k^\top W_hH_p),\quad
R_k\leftarrow\operatorname{SlotUpdate}(R_k,\sum_pA_{k,p}H_p).
\]

Each slot predicts only `{activation, embedding, soft mask}`.  Quantity is a
derived expectation `ĉ=Σ_k sigmoid(w_aᵀR_k)`, optionally supervised only when
a human/verified count label exists.  Diversity/non-overlap is a generic
regularizer, e.g. `Σ_{i≠j} mean(A_i A_j)`, not a house counter.

### 6. Temporal supervision

The temporal order is represented by signed and absolute differences.  Where
trusted labels exist, an auxiliary **generic three-channel temporal field** may
be supervised as `{changed, appeared, disappeared}`.  It is not connected to
language-rule branches.  Reversal is a consistency condition:

\[
T_{changed}(T_1,T_2)\approx T_{changed}(T_2,T_1),\quad
T_{appeared}(T_1,T_2)\approx T_{disappeared}(T_2,T_1).
\]

## Training objectives

For each ablation, activate only the losses prescribed in the matrix.

\[
\mathcal L_{global}=\mathcal L_{contrastive}(z_{text},z_{pair}).
\]

\[
\mathcal L_{mask}=\lambda_d\mathcal L_{Dice}(M,Y)+
\lambda_f\mathcal L_{Focal/Tversky}(M,Y)+
\lambda_b\mathcal L_{boundary}(M,Y)\;\text{(only where masks exist)}.
\]

`Dice`, `IoU`, precision/recall and false-positive rate are reported separately
for empty and non-empty masks; `L_boundary` is optional and selected on a
validation-only comparison, never tuned on test.

\[
\mathcal L_{reverse}=D(T_{changed}^{12},T_{changed}^{21})+
D(T_{appeared}^{12},T_{disappeared}^{21}).
\]

The ranking loss is contrastive over a positive and candidates in `C_N`; parser
categories may construct/audit negatives, but do not create a mandatory model
head or define its representation.

## Falsifiable hypotheses and acceptance signals

| ID | Hypothesis | Minimum evidence supporting it | Failure interpretation |
|---|---|---|---|
| H0 | v1 global retrieval remains a valid candidate generator | candidate recall@N and semantic R@K reported on fixed split | poor candidate recall limits every reranker |
| H1 | generic token–patch interaction improves composition without harming broad retrieval | positive conditional reranking gain; structured nDCG improves vs H0; global R@K within predeclared tolerance | patches alone do not resolve composition or data lacks signal |
| H2 | the faithful learned mask improves localization and reranking | non-empty Dice/IoU, pointing game, boundary IoU improve; mask/local-score parity passes | attention is not a usable query mask |
| H3 | a learned multi-scale decoder improves small-object and boundary localization over 32×32 interpolation | small-object Dice/recall and boundary IoU improve vs single-scale decoder | lateral features add cost without useful detail |
| H4 | class-agnostic region slots help multi-instance/count expressions | count bucket accuracy/MAE and multi-region mask metrics improve vs H3 | slots collapse, fragment, or data lacks verified count labels |
| H5 | rebalanced training improves rare compositional strata without damaging natural validation | gains on held-out balanced slice and no material natural-slice regression | sampling overfits rare/noisy cells |
| H6 | reversal consistency improves temporal direction | appeared/disappeared swap consistency and direction accuracy improve | labels/order/noise insufficient |

## What is deliberately deleted or demoted from v2

| v2 component | v3 disposition |
|---|---|
| `S_object`, `S_direction`, `S_location`, `S_count`, `S_relation` model heads | delete from v3 primary architecture |
| token-group masks and semantic parsers | evaluation, audit, sampling, and hard-negative metadata only |
| location-target coordinate scorer | delete; continuous coordinates enter generic patch field |
| connected-component count head | delete; optional generic slots provide region evidence |
| relation scorer | delete; cross-modal temporal field must learn relations |
| per-attribute auxiliary losses | delete from primary v3 ablations |
| v2 fused score as gallery-wide retriever | demote; global top-N then local reranking only |
| v2 query mask rendered separately from scoring | delete; one canonical mask API |

## Known limits before experiments

Architecture cannot create supervision missing from data.  “Curved road”,
“two houses”, or “top-right” require enough visually grounded, correctly split,
and independently validated examples.  A high score on parser-derived labels
is not scientific proof; the planned human-verified compositional subset is the
decisive evaluation.

## Archived v2 diagnostic

Job `99570` is archived, not promoted: `COMPLETED 0:0`, 100 steps, run
`/mnt/weka/svardanyan/rs_change_project/runs/qcpr_v2_d54072f_phasea_diagnostic_100_bs24`.
It initialized from immutable v1 checkpoint
`qcpr_e0_20260711-015629/pilot/best_retrieval.pt`; v3 must not initialize from
its checkpoint.  Its global semantic R@1/R@5 were `0.6400/0.9218`, below the
immutable v1 reference `0.7964/0.9255`; this is evidence that v2 is not a
scientific success and is retained only as a reproducibility artifact.
