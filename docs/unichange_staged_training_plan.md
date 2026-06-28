# UniChange Staged Training Plan

## Decision

The project remains on the selected Stage-1 stack:

```text
UniverSat-B
+ jinaai/jina-embeddings-v5-text-small-retrieval
+ global/local retrieval
+ event queries
+ soft-mask decoder
```

BGE-M3 is not adopted in this branch. The useful lesson from multi-vector retrieval work is the scoring pattern, not a replacement of the chosen text encoder. Jina v5 is used with `Query:` for user/query text and `Document:` for pair-caption semantic targets.

The final representation of a pair is an active set of local change events. The `36x36x768` UniverSat grid is an internal tensor, not the final retrieval index.

## Why the architecture can train

The training risk is real: if retrieval, masks, temporal direction, and generation are optimized from the first step, the model can learn whichever signal is easiest and forget older behavior. The fix is a gated curriculum:

1. Prove image-text retrieval on LEVIR-CC.
2. Add local late interaction without changing the backbone.
3. Add latent event queries after retrieval already works.
4. Supervise event masks with LEVIR-MCI while replaying retrieval batches.
5. Add temporal directional readout only after the public joint path and masks work.
6. Add pair-to-pair event matching.
7. Add language generation last from structured event evidence.

Every stage after retrieval keeps either a retrieval replay stream or a pair-embedding distillation loss from the previous checkpoint. This prevents mask training from overwriting the retrieval space.

## Stage 0: Offline probes

Goal:

```text
offline Jina load
offline UniverSat load
joint temporal input [B,2,C,H,W]
output_grid=36 -> [B,1296,768]
metadata assumptions logged
```

No training happens here. The LEVIR adapter must record unknown GSD, unknown calendar dates, and relative before/after order. A registered UniverSat RGB adapter may be used as an implementation adapter, but it must not be reported as verified LEVIR sensor identity.

Gate:

```text
reports/model_probes/jina_v5.json status == ok
reports/model_probes/universat.json status == ok
```

## Stage 1: Global text-to-pair retrieval

Dataset:

```text
LEVIR-CC
100 unique pairs for overfit first
all sibling captions preserved
```

Frozen:

```text
UniverSat
Jina v5
event decoder
directional readout
```

Trainable:

```text
visual projection
attention/global pooling
semantic prediction head
```

Loss:

```text
L = L_masked_multi_positive_sigmoid + 0.25 L_semantic
```

This is not CLIP softmax. For pair `i` and caption `j`, labels are:

```text
+1  confirmed same-pair caption
-1  explicit or conservative safe negative
 0  unknown relation, ignored
```

All captions of the same pair are positives. Other pairs are not automatically negatives. Safe negatives may come only from verified incompatible labels or a conservative semantic bottom quantile.

Semantic prediction target:

```text
t_pair = normalize(mean(Jina("Document: caption_k") for all captions of pair))
```

The visual embedding predicts `stop_gradient(t_pair)`.

Gate:

```text
finite loss
finite gradients
loss decreases
train R@5 >= 0.95
train R@10 close to 1.0
```

No local retrieval, event masks, or language generation starts before this gate.

## Stage 2: Local text-region retrieval

Frozen:

```text
UniverSat
Jina v5
event decoder
directional readout
```

Trainable:

```text
local visual projection
text token projection
global retrieval heads
semantic head
```

Loss:

```text
L = L_retrieval + 0.25 L_semantic + 0.25 L_local + 0.10 L_distill
```

Local score uses smooth top-k late interaction:

```text
Jina contextual query tokens [B,L,512]
UniverSat local change tokens [B,1296,512]
token-region similarity
smooth top-k aggregation
```

This makes text queries able to prefer a region of a pair without yet claiming supervised segmentation.

Gate:

```text
global retrieval drop <= 2 points
local reranking improves qualitative text-region maps
```

## Stage 3: Latent event queries

Dataset:

```text
LEVIR-CC
```

Trainable:

```text
event queries
event decoder
event presence
mask projection
```

Frozen:

```text
UniverSat
Jina v5
```

Loss:

```text
L = L_retrieval + L_semantic + L_local + L_overlap + L_distill
```

Events are weakly supervised in this stage. They must become diverse and text-selectable, but LEVIR-CC does not provide reliable segment-level captions. Do not claim supervised grounding from this stage.

## Stage 4: LEVIR-MCI supervised masks

Dataset mix:

```text
50% LEVIR-CC retrieval replay
50% LEVIR-MCI mask batches
```

LEVIR-MCI binary masks are split into connected components. Components are matched to event queries with Hungarian or Sinkhorn assignment. The event decoder learns:

```text
event embedding
presence logit
soft mask [36,36]
```

Loss:

```text
L = L_retrieval
  + 0.25 L_semantic
  + 0.25 L_local
  + L_BCE
  + L_Dice
  + 0.5 L_presence
  + 0.5 L_coverage
  + 0.05 L_overlap
  + 0.10 L_distill
```

The important rule is: never train masks alone. Retrieval replay and distillation keep the text-to-pair behavior alive while masks become better.

Metrics:

```text
IoU
Dice
pixel AP
pixel AUPRC
soft IoU
pointing-game accuracy
energy inside ground-truth mask
retrieval drop from previous checkpoint
```

Gate:

```text
mask metrics improve
retrieval drop <= 2 points
event duplicate rate decreases
```

## Stage 5: Directional temporal readout

This stage introduces explicit time/direction. It is not part of the first overfit experiment.

Required implementation:

```text
inspect UniverSat temporal path
expose states before temporal collapse
shape [B,2,N,768]
add BEFORE role embedding
add AFTER role embedding
learned change query per spatial position
temporal cross-attention
```

Output:

```text
directional change tokens [B,N,768]
temporal attention [B,N,2]
```

Only relative time is used unless real metadata exists:

```text
before = 0
after = 1
```

Never invent calendar dates.

Loss:

```text
same mixed retrieval + mask losses
+ stronger distillation from previous public-joint checkpoint
```

Compare only:

```text
universat_joint_public
universat_directional_readout
```

Gate:

```text
better or equal retrieval
better local masks or local reranking
retrieval drop <= 2 points
```

## Stage 6: Pair-to-pair event matching

Offline indexing produces:

```text
pair_index: one row per pair
event_index: one row per active event
```

Each event stores:

```text
event_id
pair_id
event_embedding [512]
presence
mask_rle or compressed 36x36 mask
bbox
transition label if known
short evidence text if known
```

Text-to-pair retrieval:

```text
query text -> Jina Query embedding
global pair search
candidate event rerank
return pair + matched event masks
```

Pair-to-pair retrieval:

```text
query pair -> event set
candidate pairs -> event sets
symmetric smooth MaxSim or Sinkhorn event matching
return matched masks on both pairs
```

LEVIR-CC pair-to-pair remains exploratory. Strict semantic event evaluation comes from SECOND-CC and Hi-UCD style transition labels.

## Stage 7: Explanations

No large VLM is added during retrieval milestones.

First output structured evidence:

```json
{
  "query": "...",
  "retrieved_pair_id": "...",
  "matched_events": [],
  "masks": [],
  "scores": {},
  "evidence_sentences": []
}
```

Only after retrieval and grounding gates pass, attach a small language decoder. It generates from event embeddings, masks, bounding boxes, transition labels, and retrieved evidence, not from raw image tokens alone.

## Source-backed design references

- UniverSat official repo and README: joint temporal inputs and `output_grid=36` dense features.
- Jina v5 retrieval model card: `Query:` / `Document:` prompting, 1024-dimensional text features, Matryoshka truncation.
- SigLIP: pairwise sigmoid image-text loss instead of batch softmax competition.
- ColBERT and FILIP: late interaction between query tokens and precomputed region/image tokens.
- Mask2Former: query-based mask decoding with localized masks.
- I-JEPA and VL-JEPA: predict semantic embeddings before text generation.
- LEVIR-CC, LEVIR-MCI, SECOND-CC, Hi-UCD: staged data curriculum for captions, masks, semantic transitions, and direction.
