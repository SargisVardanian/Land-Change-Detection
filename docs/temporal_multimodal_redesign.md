# Temporal Multimodal Redesign

## Status

The existing 100-pair, 10-epoch run is retained only as a smoke/overfit gate. It is not a scientifically valid training run and must not be used as a final benchmark.

## Design goals

The target model must support:

1. Explicit reasoning over two or more observations of the same location.
2. Text-to-sequence retrieval.
3. Change caption generation.
4. Query-conditioned soft segmentation.
5. Binary and semantic change segmentation.
6. Reuse of one shared temporal representation for downstream tasks.

## Visual encoding

UniverSat already supports snapshot inputs shaped `[B, C, H, W]` and temporal inputs shaped `[B, T, C, H, W]` with date metadata. The redesign should nevertheless preserve explicit per-time features instead of relying only on UniverSat's internal temporal collapse.

Two visual paths should be evaluated:

### A. Explicit per-frame path (default)

- Encode every timestamp with the same frozen UniverSat backbone.
- Produce `F` with shape `[B, T, N, 768]`.
- Add temporal role/date embeddings.
- Feed `F` into a trainable factorized temporal fusion transformer.

This path exposes before/after and longer-sequence states directly and makes temporal attention inspectable.

### B. Joint UniverSat temporal path (ablation)

- Pass the complete sequence to UniverSat as `[B, T, C, H, W]` plus dates.
- Use the resulting dense feature grid as a joint sequence representation.

### C. Dual-path fusion (later)

- Concatenate or cross-attend explicit per-frame features with joint UniverSat temporal features.
- Use this only after the simpler baselines are stable.

## Temporal fusion transformer

Initial specification:

- Input dimension: 768.
- Working dimension: 512 or 768.
- Depth: 4 transformer blocks.
- Attention heads: 8 or 12.
- FFN ratio: 4.
- Variable temporal depth: `T >= 2`.
- Spatial grid: 64 x 64 at the main supervised stage.
- Event queries: 32.

Full attention over all `T*N` tokens is too expensive at high resolution. Use factorized attention:

1. Temporal attention independently at each spatial position.
2. Windowed spatial attention inside each timestamp or fused map.
3. A small set of global/event tokens for long-range communication.

The model should emit:

- `sequence_global_embedding [B, 512]`
- `fused_change_tokens [B, N, D]`
- `event_tokens [B, K, D]`
- `per_time_tokens [B, T, N, D]`
- optional temporal attention maps

## Task heads

### Retrieval head

- Project the visual sequence embedding to 512 dimensions.
- Use frozen Jina v5 text embeddings initially.
- Train symmetric image-text InfoNCE/CLIP loss.
- Use a memory queue or all-gather so the effective negative set is much larger than the physical batch.

### Caption generation head

Do not train a language decoder from scratch on 10k pairs. Use a pretrained causal decoder with cross-attention or a prefix adapter. First candidate:

- 0.5B-1.5B language decoder.
- Freeze most language weights.
- Train visual projector + cross-attention + LoRA.
- Context length: at least 128 tokens; 256 preferred for instruction-style tasks.

### Query-conditioned segmentation head

- Encode the user query with Jina.
- Cross-attend query tokens to fused spatial tokens and event tokens.
- Produce a soft mask at 64 x 64 or 128 x 128.
- Upsample with a lightweight decoder and skip features.

This head requires text-specific masks. A union change mask is not enough when one pair contains several different changes.

### Change segmentation heads

- Binary change head.
- Semantic transition head where labels exist.
- Event presence/objectness head.
- Optional boundary head.

## Resolution policy

- LEVIR-MCI: use the native 256 x 256 images; do not downsample to 224.
- SECOND-CC: use native 256 x 256 patches first.
- High-resolution datasets: random 384 or 512 crops for training, sliding-window evaluation.
- Main feature grid: 64 x 64.
- High-resolution ablation: 128 x 128 through UniverSat's high-resolution output path.

## Training curriculum

### Stage 0: smoke and shape validation

- 4-16 samples.
- 1-10 optimizer steps.
- Verify shapes, finite losses, gradients, frozen parameters and checkpoints.

### Stage 1: temporal self-supervised adaptation

Train only the temporal fusion module and projection heads on large unlabeled sequences.

Objectives:

- Temporal JEPA: predict masked/future latent regions from visible temporal context.
- DINO-style teacher/student consistency across spatial and temporal views.
- Pair-order discrimination only when labels and captions remain semantically valid.
- Optional reconstruction-free masked latent prediction.

Recommended scale:

- Minimum useful pilot: 50k sequences.
- Preferred first run: 250k locations/sequences.
- 50k-200k optimizer steps.
- Global batch target: 128-512.

### Stage 2: CLIP-style change alignment

Train sequence-text alignment on captioned bi-temporal data.

- Symmetric text-to-sequence and sequence-to-text loss.
- Hard-negative mining by geography, scene type and lexical similarity.
- Multi-positive handling for five captions per image pair.
- Preserve Stage 1 SSL loss at a low replay weight to prevent forgetting.

Recommended scale:

- At least 20k captioned pairs.
- Prefer 50k-200k pairs when RSCC/CC-Foundation-style data are available.
- 20k-100k optimizer steps.

### Stage 3: joint supervised training

Train retrieval, captioning and segmentation heads together.

- Use LEVIR-MCI and SECOND-CC first.
- 20-40 epochs with early stopping.
- Balance datasets with a task-aware sampler.
- Use loss normalization or uncertainty weighting instead of manually summing unrelated loss scales.

### Stage 4: partial backbone fine-tuning

Only after the frozen-backbone baseline is stable:

- Unfreeze the last 2-4 UniverSat blocks.
- Backbone LR: approximately 10x smaller than head LR.
- Keep Jina frozen initially.
- Compare against the frozen baseline on validation sets.

## Batch policy

For CLIP-style training, physical batch size matters because it determines the in-batch negative set.

- Direct end-to-end at 256-384 resolution: target physical batch 16-32 on one H100.
- With cached frozen features: target batch 64-256.
- Add a queue of 4k-32k negative embeddings.
- Gradient accumulation alone does not create additional in-batch negatives.

## Dataset priorities

### Priority A: joint text + masks

1. LEVIR-MCI: 10,077 pairs, five captions per pair, binary/category change masks.
2. SECOND-CC: 6,041 pairs, 30,205 captions, semantic maps and transition labels.
3. Forest-Change / LEVIR-MCI-Trees: domain-specific forest adaptation when publicly available.

### Priority B: large change-caption corpora

1. RSCC: 62,315 disaster pre/post pairs with captions.
2. ChangeChat-87k: instruction data for captioning, counting and localization.
3. CC-Foundation: reported 200k pairs and 1.2M captions; use only the publicly released subset and record provenance.
4. ChangeIMTI: multi-task change captioning, classification, counting and localization; verify released files and licensing before ingestion.

### Priority C: temporal/segmentation supervision

1. DynamicEarthNet: daily multispectral sequences and monthly 7-class masks.
2. SpaceNet 7: monthly sequences, tracked building polygons and construction/demolition events.
3. Hi-UCD: three temporal phases, 0.1 m imagery and nine semantic classes.
4. S2Looking: 5,000 side-looking bi-temporal pairs and building-change masks.
5. xBD/xBD-S12: pre/post disaster imagery and building damage labels.
6. HRSCD, SECOND, LEVIR-CD and TERRA-CD for semantic/binary change segmentation.

### Priority D: large self-supervised temporal corpora

1. SSL4EO-S12: 250k locations, four seasons, Sentinel-1/2, roughly 3M patches.
2. SpaceNet 7 sequences.
3. DynamicEarthNet sequences.
4. PASTIS/PASTIS-HD and other registered UniverSat pretraining datasets.

### Priority E: general EO language alignment

1. RS5M: 5M image-text pairs.
2. SkyScript: 2.6M image-text pairs.
3. BigEarthNet.txt: 464,044 co-registered S1/S2 images with 9.6M text annotations.

These datasets teach general EO semantics but do not replace temporal change-caption data.

## Metrics

### Retrieval

- Text-to-sequence R@1, R@5, R@10.
- Sequence-to-text R@1, R@5, R@10.
- MRR, median rank and mAP.
- Geographic and event-category stratified results.

### Captioning

- BLEU-1/4.
- METEOR.
- ROUGE-L.
- CIDEr.
- SPICE.
- BERTScore or sentence-embedding semantic similarity.
- Hallucination/error rate on object presence, count and direction.

### Segmentation and grounding

- Binary IoU, Dice/F1, precision and recall.
- Semantic mIoU and per-transition IoU.
- Boundary F1.
- Text-conditioned soft IoU/Dice.
- Energy inside ground-truth mask.
- Object/component recall and matching IoU.

### Temporal reasoning

- Order sensitivity.
- Event counting accuracy.
- Construction/demolition direction accuracy.
- Performance versus temporal gap and sequence length.

## Required baselines

1. Frozen UniverSat + linear retrieval head.
2. Per-frame UniverSat + simple `concat(F1, F2, |F2-F1|, F1*F2)` fusion.
3. Joint UniverSat temporal encoding.
4. Four-layer temporal fusion transformer.
5. Temporal fusion + JEPA.
6. Temporal fusion + DINO consistency.
7. Full retrieval + caption + segmentation model.
8. Learned Jina projection versus Matryoshka truncation.

## Immediate implementation order

1. Add a dataset registry and local asset auditor.
2. Replace the fixed `t1/t2` contract with a sequence contract `[B, T, C, H, W]` plus timestamps.
3. Add explicit per-frame UniverSat encoding.
4. Add the factorized temporal fusion transformer.
5. Add a feature-cache format for frozen UniverSat and Jina outputs.
6. Implement symmetric CLIP loss with a memory queue.
7. Add separate train/validation/test loaders.
8. Add caption generation and text-conditioned segmentation after the retrieval baseline is stable.
