# UniChange v2 Architecture

## Decision

The main model line is a staged CLIP-like multimodal temporal architecture, not
a single summed joint loss and not a standalone T1/T2 semantic baseline.

```text
images [B,T,C,H,W]
  -> UniverSat per timestamp
  -> features [B,T,N,768]
  -> TemporalChangeEncoder
  -> pair_embedding + change_tokens + event_tokens
  -> retrieval, captioning, segmentation, grounding, event descriptions
```

`SemanticChangeBaseline` remains useful for supervised T1/T2 maps, transition
metrics, and segmentation supervision. It does not replace the shared temporal
representation used by retrieval and language tasks.

## Stage Contract

- Stage 1 retrieval: train temporal encoder, global/event tokens, and retrieval
  head with CLIP-like text-to-pair InfoNCE plus supervised pair-to-pair
  contrastive batches. One optimizer step uses one objective.
- Stage 2 captioning: load `best_retrieval.pt`, freeze visual retrieval
  encoder, train visual prefix adapter plus language-decoder LoRA with token
  cross-entropy only.
- Stage 3 segmentation: load `best_retrieval.pt`, train binary/semantic/event
  heads, then optionally unfreeze the last temporal blocks.
- Stage 4 text grounding: train `TextConditionedMaskDecoder` only on
  text-region aligned samples.
- Stage 5 event descriptions: train event-to-prefix adapter and decoder LoRA
  from event components and semantic-transition captions.

## Initial Implemented Modules

- `data.temporal_sample` and `data.temporal_collate`: common temporal sample and
  batch contract with variable `T`.
- `backbones.sequence_universat`: per-timestamp UniverSat wrapper preserving
  `[B,T,N,D]` features.
- `models.temporal_change_encoder`: temporal attention, shifted windowed
  spatial attention, global tokens, and normalized pair embeddings.
- `models.event_decoder`: event queries, presence logits, and upsampled masks.
- `models.retrieval_heads`: CLIP-like text-to-pair and supervised pair-to-pair
  losses.
- `models.caption_prefix_adapter`: visual prefix tokens for a frozen/LoRA
  causal decoder.
- `models.text_conditioned_mask_decoder`: query-conditioned soft mask decoder.
- `training.contrastive_queue` and `training.task_sampler`: queue-backed
  contrastive setup and one-objective-per-step route schedules.
- `scripts/train_unichange_v2_retrieval.py`: minimal Stage-1 retrieval smoke
  trainer for LEVIR-MCI with real per-frame UniverSat, frozen Jina, temporal
  encoder, text-to-pair InfoNCE, gradient audit, memory report, split leakage
  check, and checkpoint roundtrip.

## Cluster Readiness Status

The current repository is not full-training ready on H100 until the new smoke
job completes on Slurm with state `COMPLETED` and `ExitCode 0:0`.

First command to submit:

```bash
sbatch cluster/ysu/smoke_unichange_v2_retrieval.sbatch
```

Expected smoke artifacts:

```text
$RS_PROJECT_ROOT/runs/unichange_v2_retrieval_smoke/run_config.json
$RS_PROJECT_ROOT/runs/unichange_v2_retrieval_smoke/smoke_report.json
$RS_PROJECT_ROOT/runs/unichange_v2_retrieval_smoke/metrics_history.jsonl
$RS_PROJECT_ROOT/runs/unichange_v2_retrieval_smoke/best_retrieval.pt
$RS_PROJECT_ROOT/runs/unichange_v2_retrieval_smoke/last_retrieval.pt
```

Smoke PASS requires:

- Slurm state `COMPLETED` and `ExitCode 0:0`.
- finite loss and finite nonzero gradients in `TemporalChangeEncoder` and
  retrieval head.
- no gradients in frozen UniverSat or frozen Jina.
- train/validation pair IDs are disjoint.
- checkpoint save/reload preserves a validation embedding within `1e-5`.
- `smoke_report.json` records GPU name, PyTorch/CUDA versions, peak allocated
  and reserved VRAM, parameter counts, shapes, and validation retrieval metrics.

Do not launch `cluster/ysu/train_unichange_v2_retrieval.sbatch` until the smoke
job passes. Captioning, grounding, semantic segmentation integration, and
pair-to-pair retrieval are still later stages.

## Explicit Non-Goals For The First Pipeline

- No DINO objective in the main training pipeline.
- No VL-JEPA objective in the first working pipeline.
- No single `retrieval + caption + segmentation + grounding` loss.
- No validation embeddings in the contrastive queue.
- No direction loss unless reverse captions are semantically rewritten.
