from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

import train_unichange_v2_retrieval as base
from land_change_detection.models.retrieval_heads import (
    semantic_text_to_pair_set_loss,
    stable_caption_group_ids,
)
from ucv2_cluster_common import build_model, run_metadata, strict_device
from ucv2_stage1_next_core import (
    Stage1NextConfig,
    _build_stage1_datasets,
    _stage1_data_metadata,
    caption_frequencies,
    make_optimizer,
    make_train_loader,
    trainable_stage1_parameters,
)


def run(
    data_root: Path,
    output_dir: Path,
    universat_source: Path,
    universat_checkpoint: Path,
    jina_model: Path,
    temporal_depth: int = 6,
    text_max_length: int = 256,
    enable_patch_reranker: bool = False,
    qcpr_architecture_version: str = "v1",
    enable_temporal_explanation_channels: bool = False,
) -> int:
    device = strict_device("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Stage1NextConfig(
        data_root=str(data_root),
        output_dir=str(output_dir),
        universat_source=str(universat_source),
        universat_checkpoint=str(universat_checkpoint),
        jina_model=str(jina_model),
        batch_size=2,
        max_train_samples=64,
        max_val_samples=2,
        max_steps=2,
        epochs=1,
        num_workers=0,
        temporal_depth=temporal_depth,
        train_eval_pairs=0,
        train_eval_interval=0,
        checkpoint_interval_steps=0,
        text_max_length=text_max_length,
        enable_patch_reranker=enable_patch_reranker,
        qcpr_architecture_version=qcpr_architecture_version,
        enable_temporal_explanation_channels=enable_temporal_explanation_channels,
    )
    train, val = _build_stage1_datasets(config)
    data_metadata = _stage1_data_metadata(config, train, val)
    frequencies = caption_frequencies(train)
    model = build_model(config, device)
    optimizer = make_optimizer(model, config)
    rows = []

    for batch_size in (8, 16, 24, 32):
        batch_config = replace(config, batch_size=batch_size)
        loader = make_train_loader(train, batch_config, frequencies, epoch=0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        success = True
        error = ""
        shapes = {}
        try:
            for index, batch in enumerate(loader):
                if index == 2:
                    break
                model.train()
                batch = base._move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                groups = stable_caption_group_ids(batch["captions"], device=device)
                with base._amp_context(device, config.use_bf16):
                    output = model(
                        batch["images"],
                        batch["captions"],
                        batch["caption_to_pair"],
                        batch["temporal_valid_mask"],
                    )
                    loss = semantic_text_to_pair_set_loss(
                        output.pair_embedding,
                        output.text_embedding,
                        output.teacher_text_embedding,
                        batch["captions"],
                        batch["caption_to_pair"],
                        groups,
                        logit_scale=model.retrieval_head.similarity_scale(),
                        text_to_pair_weight=config.text_to_pair_weight,
                        pair_to_text_weight=config.pair_to_text_weight,
                        semantic_soft_target_weight=config.semantic_soft_target_weight,
                        semantic_teacher_top_k=config.semantic_teacher_top_k,
                        semantic_teacher_temperature=config.semantic_teacher_temperature,
                        logits_text_to_pair=(
                            output.final_scores * model.retrieval_head.similarity_scale()
                            if output.final_scores is not None and enable_patch_reranker
                            else None
                        ),
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_stage1_parameters(model),
                    config.grad_clip_norm,
                )
                optimizer.step()
                shapes = {
                    "images": list(batch["images"].shape),
                    "captions": len(batch["captions"]),
                }
        except torch.cuda.OutOfMemoryError as exc:
            success = False
            error = str(exc)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        rows.append(
            {
                "batch_size": batch_size,
                "success": success,
                "error": error,
                "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
                "shapes": shapes,
            }
        )
        if not success:
            break

    passed = [row["batch_size"] for row in rows if row["success"]]
    report = {
        **run_metadata(),
        "status": "PASS" if passed else "FAIL",
        "stage1_next": True,
        "loss": "semantic_soft_target_text_to_pair",
        "stable_caption_groups": True,
        "use_direction_embeddings": True,
        "use_explicit_change_fusion": True,
        "trainable_temperature": True,
        "temporal_depth": config.temporal_depth,
        "text_adapter_enabled": config.use_text_adapter,
        "text_max_length": config.text_max_length,
        "patch_reranker_available": enable_patch_reranker,
        "qcpr_score_mode": "fused" if enable_patch_reranker else "global",
        "semantic_soft_target_weight": config.semantic_soft_target_weight,
        "semantic_teacher_top_k": config.semantic_teacher_top_k,
        "semantic_teacher_temperature": config.semantic_teacher_temperature,
        "gpu_name": torch.cuda.get_device_name(device),
        "results": rows,
        "recommended_batch_size": max(passed) if passed else None,
        "batch_32_passed": 32 in passed,
        "memory_data_mode": "shape_probe",
        "memory_probe_scope": "architecture/batch memory validation only; not a SECOND-CC adapter validation",
        **data_metadata,
    }
    (output_dir / "memory_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if passed else 1
