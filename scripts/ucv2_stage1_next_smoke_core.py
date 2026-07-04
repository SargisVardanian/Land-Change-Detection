from __future__ import annotations

import json
from dataclasses import asdict
from itertools import cycle
from pathlib import Path
from collections import Counter

import torch

import train_unichange_v2_retrieval as base
from land_change_detection.models.retrieval_heads import (
    semantic_text_to_pair_set_loss,
    stable_caption_group_ids,
)
from land_change_detection.training.temporal_caption_dataset import parse_dataset_weights
from ucv2_cluster_common import build_model, run_metadata, strict_device
from ucv2_retrieval_metrics import relevance_aware_retrieval_metrics
from ucv2_stage1_next_core import (
    Stage1NextConfig,
    _build_stage1_full_count_datasets,
    _dataset_index_counts,
    _mixed_subset_coverage_details,
    caption_frequencies,
    make_eval_loader,
    make_optimizer,
    make_train_loader,
    save_checkpoint,
    trainable_stage1_parameters,
)


def run(
    data_root: Path,
    output_dir: Path,
    universat_source: Path,
    universat_checkpoint: Path,
    jina_model: Path,
    temporal_depth: int = 6,
    train_manifests: tuple[Path, ...] = (),
    val_manifests: tuple[Path, ...] = (),
    dataset_config: Path | None = None,
    dataset_sampling_weights: tuple[str, ...] = ("levir_mci=0.55", "second_cc=0.45"),
    text_max_length: int = 256,
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
        max_train_samples=20,
        max_val_samples=16,
        max_steps=10,
        epochs=2,
        num_workers=0,
        temporal_depth=temporal_depth,
        train_eval_pairs=0,
        train_eval_interval=0,
        checkpoint_interval_steps=0,
        smoke=True,
        train_manifests=tuple(str(path) for path in train_manifests),
        val_manifests=tuple(str(path) for path in val_manifests),
        dataset_config=str(dataset_config) if dataset_config else None,
        dataset_sampling_weights=dataset_sampling_weights,
        text_max_length=text_max_length,
    )
    base._set_seed(config.seed)
    base._write_json(
        output_dir / "run_config.json",
        asdict(config)
        | {
            "stage1_next": True,
            "loss": "semantic_soft_target_text_to_pair",
            "stable_caption_groups": True,
        },
    )

    from ucv2_stage1_next_core import _build_stage1_datasets, _stage1_data_metadata

    train, val = _build_stage1_datasets(config)
    full_train, full_val = _build_stage1_full_count_datasets(config)
    from ucv2_stage1_next_core import _assert_stage1_disjoint

    _assert_stage1_disjoint(train, val)
    data_metadata = _stage1_data_metadata(config, train, val, full_train=full_train, full_val=full_val)
    configured_dataset_weights = {name: value for name, value in sorted(parse_dataset_weights(config.dataset_sampling_weights).items()) if float(value) > 0.0}
    train_sample_counts = _dataset_index_counts(train)
    validation_sample_counts = _dataset_index_counts(val)
    train_mixed_subset_coverage: dict[str, object] = {}
    validation_mixed_subset_coverage: dict[str, object] = {}
    mixed_subset_coverage_passed = data_metadata["data_mode"] != "mixed"
    if data_metadata["data_mode"] == "mixed":
        train_mixed_subset_coverage = _mixed_subset_coverage_details(
            train,
            configured_weights=configured_dataset_weights,
            max_pairs=config.max_train_samples,
            subset_name="train",
        )
        validation_mixed_subset_coverage = _mixed_subset_coverage_details(
            val,
            configured_weights=configured_dataset_weights,
            max_pairs=config.max_val_samples,
            subset_name="validation",
        )
        mixed_subset_coverage_passed = bool(
            train_mixed_subset_coverage.get("mixed_subset_coverage_passed")
            and validation_mixed_subset_coverage.get("mixed_subset_coverage_passed")
        )
    frequencies = caption_frequencies(train)
    train_loader = make_train_loader(train, config, frequencies, epoch=0)
    val_loader = make_eval_loader(val, config)
    model = build_model(config, device)
    optimizer = make_optimizer(model, config)
    scheduler = base._make_scheduler(optimizer, config.max_steps or 10, config)

    gradient_audit = None
    finite_loss = True
    step = 0
    losses: list[float] = []
    batch_dataset_counts: Counter[str] = Counter()
    for batch in cycle(train_loader):
        if step >= 10:
            break
        model.train()
        batch_dataset_counts.update(str(name) for name in batch.get("dataset_names", []))
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
            )
        if not torch.isfinite(loss):
            finite_loss = False
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable_stage1_parameters(model),
            config.grad_clip_norm,
        )
        if gradient_audit is None:
            gradient_audit = base._gradient_audit(model)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        step += 1

    metrics = relevance_aware_retrieval_metrics(model, val_loader, device, config)
    checkpoint = output_dir / "last_retrieval.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        config,
        epoch_index=1,
        next_batch_index=0,
        step=step,
        best_scores={},
        metrics=metrics,
    )
    roundtrip = base._checkpoint_roundtrip(
        model,
        optimizer,
        scheduler,
        checkpoint,
        val_loader,
        device,
        config,
    )
    frozen_grad_violations, missing_gradients = base._audit_failures(gradient_audit)
    gradient_audit_passed = not frozen_grad_violations and not missing_gradients
    checkpoint_roundtrip_passed = bool(roundtrip["ok"])
    status = "PASS" if finite_loss and step == 10 and gradient_audit_passed and checkpoint_roundtrip_passed else "FAIL"
    device_type = device.type
    bf16_active = bool(device_type == "cuda" and config.use_bf16 and torch.cuda.is_bf16_supported())
    metadata = run_metadata()
    slurm_job_id = metadata.get("slurm_job_id", "")
    real_cluster_smoke_passed = bool(
        status == "PASS"
        and slurm_job_id
        and device_type == "cuda"
        and step == 10
        and finite_loss
        and gradient_audit_passed
        and checkpoint_roundtrip_passed
    )
    report = {
        **metadata,
        "status": status,
        "cluster_ready": real_cluster_smoke_passed,
        "cluster_ready_requires": "Slurm COMPLETED/0:0 and cluster report finalization.",
        "real_cluster_smoke_passed": real_cluster_smoke_passed,
        "device_type": device_type,
        "gpu_name": torch.cuda.get_device_name(device) if device_type == "cuda" else "cpu",
        "bf16_requested": bool(config.use_bf16),
        "bf16_active": bf16_active,
        "image_size": config.image_size,
        "output_grid": config.output_grid,
        "stage1_next": True,
        "loss": "semantic_soft_target_text_to_pair",
        "stable_caption_groups": True,
        "use_direction_embeddings": config.use_direction_embeddings,
        "use_explicit_change_fusion": config.use_explicit_change_fusion,
        "trainable_temperature": config.trainable_temperature,
        "temporal_depth": config.temporal_depth,
        "text_adapter_enabled": config.use_text_adapter,
        "text_max_length": config.text_max_length,
        "semantic_soft_target_weight": config.semantic_soft_target_weight,
        "semantic_teacher_top_k": config.semantic_teacher_top_k,
        "semantic_teacher_temperature": config.semantic_teacher_temperature,
        "max_captions_per_pair": config.max_captions_per_pair,
        "fake_backbones": False,
        "samples": {"train": len(train), "validation": len(val)},
        **data_metadata,
        "full_train_row_count": data_metadata["train_row_count"],
        "full_validation_row_count": data_metadata["validation_row_count"],
        "selected_train_row_count": len(train),
        "selected_validation_row_count": len(val),
        "mixed_smoke": data_metadata["data_mode"] == "mixed",
        "batch_dataset_counts": dict(sorted(batch_dataset_counts.items())),
        "sample_counts_by_dataset": train_sample_counts,
        "validation_sample_counts_by_dataset": validation_sample_counts,
        "configured_dataset_weights": configured_dataset_weights,
        "requested_max_train_samples": config.max_train_samples,
        "requested_max_val_samples": config.max_val_samples,
        "mixed_subset_coverage_passed": mixed_subset_coverage_passed,
        "train_mixed_subset_coverage": train_mixed_subset_coverage,
        "validation_mixed_subset_coverage": validation_mixed_subset_coverage,
        "train_val_disjoint": True,
        "steps_completed": step,
        "finite_loss": finite_loss,
        "losses": losses,
        "gradient_audit": gradient_audit,
        "gradient_audit_passed": gradient_audit_passed,
        "frozen_grad_violations": frozen_grad_violations,
        "missing_gradients": missing_gradients,
        "checkpoint_roundtrip": roundtrip,
        "checkpoint_roundtrip_passed": checkpoint_roundtrip_passed,
        "parameter_counts": base._parameter_counts(model),
        "final_validation": metrics,
        "memory": {
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }
    (output_dir / "smoke_report.json").write_text(json.dumps(base._json_ready(report), indent=2), encoding="utf-8")
    return 0 if status == "PASS" else 1
