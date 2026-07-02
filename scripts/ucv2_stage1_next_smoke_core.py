from __future__ import annotations

import json
from dataclasses import asdict
from itertools import cycle
from pathlib import Path

import torch

import train_unichange_v2_retrieval as base
from land_change_detection.models.retrieval_heads import (
    multi_positive_set_info_nce,
    stable_caption_group_ids,
)
from ucv2_cluster_common import build_model, strict_device
from ucv2_retrieval_metrics import relevance_aware_retrieval_metrics
from ucv2_stage1_next_core import (
    Stage1NextConfig,
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
        train_eval_pairs=0,
        train_eval_interval=0,
        checkpoint_interval_steps=0,
        smoke=True,
    )
    base._set_seed(config.seed)
    base._write_json(
        output_dir / "run_config.json",
        asdict(config)
        | {
            "stage1_next": True,
            "loss": "multi_positive_set_info_nce",
            "stable_caption_groups": True,
        },
    )

    train, val = base._build_datasets(config)
    base._assert_disjoint(train, val)
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
    for batch in cycle(train_loader):
        if step >= 10:
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
            loss = multi_positive_set_info_nce(
                output.pair_embedding,
                output.text_embedding,
                batch["caption_to_pair"],
                groups,
                logit_scale=model.retrieval_head.similarity_scale(),
                text_to_pair_weight=config.text_to_pair_weight,
                pair_to_text_weight=config.pair_to_text_weight,
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
    report = {
        "status": status,
        "cluster_ready": False,
        "cluster_ready_requires": "Slurm COMPLETED/0:0 and cluster report finalization.",
        "stage1_next": True,
        "loss": "multi_positive_set_info_nce",
        "stable_caption_groups": True,
        "use_direction_embeddings": config.use_direction_embeddings,
        "use_explicit_change_fusion": config.use_explicit_change_fusion,
        "trainable_temperature": config.trainable_temperature,
        "max_captions_per_pair": config.max_captions_per_pair,
        "fake_backbones": False,
        "samples": {"train": len(train), "validation": len(val)},
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
