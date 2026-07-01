from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import train_unichange_v2_retrieval as base
from ucv2_cluster_common import build_model, strict_device
from ucv2_retrieval_metrics import relevance_aware_retrieval_metrics


def save_checkpoint(path, model, optimizer, scheduler, config, epoch, batch_index, step, best_metric, metrics):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": base._rng_state(),
        "config": asdict(config),
        "epoch": epoch,
        "batch_index": batch_index,
        "step": step,
        "best_metric": best_metric,
        "metrics": metrics,
    }, path)


def _append_history(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def run(data_root: Path, output_dir: Path, universat_source: Path, universat_checkpoint: Path, jina_model: Path, batch_size: int, epochs: int, num_workers: int, resume: Path | None = None) -> int:
    device = strict_device("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = base.RetrievalConfig(
        data_root=str(data_root), output_dir=str(output_dir),
        universat_source=str(universat_source), universat_checkpoint=str(universat_checkpoint),
        jina_model=str(jina_model), image_size=256, output_grid=32,
        batch_size=batch_size, epochs=epochs, num_workers=num_workers,
        learning_rate=2e-4, retrieval_head_lr=3e-4, weight_decay=0.05,
        warmup_ratio=0.05, min_lr=1e-6, grad_clip_norm=1.0,
        use_bf16=True, device="cuda",
    )
    base._write_json(output_dir / "run_config.json", asdict(config) | {
        "resume": str(resume) if resume else None,
        "duplicate_caption_aware": True,
        "history_schema": 2,
    })
    train, val = base._build_datasets(config)
    base._assert_disjoint(train, val)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=base._collate_temporal, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=base._collate_temporal, pin_memory=True)
    total_steps = len(train_loader) * epochs
    model = build_model(config, device)
    optimizer = base._make_optimizer(model, config)
    scheduler = base._make_scheduler(optimizer, total_steps, config)
    start_epoch = 0
    resume_batch = 0
    step = 0
    best_metric = -1.0
    if resume:
        payload = torch.load(resume, map_location=device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        base._restore_rng_state(payload.get("rng", {}))
        start_epoch = int(payload.get("epoch", 0))
        resume_batch = int(payload.get("batch_index", 0))
        step = int(payload.get("step", 0))
        best_metric = float(payload.get("best_metric", -1.0))

    history_path = output_dir / "metrics_history.jsonl"
    for epoch in range(start_epoch, epochs):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        epoch_pair_count = 0

        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            model.train()
            batch = base._move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with base._amp_context(device, True):
                output = model(batch["images"], batch["captions"], batch["caption_to_pair"], batch["temporal_valid_mask"])
                loss = base.multi_positive_symmetric_info_nce(output.pair_embedding, output.text_embedding, batch["caption_to_pair"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.temporal_encoder.parameters()) + list(model.retrieval_head.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            loss_value = float(loss.detach().cpu())
            pair_count = int(batch["images"].shape[0])
            epoch_loss_sum += loss_value
            epoch_steps += 1
            epoch_pair_count += pair_count
            _append_history(history_path, {
                "record_type": "train_step",
                "epoch": epoch + 1,
                "step": step,
                "loss": loss_value,
                "batch_pairs": pair_count,
                "captions": len(batch["captions"]),
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            })
            if step % 500 == 0:
                save_checkpoint(output_dir / f"step_{step}.pt", model, optimizer, scheduler, config, epoch, batch_index + 1, step, best_metric, {})

        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        metrics = relevance_aware_retrieval_metrics(model, val_loader, device, config)
        validation_seconds = time.perf_counter() - validation_started
        epoch_record = {
            "record_type": "epoch",
            "epoch": epoch + 1,
            "step": step,
            "train_loss_mean": epoch_loss_sum / max(epoch_steps, 1),
            "train_steps": epoch_steps,
            "train_pair_count": epoch_pair_count,
            "train_seconds": train_seconds,
            "train_pairs_per_second": epoch_pair_count / max(train_seconds, 1e-12),
            "validation_seconds": validation_seconds,
            "training_peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "training_peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
            **metrics,
        }
        _append_history(history_path, epoch_record)
        base._write_json(output_dir / "latest_metrics.json", epoch_record)

        if float(metrics["text_to_pair_R@1"]) > best_metric:
            best_metric = float(metrics["text_to_pair_R@1"])
            save_checkpoint(output_dir / "best_retrieval.pt", model, optimizer, scheduler, config, epoch + 1, 0, step, best_metric, metrics)
        save_checkpoint(output_dir / "last_retrieval.pt", model, optimizer, scheduler, config, epoch + 1, 0, step, best_metric, metrics)
        resume_batch = 0
    return 0
