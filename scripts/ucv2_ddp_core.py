from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import train_unichange_v2_retrieval as base
from ucv2_cluster_common import build_model
from ucv2_full_core import save_checkpoint
from land_change_detection.training.distributed_retrieval import distributed_multi_positive_info_nce


def run(
    data_root: Path,
    output_dir: Path,
    universat_source: Path,
    universat_checkpoint: Path,
    jina_model: Path,
    local_batch_size: int,
    epochs: int,
    num_workers: int,
    resume: Path | None = None,
) -> int:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    global_batch_size = local_batch_size * world_size
    if global_batch_size < 32:
        raise RuntimeError(f"Global contrastive batch must be at least 32, got {global_batch_size}.")

    base._set_seed(20260701 + rank)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = base.RetrievalConfig(
        data_root=str(data_root), output_dir=str(output_dir),
        universat_source=str(universat_source), universat_checkpoint=str(universat_checkpoint),
        jina_model=str(jina_model), image_size=256, output_grid=32,
        batch_size=local_batch_size, epochs=epochs, num_workers=num_workers,
        learning_rate=2e-4, retrieval_head_lr=3e-4, weight_decay=0.05,
        warmup_ratio=0.05, min_lr=1e-6, grad_clip_norm=1.0,
        use_bf16=True, device="cuda",
    )
    train_dataset, val_dataset = base._build_datasets(config)
    base._assert_disjoint(train_dataset, val_dataset)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=base._collate_temporal,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=local_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=base._collate_temporal,
            pin_memory=True,
        )
        base._write_json(
            output_dir / "run_config.json",
            asdict(config) | {
                "distributed": True,
                "world_size": world_size,
                "local_batch_size": local_batch_size,
                "global_batch_size": global_batch_size,
                "resume": str(resume) if resume else None,
            },
        )

    model = build_model(config, device)
    optimizer = base._make_optimizer(model, config)
    total_steps = len(train_loader) * epochs
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

    ddp_model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    history_path = output_dir / "metrics_history.jsonl"
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            ddp_model.train()
            batch = base._move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with base._amp_context(device, True):
                output = ddp_model(batch["images"], batch["captions"], batch["caption_to_pair"], batch["temporal_valid_mask"])
                loss = distributed_multi_positive_info_nce(output.pair_embedding, output.text_embedding, batch["caption_to_pair"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite distributed loss at step {step}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.temporal_encoder.parameters()) + list(model.retrieval_head.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if rank == 0:
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"epoch": epoch, "step": step, "loss": float(loss.detach().cpu()), "global_batch_size": global_batch_size}) + "\n")
                if step % 500 == 0:
                    save_checkpoint(output_dir / f"step_{step}.pt", model, optimizer, scheduler, config, epoch, batch_index + 1, step, best_metric, {})

        dist.barrier()
        if rank == 0:
            assert val_loader is not None
            metrics = base._retrieval_metrics(model, val_loader, device, config)
            metrics["global_batch_size"] = global_batch_size
            if metrics["text_to_pair_R@1"] > best_metric:
                best_metric = metrics["text_to_pair_R@1"]
                save_checkpoint(output_dir / "best_retrieval.pt", model, optimizer, scheduler, config, epoch + 1, 0, step, best_metric, metrics)
            save_checkpoint(output_dir / "last_retrieval.pt", model, optimizer, scheduler, config, epoch + 1, 0, step, best_metric, metrics)
        dist.barrier()
        resume_batch = 0

    dist.destroy_process_group()
    return 0
