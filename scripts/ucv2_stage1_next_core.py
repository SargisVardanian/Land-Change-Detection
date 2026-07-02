from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

import train_unichange_v2_retrieval as base
from land_change_detection.models.retrieval_heads import (
    multi_positive_set_info_nce,
    normalize_caption_text,
    stable_caption_group_ids,
)
from ucv2_cluster_common import build_model, run_metadata, strict_device
from ucv2_retrieval_metrics import relevance_aware_retrieval_metrics


@dataclass(frozen=True)
class Stage1NextConfig:
    data_root: str
    output_dir: str
    universat_source: str
    universat_checkpoint: str
    jina_model: str
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = 256
    output_grid: int = 32
    batch_size: int = 32
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_steps: int | None = None
    epochs: int = 25
    num_workers: int = 8
    learning_rate: float = 2e-4
    retrieval_head_lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    min_lr: float = 1e-6
    grad_clip_norm: float = 1.0
    use_bf16: bool = True
    device: str = "cuda"
    seed: int = 20260701
    smoke: bool = False
    fake_backbones: bool = False
    synthetic_data: bool = False

    temporal_depth: int = 4
    use_direction_embeddings: bool = True
    use_explicit_change_fusion: bool = True
    trainable_temperature: bool = True
    initial_temperature: float = 0.07
    max_logit_scale: float = 100.0

    max_captions_per_pair: int = 2
    caption_frequency_power: float = 0.5
    text_to_pair_weight: float = 0.75
    pair_to_text_weight: float = 0.25

    train_eval_pairs: int = 1024
    train_eval_interval: int = 2
    checkpoint_interval_steps: int = 500


class FrequencyBalancedCaptionCollator:
    def __init__(
        self,
        caption_frequencies: dict[str, int],
        *,
        max_captions_per_pair: int | None,
        frequency_power: float,
        seed: int,
        epoch: int,
        training: bool,
    ):
        self.caption_frequencies = caption_frequencies
        self.max_captions_per_pair = max_captions_per_pair
        self.frequency_power = float(frequency_power)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.training = bool(training)

    def _item_seed(self, pair_id: str) -> int:
        payload = f"{self.seed}:{self.epoch}:{pair_id}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & 0x7FFF_FFFF_FFFF_FFFF

    def _select_captions(self, pair_id: str, captions: list[str]) -> list[str]:
        limit = self.max_captions_per_pair
        if not self.training or limit is None or limit <= 0 or len(captions) <= limit:
            return captions
        weights = torch.tensor(
            [
                max(self.caption_frequencies.get(normalize_caption_text(caption), 1), 1)
                ** (-self.frequency_power)
                for caption in captions
            ],
            dtype=torch.float64,
        )
        generator = torch.Generator().manual_seed(self._item_seed(pair_id))
        indices = torch.multinomial(weights, num_samples=limit, replacement=False, generator=generator)
        return [captions[index] for index in sorted(indices.tolist())]

    def __call__(self, items: list[Any]) -> dict[str, Any]:
        images = torch.stack(
            [torch.stack([item.t1, item.t2], dim=0) for item in items],
            dim=0,
        )
        captions: list[str] = []
        caption_to_pair: list[int] = []
        for pair_index, item in enumerate(items):
            selected = self._select_captions(str(item.pair_id), list(item.captions))
            if not selected:
                raise RuntimeError(f"Pair {item.pair_id} has no usable captions")
            captions.extend(selected)
            caption_to_pair.extend([pair_index] * len(selected))
        return {
            "pair_ids": [str(item.pair_id) for item in items],
            "images": images,
            "timestamps": torch.tensor([[0.0, 1.0] for _ in items], dtype=torch.float32),
            "temporal_valid_mask": torch.ones(len(items), 2, dtype=torch.bool),
            "captions": captions,
            "caption_to_pair": torch.tensor(caption_to_pair, dtype=torch.long),
            "mask_fractions": torch.tensor(
                [float(item.mask.float().mean().item()) for item in items],
                dtype=torch.float32,
            ),
        }


def _captions_from_sample(sample: Any) -> list[str]:
    captions = [str(caption) for caption in getattr(sample, "captions", []) if str(caption).strip()]
    fallback = str(getattr(sample, "caption", ""))
    if not captions and fallback.strip():
        captions = [fallback]
    return captions


def caption_frequencies(dataset: Dataset) -> dict[str, int]:
    frequencies: Counter[str] = Counter()
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        for sample in samples:
            frequencies.update(normalize_caption_text(caption) for caption in _captions_from_sample(sample))
    else:
        for index in range(len(dataset)):
            item = dataset[index]
            frequencies.update(normalize_caption_text(caption) for caption in item.captions)
    return dict(frequencies)


def _make_collator(
    dataset: Dataset,
    config: Stage1NextConfig,
    *,
    epoch: int,
    training: bool,
    frequencies: dict[str, int] | None = None,
) -> FrequencyBalancedCaptionCollator:
    return FrequencyBalancedCaptionCollator(
        frequencies if frequencies is not None else (caption_frequencies(dataset) if training else {}),
        max_captions_per_pair=config.max_captions_per_pair if training else None,
        frequency_power=config.caption_frequency_power,
        seed=config.seed,
        epoch=epoch,
        training=training,
    )


def make_train_loader(
    dataset: Dataset,
    config: Stage1NextConfig,
    frequencies: dict[str, int],
    epoch: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        collate_fn=_make_collator(dataset, config, epoch=epoch, training=True, frequencies=frequencies),
        pin_memory=True,
        persistent_workers=False,
    )


def make_eval_loader(dataset: Dataset, config: Stage1NextConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_make_collator(dataset, config, epoch=0, training=False),
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )


def _no_decay(name: str, parameter: nn.Parameter) -> bool:
    lowered = name.casefold()
    return (
        parameter.ndim <= 1
        or lowered.endswith("bias")
        or "norm" in lowered
        or "global_queries" in lowered
        or "direction_embeddings" in lowered
        or "logit_scale" in lowered
    )


def _module_parameter_groups(
    module: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if _no_decay(name, parameter) else decay).append(parameter)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "lr": learning_rate, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": learning_rate, "weight_decay": 0.0})
    return groups


def make_optimizer(model: nn.Module, config: Stage1NextConfig) -> torch.optim.Optimizer:
    groups = [
        *_module_parameter_groups(
            model.temporal_encoder,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
        *_module_parameter_groups(
            model.retrieval_head,
            learning_rate=config.retrieval_head_lr,
            weight_decay=config.weight_decay,
        ),
    ]
    if not groups:
        raise RuntimeError("No trainable Stage-1 parameters were found")
    return torch.optim.AdamW(groups)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Stage1NextConfig,
    *,
    epoch_index: int,
    next_batch_index: int,
    step: int,
    best_scores: dict[str, float],
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "stage1_next": True,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": base._rng_state(),
            "config": asdict(config),
            "epoch_index": epoch_index,
            "next_batch_index": next_batch_index,
            "step": step,
            "best_scores": best_scores,
            "metrics": metrics,
        },
        path,
    )


def _append_history(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(base._json_ready(payload)) + "\n")


def _prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def composite_score(metrics: dict[str, Any]) -> float:
    score = (
        0.30 * float(metrics.get("text_to_pair_R@1", 0.0))
        + 0.20 * float(metrics.get("text_to_pair_R@5", 0.0))
        + 0.15 * float(metrics.get("text_to_pair_R@10", 0.0))
        + 0.20 * float(metrics.get("MRR", 0.0))
        + 0.15 * float(metrics.get("unique_caption_R@5", 0.0))
    )
    return float(score)


def _selection_scores(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "r1": float(metrics.get("text_to_pair_R@1", 0.0)),
        "mrr": float(metrics.get("MRR", 0.0)),
        "r10": float(metrics.get("text_to_pair_R@10", 0.0)),
        "exact_r10": float(metrics.get("exact_pair_R@10", 0.0)),
        "unique_r5": float(metrics.get("unique_caption_R@5", 0.0)),
        "composite": composite_score(metrics),
    }


def _critical_resume_config(config: Stage1NextConfig) -> dict[str, Any]:
    keys = (
        "image_size",
        "output_grid",
        "batch_size",
        "temporal_depth",
        "use_direction_embeddings",
        "use_explicit_change_fusion",
        "trainable_temperature",
        "max_captions_per_pair",
        "caption_frequency_power",
        "text_to_pair_weight",
        "pair_to_text_weight",
    )
    payload = asdict(config)
    return {key: payload[key] for key in keys}


def _validate_resume_config(config: Stage1NextConfig, payload: dict[str, Any]) -> None:
    previous = payload.get("config", {})
    expected = _critical_resume_config(config)
    mismatches = {
        key: (previous.get(key), value)
        for key, value in expected.items()
        if previous.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Resume checkpoint configuration mismatch: {mismatches}")


def run(
    data_root: Path,
    output_dir: Path,
    universat_source: Path,
    universat_checkpoint: Path,
    jina_model: Path,
    batch_size: int,
    epochs: int,
    num_workers: int,
    resume: Path | None = None,
    *,
    temporal_depth: int = 4,
    max_captions_per_pair: int = 2,
    caption_frequency_power: float = 0.5,
    train_eval_pairs: int = 1024,
    train_eval_interval: int = 2,
) -> int:
    device = strict_device("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Stage1NextConfig(
        data_root=str(data_root),
        output_dir=str(output_dir),
        universat_source=str(universat_source),
        universat_checkpoint=str(universat_checkpoint),
        jina_model=str(jina_model),
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers,
        temporal_depth=temporal_depth,
        max_captions_per_pair=max_captions_per_pair,
        caption_frequency_power=caption_frequency_power,
        train_eval_pairs=train_eval_pairs,
        train_eval_interval=train_eval_interval,
    )
    base._set_seed(config.seed)
    base._write_json(
        output_dir / "run_config.json",
        asdict(config)
        | {
            "stage1_next": True,
            "loss": "multi_positive_set_info_nce",
            "stable_caption_groups": True,
            "resume": str(resume) if resume else None,
            "history_schema": 3,
            **run_metadata(),
        },
    )

    train, val = base._build_datasets(config)
    base._assert_disjoint(train, val)
    frequencies = caption_frequencies(train)
    val_loader = make_eval_loader(val, config)
    train_eval_loader: DataLoader | None = None
    if config.train_eval_pairs > 0:
        count = min(config.train_eval_pairs, len(train))
        train_eval_dataset: Dataset = Subset(train, range(count))
        train_eval_loader = make_eval_loader(train_eval_dataset, config)

    steps_per_epoch = math.ceil(len(train) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    model = build_model(config, device)
    optimizer = make_optimizer(model, config)
    scheduler = base._make_scheduler(optimizer, total_steps, config)

    start_epoch = 0
    resume_batch = 0
    step = 0
    best_scores = {
        "r1": -1.0,
        "mrr": -1.0,
        "r10": -1.0,
        "exact_r10": -1.0,
        "unique_r5": -1.0,
        "composite": -1.0,
    }
    if resume:
        payload = torch.load(resume, map_location=device)
        if not payload.get("stage1_next"):
            raise RuntimeError("Refusing to resume Stage-1-next from a baseline checkpoint")
        _validate_resume_config(config, payload)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        base._restore_rng_state(payload.get("rng", {}))
        start_epoch = int(payload.get("epoch_index", 0))
        resume_batch = int(payload.get("next_batch_index", 0))
        step = int(payload.get("step", 0))
        best_scores.update({key: float(value) for key, value in payload.get("best_scores", {}).items()})

    history_path = output_dir / "metrics_history.jsonl"
    for epoch in range(start_epoch, config.epochs):
        train_loader = make_train_loader(train, config, frequencies, epoch)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        epoch_pair_count = 0
        grad_norms: list[float] = []
        clipped_steps = 0

        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            model.train()
            batch = base._move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            caption_groups = stable_caption_group_ids(batch["captions"], device=device)
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
                    caption_groups,
                    logit_scale=model.retrieval_head.similarity_scale(),
                    text_to_pair_weight=config.text_to_pair_weight,
                    pair_to_text_weight=config.pair_to_text_weight,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {float(loss.detach().cpu())}")
            loss.backward()
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                list(model.temporal_encoder.parameters())
                + list(model.retrieval_head.parameters()),
                config.grad_clip_norm,
            )
            grad_norm = float(torch.as_tensor(grad_norm_tensor).detach().cpu())
            grad_norms.append(grad_norm)
            clipped_steps += int(grad_norm > config.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            step += 1

            loss_value = float(loss.detach().cpu())
            pair_count = int(batch["images"].shape[0])
            epoch_loss_sum += loss_value
            epoch_steps += 1
            epoch_pair_count += pair_count
            _append_history(
                history_path,
                {
                    "record_type": "train_step",
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": loss_value,
                    "batch_pairs": pair_count,
                    "captions": len(batch["captions"]),
                    "grad_norm_before_clip": grad_norm,
                    "gradient_was_clipped": grad_norm > config.grad_clip_norm,
                    "logit_scale": float(model.retrieval_head.similarity_scale().detach().cpu()),
                    "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                },
            )
            if config.checkpoint_interval_steps > 0 and step % config.checkpoint_interval_steps == 0:
                save_checkpoint(
                    output_dir / f"step_{step}.pt",
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch_index=epoch,
                    next_batch_index=batch_index + 1,
                    step=step,
                    best_scores=best_scores,
                    metrics={},
                )
            if config.max_steps is not None and step >= config.max_steps:
                break

        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        training_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        training_peak_reserved = int(torch.cuda.max_memory_reserved(device))

        validation_started = time.perf_counter()
        metrics = relevance_aware_retrieval_metrics(model, val_loader, device, config)
        validation_seconds = time.perf_counter() - validation_started
        train_eval_metrics: dict[str, Any] = {}
        if (
            train_eval_loader is not None
            and config.train_eval_interval > 0
            and ((epoch + 1) % config.train_eval_interval == 0 or epoch + 1 == config.epochs)
        ):
            train_eval_metrics = _prefix_metrics(
                relevance_aware_retrieval_metrics(model, train_eval_loader, device, config),
                "train_subset_",
            )

        grad_tensor = torch.tensor(grad_norms, dtype=torch.float32)
        epoch_record: dict[str, Any] = {
            "record_type": "epoch",
            "epoch": epoch + 1,
            "step": step,
            "train_loss_mean": epoch_loss_sum / max(epoch_steps, 1),
            "train_steps": epoch_steps,
            "train_pair_count": epoch_pair_count,
            "train_seconds": train_seconds,
            "train_pairs_per_second": epoch_pair_count / max(train_seconds, 1e-12),
            "validation_seconds": validation_seconds,
            "training_peak_allocated_vram_bytes": training_peak_allocated,
            "training_peak_reserved_vram_bytes": training_peak_reserved,
            "grad_norm_mean": float(grad_tensor.mean().item()) if grad_tensor.numel() else 0.0,
            "grad_norm_p90": float(torch.quantile(grad_tensor, 0.9).item()) if grad_tensor.numel() else 0.0,
            "gradient_clipping_fraction": clipped_steps / max(epoch_steps, 1),
            "logit_scale": float(model.retrieval_head.similarity_scale().detach().cpu()),
            "effective_temperature": float(1.0 / model.retrieval_head.similarity_scale().detach().cpu()),
            **metrics,
            **train_eval_metrics,
        }
        epoch_record["composite_score"] = composite_score(epoch_record)
        _append_history(history_path, epoch_record)
        base._write_json(output_dir / "latest_metrics.json", epoch_record)

        selection = _selection_scores(epoch_record)
        for name, score in selection.items():
            if score > best_scores[name]:
                best_scores[name] = score
                save_checkpoint(
                    output_dir / f"best_{name}.pt",
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch_index=epoch + 1,
                    next_batch_index=0,
                    step=step,
                    best_scores=best_scores,
                    metrics=epoch_record,
                )
                if name == "composite":
                    save_checkpoint(
                        output_dir / "best_retrieval.pt",
                        model,
                        optimizer,
                        scheduler,
                        config,
                        epoch_index=epoch + 1,
                        next_batch_index=0,
                        step=step,
                        best_scores=best_scores,
                        metrics=epoch_record,
                    )

        save_checkpoint(
            output_dir / "last_retrieval.pt",
            model,
            optimizer,
            scheduler,
            config,
            epoch_index=epoch + 1,
            next_batch_index=0,
            step=step,
            best_scores=best_scores,
            metrics=epoch_record,
        )
        base._write_json(
            output_dir / "training_report.json",
            {
                "stage1_next": True,
                "status": "RUNNING" if epoch + 1 < config.epochs else "COMPLETED",
                "completed_epochs": epoch + 1,
                "step": step,
                "best_scores": best_scores,
                "latest_metrics": epoch_record,
                **run_metadata(),
            },
        )
        resume_batch = 0
        if config.max_steps is not None and step >= config.max_steps:
            break

    return 0
