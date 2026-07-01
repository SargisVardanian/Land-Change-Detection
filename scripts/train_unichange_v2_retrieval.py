#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from land_change_detection.backbones.jina_v5_text import JinaV5TextConfig, JinaV5TextEncoder, TextFeatures
from land_change_detection.backbones.sequence_universat import SequenceUniverSatEncoder
from land_change_detection.backbones.universat_backend import UniverSatBackendConfig, UniverSatJointBackend
from land_change_detection.data.unichange_mci import UniChangeMciDataset, UniChangeMciItem
from land_change_detection.data.unichange_subset import resolve_levir_mci_root
from land_change_detection.models.retrieval_heads import RetrievalProjectionHead, multi_positive_symmetric_info_nce
from land_change_detection.models.temporal_change_encoder import TemporalChangeEncoder, TemporalChangeEncoderConfig
from land_change_detection.models.unichange_v2_retrieval import UniChangeV2RetrievalModel


@dataclass(frozen=True)
class RetrievalConfig:
    data_root: str
    output_dir: str
    universat_source: str
    universat_checkpoint: str
    jina_model: str
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = 256
    output_grid: int = 32
    batch_size: int = 16
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_steps: int | None = None
    epochs: int = 1
    num_workers: int = 2
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


def parse_args() -> argparse.Namespace:
    root = Path(os.environ.get("RS_PROJECT_ROOT", Path.cwd()))
    parser = argparse.ArgumentParser(description="Train/smoke UniChange v2 Stage-1 text-to-pair retrieval.")
    parser.add_argument("--data-root", type=Path, default=root / "datasets" / "processed" / "LEVIR-MCI")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--universat-source", type=Path, default=root / "external" / "UniverSat")
    parser.add_argument("--universat-checkpoint", type=Path, default=root / "models" / "universat-base")
    parser.add_argument("--jina-model", type=Path, default=root / "models" / "jina-v5-text-small-retrieval")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output-grid", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--retrieval-head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--use-bf16", dest="use_bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="use_bf16", action="store_false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fake-backbones", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--synthetic-data", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _SyntheticMciDataset(Dataset[UniChangeMciItem]):
    def __init__(self, split: str, count: int, image_size: int):
        self.split = split
        self.count = count
        self.image_size = image_size

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> UniChangeMciItem:
        from land_change_detection.data.event_targets import ComponentTargets, TemporalContext

        generator = torch.Generator().manual_seed(index + (0 if self.split == "train" else 10_000))
        t1 = torch.rand(3, self.image_size, self.image_size, generator=generator)
        t2 = torch.rand(3, self.image_size, self.image_size, generator=generator)
        mask = torch.zeros(self.image_size, self.image_size)
        empty_components = ComponentTargets(masks=torch.empty(0, 32 * 32), full_resolution_mask=mask, areas=[])
        return UniChangeMciItem(
            pair_id=f"{self.split}_{index}",
            t1=t1,
            t2=t2,
            captions=[f"{self.split} synthetic change {index}"],
            mask=mask,
            components=empty_components,
            temporal_context=TemporalContext(),
            metadata={"split": self.split},
        )


def _collate_temporal(items: list[UniChangeMciItem]) -> dict[str, Any]:
    images = torch.stack([torch.stack([item.t1, item.t2], dim=0) for item in items], dim=0)
    captions: list[str] = []
    caption_to_pair: list[int] = []
    for pair_index, item in enumerate(items):
        captions.extend(item.captions)
        caption_to_pair.extend([pair_index] * len(item.captions))
    return {
        "pair_ids": [item.pair_id for item in items],
        "images": images,
        "timestamps": torch.tensor([[0.0, 1.0] for _ in items], dtype=torch.float32),
        "temporal_valid_mask": torch.ones(len(items), 2, dtype=torch.bool),
        "captions": captions,
        "caption_to_pair": torch.tensor(caption_to_pair, dtype=torch.long),
    }


def _build_datasets(config: RetrievalConfig) -> tuple[Dataset, Dataset]:
    if config.synthetic_data:
        return (
            _SyntheticMciDataset("train", config.max_train_samples or 16, config.image_size),
            _SyntheticMciDataset("val", config.max_val_samples or 8, config.image_size),
        )
    root = resolve_levir_mci_root(config.data_root).root
    train = UniChangeMciDataset(
        root,
        split=config.train_split,
        image_size=config.image_size,
        output_grid=config.output_grid,
        max_pairs=config.max_train_samples,
    )
    val = UniChangeMciDataset(
        root,
        split=config.val_split,
        image_size=config.image_size,
        output_grid=config.output_grid,
        max_pairs=config.max_val_samples,
    )
    return train, val


class _FakeImageEncoder(nn.Module):
    def __init__(self, tokens: int, dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(3, dim)
        self.tokens = tokens

    def forward(self, images: Tensor, output_grid: int | None = None) -> Tensor:
        pooled = images.mean(dim=(-2, -1))
        token = self.proj(pooled)
        return token.unsqueeze(1).expand(images.shape[0], self.tokens, token.shape[-1])


class _FakeTextEncoder(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(8192, dim)

    def forward(self, texts: list[str], role: str = "document") -> TextFeatures:
        device = self.embedding.weight.device
        ids = torch.tensor([abs(hash(text)) % 8192 for text in texts], device=device)
        global_embedding = F.normalize(self.embedding(ids), dim=-1)
        return TextFeatures(
            global_embedding=global_embedding,
            token_embeddings=global_embedding.unsqueeze(1),
            attention_mask=torch.ones(len(texts), 1, dtype=torch.long, device=device),
            role="document",  # type: ignore[arg-type]
            metadata={"fake": True},
        )


def _build_model(config: RetrievalConfig, device: torch.device) -> UniChangeV2RetrievalModel:
    if config.fake_backbones:
        image_encoder = _FakeImageEncoder(config.output_grid * config.output_grid)
        text_encoder: nn.Module = _FakeTextEncoder()
    else:
        joint = UniverSatJointBackend(
            UniverSatBackendConfig(
                source_dir=config.universat_source,
                checkpoint_dir=config.universat_checkpoint,
                output_grid=config.output_grid,
                freeze=True,
            )
        )
        image_encoder = joint.model
        text_encoder = JinaV5TextEncoder(
            JinaV5TextConfig(
                model_path=config.jina_model,
                max_length=96,
                global_projection_mode="matryoshka_truncate",
                freeze=True,
            )
        )
    visual = SequenceUniverSatEncoder(
        image_encoder,
        output_grid=config.output_grid,
        visual_dim=768,
        freeze=True,
    )
    temporal = TemporalChangeEncoder(
        TemporalChangeEncoderConfig(
            input_dim=768,
            hidden_dim=512,
            depth=4,
            heads=8,
            ffn_dim=2048,
            grid_size=config.output_grid,
            window_size=8,
            global_tokens=4,
        )
    )
    model = UniChangeV2RetrievalModel(
        visual_encoder=visual,
        temporal_encoder=temporal,
        text_encoder=text_encoder,
        retrieval_head=RetrievalProjectionHead(dim=512),
    )
    return model.to(device)


def _assert_disjoint(train: Dataset, val: Dataset) -> None:
    def ids(dataset: Dataset) -> set[str]:
        samples = getattr(dataset, "samples", None)
        if samples is not None:
            return {str(getattr(sample, "sample_id")) for sample in samples}
        return {str(getattr(dataset[index], "pair_id")) for index in range(len(dataset))}

    train_ids = ids(train)
    val_ids = ids(val)
    overlap = train_ids & val_ids
    if overlap:
        preview = sorted(overlap)[:10]
        raise RuntimeError(f"Train/validation pair ID leakage detected: {preview}")


def _make_optimizer(model: UniChangeV2RetrievalModel, config: RetrievalConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": model.temporal_encoder.parameters(), "lr": config.learning_rate},
            {"params": model.retrieval_head.parameters(), "lr": config.retrieval_head_lr},
        ],
        weight_decay=config.weight_decay,
    )


def _make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, config: RetrievalConfig):
    warmup = max(1, int(total_steps * config.warmup_ratio))
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max((step + 1) / warmup, 1e-6)
        progress = min(max((step - warmup) / max(total_steps - warmup, 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = min(config.min_lr / max(lr, config.min_lr) for lr in base_lrs)
        return max(cosine, min_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _amp_context(device: torch.device, use_bf16: bool):
    enabled = device.type == "cuda" and use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _gradient_audit(model: UniChangeV2RetrievalModel) -> dict[str, Any]:
    def stats(parameters) -> dict[str, Any]:
        grads = [param.grad.detach() for param in parameters if param.grad is not None]
        if not grads:
            return {"has_grad": False, "finite": True, "nonzero": False}
        finite = all(torch.isfinite(grad).all().item() for grad in grads)
        nonzero = any(bool((grad.abs() > 0).any().item()) for grad in grads)
        return {"has_grad": True, "finite": bool(finite), "nonzero": bool(nonzero)}

    return {
        "visual_backbone": stats(model.visual_encoder.image_encoder.parameters()),
        "text_encoder": stats(model.text_encoder.parameters()),
        "temporal_encoder": stats(model.temporal_encoder.parameters()),
        "retrieval_head": stats(model.retrieval_head.parameters()),
    }


def _parameter_counts(model: UniChangeV2RetrievalModel) -> dict[str, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    return {"trainable": trainable, "frozen": frozen}


def _audit_failures(gradient_audit: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if gradient_audit is None:
        return ["visual_backbone", "text_encoder"], ["temporal_encoder", "retrieval_head"]
    frozen_grad_violations = [
        name
        for name in ("visual_backbone", "text_encoder")
        if bool(gradient_audit.get(name, {}).get("has_grad", False))
    ]
    missing_gradients = [
        name
        for name in ("temporal_encoder", "retrieval_head")
        if not (
            bool(gradient_audit.get(name, {}).get("has_grad", False))
            and bool(gradient_audit.get(name, {}).get("finite", False))
            and bool(gradient_audit.get(name, {}).get("nonzero", False))
        )
    ]
    return frozen_grad_violations, missing_gradients


def _retrieval_metrics(model: UniChangeV2RetrievalModel, loader: DataLoader, device: torch.device, config: RetrievalConfig) -> dict[str, float]:
    model.eval()
    pair_embeddings: list[Tensor] = []
    text_embeddings: list[Tensor] = []
    caption_to_pair_all: list[Tensor] = []
    pair_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            with _amp_context(device, config.use_bf16):
                output = model(batch["images"], batch["captions"], batch["caption_to_pair"], batch["temporal_valid_mask"])
            pair_embeddings.append(output.pair_embedding.float().cpu())
            text_embeddings.append(output.text_embedding.float().cpu())
            caption_to_pair_all.append(batch["caption_to_pair"].cpu() + pair_offset)
            pair_offset += output.pair_embedding.shape[0]
    if not pair_embeddings:
        return {"text_to_pair_R@1": 0.0, "text_to_pair_R@5": 0.0, "MRR": 0.0, "median_rank": 0.0}
    pairs = torch.cat(pair_embeddings)
    texts = torch.cat(text_embeddings)
    caption_to_pair = torch.cat(caption_to_pair_all)
    sims = texts @ pairs.T
    ranks = []
    for caption_index, pair_index in enumerate(caption_to_pair.tolist()):
        order = torch.argsort(sims[caption_index], descending=True)
        rank = int((order == pair_index).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append(rank)
    rank_tensor = torch.tensor(ranks, dtype=torch.float32)
    return {
        "text_to_pair_R@1": float((rank_tensor <= 1).float().mean().item()),
        "text_to_pair_R@5": float((rank_tensor <= 5).float().mean().item()),
        "text_to_pair_R@10": float((rank_tensor <= 10).float().mean().item()),
        "MRR": float((1.0 / rank_tensor).mean().item()),
        "median_rank": float(rank_tensor.median().item()),
    }


def _rng_state() -> dict[str, Any]:
    state = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"].cpu() if hasattr(state["torch_cpu"], "cpu") else state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_checkpoint(path: Path, model, optimizer, scheduler, config: RetrievalConfig, step: int, metrics: dict[str, Any]) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(),
            "config": asdict(config),
            "step": step,
            "metrics": metrics,
        },
        path,
    )


def _checkpoint_roundtrip(model, optimizer, scheduler, checkpoint: Path, val_loader: DataLoader, device: torch.device, config: RetrievalConfig) -> dict[str, Any]:
    batch = _move_batch(next(iter(val_loader)), device)
    model.eval()
    with torch.no_grad(), _amp_context(device, config.use_bf16):
        before, _ = model.encode_pairs(batch["images"], batch["temporal_valid_mask"])
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng_state(payload.get("rng", {}))
    with torch.no_grad(), _amp_context(device, config.use_bf16):
        after, _ = model.encode_pairs(batch["images"], batch["temporal_valid_mask"])
    max_abs_diff = float((before.float() - after.float()).abs().max().item())
    return {"ok": max_abs_diff <= 1e-5, "max_abs_diff": max_abs_diff}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.max_train_samples = min(args.max_train_samples or 16, 16)
        args.max_val_samples = min(args.max_val_samples or 16, 16)
        args.batch_size = 2
        args.max_steps = min(args.max_steps or 10, 10)
        args.num_workers = 0
    config = RetrievalConfig(
        data_root=str(args.data_root),
        output_dir=str(args.output_dir),
        universat_source=str(args.universat_source),
        universat_checkpoint=str(args.universat_checkpoint),
        jina_model=str(args.jina_model),
        train_split=args.train_split,
        val_split=args.val_split,
        image_size=args.image_size,
        output_grid=args.output_grid,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_steps=args.max_steps,
        epochs=args.epochs,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        retrieval_head_lr=args.retrieval_head_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_lr,
        grad_clip_norm=args.grad_clip_norm,
        use_bf16=args.use_bf16,
        device=args.device,
        seed=args.seed,
        smoke=args.smoke,
        fake_backbones=args.fake_backbones,
        synthetic_data=args.synthetic_data,
    )
    _set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_config.json", asdict(config) | {"cluster_ready": False, "readiness_note": "Requires real Slurm COMPLETED/0:0 smoke report."})

    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_dataset, val_dataset = _build_datasets(config)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"Train/validation datasets must be non-empty, got train={len(train_dataset)} val={len(val_dataset)}")
    _assert_disjoint(train_dataset, val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=_collate_temporal)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=_collate_temporal)
    total_batches = len(train_loader) * max(config.epochs, 1)
    total_steps = config.max_steps or total_batches
    model = _build_model(config, device)
    optimizer = _make_optimizer(model, config)
    scheduler = _make_scheduler(optimizer, total_steps, config)

    history_path = output_dir / "metrics_history.jsonl"
    best_metric = -1.0
    step = 0
    gradient_audit: dict[str, Any] | None = None
    shape_report: dict[str, Any] = {}
    for epoch in range(config.epochs):
        for batch in train_loader:
            if config.max_steps is not None and step >= config.max_steps:
                break
            model.train()
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, config.use_bf16):
                output = model(batch["images"], batch["captions"], batch["caption_to_pair"], batch["temporal_valid_mask"])
                loss = multi_positive_symmetric_info_nce(output.pair_embedding, output.text_embedding, batch["caption_to_pair"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.temporal_encoder.parameters()) + list(model.retrieval_head.parameters()),
                config.grad_clip_norm,
            )
            if gradient_audit is None:
                gradient_audit = _gradient_audit(model)
            optimizer.step()
            scheduler.step()
            shape_report = {
                "images": list(batch["images"].shape),
                "pair_embedding": list(output.pair_embedding.shape),
                "text_embedding": list(output.text_embedding.shape),
                "logits": list(output.logits.shape),
                "visual_metadata": output.visual_metadata,
            }
            row = {
                "step": step + 1,
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "lr": [group["lr"] for group in optimizer.param_groups],
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_ready(row)) + "\n")
            step += 1
        if config.max_steps is not None and step >= config.max_steps:
            break
        metrics = _retrieval_metrics(model, val_loader, device, config)
        if metrics["text_to_pair_R@1"] > best_metric:
            best_metric = metrics["text_to_pair_R@1"]
            _save_checkpoint(output_dir / "best_retrieval.pt", model, optimizer, scheduler, config, step, metrics)

    final_metrics = _retrieval_metrics(model, val_loader, device, config)
    if not (output_dir / "best_retrieval.pt").exists():
        _save_checkpoint(output_dir / "best_retrieval.pt", model, optimizer, scheduler, config, step, final_metrics)
    _save_checkpoint(output_dir / "last_retrieval.pt", model, optimizer, scheduler, config, step, final_metrics)
    roundtrip = _checkpoint_roundtrip(model, optimizer, scheduler, output_dir / "last_retrieval.pt", val_loader, device, config)
    frozen_grad_violations, missing_gradients = _audit_failures(gradient_audit)
    gradient_audit_passed = not frozen_grad_violations and not missing_gradients
    checkpoint_roundtrip_passed = bool(roundtrip["ok"])
    smoke_status = "PASS" if gradient_audit_passed and checkpoint_roundtrip_passed else "FAIL"
    memory = {
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
    }
    report = {
        "status": smoke_status,
        "cluster_ready": False,
        "cluster_ready_requires": "Slurm job state COMPLETED and ExitCode 0:0 with this smoke_report.json.",
        "fake_backbones": config.fake_backbones,
        "universat_checkpoint": config.universat_checkpoint,
        "jina_model": config.jina_model,
        "samples": {"train": len(train_dataset), "validation": len(val_dataset)},
        "train_val_disjoint": True,
        "steps_completed": step,
        "finite_loss": True,
        "gradient_audit": gradient_audit,
        "gradient_audit_passed": gradient_audit_passed,
        "frozen_grad_violations": frozen_grad_violations,
        "missing_gradients": missing_gradients,
        "checkpoint_roundtrip": roundtrip,
        "checkpoint_roundtrip_passed": checkpoint_roundtrip_passed,
        "parameter_counts": _parameter_counts(model),
        "shape_report": shape_report,
        "final_validation": final_metrics,
        "memory": memory,
        "not_ready": [
            "No claim of H100 readiness until this exact script completes under Slurm with ExitCode 0:0.",
            "Captioning, grounding, semantic segmentation and pair-to-pair retrieval are intentionally not connected in this Stage-1 smoke.",
        ],
    }
    _write_json(output_dir / "smoke_report.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("Smoke audit failed; see smoke_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
