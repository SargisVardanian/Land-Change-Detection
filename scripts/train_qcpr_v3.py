#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ucv2_cluster_common as common
import ucv2_stage1_next_core as legacy_data
from land_change_detection.models.qcpr_v3_losses import query_mask_loss, query_mask_metrics
from land_change_detection.models.qcpr_v3_phases import apply_phase_to_model, resolve_training_phase
from land_change_detection.models.qcpr_v3_runtime import IMMUTABLE_V1, assert_teacher_not_in_optimizer, build_v3_and_teacher
from land_change_detection.models.qcpr_v3_teacher import teacher_preservation_losses


def _seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def _retrieval_loss(scores: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(scores, mapping)


def _positive_margin(scores: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    positive = scores.gather(1, mapping[:, None]).squeeze(1)
    negative = scores.masked_fill(F.one_hot(mapping, scores.shape[1]).bool(), float("-inf")).max(dim=1).values
    return (positive - negative).mean()


def _save_panel(path: Path, images: torch.Tensor, mask_logits: torch.Tensor, target: torch.Tensor) -> None:
    tiles = []
    for tensor in (images[0, 0], images[0, 1]):
        array = tensor.detach().float().cpu().permute(1, 2, 0).numpy()
        array = (array - array.min()) / max(float(array.max() - array.min()), 1e-6)
        tiles.append((array * 255).astype(np.uint8))
    for tensor in (target, mask_logits.sigmoid()):
        array = tensor.detach().float().cpu().numpy()
        array = np.stack([array, array, array], axis=-1)
        tiles.append((array.clip(0, 1) * 255).astype(np.uint8))
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("QCPR v3 real run requires CUDA")
    device = torch.device("cuda")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    _seed(args.seed)
    payload = torch.load(args.v1_checkpoint, map_location="cpu", weights_only=False)
    config_dict = dict(payload["config"])
    config_dict.update(batch_size=args.batch_size, num_workers=args.num_workers, max_steps=args.steps, seed=args.seed)
    if args.phase == "mask_grounding":
        derived = Path(args.derived_manifest_dir)
        config_dict.update(
            dataset_config=None,
            train_manifests=(str(derived / "balanced_train_manifest.jsonl"),),
            val_manifests=(str(derived / "natural_validation_manifest.jsonl"),),
            dataset_sampling_weights=(),
        )
    config = legacy_data.Stage1NextConfig(**{key: value for key, value in config_dict.items() if key in legacy_data.Stage1NextConfig.__dataclass_fields__})
    train, val = legacy_data._build_stage1_datasets(config)
    legacy_data._assert_stage1_disjoint(train, val)
    frequencies = legacy_data.caption_frequencies(train)
    loader = legacy_data.make_train_loader(train, config, frequencies, epoch=0)

    student, teacher, initialization = build_v3_and_teacher(args.v1_checkpoint, device=device, build_legacy_model=common.build_model)
    profile = resolve_training_phase(args.phase)
    optimizer_audit = apply_phase_to_model(student, profile)
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    history = []
    student.train()
    iterator = iter(loader)
    observed_datasets: set[str] = set()
    for step in range(1, args.steps + 1):
        attempts = 0
        while True:
            try: batch = next(iterator)
            except StopIteration:
                iterator = iter(loader); batch = next(iterator)
            attempts += 1
            names = {str(name) for name in batch.get("dataset_names", [])}
            if step > 1 or not args.required_dataset or args.required_dataset in names:
                break
            if attempts > len(loader):
                raise RuntimeError(f"required smoke dataset {args.required_dataset!r} was not found")
        observed_datasets.update(names)
        images = batch["images"].to(device, non_blocking=True)
        mapping = batch["caption_to_pair"].to(device)
        temporal_mask = batch["temporal_valid_mask"].to(device)
        selection = legacy_data.retrieval_supervision_selection(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            result = student(images, batch["captions"], mapping, temporal_mask)
            losses: dict[str, torch.Tensor] = {}
            selected_scores = lambda scores: scores[selection["selected_queries"]][:, selection["selected_pairs"]]
            if "global_contrastive" in profile.active_losses:
                losses["global_contrastive"] = _retrieval_loss(selected_scores(result.scores.global_score), selection["selected_mapping"])
            if "local_contrastive" in profile.active_losses:
                if selection["selected_queries"].numel():
                    losses["local_contrastive"] = _retrieval_loss(selected_scores(result.scores.reranked_score), selection["selected_mapping"])
            if "teacher_distillation" in profile.active_losses:
                target = teacher(images, batch["captions"], mapping, temporal_mask)
                losses.update(teacher_preservation_losses(result.pair_embedding, result.text_embedding, result.scores.global_score, target))
            if "mask_dice" in profile.active_losses:
                supervised_pairs = batch["segmentation_supervision"].to(device).bool()
                supervised_queries = supervised_pairs[mapping]
                query_indices = torch.nonzero(supervised_queries, as_tuple=False).flatten()
                if query_indices.numel():
                    selected = result.scores.decoded_mask_logits[query_indices, mapping[query_indices]]
                    targets = F.interpolate(batch["masks"].to(device).float().unsqueeze(1), selected.shape[-2:], mode="nearest").squeeze(1)[mapping[query_indices]]
                    mask = query_mask_loss(selected, targets)
                    losses["query_mask"] = mask.total
            if not losses:
                raise RuntimeError("batch has no supervision for the selected phase")
            total = sum(losses.values())
        if not torch.isfinite(total): raise FloatingPointError(f"non-finite loss at step {step}")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm)
        if not torch.isfinite(norm): raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        if step == 1:
            first_query = 0
            first_pair = int(mapping[first_query])
            target_panel = F.interpolate(batch["masks"][first_pair:first_pair + 1].float().unsqueeze(1), result.scores.decoded_mask_logits.shape[-2:], mode="nearest")[0, 0]
            _save_panel(output / "soft_segmentation_panel.png", images, result.scores.decoded_mask_logits[first_query, first_pair], target_panel)
            _save_panel(output / "retrieval_panel.png", images, result.scores.decoded_mask_logits[first_query, first_pair], target_panel)
        margin = _positive_margin(selected_scores(result.scores.local_score.detach()), selection["selected_mapping"]) if selection["selected_queries"].numel() else torch.tensor(float("nan"))
        row = {"step": step, "loss": float(total.detach()), "grad_norm": float(norm), "local_margin": float(margin)}
        row.update({name: float(value.detach()) for name, value in losses.items()})
        history.append(row)

    checkpoint = output / "last.pt"
    torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict(), "phase": profile.name, "step": args.steps, "initialization": asdict(initialization)}, checkpoint)
    restored = {key: value.cpu() for key, value in student.state_dict().items()}
    round_trip = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    if restored.keys() != round_trip.keys() or any(not torch.equal(restored[key], round_trip[key]) for key in restored):
        raise RuntimeError("checkpoint round-trip mismatch")
    report = {
        "status": "PASS", "phase": profile.to_dict(), "initialization": asdict(initialization),
        "optimizer_audit": optimizer_audit, "history": history,
        "observed_datasets": sorted(observed_datasets),
        "peak_gpu_allocated": torch.cuda.max_memory_allocated(), "peak_gpu_reserved": torch.cuda.max_memory_reserved(),
        "git_sha": os.popen("git rev-parse HEAD").read().strip(), "checkpoint": str(checkpoint),
    }
    (output / "resolved_config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("global_recovery", "late_interaction", "mask_grounding", "region_slots", "joint_finetune"), required=True)
    parser.add_argument("--v1-checkpoint", type=Path, default=IMMUTABLE_V1)
    parser.add_argument("--derived-manifest-dir", type=Path, default=Path("/mnt/weka/svardanyan/rs_change_project/manifests/qcpr_v3"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--required-dataset", default="")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
