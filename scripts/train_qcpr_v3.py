#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import qcpr_v3_data_compat as data_compat
from land_change_detection.models.qcpr_v3_losses import (
    foreground_preserving_resize,
    m0_aligned_symmetric_query_swap_loss,
    m0_balanced_mask_loss,
    m0_symmetric_query_swap_loss,
    query_mask_metrics,
    separated_query_mask_losses,
)
from land_change_detection.models.qcpr_v3_losses import (
    a0_physical_pair_contrastive_masks,
    a0_symmetric_physical_pair_contrastive_loss,
    multi_positive_contrastive_loss,
)
from land_change_detection.models.qcpr_v3_phases import apply_phase_to_model, resolve_training_phase
from land_change_detection.models.qcpr_v3_factory import QCPRV3BackboneConfig
from land_change_detection.models.qcpr_v3_runtime import IMMUTABLE_V1, assert_teacher_not_in_optimizer, build_clean_v3, build_v3_and_teacher
from land_change_detection.models.qcpr_v3_teacher import teacher_preservation_losses
from land_change_detection.models.qcpr_v3_adapters import CanonicalV3Inputs, evaluator_score, renderer_score, trainer_score
from land_change_detection.models.qcpr_v3_experiment import parser_derived_late_interaction_loss
from land_change_detection.models.qcpr_v3_data import CappedCompositionalBatchSampler, DirectionalCurriculumBatchSampler, DirectionalM0BatchSampler, DirectionalSanitySampler
from land_change_detection.models.qcpr_v3 import stable_global_top_n
from land_change_detection.models.qcpr_v31_encoder_ablation import load_global_retrieval_modules_strict
from land_change_detection.models.qcpr_v3_mask_diagnostic import (
    append_jsonl,
    fixed_probe_contract,
    fixed_probe_subset,
    query_swap_metrics,
    resolve_mask_objective,
    swapped_direction_indices,
)
from land_change_detection.models.retrieval_heads import semantic_teacher_relevance_matrix, stable_caption_group_ids
from land_change_detection.training.runtime_device import assert_runtime_tensor_devices, resolve_runtime_device
from qcpr_v3_progress import write_progress


def _seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _capture_rng_state() -> dict[str, object]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state

def _restore_rng_state(state: dict[str, object] | None) -> None:
    if not state:
        return
    if "python" in state: random.setstate(state["python"])
    if "numpy" in state: np.random.set_state(state["numpy"])
    if "torch_cpu" in state: torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])

def _save_training_checkpoint(path, student, optimizer, step, metadata):
    torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "rng_state": _capture_rng_state(), **metadata}, path)

def _retrieval_loss(scores: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(scores, mapping)


def _positive_margin(scores: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    positive = scores.gather(1, mapping[:, None]).squeeze(1)
    negative = scores.masked_fill(F.one_hot(mapping, scores.shape[1]).bool(), float("-inf")).max(dim=1).values
    return (positive - negative).mean()


def _save_panel(
    path: Path, images: torch.Tensor, mask_logits: torch.Tensor, target: torch.Tensor,
    *, query: str, pair_id: str,
) -> None:
    """Write a self-explanatory grounding panel, never an unlabeled tile strip."""
    import matplotlib.pyplot as plt

    def rgb(tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().float().cpu().permute(1, 2, 0).numpy()
        return np.clip(array, 0.0, 1.0)

    t1, t2 = rgb(images[0, 0]), rgb(images[0, 1])
    probability = mask_logits.detach().float().sigmoid().cpu().numpy()
    truth = target.detach().float().cpu().numpy() >= 0.5
    predicted = probability >= 0.5
    tp, fp, fn = predicted & truth, predicted & ~truth, ~predicted & truth
    error = np.zeros((*truth.shape, 3), dtype=np.float32)
    error[tp, 1] = 1.0
    error[fp, 0] = 1.0
    error[fn, 2] = 1.0
    metrics = query_mask_metrics(mask_logits[None].float(), target[None].float())

    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], t1, "T1 — before", None),
        (axes[0, 1], t2, "T2 — after", None),
        (axes[0, 2], truth, "Directional query GT", "gray"),
        (axes[0, 3], probability, "Predicted soft probability", "magma"),
        (axes[1, 0], t2, "T2 + soft-mask overlay", None),
        (axes[1, 1], predicted, "Thresholded prediction @ 0.5", "gray"),
        (axes[1, 2], error, "Errors: TP green / FP red / FN blue", None),
    )
    for axis, value, title, cmap in panels:
        axis.imshow(value, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    axes[1, 0].imshow(probability, cmap="magma", vmin=0, vmax=1, alpha=0.55)
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.0, 1.0,
        "Unsmoothed validation metrics\n"
        f"Dice: {metrics['nonempty_dice']:.4f}\n"
        f"IoU: {metrics['nonempty_iou']:.4f}\n"
        f"Precision: {metrics['nonempty_precision']:.4f}\n"
        f"Recall: {metrics['nonempty_recall']:.4f}\n"
        f"Pixel AP: {metrics['pixel_average_precision']:.4f}\n"
        f"Soft Dice: {metrics['nonempty_soft_dice']:.4f}\n"
        f"Soft IoU: {metrics['nonempty_soft_iou']:.4f}\n"
        f"Localization margin: {metrics['localization_margin']:+.4f}\n"
        f"Empty mean p: {metrics['empty_mean_probability']:.4f}\n"
        f"Target area: {metrics['target_area']:.4f}\n"
        f"Predicted area: {metrics['predicted_area']:.4f}",
        va="top", ha="left", family="monospace",
    )
    figure.suptitle(f"Query: {query}\nPair: {pair_id}", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _assert_cuda_selection(selection: dict[str, torch.Tensor], device: torch.device) -> None:
    """Smoke-visible enforcement of the collator-to-CUDA indexing boundary."""
    resolved_device = resolve_runtime_device(device)
    if resolved_device.type != "cuda":
        raise RuntimeError("real QCPR v3 run must use CUDA")
    assert_runtime_tensor_devices(selection, resolved_device)


def _caption_by_pair(batch: dict, mapping: torch.Tensor) -> list[str]:
    captions = [""] * len(batch["pair_ids"])
    for query_index, pair_index in enumerate(mapping.detach().cpu().tolist()):
        if not captions[pair_index]:
            captions[pair_index] = str(batch["captions"][query_index])
    if any(not caption for caption in captions):
        raise RuntimeError("every fixed-probe pair must have one caption")
    return captions


def _load_v31_parent(student, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Strictly migrate a v3 parent while deliberately reinitializing only the v3.1 FPN."""
    if student.grounding_backbone is not None:
        global_audit = load_global_retrieval_modules_strict(student, state_dict)
        return {
            "missing_reinitialized": sorted(
                name for name in student.state_dict() if name.startswith("grounder.")
            ),
            "obsolete_parent_keys": sorted(
                name for name in state_dict if name.startswith("grounder.")
            ),
            "global_path_load": global_audit,
            "grounder_migration": "REINITIALIZED_FOR_SIGLIP2_DENSE_CONTRACT",
        }
    result = student.load_state_dict(state_dict, strict=False)
    missing = sorted(result.missing_keys)
    unexpected = sorted(result.unexpected_keys)
    reinitialized_prefixes = (
        "grounder.mask_decoder.",
        "grounder.grounding_decoder.query_modulation.",
        "grounder.temporal_field.source_projection.",
    )
    invalid_missing = [name for name in missing if not name.startswith(reinitialized_prefixes)]
    invalid_unexpected = [name for name in unexpected if not name.startswith("grounder.mask_decoder.")]
    if invalid_missing or invalid_unexpected:
        raise RuntimeError(
            f"v3.1 parent migration mismatch: missing={invalid_missing}, unexpected={invalid_unexpected}"
        )
    return {"missing_reinitialized": missing, "obsolete_parent_keys": unexpected}


def _swapped_query_forward(student, batch: dict, device: torch.device):
    mapping = batch["caption_to_pair"].to(device)
    query_directions = list(batch.get("query_change_types", []))
    if query_directions and len(query_directions) == mapping.numel():
        lookup = {
            (int(pair_index), str(direction)): query_index
            for query_index, (pair_index, direction) in enumerate(
                zip(mapping.detach().cpu().tolist(), query_directions)
            )
            if direction in {"appeared", "disappeared"}
        }
        source_indices: list[int] = []
        opposite_query_indices: list[int] = []
        for query_index, (pair_index, direction) in enumerate(
            zip(mapping.detach().cpu().tolist(), query_directions)
        ):
            opposite = "disappeared" if direction == "appeared" else "appeared"
            if direction in {"appeared", "disappeared"} and (pair_index, opposite) in lookup:
                source_indices.append(query_index)
                opposite_query_indices.append(lookup[(pair_index, opposite)])
        if source_indices:
            source_pairs = mapping.index_select(
                0, torch.tensor(source_indices, device=device, dtype=torch.long)
            )
            wrong_captions = [batch["captions"][index] for index in opposite_query_indices]
            wrong_mapping = torch.arange(len(source_indices), device=device)
            wrong = student(
                batch["images"].to(device).index_select(0, source_pairs), wrong_captions, wrong_mapping,
                batch["temporal_valid_mask"].to(device).index_select(0, source_pairs),
            )
            logits = wrong.scores.decoded_mask_logits[wrong_mapping, wrong_mapping]
            return (wrong, logits), source_indices
    captions_by_pair = _caption_by_pair(batch, mapping)
    opposites = swapped_direction_indices(batch["pair_ids"], batch["change_types"])
    source_indices = [index for index, opposite in enumerate(opposites) if opposite >= 0]
    opposite_indices = [opposites[index] for index in source_indices]
    if not source_indices:
        return None, source_indices
    source = torch.tensor(source_indices, device=device, dtype=torch.long)
    wrong_captions = [captions_by_pair[index] for index in opposite_indices]
    wrong_mapping = torch.arange(len(source_indices), device=device)
    wrong = student(
        batch["images"].to(device).index_select(0, source), wrong_captions, wrong_mapping,
        batch["temporal_valid_mask"].to(device).index_select(0, source),
    )
    logits = wrong.scores.decoded_mask_logits[
        torch.arange(len(source_indices), device=device), wrong_mapping
    ]
    return (wrong, logits), source_indices


@torch.no_grad()
def _evaluate_fixed_mask_probe(
    student, batch: dict, device: torch.device, *, panel_path: Path | None = None,
) -> dict[str, float | int]:
    was_training = student.training
    student.eval()
    images = batch["images"].to(device)
    mapping = batch["caption_to_pair"].to(device)
    temporal_mask = batch["temporal_valid_mask"].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        result = student(images, batch["captions"], mapping, temporal_mask)
        query_indices = torch.arange(mapping.numel(), device=device)
        correct = result.scores.decoded_mask_logits[query_indices, mapping]
        if "query_masks" in batch:
            targets = foreground_preserving_resize(
                batch["query_masks"].to(device).float(), correct.shape[-2:],
            )
        else:
            targets = foreground_preserving_resize(
                batch["masks"].to(device).float(), correct.shape[-2:],
            ).index_select(0, mapping)
        swapped_payload, source_indices = _swapped_query_forward(
            student, batch, device,
        )
    metrics = query_mask_metrics(correct.float(), targets.float())
    if panel_path is not None:
        pair_targets = targets.detach().flatten(1).sum(dim=1)
        panel_query = int(torch.nonzero(pair_targets > 0, as_tuple=False)[0])
        panel_pair = int(mapping[panel_query])
        _save_panel(
            panel_path, images[panel_pair:panel_pair + 1], correct[panel_query], targets[panel_query],
            query=str(batch["captions"][panel_query]), pair_id=str(batch["pair_ids"][panel_pair]),
        )
    validity = result.scores.mask_validity[query_indices, mapping]
    metrics["selected_near_empty_rate"] = float((validity <= 0).float().mean())
    metrics["selected_sample_count"] = int(correct.shape[0])
    metrics["true_positive_sample_fraction"] = float(metrics["samples_with_true_positive"]) / max(int(metrics["nonempty_count"]), 1)
    if swapped_payload is None:
        metrics.update(
            correct_query_iou=0.0, swapped_query_iou=0.0, query_swap_gap=0.0,
            correct_query_soft_iou=0.0, swapped_query_soft_iou=0.0,
            soft_query_swap_iou_gap=0.0, negative_pair_active_rate=0.0,
            query_swap_count=0,
        )
    else:
        swapped_result, swapped_logits = swapped_payload
        source = torch.tensor(source_indices, device=device, dtype=torch.long)
        metrics.update(query_swap_metrics(correct.index_select(0, source).float(), swapped_logits.float(), targets.index_select(0, source).float()))
        query_directions = list(batch.get("query_change_types", []))
        for direction in ("appeared", "disappeared"):
            local = [
                index for index, query_index in enumerate(source_indices)
                if query_directions and query_directions[query_index] == direction
            ]
            if local:
                local_tensor = torch.tensor(local, device=device, dtype=torch.long)
                directional = query_swap_metrics(
                    correct.index_select(0, source).index_select(0, local_tensor).float(),
                    swapped_logits.index_select(0, local_tensor).float(),
                    targets.index_select(0, source).index_select(0, local_tensor).float(),
                )
                metrics.update({f"{direction}_{name}": value for name, value in directional.items()})
        wrong_validity = swapped_result.scores.mask_validity[
            torch.arange(len(source_indices), device=device), torch.arange(len(source_indices), device=device)
        ]
        metrics["negative_pair_active_rate"] = float((wrong_validity > 0).float().mean())
        metrics["query_swap_count"] = len(source_indices)
    if was_training:
        student.train()
    return metrics


def _micro_overfit_gate(
    history: list[dict[str, float | int]], *, gradients_finite: bool,
) -> dict[str, object]:
    first, last = history[0], history[-1]
    checks = {
        "train_soft_dice_increased": last["nonempty_soft_dice"] > first["nonempty_soft_dice"],
        "train_soft_iou_increased": last["nonempty_soft_iou"] > first["nonempty_soft_iou"],
        "train_localization_margin_increased": last["localization_margin"] > first["localization_margin"],
        "train_empty_mean_probability_not_increased": last["empty_mean_probability"] <= first["empty_mean_probability"],
        "train_soft_query_swap_gap_positive": last["soft_query_swap_iou_gap"] > 0.0,
        "train_appeared_soft_query_swap_gap_positive": last["appeared_soft_query_swap_iou_gap"] > 0.0,
        "train_disappeared_soft_query_swap_gap_positive": last["disappeared_soft_query_swap_iou_gap"] > 0.0,
        "train_mask_not_empty_or_full": 0.0 < last["predicted_area"] < 1.0,
        "gradients_finite": gradients_finite,
    }
    return {"passed": all(checks.values()), "checks": checks, "step0": first, "final": last}


def _validation_direction(history: list[dict[str, float | int]]) -> dict[str, object]:
    first, last = history[0], history[-1]
    checks = {
        "validation_soft_dice_increased": last["nonempty_soft_dice"] > first["nonempty_soft_dice"],
        "validation_soft_iou_increased": last["nonempty_soft_iou"] > first["nonempty_soft_iou"],
        "validation_localization_margin_increased": last["localization_margin"] > first["localization_margin"],
        "validation_empty_mean_probability_not_increased": last["empty_mean_probability"] <= first["empty_mean_probability"],
        "validation_query_swap_count_positive": last["query_swap_count"] > 0,
        "validation_soft_query_swap_gap_positive": last["soft_query_swap_iou_gap"] > 0.0,
        "validation_appeared_soft_query_swap_gap_positive": last["appeared_soft_query_swap_iou_gap"] > 0.0,
        "validation_disappeared_soft_query_swap_gap_positive": last["disappeared_soft_query_swap_iou_gap"] > 0.0,
        "validation_mask_not_empty_or_full": 0.0 < last["predicted_area"] < 1.0,
    }
    return {"positive": all(checks.values()), "checks": checks, "step0": first, "final": last}


def _direction_query_separation_loss(
    all_logits: torch.Tensor,
    mapping: torch.Tensor,
    query_indices: torch.Tensor,
    changes: list[str | None],
    targets: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, int]:
    """Rank correct text above opposite text by query-target soft IoU.

    The scientific query-swap gate is defined by soft IoU.  Optimizing a
    different foreground-minus-background statistic left a gap where training
    could improve localization while making the opposite query overlap a target
    more strongly.  This pairwise logistic objective uses the same spatial
    quantity as the gate without introducing a direction-specific model head.
    """
    if all_logits.ndim != 4 or targets.ndim != 3:
        raise ValueError("direction separation expects [Q,C,H,W] logits and [S,H,W] targets")
    if query_indices.numel() != len(changes) or targets.shape[0] != len(changes):
        raise ValueError("selected directional query metadata must align")
    if temperature <= 0:
        raise ValueError("direction separation temperature must be positive")
    selected_query_ids = [int(value) for value in query_indices.detach().cpu().tolist()]
    lookup = {
        (int(mapping[query_id]), str(change)): local_index
        for local_index, (query_id, change) in enumerate(zip(selected_query_ids, changes, strict=True))
        if change in {"appeared", "disappeared"}
    }
    terms: list[torch.Tensor] = []
    for local_index, (query_id, change) in enumerate(zip(selected_query_ids, changes, strict=True)):
        if change not in {"appeared", "disappeared"}:
            continue
        opposite = "disappeared" if change == "appeared" else "appeared"
        opposite_local = lookup.get((int(mapping[query_id]), opposite))
        if opposite_local is None:
            continue
        target = targets[local_index] >= 0.5
        if not bool(target.any()) or not bool((~target).any()):
            continue
        candidate_id = int(mapping[query_id])
        opposite_query_id = selected_query_ids[opposite_local]
        target_float = target.to(dtype=all_logits.dtype)
        correct_probability = all_logits[query_id, candidate_id].sigmoid()
        opposite_probability = all_logits[opposite_query_id, candidate_id].sigmoid()

        def soft_iou(probability: torch.Tensor) -> torch.Tensor:
            intersection = (probability * target_float).sum()
            union = probability.sum() + target_float.sum() - intersection
            return (intersection + 1e-6) / (union + 1e-6)

        correct_overlap = soft_iou(correct_probability)
        opposite_overlap = soft_iou(opposite_probability)
        # Pairwise logistic ranking remains differentiable at equal scores and
        # directly rewards a positive soft query-swap IoU gap.
        terms.append(F.softplus((opposite_overlap - correct_overlap) / float(temperature)))
    if not terms:
        return all_logits.sum() * 0.0, 0
    return torch.stack(terms).mean(), len(terms)



@torch.no_grad()
def _evaluate_m0_development(student, loader: DataLoader, device: torch.device) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    """Evaluate an immutable M0 development split without target-centred crops."""
    was_training = student.training
    student.eval()
    records: list[dict[str, float | int | str]] = []
    for batch in loader:
        mapping_cpu = batch["caption_to_pair"].long()
        mapping = mapping_cpu.to(device)
        directions = [str(value) for value in batch.get("query_change_types", [])]
        images = batch["images"].to(device, non_blocking=True)
        temporal_mask = batch["temporal_valid_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result = student.forward_aligned_masks(
                images, batch["captions"], mapping, temporal_mask
            )
        correct = result.masks.decoded_mask_logits.float()
        targets = foreground_preserving_resize(
            batch["query_masks"].to(device).float(), correct.shape[-2:]
        )
        lookup = {
            (int(pair), direction): query
            for query, (pair, direction) in enumerate(
                zip(mapping_cpu.tolist(), directions, strict=True)
            )
            if direction in {"appeared", "disappeared"}
        }
        for query, direction in enumerate(directions):
            if direction not in {"appeared", "disappeared"}:
                continue
            pair = int(mapping_cpu[query])
            opposite_query = lookup.get(
                (pair, "disappeared" if direction == "appeared" else "appeared")
            )
            if opposite_query is None:
                continue
            target = targets[query:query + 1]
            metric = query_mask_metrics(correct[query:query + 1], target)
            swapped_metric = query_mask_metrics(
                correct[opposite_query:opposite_query + 1], target
            )
            probability = correct[query].sigmoid()
            edge = torch.cat((
                probability[0].flatten(), probability[-1].flatten(),
                probability[:, 0].flatten(), probability[:, -1].flatten(),
            )).mean()
            center = probability[
                probability.shape[-2] // 4:3 * probability.shape[-2] // 4,
                probability.shape[-1] // 4:3 * probability.shape[-1] // 4,
            ].mean().clamp_min(1e-6)
            target_area = float(metric["target_area"])
            records.append({
                "pair_id": str(batch["pair_ids"][pair]),
                "direction": direction,
                "soft_dice": float(metric["nonempty_soft_dice"]),
                "soft_iou": float(metric["nonempty_soft_iou"]),
                "localization_margin": float(metric["localization_margin"]),
                "predicted_area": float(metric["predicted_area"]),
                "target_area": target_area,
                "predicted_target_area_ratio": (
                    float(metric["predicted_area"]) / max(target_area, 1e-8)
                ),
                "soft_query_swap_iou_gap": (
                    float(metric["nonempty_soft_iou"])
                    - float(swapped_metric["nonempty_soft_iou"])
                ),
                "edge_center_ratio": float(edge / center),
                "all_empty_or_foreground": int(
                    float(metric["predicted_area"]) <= 0.0
                    or float(metric["predicted_area"]) >= 1.0
                ),
            })
    if was_training:
        student.train()
    if not records:
        raise RuntimeError("M0 development evaluation produced no paired directional records")
    by_direction = {direction: [record for record in records if record["direction"] == direction] for direction in ("appeared", "disappeared")}
    pair_groups: dict[str, list[dict[str, float | int | str]]] = {}
    for record in records:
        pair_groups.setdefault(str(record["pair_id"]), []).append(record)
    pair_positive = [all(float(item["soft_query_swap_iou_gap"]) > 0.0 for item in items) for items in pair_groups.values() if len(items) == 2]
    def median(name: str, rows: list[dict[str, float | int | str]] = records) -> float:
        return float(np.median([float(row[name]) for row in rows]))
    return {
        "query_record_count": len(records), "pair_count": len(pair_groups),
        "soft_dice_median": median("soft_dice"), "soft_iou_median": median("soft_iou"),
        "localization_margin_median": median("localization_margin"),
        "predicted_target_area_ratio_median": median("predicted_target_area_ratio"),
        "all_empty_or_foreground_fraction": float(np.mean([int(row["all_empty_or_foreground"]) for row in records])),
        "edge_center_ratio_median": median("edge_center_ratio"),
        "appeared_soft_query_swap_iou_gap_median": median("soft_query_swap_iou_gap", by_direction["appeared"]),
        "disappeared_soft_query_swap_iou_gap_median": median("soft_query_swap_iou_gap", by_direction["disappeared"]),
        "positive_pair_swap_gap_fraction": float(np.mean(pair_positive)) if pair_positive else 0.0,
    }, records


def _m0_gate(history: list[dict[str, float | int]], *, gradients_finite: bool) -> dict[str, object]:
    if len(history) < 2:
        raise ValueError("M0 gate requires step-zero and final development evaluation")
    first, final = history[0], history[-1]
    improved = float(final["soft_dice_median"]) >= float(first["soft_dice_median"]) + 0.01 or float(final["soft_dice_median"]) >= float(first["soft_dice_median"]) * 1.20
    checks = {
        "development_soft_dice_improved": improved,
        "median_localization_margin_positive": float(final["localization_margin_median"]) > 0.0,
        "appeared_median_swap_gap_nonnegative": float(final["appeared_soft_query_swap_iou_gap_median"]) >= 0.0,
        "disappeared_median_swap_gap_nonnegative": float(final["disappeared_soft_query_swap_iou_gap_median"]) >= 0.0,
        "positive_pair_swap_gaps_at_least_60pct": float(final["positive_pair_swap_gap_fraction"]) >= 0.60,
        "median_area_ratio_in_range": 0.25 <= float(final["predicted_target_area_ratio_median"]) <= 4.0,
        "collapse_fraction_at_most_10pct": float(final["all_empty_or_foreground_fraction"]) <= 0.10,
        "no_systematic_edge_artifact": 0.5 <= float(final["edge_center_ratio_median"]) <= 2.0,
        "finite_gradients": gradients_finite,
    }
    return {"passed": all(checks.values()), "checks": checks, "step0": first, "final": final}


def _m0_swap_weight(step: int, total_steps: int) -> float:
    warmup = max(1, math.ceil(total_steps * 0.20))
    return 0.0 if step <= warmup else 0.1 * min(1.0, (step - warmup) / max(total_steps - warmup, 1))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_parameter_fingerprint(model: torch.nn.Module, prefixes: tuple[str, ...]) -> str:
    """SHA256 of exact frozen backbone parameters for A0 drift detection."""
    digest = hashlib.sha256()
    selected = [
        (name, parameter) for name, parameter in model.named_parameters()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    ]
    if not selected:
        raise RuntimeError(f"no parameters found for frozen fingerprint prefixes {prefixes!r}")
    for name, parameter in selected:
        digest.update(name.encode())
        tensor = parameter.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("QCPR v3 real run requires CUDA")
    requested_device = torch.device("cuda")
    device = resolve_runtime_device(requested_device)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    _seed(args.seed)
    m0_mode = args.phase == "mask_only_diagnostic" and args.mask_diagnostic_mode == "multi_pair"
    derived = Path(args.derived_manifest_dir)
    backbone_config = QCPRV3BackboneConfig(
        universat_source=str(args.universat_source),
        universat_checkpoint=str(args.universat_checkpoint),
        jina_model=str(args.jina_model),
        temporal_depth=args.temporal_depth,
        text_max_length=args.text_max_length,
        grounding_backbone_kind=args.grounding_backbone,
        siglip2_model=str(args.siglip2_model) if args.grounding_backbone == "siglip2" else None,
    )
    if args.initialization_mode == "historical_e0":
        payload = torch.load(args.v1_checkpoint, map_location="cpu", weights_only=False)
        config_dict = dict(payload["config"])
    else:
        config_dict = {
            "data_root": str(args.data_root),
            "output_dir": str(output),
            **asdict(backbone_config),
        }
    retrieval_phase = args.phase in {"global_bootstrap", "late_interaction"}
    train_manifest = "m0_train_manifest.jsonl" if m0_mode else ("natural_train_retrieval_manifest.jsonl" if retrieval_phase else "natural_train_manifest.jsonl")
    val_manifest = "m0_dev_manifest.jsonl" if m0_mode else ("natural_validation_retrieval_manifest.jsonl" if retrieval_phase else "natural_validation_manifest.jsonl")
    config_dict.update(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_steps=args.steps,
        seed=args.seed,
        dataset_config=None,
        train_manifests=(str(derived / train_manifest),),
        val_manifests=(str(derived / val_manifest),),
        dataset_sampling_weights=(),
        target_aware_mask_crop=args.phase in {"mask_only_diagnostic", "mask_grounding"},
        validation_target_aware_mask_crop=False if m0_mode else args.phase in {"mask_only_diagnostic", "mask_grounding"},
        target_crop_context=args.target_crop_context,
        direction_only_captions=args.direction_only_probe_captions,
    )
    config = data_compat.Stage1NextConfig(**{key: value for key, value in config_dict.items() if key in data_compat.Stage1NextConfig.__dataclass_fields__})
    train, val = data_compat.build_datasets(config)
    data_compat.assert_disjoint(train, val)
    # Metadata is needed before choosing samplers; model/optimizer restoration
    # remains below, after model construction.
    resume_payload_hint = (
        torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if args.resume_checkpoint is not None
        else None
    )
    resume_step_hint = int(resume_payload_hint.get("step", 0)) if resume_payload_hint else 0
    fixed_train_batch = None
    fixed_validation_batch = None
    probe_contract = None
    if args.phase == "mask_only_diagnostic" and not m0_mode:
        train = fixed_probe_subset(train, count=args.micro_train_samples, pair_ids=tuple(args.train_probe_pair_id))
        val = fixed_probe_subset(val, count=args.validation_probe_samples, pair_ids=tuple(args.validation_probe_pair_id))
        probe_contract = fixed_probe_contract(
            train_subset=train, validation_subset=val,
            train_manifest=derived / train_manifest, validation_manifest=derived / val_manifest,
        )
        (output / "fixed_probe_contract.json").write_text(json.dumps(probe_contract, indent=2, sort_keys=True))
    frequencies = data_compat.caption_frequencies(train)
    if args.phase in {"global_bootstrap", "late_interaction"}:
        # One pair per batch entry: caption paraphrases are selected by the
        # collator, never emitted as five independent no-change examples.
        sampler = CappedCompositionalBatchSampler(
            list(getattr(train, "samples", [])), config.batch_size, seed=config.seed,
            no_change_fraction_cap=args.no_change_batch_fraction_cap,
        )
        loader = DataLoader(
            train, batch_sampler=sampler, num_workers=config.num_workers,
            collate_fn=data_compat.make_collator(train, config, epoch=0, training=True, frequencies=frequencies),
            pin_memory=True, persistent_workers=False,
        )
    elif m0_mode:
        m0_sampler = DirectionalM0BatchSampler(
            list(getattr(train, "samples", [])),
            config.batch_size,
            seed=config.seed,
            start_step=resume_step_hint,
        )
        loader = DataLoader(
            train, batch_sampler=m0_sampler, num_workers=config.num_workers,
            collate_fn=data_compat.make_collator(train, config, epoch=0, training=True, frequencies=frequencies),
            pin_memory=True, persistent_workers=False,
        )
        m0_validation_loader = DataLoader(
            val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
            collate_fn=data_compat.make_collator(val, config, epoch=0, training=False),
            pin_memory=True, persistent_workers=False,
        )
    elif resume_step_hint == 0 and args.phase == "mask_only_diagnostic" and not m0_mode:
        fixed_train_loader = DataLoader(
            train, batch_size=len(train), shuffle=False, num_workers=0,
            collate_fn=data_compat.make_collator(train, config, epoch=0, training=False),
        )
        validation_loader = DataLoader(
            val, batch_size=len(val), shuffle=False, num_workers=0,
            collate_fn=data_compat.make_collator(val, config, epoch=0, training=False),
        )
        fixed_train_batch = next(iter(fixed_train_loader))
        fixed_validation_batch = next(iter(validation_loader))
        if len(train.indices) != 1:
            raise ValueError("directional sanity curriculum requires exactly one selected base pair")
        loader = DataLoader(
            train.dataset,
            batch_sampler=DirectionalSanitySampler(
                row_index=int(train.indices[0]), steps=args.steps, seed=config.seed,
            ),
            num_workers=0,
            collate_fn=data_compat.make_collator(train.dataset, config, epoch=0, training=False),
        )
    elif args.phase == "mask_grounding" and any(row.get("quality_tier") for row in getattr(train, "samples", [])):
        sampler = DirectionalCurriculumBatchSampler(
            list(getattr(train, "samples", [])), config.batch_size, seed=config.seed,
        )
        loader = DataLoader(
            train, batch_sampler=sampler, num_workers=config.num_workers,
            collate_fn=data_compat.make_collator(train, config, epoch=0, training=True, frequencies=frequencies),
            pin_memory=True, persistent_workers=False,
        )
    else:
        loader = data_compat.make_train_loader(train, config, frequencies, epoch=0)

    migration_audit = {"missing_reinitialized": [], "obsolete_parent_keys": []}
    if args.initialization_mode == "clean_pretrained":
        student, teacher, initialization = build_clean_v3(backbone_config, device=device)
        if args.phase != "global_bootstrap" and not m0_mode:
            if args.v3_checkpoint is None:
                raise ValueError("clean_pretrained phases after global_bootstrap require --v3-checkpoint")
            prior = torch.load(args.v3_checkpoint, map_location="cpu", weights_only=False)
            expected_parent = {
                "late_interaction": "global_bootstrap",
                "mask_grounding": "late_interaction",
                "mask_only_diagnostic": "late_interaction",
            }.get(args.phase)
            if expected_parent is None or prior.get("phase") != expected_parent:
                raise ValueError(
                    f"clean v3 {args.phase} requires an accepted {expected_parent} checkpoint; "
                    f"got {prior.get('phase')!r}"
                )
            migration_audit = _load_v31_parent(student, prior["model"])
    else:
        from qcpr_v3_historical_compat import build_historical_model

        student, teacher, initialization = build_v3_and_teacher(
            args.v1_checkpoint, device=device, build_legacy_model=build_historical_model
        )
    profile = resolve_training_phase(args.phase)
    optimizer_audit = apply_phase_to_model(student, profile)
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    a0_frozen_fingerprint_before = _frozen_parameter_fingerprint(student, ("visual_encoder", "text_encoder")) if args.phase == "global_bootstrap" else None
    text_adapter_ids = {
        id(parameter) for parameter in student.text_adapter.parameters()
    } if student.text_adapter is not None else set()
    text_adapter_parameters = [parameter for parameter in parameters if id(parameter) in text_adapter_ids]
    primary_parameters = [parameter for parameter in parameters if id(parameter) not in text_adapter_ids]
    optimizer_groups = []
    if primary_parameters:
        optimizer_groups.append({"params": primary_parameters, "lr": args.learning_rate, "name": "v3_primary"})
    if text_adapter_parameters:
        optimizer_groups.append({"params": text_adapter_parameters, "lr": args.text_adapter_learning_rate, "name": "text_adapter_low_lr"})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    named_parameter_by_id = {id(parameter): name for name, parameter in student.named_parameters()}
    optimizer_audit["optimizer_parameter_names"] = sorted(
        named_parameter_by_id[id(parameter)] for group in optimizer.param_groups for parameter in group["params"]
    )
    if teacher is not None:
        assert_teacher_not_in_optimizer(teacher, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    history = []
    gradient_presence = {name: False for name, parameter in student.named_parameters() if parameter.requires_grad}
    checkpoint_metadata = {
        "phase": profile.name,
        "v31_parent_migration": migration_audit,
        "initialization": asdict(initialization),
        "backbone_config": asdict(backbone_config),
        "data_config": asdict(config),
        "grounding_config": asdict(student.grounder.config),
        "parent_checkpoint": None if args.v3_checkpoint is None else str(args.v3_checkpoint),
        "parent_phase": None if args.v3_checkpoint is None else prior.get("phase"),
        "grounding_backbone_provenance": (
            None
            if student.grounding_backbone is None
            else student.grounding_backbone.provenance()
        ),
        "parent_scientific_status": "ENGINEERING_PARENT_NOT_ACCEPTED_STAGE_B" if args.phase == "mask_only_diagnostic" else None,
        "fixed_probe_contract": probe_contract,
        "a0_frozen_fingerprint_before": a0_frozen_fingerprint_before,
        "mask_objective": None if args.phase != "mask_only_diagnostic" else asdict(resolve_mask_objective(args.mask_objective)),
    }
    resume_payload = resume_payload_hint
    start_step = resume_step_hint
    if resume_payload is not None:
        student.load_state_dict(resume_payload["model"], strict=True)
        if "optimizer" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer"])
        _restore_rng_state(resume_payload.get("rng_state"))
        previous_schedule_steps = int(
            resume_payload.get("data_config", {}).get("max_steps", args.steps)
        )
        m0_swap_schedule_total_steps = previous_schedule_steps if m0_mode else args.steps
        checkpoint_metadata.update(
            {
                "git_sha": os.popen("git rev-parse HEAD").read().strip(),
                "manifest_sha256": _sha256_file(derived / train_manifest),
                "resume_checkpoint": str(args.resume_checkpoint),
                "resume_checkpoint_sha256": _sha256_file(args.resume_checkpoint),
                "resume_checkpoint_step": start_step,
                "resume_mode": (
                    "optimizer_preserving_warm_start"
                    if "rng_state" not in resume_payload
                    else "exact_rng_resume"
                ),
                "resume_optimizer_state_restored": "optimizer" in resume_payload,
                "resume_rng_state_restored": "rng_state" in resume_payload,
                "resume_scheduler_state_available": "scheduler" in resume_payload,
                "resume_scaler_state_available": "scaler" in resume_payload,
                "m0_swap_schedule_total_steps": m0_swap_schedule_total_steps,
                "m0_sampler_start_step": start_step,
            }
        )
        (output / "resume_audit.json").write_text(
            json.dumps(
                {
                    "checkpoint_step": start_step,
                    "start_step": start_step + 1,
                    "model_state_loaded": True,
                    "optimizer_state_loaded": "optimizer" in resume_payload,
                    "scheduler_state_available": "scheduler" in resume_payload,
                    "rng_state_available": "rng_state" in resume_payload,
                    "scaler_state_available": "scaler" in resume_payload,
                    "effective_lr_at_resume": [group["lr"] for group in optimizer.param_groups],
                    "m0_swap_weight_at_resume": _m0_swap_weight(
                        start_step + 1, m0_swap_schedule_total_steps
                    ),
                    "sampler_offset": start_step,
                    "resume_checkpoint_sha256": checkpoint_metadata[
                        "resume_checkpoint_sha256"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        m0_swap_schedule_total_steps = args.steps
        checkpoint_metadata.update(
            {
                "git_sha": os.popen("git rev-parse HEAD").read().strip(),
                "manifest_sha256": _sha256_file(derived / train_manifest),
                "m0_swap_schedule_total_steps": m0_swap_schedule_total_steps,
            }
        )
        torch.save({"model": student.state_dict(), "step": 0, **checkpoint_metadata}, output / "initial.pt")
    student.train()
    iterator = iter(loader)
    observed_datasets: set[str] = set()
    started = time.perf_counter()
    progress_path = output / "progress.json"
    write_progress(progress_path, stage="training", completed=start_step, total=args.steps, started=started)
    fixed_train_history: list[dict[str, float | int]] = []
    fixed_validation_history: list[dict[str, float | int]] = []
    m0_development_history: list[dict[str, float | int]] = []
    if m0_mode and not args.skip_development_evaluation:
        # Reproduce the loaded state before the first resumed optimizer update.
        initial_m0, initial_m0_records = _evaluate_m0_development(student, m0_validation_loader, device)
        initial_m0 = {"step": start_step, **initial_m0}
        m0_development_history.append(initial_m0)
        append_jsonl(output / "m0_development_history.jsonl", initial_m0)
        (output / f"m0_development_step{start_step:04d}.json").write_text(
            json.dumps({"summary": initial_m0, "records": initial_m0_records}, indent=2, sort_keys=True)
        )
        student.train()
    elif args.phase == "mask_only_diagnostic" and not m0_mode:
        assert fixed_train_batch is not None and fixed_validation_batch is not None
        initial_train = {"step": 0, **_evaluate_fixed_mask_probe(
            student, fixed_train_batch, device,
            panel_path=output / "soft_segmentation_train_step0000.png",
        )}
        initial_validation = {"step": 0, **_evaluate_fixed_mask_probe(
            student, fixed_validation_batch, device,
            panel_path=output / "soft_segmentation_validation_step0000.png",
        )}
        fixed_train_history.append(initial_train)
        fixed_validation_history.append(initial_validation)
        append_jsonl(output / "fixed_train_history.jsonl", initial_train)
        append_jsonl(output / "fixed_validation_history.jsonl", initial_validation)
        student.train()
    for step in range(start_step + 1, args.steps + 1):
        attempts = 0
        while True:
            try: batch = next(iterator)
            except StopIteration:
                if m0_mode:
                    m0_sampler.start_step = step - 1
                iterator = iter(loader); batch = next(iterator)
            attempts += 1
            names = {str(name) for name in batch.get("dataset_names", [])}
            required = {name for name in args.required_datasets.split(",") if name}
            if not required or required.issubset(names):
                break
            if attempts > len(loader):
                raise RuntimeError(f"required smoke datasets {sorted(required)!r} were not found together")
        observed_datasets.update(names)
        images = batch["images"].to(device, non_blocking=True)
        mapping = batch["caption_to_pair"].to(device)
        temporal_mask = batch["temporal_valid_mask"].to(device)
        selection = data_compat.retrieval_supervision_selection(batch, device)
        _assert_cuda_selection(selection, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            global_only = args.phase == "global_bootstrap"
            result = (
                student.forward_global(images, batch["captions"], temporal_mask)
                if global_only
                else student.forward_aligned_masks(
                    images, batch["captions"], mapping, temporal_mask
                )
                if m0_mode
                else student(
                    images,
                    batch["captions"],
                    mapping,
                    temporal_mask,
                    decode_mask=args.phase != "late_interaction",
                )
            )
            losses: dict[str, torch.Tensor] = {}
            global_scores = (
                result.global_score
                if global_only
                else result.text_embedding @ result.pair_embedding.T
                if m0_mode
                else result.scores.global_score
            )
            selected_scores = lambda scores: scores[selection["selected_queries"]][:, selection["selected_pairs"]]
            selected_mapping = selection["selected_mapping"]
            selected_captions = [batch["captions"][index] for index in selection["selected_queries"].tolist()]
            selected_groups = (
                stable_caption_group_ids(selected_captions, device=device)
                if selected_mapping.numel()
                else torch.empty(0, dtype=torch.long, device=device)
            )
            contrastive_masks = (
                a0_physical_pair_contrastive_masks(
                    selected_mapping, selected_groups, pair_count=selection["selected_pairs"].numel()
                )
                if selected_mapping.numel()
                else None
            )
            positive_mask = (
                contrastive_masks.text_to_pair_positive
                if contrastive_masks is not None
                else None
            )
            if "global_physical_pair_contrastive" in profile.active_losses and contrastive_masks is not None:
                losses["global_physical_pair_contrastive"] = (
                    a0_symmetric_physical_pair_contrastive_loss(
                        selected_scores(global_scores),
                        selected_mapping,
                        selected_groups,
                        temperature=args.contrastive_temperature,
                    )
                )
            if "base_text_preservation" in profile.active_losses and selected_mapping.numel():
                selected_queries = selection["selected_queries"]
                losses["base_text_preservation"] = args.base_text_preservation_weight * (1.0 - F.cosine_similarity(
                    result.text_embedding[selected_queries], result.base_text_embedding[selected_queries], dim=-1
                ).mean())
            if "global_contrastive" in profile.active_losses:
                losses["global_contrastive"] = _retrieval_loss(selected_scores(global_scores), selection["selected_mapping"])
            local_audit: dict[str, float | int] = {}
            mask_audit: dict[str, float | int] = {}
            local_loss_name = next((name for name in ("local_contrastive", "masked_local_contrastive") if name in profile.active_losses), None)
            if local_loss_name is not None and selection["selected_queries"].numel():
                branch = result.scores.token_patch_score if args.phase == "late_interaction" else result.scores.local_score
                local_scores = selected_scores(branch)
                broad = semantic_teacher_relevance_matrix(
                    result.base_text_embedding[selection["selected_queries"]].detach(), selected_captions,
                    selection["selected_mapping"], selected_groups, pair_count=selection["selected_pairs"].numel(), top_k=0,
                ) > 0
                structured_loss, local_audit = parser_derived_late_interaction_loss(
                    local_scores, selection["selected_mapping"], selected_captions,
                    selected_scores(result.scores.global_score).detach(), broad, top_n=min(50, local_scores.shape[1]),
                )
                # Generic token-patch contrastive learning plus only verified-as-negative parser conflicts.
                losses[local_loss_name] = multi_positive_contrastive_loss(
                    local_scores, positive_mask, exclusion_mask=broad & ~positive_mask,
                    temperature=args.contrastive_temperature,
                ) + structured_loss
                if "rerank_contrastive" in profile.active_losses:
                    losses["rerank_contrastive"] = multi_positive_contrastive_loss(
                        selected_scores(result.scores.reranked_score), positive_mask,
                        exclusion_mask=broad & ~positive_mask,
                        temperature=args.contrastive_temperature,
                    )
            if "teacher_distillation" in profile.active_losses:
                if teacher is None:
                    raise RuntimeError("historical-global-teacher loss requested without historical E0")
                target = teacher(images, batch["captions"], mapping, temporal_mask)
                losses.update(teacher_preservation_losses(result.pair_embedding, result.text_embedding, result.scores.global_score, target))
            if "mask_dice" in profile.active_losses or "mask_supervised_only" in profile.active_losses:
                if "query_segmentation_supervision" in batch:
                    supervised_queries = batch["query_segmentation_supervision"].to(device).bool()
                else:
                    supervised_pairs = batch["segmentation_supervision"].to(device).bool()
                    supervised_queries = supervised_pairs[mapping]
                query_indices = torch.nonzero(supervised_queries, as_tuple=False).flatten()
                if query_indices.numel():
                    selected = (
                        result.masks.decoded_mask_logits.index_select(0, query_indices)
                        if m0_mode
                        else result.scores.decoded_mask_logits[query_indices, mapping[query_indices]]
                    )
                    if "query_masks" in batch:
                        targets = foreground_preserving_resize(
                            batch["query_masks"].to(device).float(), selected.shape[-2:]
                        )[query_indices]
                    else:
                        targets = foreground_preserving_resize(
                            batch["masks"].to(device).float(), selected.shape[-2:]
                        )[mapping[query_indices]]
                    kinds = [str(batch["segmentation_target_kinds"][int(mapping[index])]) for index in query_indices.tolist()]
                    if "query_change_types" in batch:
                        changes = [batch["query_change_types"][index] for index in query_indices.tolist()]
                    else:
                        changes = [batch.get("change_types", [None] * images.shape[0])[int(mapping[index])] for index in query_indices.tolist()]
                    if m0_mode:
                        with torch.autocast("cuda", enabled=False):
                            mask_losses = m0_balanced_mask_loss(selected.float(), targets.float(), area_weight=0.1, empty_weight=1.0)
                        losses["m0_balanced_mask"] = mask_losses["total"]
                    else:
                        objective = resolve_mask_objective(args.mask_objective)
                        with torch.autocast("cuda", enabled=False):
                            mask_losses = separated_query_mask_losses(
                                selected.float(), targets.float(), kinds, changes,
                                verified_mismatched_logits=None, dice_weight=objective.dice_weight,
                                focal_weight=objective.focal_weight, tversky_weight=objective.tversky_weight,
                                positive_focal_weight=objective.positive_focal_weight,
                                negative_focal_weight=objective.negative_focal_weight, empty_weight=0.25,
                            )
                        losses["query_mask"] = mask_losses["total"]
                    probabilities = selected.sigmoid()
                    with torch.autocast("cuda", enabled=False):
                        if m0_mode:
                            separation_loss, separation_count = m0_aligned_symmetric_query_swap_loss(
                                selected.float(),
                                mapping.index_select(0, query_indices),
                                changes,
                                targets.float(),
                                margin=0.02,
                            )
                            swap_weight = _m0_swap_weight(step, m0_swap_schedule_total_steps)
                            if separation_count and swap_weight > 0:
                                losses["m0_symmetric_query_swap"] = swap_weight * separation_loss
                        else:
                            separation_loss, separation_count = _direction_query_separation_loss(
                                result.scores.decoded_mask_logits.float(), mapping, query_indices, changes, targets.float(),
                            )
                            if separation_count:
                                losses["direction_query_separation"] = separation_loss
                    foreground = targets >= 0.5
                    nonempty = foreground.flatten(1).any(dim=1)
                    background_nonempty = nonempty[:, None, None] & ~foreground
                    empty_rows = ~nonempty
                    edge = torch.cat((
                        probabilities[..., 0, :].flatten(), probabilities[..., -1, :].flatten(),
                        probabilities[..., :, 0].flatten(), probabilities[..., :, -1].flatten(),
                    )).mean()
                    center = probabilities[..., probabilities.shape[-2] // 4: 3 * probabilities.shape[-2] // 4, probabilities.shape[-1] // 4: 3 * probabilities.shape[-1] // 4].mean()
                    scientific_mask_metrics = query_mask_metrics(selected.detach(), targets.detach())
                    mask_audit = {
                        **scientific_mask_metrics,
                        "mask_foreground_probability": float(probabilities[foreground].detach().mean()) if bool(foreground.any()) else 0.0,
                        "mask_background_probability_nonempty": float(probabilities[background_nonempty].detach().mean()) if bool(background_nonempty.any()) else 0.0,
                        "mask_empty_mean_probability": float(probabilities[empty_rows].detach().mean()) if bool(empty_rows.any()) else 0.0,
                        "mask_localization_margin": float(scientific_mask_metrics["localization_margin"]),
                        "mask_edge_center_ratio": float((edge / center.clamp_min(1e-6)).detach()),
                        "verified_query_specific_count": sum(kind in {"query_specific", "verified_query_specific", "localization_only"} for kind in kinds),
                        "verified_mismatch_count": 0,
                        "direction_query_separation_count": separation_count,
                        "verified_mismatch_training_status": "DISABLED_DUPLICATE_DIRECTIONAL_ROW",
                    }
            if not losses:
                raise RuntimeError("batch has no supervision for the selected phase")
            total = sum(losses.values())
        if not torch.isfinite(total): raise FloatingPointError(f"non-finite loss at step {step}")
        total.backward()
        for name, parameter in student.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                finite_nonzero = bool(torch.isfinite(parameter.grad).all() and parameter.grad.detach().abs().sum() > 0)
                gradient_presence[name] = gradient_presence[name] or finite_nonzero

        norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm)
        if not torch.isfinite(norm): raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        if (
            m0_mode
            and not args.skip_development_evaluation
            and (step % args.m0_eval_interval == 0 or step == args.steps)
        ):
            m0_summary, m0_records = _evaluate_m0_development(student, m0_validation_loader, device)
            m0_summary = {"step": step, **m0_summary}
            m0_development_history.append(m0_summary)
            append_jsonl(output / "m0_development_history.jsonl", m0_summary)
            (output / f"m0_development_step{step:04d}.json").write_text(
                json.dumps({"summary": m0_summary, "records": m0_records}, indent=2, sort_keys=True)
            )
            student.train()

        if args.phase == "mask_only_diagnostic" and not m0_mode and step % args.probe_interval == 0:
            assert fixed_train_batch is not None and fixed_validation_batch is not None
            train_probe = {"step": step, **_evaluate_fixed_mask_probe(
                student, fixed_train_batch, device,
                panel_path=(output / f"soft_segmentation_train_step{step:04d}.png") if step == args.steps else None,
            )}
            validation_probe = {"step": step, **_evaluate_fixed_mask_probe(
                student, fixed_validation_batch, device,
                panel_path=(output / f"soft_segmentation_validation_step{step:04d}.png") if step == args.steps else None,
            )}
            fixed_train_history.append(train_probe)
            fixed_validation_history.append(validation_probe)
            append_jsonl(output / "fixed_train_history.jsonl", train_probe)
            append_jsonl(output / "fixed_validation_history.jsonl", validation_probe)
            student.train()
        if step == 1 and not global_only and not m0_mode:
            first_query = 0
            first_pair = int(mapping[first_query])
            target_panel = F.interpolate(batch["masks"][first_pair:first_pair + 1].float().unsqueeze(1), result.scores.decoded_mask_logits.shape[-2:], mode="nearest")[0, 0]
            if args.phase != "mask_only_diagnostic":
                _save_panel(
                    output / "soft_segmentation_panel.png", images,
                    result.scores.decoded_mask_logits[first_query, first_pair], target_panel,
                    query=str(batch["captions"][first_query]), pair_id=str(batch["pair_ids"][first_pair]),
                )
                _save_panel(
                    output / "retrieval_panel.png", images,
                    result.scores.decoded_mask_logits[first_query, first_pair], target_panel,
                    query=str(batch["captions"][first_query]), pair_id=str(batch["pair_ids"][first_pair]),
                )
            with torch.no_grad():
                canonical = CanonicalV3Inputs(
                    global_query_embeddings=result.text_embedding.detach(),
                    text_token_embeddings=result.text_token_embeddings.detach(),
                    text_attention_mask=result.text_attention_mask.detach(),
                    text_content_mask=result.text_content_mask.detach(),
                    pair_embeddings=result.pair_embedding.detach(),
                    per_time_tokens=result.per_time_tokens.detach(),
                )
                was_training = student.training
                student.eval()
                try:
                    parity = (trainer_score(student, canonical), evaluator_score(student, canonical), renderer_score(student, canonical))
                    for candidate in parity[1:]:
                        torch.testing.assert_close(parity[0].reranked_score, candidate.reranked_score)
                        torch.testing.assert_close(parity[0].decoded_mask_logits, candidate.decoded_mask_logits)
                finally:
                    student.train(was_training)
        margin = (
            _positive_margin(selected_scores(result.scores.local_score.detach()), selection["selected_mapping"])
            if not global_only and not m0_mode and selection["selected_queries"].numel()
            else None
        )
        post_clip_sq = sum(float(parameter.grad.detach().float().square().sum()) for parameter in parameters if parameter.grad is not None)
        row = {"step": step, "loss": float(total.detach()), "grad_norm_before_clip": float(norm), "grad_norm_after_clip": math.sqrt(post_clip_sq), "gradient_was_clipped": bool(norm > args.grad_clip_norm), "local_margin": float(margin) if margin is not None else None}
        if not global_only and not m0_mode:
            row.update(
                mask_mass_mean=float(result.scores.mask_mass.detach().mean()),
                mask_entropy_mean=float(result.scores.mask_entropy.detach().mean()),
                mask_effective_patch_count_mean=float(result.scores.mask_effective_patch_count.detach().mean()),
                near_empty_mask_fraction=float((result.scores.mask_validity.detach() <= 0).float().mean()),
            )
        elif m0_mode:
            aligned_probability = result.masks.patch_mask_logits.detach().sigmoid()
            normalized = aligned_probability / aligned_probability.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            row.update(
                mask_mass_mean=float(aligned_probability.mean()),
                mask_entropy_mean=float(-(
                    aligned_probability.clamp(1e-6, 1 - 1e-6)
                    * aligned_probability.clamp(1e-6, 1 - 1e-6).log()
                    + (1 - aligned_probability).clamp(1e-6, 1 - 1e-6)
                    * (1 - aligned_probability).clamp(1e-6, 1 - 1e-6).log()
                ).mean()),
                mask_effective_patch_count_mean=float(
                    normalized.square().sum(dim=-1).clamp_min(1e-6).reciprocal().mean()
                ),
                near_empty_mask_fraction=float(
                    (aligned_probability.amax(dim=-1) < 0.10).float().mean()
                ),
            )
        row.update(local_audit)
        row.update(mask_audit)
        row.update({name: float(value.detach()) for name, value in losses.items()})
        history.append(row)
        write_progress(
            progress_path,
            stage="training",
            completed=step,
            total=args.steps,
            started=started,
            metrics=row,
        )
        if step % args.checkpoint_interval == 0 or step == args.steps:
            _save_training_checkpoint(output / f"step{step:04d}.pt", student, optimizer, step, checkpoint_metadata)

    checkpoint = output / "last.pt"
    _save_training_checkpoint(checkpoint, student, optimizer, args.steps, checkpoint_metadata)
    restored = {key: value.cpu() for key, value in student.state_dict().items()}
    round_trip = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    if restored.keys() != round_trip.keys() or any(not torch.equal(restored[key], round_trip[key]) for key in restored):
        raise RuntimeError("checkpoint round-trip mismatch")
    a0_frozen_fingerprint_after = _frozen_parameter_fingerprint(student, ("visual_encoder", "text_encoder")) if args.phase == "global_bootstrap" else None
    a0_frozen_fingerprints_match = (a0_frozen_fingerprint_before == a0_frozen_fingerprint_after) if args.phase == "global_bootstrap" else None
    missing_gradients = sorted(name for name, present in gradient_presence.items() if not present)
    gradients_ok = bool(gradient_presence) and not missing_gradients
    all_finite = all(math.isfinite(value) for row in history for value in row.values() if isinstance(value, float))
    engineering_ok = all_finite and gradients_ok
    micro_gate = _micro_overfit_gate(
        fixed_train_history, gradients_finite=engineering_ok,
    ) if args.phase == "mask_only_diagnostic" and not m0_mode else None
    validation_signal = _validation_direction(
        fixed_validation_history,
    ) if args.phase == "mask_only_diagnostic" and not m0_mode else None
    m0_gate = (
        _m0_gate(m0_development_history, gradients_finite=engineering_ok)
        if m0_mode and not args.skip_development_evaluation
        else None
    )
    diagnostic_passed = bool(m0_gate and m0_gate["passed"]) if m0_mode else bool(micro_gate and micro_gate["passed"] and validation_signal and validation_signal["positive"])
    report = {
        "runtime_status": "PASS" if all_finite else "FAIL",
        "gradient_status": "PASS" if gradients_ok else "FAIL",
        "anti_empty_collapse_status": "NOT_EVALUATED",
        "localization_status": "M0_MULTI_PAIR_PASS" if m0_gate and m0_gate["passed"] else ("M0_MULTI_PAIR_HOLD" if m0_mode else ("TRAIN_FIT_PASS" if micro_gate and micro_gate["passed"] else ("TRAIN_FIT_FAIL" if args.phase == "mask_only_diagnostic" else "NOT_EVALUATED"))),
        "query_specificity_status": "PASS" if m0_gate and m0_gate["passed"] else ("FAIL_OR_INCONCLUSIVE" if args.phase == "mask_only_diagnostic" else "NOT_EVALUATED"),
        "validation_generalization_status": "M0_MULTI_PAIR_PASS" if m0_gate and m0_gate["passed"] else ("M0_MULTI_PAIR_HOLD" if m0_mode else ("POSITIVE_DIRECTION" if validation_signal and validation_signal["positive"] else ("NO_POSITIVE_DIRECTION" if args.phase == "mask_only_diagnostic" else "NOT_EVALUATED"))),
        "scientific_status": ("M0_DIAGNOSTIC_PASS" if diagnostic_passed else "M0_SCIENTIFIC_HOLD") if m0_mode else ("DIAGNOSTIC_PASS" if diagnostic_passed else ("SCIENTIFIC_HOLD" if args.phase == "mask_only_diagnostic" else "NOT_EVALUATED")),
        "m0_500_step_authorized": diagnostic_passed if m0_mode else False,
        "stage_c_100_step_authorized": diagnostic_passed if not m0_mode else False,
        "phase": profile.to_dict(), "initialization": asdict(initialization),
        "baseline_identity": "clean_v3_pretrained_bootstrap" if args.initialization_mode == "clean_pretrained" else "historical_e0_continuation",
        "historical_e0_continuity": args.initialization_mode == "historical_e0",
        "historical_global_teacher_used": teacher is not None,
        "text_derived_semantic_relevance_is_ground_truth": False,
        "optimizer_audit": optimizer_audit, "history": history,
        "fixed_train_history": fixed_train_history,
        "fixed_validation_history": fixed_validation_history,
        "micro_overfit_gate": micro_gate,
        "validation_direction": validation_signal,
        "m0_development_history": m0_development_history,
        "m0_gate": m0_gate,
        "batch_accounting": {
            "physical_pair_batch_size": int(args.batch_size),
            "directional_query_batch_size": (
                int(args.batch_size * 2) if m0_mode else int(args.batch_size)
            ),
            "physical_pair_microbatch_size": int(args.batch_size),
            "directional_query_microbatch_size": (
                int(args.batch_size * 2) if m0_mode else int(args.batch_size)
            ),
            "gradient_accumulation_steps": 1,
            "effective_optimizer_physical_pair_batch_size": int(args.batch_size),
            "effective_optimizer_directional_query_batch_size": (
                int(args.batch_size * 2) if m0_mode else int(args.batch_size)
            ),
        },
        "aligned_mask_execution": (
            {
                "enabled": True,
                "decoded_mask_count": int(result.masks.decoded_mask_logits.shape[0]),
                **result.masks.memory_diagnostics,
            }
            if m0_mode else {"enabled": False}
        ),
        "m0_contract": {"chronology": "Image2(before)->Image1(after); Label1=appeared; Label2=disappeared", "pre_a936f116_mask_local_artifacts": "INVALID", "development_target_centered_crops": False} if m0_mode else None,
        "observed_datasets": sorted(observed_datasets),
        "runtime_device_contract": {
            "requested_device": str(requested_device),
            "resolved_device": str(device),
            "selected_pairs_device": str(selection["selected_pairs"].device),
            "selected_queries_device": str(selection["selected_queries"].device),
            "selected_mapping_device": str(selection["selected_mapping"].device),
            "cuda_current_device": torch.cuda.current_device(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "a0_frozen_fingerprint_before": a0_frozen_fingerprint_before,
        "a0_frozen_fingerprint_after": a0_frozen_fingerprint_after,
        "a0_frozen_fingerprints_match": a0_frozen_fingerprints_match,
        "all_finite": all_finite,
        "gradient_audit": {
            "expected_trainable_gradients_finite_nonzero": gradients_ok,
            "parameters_with_gradient": sorted(name for name, present in gradient_presence.items() if present),
            "parameters_without_finite_nonzero_gradient": missing_gradients,
        },
        "peak_gpu_allocated": torch.cuda.max_memory_allocated(), "peak_gpu_reserved": torch.cuda.max_memory_reserved(),
        "git_sha": os.popen("git rev-parse HEAD").read().strip(), "checkpoint": str(checkpoint),
    }
    (output / "resolved_config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2))
    write_progress(
        progress_path, stage="complete", completed=args.steps, total=args.steps,
        started=started, complete=True,
        metrics={"runtime_status": report["runtime_status"], "scientific_status": report["scientific_status"], "last_loss": history[-1]["loss"] if history else None},
    )
    if not engineering_ok:
        raise RuntimeError(
            "QCPR v3 runtime contract failed: "
            f"all_finite={all_finite}, missing_gradients={missing_gradients}"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--phase",
        choices=("global_bootstrap", "late_interaction", "mask_grounding", "mask_only_diagnostic"),
        required=True,
    )
    parser.add_argument("--initialization-mode", choices=("clean_pretrained",), required=True)
    parser.add_argument("--v1-checkpoint", type=Path, default=IMMUTABLE_V1)
    parser.add_argument("--v3-checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/weka/svardanyan/rs_change_project/datasets/processed/LEVIR-MCI"))
    parser.add_argument("--universat-source", type=Path, default=Path("/mnt/weka/svardanyan/rs_change_project/external/UniverSat"))
    parser.add_argument("--universat-checkpoint", type=Path, default=Path("/mnt/weka/svardanyan/rs_change_project/models/universat-base"))
    parser.add_argument("--jina-model", type=Path, default=Path("/mnt/weka/svardanyan/rs_change_project/models/jina-v5-text-small-retrieval"))
    parser.add_argument("--grounding-backbone", choices=("universat", "siglip2"), default="universat")
    parser.add_argument("--siglip2-model", type=Path, default=Path("/mnt/weka/svardanyan/rs_change_project/models/siglip2-base-patch16-256"))
    parser.add_argument("--derived-manifest-dir", type=Path, default=Path("/mnt/weka/svardanyan/rs_change_project/manifests/qcpr_v3_clean_058779b7"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--text-adapter-learning-rate", type=float, default=2e-5)
    parser.add_argument("--base-text-preservation-weight", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=13.41814250946045, help="Clean-bootstrap probe p80 threshold; target clipping <=20%%")
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--temporal-depth", type=int, default=6)
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--no-change-batch-fraction-cap", type=float, default=0.25)
    parser.add_argument("--required-datasets", default="")
    parser.add_argument("--mask-objective", choices=("A",), default="A")
    parser.add_argument("--mask-diagnostic-mode", choices=("single_pair", "multi_pair"), default="single_pair")
    parser.add_argument("--m0-eval-interval", type=int, default=10)
    parser.add_argument("--skip-development-evaluation", action="store_true")
    parser.add_argument("--micro-train-samples", type=int, default=8)
    parser.add_argument("--validation-probe-samples", type=int, default=16)
    parser.add_argument("--probe-interval", type=int, default=5)
    parser.add_argument("--target-crop-context", type=float, default=2.0)
    parser.add_argument("--direction-only-probe-captions", action="store_true")
    parser.add_argument("--train-probe-pair-id", action="append", default=[])
    parser.add_argument("--validation-probe-pair-id", action="append", default=[])
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
