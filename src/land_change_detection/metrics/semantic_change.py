from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ConfusionMetrics:
    pixel_accuracy: float
    mean_iou: float
    macro_f1: float


def _safe_div(num: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    return torch.where(denom > 0, num / denom.clamp_min(1), torch.zeros_like(num, dtype=torch.float64))


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, classes: int, ignore_index: int = -1) -> torch.Tensor:
    pred = pred.to(torch.int64).reshape(-1)
    target = target.to(torch.int64).reshape(-1)
    valid = (target != ignore_index) & (target >= 0) & (target < classes) & (pred >= 0) & (pred < classes)
    if not torch.any(valid):
        return torch.zeros((classes, classes), dtype=torch.float64, device=pred.device)
    encoded = target[valid] * classes + pred[valid]
    return torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes).to(torch.float64)


def metrics_from_confusion(confusion: torch.Tensor) -> ConfusionMetrics:
    confusion = confusion.to(torch.float64)
    true_positive = torch.diag(confusion)
    row_sum = confusion.sum(dim=1)
    col_sum = confusion.sum(dim=0)
    total = confusion.sum()
    accuracy = float(true_positive.sum().div(total).item()) if total > 0 else 0.0
    union = row_sum + col_sum - true_positive
    iou = _safe_div(true_positive, union)
    f1 = _safe_div(2.0 * true_positive, row_sum + col_sum)
    present_iou = union > 0
    present_f1 = (row_sum + col_sum) > 0
    mean_iou = float(iou[present_iou].mean().item()) if torch.any(present_iou) else 0.0
    macro_f1 = float(f1[present_f1].mean().item()) if torch.any(present_f1) else 0.0
    return ConfusionMetrics(pixel_accuracy=accuracy, mean_iou=mean_iou, macro_f1=macro_f1)


def binary_change_metrics(pred: torch.Tensor, target: torch.Tensor, ignore_mask: torch.Tensor | None = None) -> dict[str, float]:
    pred = pred.to(torch.bool)
    target = target.to(torch.bool)
    valid = torch.ones_like(target, dtype=torch.bool) if ignore_mask is None else ~ignore_mask.to(torch.bool)
    pred = pred[valid]
    target = target[valid]
    if pred.numel() == 0:
        return {
            "binary_change_precision": 0.0,
            "binary_change_recall": 0.0,
            "binary_change_f1": 0.0,
            "binary_change_iou": 0.0,
            "binary_change_dice": 0.0,
        }
    tp = torch.logical_and(pred, target).sum().to(torch.float64)
    fp = torch.logical_and(pred, ~target).sum().to(torch.float64)
    fn = torch.logical_and(~pred, target).sum().to(torch.float64)
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = (2.0 * precision * recall) / (precision + recall).clamp_min(1e-12)
    iou = tp / (tp + fp + fn).clamp_min(1)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1)
    return {
        "binary_change_precision": float(precision.item()),
        "binary_change_recall": float(recall.item()),
        "binary_change_f1": float(f1.item()),
        "binary_change_iou": float(iou.item()),
        "binary_change_dice": float(dice.item()),
    }


def transition_counts(transition_map: torch.Tensor, num_classes: int, valid_mask: torch.Tensor | None = None) -> dict[str, int]:
    flat = transition_map.to(torch.int64).reshape(-1)
    if valid_mask is not None:
        flat = flat[valid_mask.reshape(-1)]
    if flat.numel() == 0:
        return {}
    counts = torch.bincount(flat.cpu(), minlength=num_classes * num_classes)
    output: dict[str, int] = {}
    for index, value in enumerate(counts.tolist()):
        if value:
            output[f"{index // num_classes}->{index % num_classes}"] = int(value)
    return output


def _dominant_transition_labels(transition_map: torch.Tensor, num_classes: int, valid_mask: torch.Tensor) -> list[str]:
    labels: list[str] = []
    for sample_index in range(transition_map.shape[0]):
        flat = transition_map[sample_index][valid_mask[sample_index]].to(torch.int64)
        if flat.numel() == 0:
            labels.append("")
            continue
        counts = torch.bincount(flat.cpu(), minlength=num_classes * num_classes)
        index = int(torch.argmax(counts).item())
        labels.append(f"{index // num_classes}->{index % num_classes}")
    return labels


def semantic_change_metrics(output: Any, batch: dict[str, Any], num_classes: int, ignore_index: int = -1) -> dict[str, Any]:
    before_target = batch["before_target"].to(output.before_logits.device)
    after_target = batch["after_target"].to(output.after_logits.device)
    change_target = batch["change_target"].to(output.change_logits.device)
    before_pred = output.before_semantic_map
    after_pred = output.after_semantic_map
    valid_semantic = (before_target != ignore_index) & (after_target != ignore_index)

    before = metrics_from_confusion(confusion_matrix(before_pred, before_target, num_classes, ignore_index))
    after = metrics_from_confusion(confusion_matrix(after_pred, after_target, num_classes, ignore_index))

    pred_transition = output.transition_map()
    target_transition = before_target.to(torch.int64) * int(num_classes) + after_target.to(torch.int64)
    transition_confusion = confusion_matrix(
        pred_transition[valid_semantic],
        target_transition[valid_semantic],
        num_classes * num_classes,
        ignore_index=ignore_index,
    )
    transition = metrics_from_confusion(transition_confusion)

    if change_target.ndim == 4:
        change_target_2d = change_target.squeeze(1)
    else:
        change_target_2d = change_target
    target_change = change_target_2d >= 0.5
    pred_change = output.binary_change_map.to(change_target_2d.device).to(torch.bool)
    binary = binary_change_metrics(pred_change, target_change, ignore_mask=change_target_2d < 0)
    pred_ratio = pred_change.float().mean(dim=(1, 2))
    target_ratio = target_change.float().mean(dim=(1, 2))
    changed_area_ratio_mae = float(torch.mean(torch.abs(pred_ratio - target_ratio)).item()) if pred_ratio.numel() else 0.0

    dominant_accuracy: float | None = None
    expected = batch.get("dominant_transition")
    if expected is not None:
        predicted_labels = _dominant_transition_labels(pred_transition.detach(), num_classes, valid_semantic)
        expected_labels = [str(item) for item in expected]
        comparable = [(pred, label) for pred, label in zip(predicted_labels, expected_labels, strict=False) if label]
        if comparable:
            dominant_accuracy = sum(1 for pred, label in comparable if pred == label) / float(len(comparable))

    return {
        "before_pixel_accuracy": before.pixel_accuracy,
        "before_mean_iou": before.mean_iou,
        "before_macro_f1": before.macro_f1,
        "after_pixel_accuracy": after.pixel_accuracy,
        "after_mean_iou": after.mean_iou,
        "after_macro_f1": after.macro_f1,
        "semantic_mean_iou": (before.mean_iou + after.mean_iou) * 0.5,
        "semantic_macro_f1": (before.macro_f1 + after.macro_f1) * 0.5,
        "transition_mean_iou": transition.mean_iou,
        "transition_macro_f1": transition.macro_f1,
        "transition_pixel_accuracy": transition.pixel_accuracy,
        "changed_area_ratio_mae": changed_area_ratio_mae,
        "dominant_transition_accuracy": dominant_accuracy,
        "transition_summary": {
            "target_counts": transition_counts(target_transition.detach().cpu(), num_classes, valid_semantic.detach().cpu()),
            "predicted_counts": transition_counts(pred_transition.detach().cpu(), num_classes, valid_semantic.detach().cpu()),
        },
        **binary,
    }
