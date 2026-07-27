from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def union_mask_from_events(event_masks: Tensor, presence_logits: Tensor, threshold: float = 0.5) -> Tensor:
    if event_masks.ndim != 3 or presence_logits.ndim != 2:
        raise ValueError("event_masks must be [B,K,N] and presence_logits [B,K].")
    active = (torch.sigmoid(presence_logits) >= threshold).to(event_masks.dtype)
    union = 1.0 - torch.prod(1.0 - event_masks * active.unsqueeze(-1), dim=1)
    return union


def dice_score(pred_mask: Tensor, target_mask: Tensor, eps: float = 1e-6) -> float:
    pred = pred_mask.to(torch.float32)
    target = target_mask.to(torch.float32)
    if pred.shape != target.shape:
        pred = F.interpolate(pred.unsqueeze(1), size=target.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    pred_binary = pred >= 0.5
    target_binary = target >= 0.5
    intersection = (pred_binary & target_binary).sum(dim=tuple(range(1, pred_binary.ndim))).to(torch.float32)
    denom = pred_binary.sum(dim=tuple(range(1, pred_binary.ndim))).to(torch.float32) + target_binary.sum(dim=tuple(range(1, target_binary.ndim))).to(torch.float32)
    return float(((2 * intersection + eps) / (denom + eps)).mean().item())


def heatmap_energy_inside_mask(heatmap: Tensor, target_mask: Tensor, eps: float = 1e-6) -> float:
    heat = heatmap.to(torch.float32)
    target = target_mask.to(torch.float32)
    if heat.ndim == 3 and target.ndim == 3 and heat.shape[-2:] != target.shape[-2:]:
        heat = F.interpolate(heat.unsqueeze(1), size=target.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    numerator = (heat * target).flatten(1).sum(dim=1)
    denominator = heat.flatten(1).sum(dim=1).clamp_min(eps)
    return float((numerator / denominator).mean().item())
