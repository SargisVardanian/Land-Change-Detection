from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from land_change_detection.data.event_targets import ComponentTargets
from land_change_detection.losses.unichange_losses import (
    cosine_semantic_regression,
    dice_loss,
    event_component_coverage_loss,
    event_overlap_loss,
    masked_multi_positive_sigmoid_loss,
    smooth_topk_late_interaction_score,
)
from land_change_detection.metrics.unichange_grounding import dice_score, heatmap_energy_inside_mask, union_mask_from_events
from land_change_detection.metrics.unichange_retrieval import retrieval_metrics
from land_change_detection.training.hungarian_event_matcher import hungarian_match_events
from land_change_detection.training.safe_negative_miner import mine_safe_negative_mask


@dataclass(frozen=True)
class JointLossWeights:
    retrieval: float = 1.0
    semantic: float = 0.25
    local: float = 0.0
    mask: float = 0.0
    presence: float = 0.5
    coverage: float = 0.25
    overlap: float = 0.05


@dataclass(frozen=True)
class UniChangeJointTrainerConfig:
    run_name: str = "joint_retrieval_grounding_overfit_100"
    output_dir: Path = Path("runs/joint_retrieval_grounding_overfit_100")
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    total_steps: int | None = None
    local_max_weight: float = 0.25
    mask_max_weight: float = 1.0
    safe_negative_bottom_quantile: float = 0.35
    retrieval_temperature: float = 0.07
    local_top_k: int = 8
    presence_positive_target: float = 1.0
    presence_negative_target: float = 0.0
    grad_clip_norm: float | None = 1.0
    device: str = "cpu"


@dataclass(frozen=True)
class JointStepResult:
    loss: Tensor
    metrics: dict[str, float]


def scheduled_joint_weights(step: int, total_steps: int, local_max: float = 0.25, mask_max: float = 1.0) -> JointLossWeights:
    progress = 1.0 if total_steps <= 1 else min(max(step / float(total_steps - 1), 0.0), 1.0)

    def ramp(start: float, end: float, maximum: float) -> float:
        if progress <= start:
            return 0.0
        if progress >= end:
            return maximum
        return maximum * ((progress - start) / (end - start))

    return JointLossWeights(
        retrieval=1.0,
        semantic=0.25,
        local=ramp(0.10, 0.30, local_max),
        mask=ramp(0.20, 0.40, mask_max),
        presence=0.5,
        coverage=0.25,
        overlap=0.05,
    )


def _pair_document_targets(model: nn.Module, captions: list[str], caption_to_pair: Tensor, num_pairs: int) -> Tensor:
    with torch.no_grad():
        document_features = model.encode_text(captions, role="document")
        document_embeddings = document_features.global_embedding.detach()
        targets = torch.zeros(num_pairs, document_embeddings.shape[-1], device=document_embeddings.device, dtype=document_embeddings.dtype)
        counts = torch.zeros(num_pairs, device=document_embeddings.device, dtype=document_embeddings.dtype)
        targets.index_add_(0, caption_to_pair.long(), document_embeddings)
        counts.index_add_(0, caption_to_pair.long(), torch.ones_like(caption_to_pair, dtype=document_embeddings.dtype))
        return F.normalize(targets / counts.clamp_min(1).unsqueeze(-1), dim=-1)


def _move_components(components: list[ComponentTargets], device: torch.device) -> list[ComponentTargets]:
    moved: list[ComponentTargets] = []
    for component in components:
        moved.append(
            ComponentTargets(
                masks=component.masks.to(device),
                full_resolution_mask=component.full_resolution_mask.to(device),
                areas=component.areas,
            )
        )
    return moved


class UniChangeJointTrainer:
    def __init__(self, model: nn.Module, config: UniChangeJointTrainerConfig | None = None, optimizer: torch.optim.Optimizer | None = None):
        self.model = model
        self.config = config or UniChangeJointTrainerConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = optimizer or torch.optim.AdamW(trainable, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.history: list[dict[str, float]] = []

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(batch)
        prepared["t1"] = batch["t1"].to(self.device)
        prepared["t2"] = batch["t2"].to(self.device)
        prepared["caption_to_pair"] = batch["caption_to_pair"].to(self.device)
        prepared["masks"] = batch["masks"].to(self.device)
        prepared["components"] = _move_components(batch["components"], self.device)
        return prepared

    def compute_loss(self, batch: dict[str, Any], step: int, total_steps: int) -> JointStepResult:
        batch = self._prepare_batch(batch)
        captions: list[str] = batch["captions"]
        caption_to_pair: Tensor = batch["caption_to_pair"]
        num_pairs = int(batch["t1"].shape[0])
        weights = scheduled_joint_weights(step, total_steps, self.config.local_max_weight, self.config.mask_max_weight)
        output = self.model(batch["t1"], batch["t2"], captions, text_role="query")
        if output.text_global_embedding is None or output.text_token_embeddings is None:
            raise RuntimeError("Joint training requires text embeddings in UniChangeOutput.")
        if output.event_embeddings is None or output.event_mask_logits is None or output.event_masks is None or output.event_presence_logits is None:
            raise RuntimeError("Joint training requires event embeddings, mask logits, masks and presence logits.")
        if output.semantic_prediction is None:
            raise RuntimeError("Joint training requires semantic_prediction.")

        safe_negative_mask = mine_safe_negative_mask(
            output.text_global_embedding.detach(),
            caption_to_pair,
            num_pairs,
            bottom_quantile=self.config.safe_negative_bottom_quantile,
        )
        retrieval_loss, retrieval_stats = masked_multi_positive_sigmoid_loss(
            output.global_pair_embedding,
            output.text_global_embedding,
            caption_to_pair,
            safe_negative_mask=safe_negative_mask,
            temperature=self.config.retrieval_temperature,
        )

        semantic_target = _pair_document_targets(self.model, captions, caption_to_pair, num_pairs).to(output.semantic_prediction.device)
        semantic_loss = cosine_semantic_regression(output.semantic_prediction, semantic_target)

        selected_events = output.event_embeddings[caption_to_pair]
        local_scores = smooth_topk_late_interaction_score(
            output.text_token_embeddings,
            selected_events,
            top_k=min(self.config.local_top_k, selected_events.shape[1]),
        )
        local_loss = -local_scores.mean()

        mask_bce_terms: list[Tensor] = []
        mask_dice_terms: list[Tensor] = []
        presence_targets = torch.full_like(output.event_presence_logits, self.config.presence_negative_target)
        component_batches: list[Tensor] = []
        active_changed = 0
        for pair_index, component in enumerate(batch["components"]):
            components = component.masks.to(output.event_mask_logits.device)
            component_batches.append(components)
            if components.shape[0] > 0:
                match = hungarian_match_events(output.event_mask_logits[pair_index], components)
                if match.event_indices.numel() > 0:
                    event_idx = match.event_indices
                    comp_idx = match.component_indices
                    presence_targets[pair_index, event_idx] = self.config.presence_positive_target
                    logits = output.event_mask_logits[pair_index, event_idx]
                    targets = components[comp_idx].to(logits.dtype)
                    mask_bce_terms.append(F.binary_cross_entropy_with_logits(logits, targets))
                    mask_dice_terms.append(dice_loss(logits, targets))
                    active_changed += 1

        zero = output.global_pair_embedding.sum() * 0.0
        mask_bce = torch.stack(mask_bce_terms).mean() if mask_bce_terms else zero
        mask_dice = torch.stack(mask_dice_terms).mean() if mask_dice_terms else zero
        presence_loss = F.binary_cross_entropy_with_logits(output.event_presence_logits, presence_targets)
        coverage_terms: list[Tensor] = []
        for pair_index, components in enumerate(component_batches):
            if components.shape[0] == 0:
                continue
            coverage_terms.append(event_component_coverage_loss(output.event_masks[pair_index : pair_index + 1], components.unsqueeze(0)))
        coverage_loss = torch.stack(coverage_terms).mean() if coverage_terms else zero
        overlap_loss = event_overlap_loss(output.event_masks)
        total_loss = (
            weights.retrieval * retrieval_loss
            + weights.semantic * semantic_loss
            + weights.local * local_loss
            + weights.mask * (mask_bce + mask_dice)
            + weights.presence * presence_loss
            + weights.coverage * coverage_loss
            + weights.overlap * overlap_loss
        )

        with torch.no_grad():
            retrieval = retrieval_metrics(output.global_pair_embedding.detach(), output.text_global_embedding.detach(), caption_to_pair.detach())
            union = union_mask_from_events(output.event_masks.detach(), output.event_presence_logits.detach()).view(num_pairs, 36, 36)
            target_masks = batch["masks"].detach()
            text_mask = output.text_conditioned_mask
            if text_mask is not None and text_mask.ndim == 3:
                text_mask = text_mask[torch.arange(text_mask.shape[0], device=text_mask.device), caption_to_pair]
            if text_mask is not None:
                text_heat = text_mask.detach().view(text_mask.shape[0], 36, 36)
                caption_targets = target_masks[caption_to_pair]
                energy = heatmap_energy_inside_mask(text_heat, caption_targets)
            else:
                energy = 0.0
            metrics = {
                "loss": float(total_loss.detach().item()),
                "retrieval_loss": float(retrieval_loss.detach().item()),
                "semantic_loss": float(semantic_loss.detach().item()),
                "local_loss": float(local_loss.detach().item()),
                "mask_bce_loss": float(mask_bce.detach().item()),
                "mask_dice_loss": float(mask_dice.detach().item()),
                "presence_loss": float(presence_loss.detach().item()),
                "coverage_loss": float(coverage_loss.detach().item()),
                "overlap_loss": float(overlap_loss.detach().item()),
                "weight_local": weights.local,
                "weight_mask": weights.mask,
                "finite_loss": float(torch.isfinite(total_loss).item()),
                "changed_samples_with_active_event_ratio": float(active_changed / max(num_pairs, 1)),
                "union_mask_dice": dice_score(union, target_masks),
                "text_conditioned_energy_inside_gt": energy,
                **retrieval_stats,
                **retrieval,
            }
        return JointStepResult(total_loss, metrics)

    def train_step(self, batch: dict[str, Any], step: int, total_steps: int) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        result = self.compute_loss(batch, step=step, total_steps=total_steps)
        if not torch.isfinite(result.loss):
            raise FloatingPointError(f"Non-finite UniChange loss at step {step}: {result.metrics}")
        result.loss.backward()
        finite_gradients = True
        for param in self.model.parameters():
            if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                finite_gradients = False
                break
        if not finite_gradients:
            raise FloatingPointError(f"Non-finite UniChange gradients at step {step}.")
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        metrics = dict(result.metrics)
        metrics["step"] = float(step)
        metrics["finite_gradients"] = 1.0
        self.history.append(metrics)
        return metrics

    def save_checkpoint(self, path: str | Path, step: int, metrics: dict[str, float] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": step,
                "metrics": metrics or {},
                "config": {**asdict(self.config), "output_dir": str(self.config.output_dir)},
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return int(checkpoint.get("step", 0))

    def fit(self, train_loader: Any) -> dict[str, float]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "visuals" / "retrieval_examples").mkdir(parents=True, exist_ok=True)
        (output_dir / "visuals" / "event_masks").mkdir(parents=True, exist_ok=True)
        (output_dir / "visuals" / "text_conditioned_masks").mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps({**asdict(self.config), "output_dir": str(output_dir)}, indent=2))
        total_steps = self.config.total_steps or max(len(train_loader) * self.config.epochs, 1)
        best_metric = float("-inf")
        best_metrics: dict[str, float] = {}
        step = 0
        for _epoch in range(self.config.epochs):
            for batch in train_loader:
                metrics = self.train_step(batch, step=step, total_steps=total_steps)
                score = metrics.get("R@5", 0.0) + metrics.get("union_mask_dice", 0.0)
                if score >= best_metric:
                    best_metric = score
                    best_metrics = dict(metrics)
                    self.save_checkpoint(output_dir / "best.pt", step, metrics)
                step += 1
                if step >= total_steps:
                    break
            if step >= total_steps:
                break
        last_metrics = self.history[-1] if self.history else {}
        self.save_checkpoint(output_dir / "last.pt", max(step - 1, 0), last_metrics)
        (output_dir / "metrics.json").write_text(json.dumps({"best": best_metrics, "last": last_metrics}, indent=2))
        self._write_history(output_dir / "history.csv")
        (output_dir / "rankings.jsonl").write_text("")
        return last_metrics

    def _write_history(self, path: Path) -> None:
        if not self.history:
            path.write_text("")
            return
        fields = sorted({key for row in self.history for key in row})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.history)
