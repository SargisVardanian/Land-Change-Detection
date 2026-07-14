from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from torch import nn


@dataclass(frozen=True)
class TrainingPhaseProfile:
    name: str
    trainable_modules: tuple[str, ...]
    frozen_modules: tuple[str, ...]
    active_losses: tuple[str, ...]
    datasets: tuple[str, ...]
    sampler: str
    validation_metrics: tuple[str, ...]
    checkpoint_metric: str
    gates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PHASES: dict[str, TrainingPhaseProfile] = {
    "global_recovery": TrainingPhaseProfile(
        "global_recovery", ("temporal_encoder", "retrieval_head", "text_adapter"),
        ("visual_encoder", "text_encoder", "grounder"),
        ("global_contrastive", "teacher_distillation", "pair_preservation", "text_preservation"),
        ("levir_mci", "second_cc"), "natural", ("semantic_r1", "semantic_r5", "candidate_recall"),
        "semantic_r1", ("global_r1_within_0.01", "global_r5_within_0.01"),
    ),
    "late_interaction": TrainingPhaseProfile(
        "late_interaction", ("grounder.temporal_field", "grounder.grounding_decoder", "grounder.local_projection", "grounder.rerank_logits"),
        ("visual_encoder", "text_encoder", "temporal_encoder", "retrieval_head", "text_adapter", "grounder.mask_decoder", "grounder.region_slots"),
        ("local_contrastive",), ("levir_mci", "second_cc"), "natural",
        ("candidate_recall", "conditional_rerank", "semantic_r1", "semantic_r5", "local_margin"),
        "conditional_rerank", ("positive_local_margin", "rerank_not_worse", "finite_local_gradients"),
    ),
    "mask_grounding": TrainingPhaseProfile(
        "mask_grounding", ("grounder.temporal_field", "grounder.grounding_decoder", "grounder.mask_decoder", "grounder.local_projection", "grounder.rerank_logits", "grounder.temporal_map_head"),
        ("visual_encoder", "text_encoder", "temporal_encoder", "retrieval_head", "text_adapter", "grounder.region_slots"),
        ("local_contrastive", "mask_dice", "mask_focal", "empty_false_positive", "temporal_maps"),
        ("levir_mci", "second_cc", "s2looking_localization"), "curriculum",
        ("candidate_recall", "conditional_rerank", "nonempty_dice", "nonempty_iou", "empty_false_positive"),
        "nonempty_dice", ("mask_not_collapsed", "nonempty_recall_above_zero", "faithful_mask"),
    ),
    "region_slots": TrainingPhaseProfile(
        "region_slots", ("grounder.region_slots",),
        ("visual_encoder", "text_encoder", "temporal_encoder", "retrieval_head", "text_adapter", "grounder.temporal_field", "grounder.grounding_decoder", "grounder.mask_decoder"),
        ("slot_activation", "slot_diversity", "verified_count"),
        ("verified_count_subset",), "balanced", ("count_mae", "count_bucket_accuracy", "slot_overlap"),
        "count_bucket_accuracy", ("verified_count_coverage", "slot_noncollapse"),
    ),
    "joint_finetune": TrainingPhaseProfile(
        "joint_finetune", ("temporal_encoder", "retrieval_head", "text_adapter", "grounder"),
        ("visual_encoder", "text_encoder"),
        ("global_contrastive", "teacher_distillation", "pair_preservation", "text_preservation", "local_contrastive", "mask_dice", "mask_focal", "empty_false_positive", "temporal_reversal"),
        ("levir_mci", "second_cc", "s2looking_localization"), "curriculum",
        ("semantic_r1", "semantic_r5", "conditional_rerank", "nonempty_dice", "reversal_accuracy"),
        "joint_composite", ("all_previous_phase_gates", "teacher_preservation"),
    ),
}


def resolve_training_phase(name: str, overrides: Mapping[str, object] | None = None) -> TrainingPhaseProfile:
    if name not in PHASES:
        raise ValueError(f"Unknown training phase {name!r}; expected one of {sorted(PHASES)}")
    overrides = overrides or {}
    contradictory = sorted(key for key, value in overrides.items() if value is not None)
    if contradictory:
        raise ValueError(f"Training phases are immutable; contradictory overrides: {contradictory}")
    return PHASES[name]


def apply_phase_to_model(model: nn.Module, profile: TrainingPhaseProfile) -> dict[str, list[str]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named_modules = dict(model.named_modules())
    named_parameters = dict(model.named_parameters())
    missing: list[str] = []
    enabled: list[str] = []
    for prefix in profile.trainable_modules:
        module = named_modules.get(prefix)
        if module is None:
            parameter = named_parameters.get(prefix)
            if parameter is None:
                missing.append(prefix)
                continue
            parameter.requires_grad_(True)
            enabled.append(prefix)
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        enabled.append(prefix)
    if missing:
        raise ValueError(f"Phase {profile.name} references missing modules: {missing}")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    return {"enabled_modules": enabled, "trainable_parameters": trainable, "frozen_parameters": frozen}
