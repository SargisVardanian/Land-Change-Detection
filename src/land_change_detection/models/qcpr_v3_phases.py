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
    "global_bootstrap": TrainingPhaseProfile(
        "global_bootstrap", ("temporal_encoder", "retrieval_head.pair_projection", "text_adapter"),
        ("visual_encoder", "text_encoder", "grounder"),
        # Track A0 is deliberately global-only. The frozen Jina representation is input to the small adapter, not a historical-teacher objective.
        ("global_physical_pair_contrastive",),
        ("levir_mci", "second_cc"), "capped_natural",
        (
            "changed_text_derived_ndcg_at_10",
            "changed_candidate_recall_at_100",
            "no_change_metrics_separate",
            "exact_pair_diagnostic",
        ),
        "changed_ndcg_at_10_with_candidate_recall_at_100_constraint",
        (
            "finite_global_gradients",
            "clipping_fraction_below_0.3",
            "changed_candidate_recall_at_100_preserved",
        ),
    ),
    "late_interaction": TrainingPhaseProfile(
        "late_interaction", (
            "grounder.temporal_field",
            "grounder.direct_token_projection",
        ),
        (
            "visual_encoder", "text_encoder", "temporal_encoder",
            "retrieval_head", "text_adapter", "grounder.grounding_decoder",
            "grounder.mask_decoder",
        ),
        ("local_contrastive",), ("levir_mci", "second_cc"), "natural",
        ("candidate_recall", "conditional_rerank", "semantic_r1", "semantic_r5", "local_margin"),
        "conditional_rerank", ("positive_local_margin", "rerank_not_worse", "finite_local_gradients"),
    ),
    "mask_grounding": TrainingPhaseProfile(
        "mask_grounding", (
            "grounder.temporal_field",
            "grounder.grounding_decoder",
            "grounder.mask_decoder",
        ),
        ("visual_encoder", "text_encoder", "temporal_encoder", "retrieval_head", "text_adapter"),
        ("mask_dice", "mask_focal", "empty_false_positive"),
        ("s2looking_localization",), "pair_level_directional",
        ("nonempty_dice", "nonempty_iou", "empty_false_positive", "query_swap_gap"),
        "nonempty_dice", ("mask_not_collapsed", "nonempty_recall_above_zero", "faithful_mask"),
    ),
    "mask_only_diagnostic": TrainingPhaseProfile(
        "mask_only_diagnostic", (
            "grounder.temporal_field",
            "grounder.grounding_decoder",
            "grounder.mask_decoder",
        ),
        (
            "visual_encoder", "text_encoder", "temporal_encoder", "retrieval_head",
            "text_adapter",
        ),
        ("mask_supervised_only",), ("s2looking_localization",), "fixed_s2looking_probe",
        (
            "nonempty_dice", "nonempty_iou", "nonempty_precision", "nonempty_recall",
            "nonempty_soft_dice", "nonempty_soft_iou", "localization_margin",
            "empty_mean_probability", "pixel_average_precision", "soft_query_swap_iou_gap",
        ),
        "nonempty_dice",
        (
            "fixed_train_fit", "soft_dice_iou_margin_improve", "empty_probability_nonincrease",
            "soft_query_swap_gap_positive", "finite_mask_gradients",
        ),
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
