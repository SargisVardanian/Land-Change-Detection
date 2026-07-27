from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class UniChangeStage(str, Enum):
    PROBE = "probe"
    GLOBAL_RETRIEVAL = "global_retrieval"
    LOCAL_RETRIEVAL = "local_retrieval"
    WEAK_EVENTS = "weak_events"
    SUPERVISED_MASKS = "supervised_masks"
    DIRECTIONAL_READOUT = "directional_readout"
    PAIR_TO_PAIR = "pair_to_pair"
    EXPLANATIONS = "explanations"


@dataclass(frozen=True)
class LossWeights:
    retrieval: float = 0.0
    semantic: float = 0.0
    local: float = 0.0
    mask_bce: float = 0.0
    mask_dice: float = 0.0
    event_presence: float = 0.0
    event_coverage: float = 0.0
    event_overlap: float = 0.0
    distillation: float = 0.0
    caption: float = 0.0


@dataclass(frozen=True)
class StageGate:
    finite_loss: bool = True
    finite_gradients: bool = True
    min_train_recall_at_5: float | None = None
    min_train_recall_at_10: float | None = None
    max_retrieval_drop_points: float | None = None
    requires_mask_metric_improvement: bool = False
    requires_offline_index: bool = False


@dataclass(frozen=True)
class StageSpec:
    stage: UniChangeStage
    datasets: tuple[str, ...]
    trainable_modules: tuple[str, ...]
    frozen_modules: tuple[str, ...]
    loss_weights: LossWeights
    retrieval_replay_fraction: float
    gate: StageGate
    notes: tuple[str, ...] = ()


def default_unichange_curriculum() -> tuple[StageSpec, ...]:
    return (
        StageSpec(
            stage=UniChangeStage.PROBE,
            datasets=(),
            trainable_modules=(),
            frozen_modules=("universat", "jina_v5"),
            loss_weights=LossWeights(),
            retrieval_replay_fraction=0.0,
            gate=StageGate(),
            notes=("Offline asset and shape probe only.",),
        ),
        StageSpec(
            stage=UniChangeStage.GLOBAL_RETRIEVAL,
            datasets=("LEVIR-CC",),
            trainable_modules=("visual_projection", "attention_pool", "semantic_head"),
            frozen_modules=("universat", "jina_v5", "event_decoder", "directional_readout"),
            loss_weights=LossWeights(retrieval=1.0, semantic=0.25),
            retrieval_replay_fraction=1.0,
            gate=StageGate(min_train_recall_at_5=0.95, min_train_recall_at_10=0.99),
            notes=("Use masked multi-positive sigmoid, not CLIP softmax.", "One visual pair is encoded once for all captions."),
        ),
        StageSpec(
            stage=UniChangeStage.LOCAL_RETRIEVAL,
            datasets=("LEVIR-CC",),
            trainable_modules=("visual_projection", "attention_pool", "semantic_head", "local_projection"),
            frozen_modules=("universat", "jina_v5", "event_decoder", "directional_readout"),
            loss_weights=LossWeights(retrieval=1.0, semantic=0.25, local=0.25, distillation=0.10),
            retrieval_replay_fraction=1.0,
            gate=StageGate(min_train_recall_at_5=0.95, min_train_recall_at_10=0.99, max_retrieval_drop_points=2.0),
            notes=("Add smooth top-k token-region scoring only after global retrieval overfit passes.",),
        ),
        StageSpec(
            stage=UniChangeStage.WEAK_EVENTS,
            datasets=("LEVIR-CC",),
            trainable_modules=("event_decoder", "event_presence", "mask_projection"),
            frozen_modules=("universat", "jina_v5"),
            loss_weights=LossWeights(retrieval=1.0, semantic=0.25, local=0.25, event_overlap=0.05, distillation=0.10),
            retrieval_replay_fraction=1.0,
            gate=StageGate(max_retrieval_drop_points=2.0),
            notes=("Events are latent and qualitative here; do not claim supervised masks from LEVIR-CC.",),
        ),
        StageSpec(
            stage=UniChangeStage.SUPERVISED_MASKS,
            datasets=("LEVIR-CC", "LEVIR-MCI"),
            trainable_modules=("event_decoder", "event_presence", "mask_projection"),
            frozen_modules=("universat", "jina_v5"),
            loss_weights=LossWeights(
                retrieval=1.0,
                semantic=0.25,
                local=0.25,
                mask_bce=1.0,
                mask_dice=1.0,
                event_presence=0.5,
                event_coverage=0.5,
                event_overlap=0.05,
                distillation=0.10,
            ),
            retrieval_replay_fraction=0.5,
            gate=StageGate(max_retrieval_drop_points=2.0, requires_mask_metric_improvement=True),
            notes=("Never train masks alone; every mask step keeps retrieval replay or distillation.",),
        ),
        StageSpec(
            stage=UniChangeStage.DIRECTIONAL_READOUT,
            datasets=("LEVIR-CC", "LEVIR-MCI"),
            trainable_modules=("directional_readout", "event_decoder", "projection_heads"),
            frozen_modules=("universat_trunk", "jina_v5"),
            loss_weights=LossWeights(
                retrieval=1.0,
                semantic=0.25,
                local=0.25,
                mask_bce=1.0,
                mask_dice=1.0,
                event_presence=0.5,
                event_coverage=0.5,
                event_overlap=0.05,
                distillation=0.20,
            ),
            retrieval_replay_fraction=0.5,
            gate=StageGate(max_retrieval_drop_points=2.0, requires_mask_metric_improvement=True),
            notes=("Expose pre-collapse temporal states before using this stage.", "Relative time is ordinal before/after unless real metadata exists."),
        ),
        StageSpec(
            stage=UniChangeStage.PAIR_TO_PAIR,
            datasets=("LEVIR-CC", "LEVIR-MCI", "SECOND-CC", "Hi-UCD"),
            trainable_modules=("event_matcher", "transition_head"),
            frozen_modules=("universat", "jina_v5"),
            loss_weights=LossWeights(retrieval=1.0, local=0.25, event_coverage=0.5, distillation=0.20),
            retrieval_replay_fraction=0.4,
            gate=StageGate(max_retrieval_drop_points=2.0, requires_offline_index=True),
            notes=("Use event-set matching for pair-to-pair; LEVIR-CC pair-to-pair remains exploratory.",),
        ),
        StageSpec(
            stage=UniChangeStage.EXPLANATIONS,
            datasets=("grounded_event_records",),
            trainable_modules=("language_decoder",),
            frozen_modules=("universat", "jina_v5", "event_decoder", "retrieval_heads"),
            loss_weights=LossWeights(caption=1.0),
            retrieval_replay_fraction=0.0,
            gate=StageGate(requires_offline_index=True),
            notes=("Generate from structured event evidence, not raw image tokens.",),
        ),
    )


def stage_by_name(name: str) -> StageSpec:
    for stage in default_unichange_curriculum():
        if stage.stage.value == name:
            return stage
    raise ValueError(f"Unknown UniChange stage: {name}")
