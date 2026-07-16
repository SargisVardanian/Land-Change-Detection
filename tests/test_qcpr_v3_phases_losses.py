from __future__ import annotations

import pytest
import torch

from land_change_detection.models.qcpr_v3 import QCPRV3Config, QCPRV3GenericGrounding
from land_change_detection.models.qcpr_v3_losses import (
    duplicate_aware_positive_mask,
    multi_positive_contrastive_loss,
    query_mask_loss,
    query_mask_metrics,
)
from land_change_detection.models.qcpr_v3_phases import PHASES, apply_phase_to_model, resolve_training_phase


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = torch.nn.Linear(2, 2)
        self.text_encoder = torch.nn.Linear(2, 2)
        self.temporal_encoder = torch.nn.Linear(2, 2)
        self.retrieval_head = torch.nn.Module()
        self.retrieval_head.pair_projection = torch.nn.Linear(2, 2)
        self.retrieval_head.logit_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.text_adapter = torch.nn.Linear(2, 2)
        self.grounder = QCPRV3GenericGrounding(QCPRV3Config(input_dim=8, text_dim=8, hidden_dim=8, heads=2, decoder_layers=1, scales=(2,), output_size=(8, 8)))


def test_training_phases_are_complete_and_overrides_rejected() -> None:
    assert set(PHASES) == {"global_bootstrap", "global_recovery", "late_interaction", "mask_grounding", "mask_only_diagnostic", "region_slots", "joint_finetune"}
    with pytest.raises(ValueError, match="immutable"):
        resolve_training_phase("late_interaction", {"mask_loss": 1.0})
    with pytest.raises(ValueError, match="Unknown"):
        resolve_training_phase("object_specific")


def test_phase_enables_only_declared_modules() -> None:
    model = Wrapper()
    audit = apply_phase_to_model(model, resolve_training_phase("late_interaction"))
    assert audit["trainable_parameters"]
    assert all(name.startswith(("grounder.temporal_field", "grounder.grounding_decoder")) for name in audit["trainable_parameters"])
    assert not any("mask_head" in name for name in audit["trainable_parameters"])
    assert not any(name.startswith(("grounder.local_projection", "grounder.rerank_logits")) for name in audit["trainable_parameters"])


def test_phase_profiles_only_enable_parameters_connected_to_active_losses() -> None:
    model = Wrapper()
    global_audit = apply_phase_to_model(model, resolve_training_phase("global_bootstrap"))
    assert "retrieval_head.logit_scale" not in global_audit["trainable_parameters"]

    mask_audit = apply_phase_to_model(model, resolve_training_phase("mask_grounding"))
    assert "temporal_maps" not in resolve_training_phase("mask_grounding").active_losses
    assert any(name.startswith("grounder.local_projection") for name in mask_audit["trainable_parameters"])
    assert "grounder.rerank_logits" in mask_audit["trainable_parameters"]
    assert not any(name.startswith("grounder.temporal_map_head") for name in mask_audit["trainable_parameters"])

    mask_only = apply_phase_to_model(model, resolve_training_phase("mask_only_diagnostic"))
    assert resolve_training_phase("mask_only_diagnostic").active_losses == ("mask_supervised_only",)
    assert not any(name.startswith("grounder.local_projection") for name in mask_only["trainable_parameters"])
    assert "grounder.rerank_logits" not in mask_only["trainable_parameters"]


def test_mask_losses_separate_empty_and_nonempty() -> None:
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 3, 3] = 1
    losses = query_mask_loss(logits, target)
    losses.total.backward()
    assert torch.isfinite(logits.grad).all()
    metrics = query_mask_metrics(logits.detach(), target)
    assert metrics["nonempty_count"] == 1
    assert metrics["empty_count"] == 1
    assert metrics["empty_false_positive_area"] == pytest.approx(1.0)
    assert metrics["nonempty_dice"] == pytest.approx(2.0 / 65.0)
    assert metrics["nonempty_iou"] == pytest.approx(1.0 / 64.0)


def test_duplicate_aware_multi_positive_loss_does_not_penalize_equivalent_pairs() -> None:
    mapping = torch.tensor([0, 1, 2])
    groups = torch.tensor([7, 7, 9])
    positives = duplicate_aware_positive_mask(mapping, groups, pair_count=3)
    assert positives.tolist() == [[True, True, False], [True, True, False], [False, False, True]]
    scores = torch.tensor([[1.0, 0.9, -1.0], [0.8, 1.0, -1.0], [-1.0, -1.0, 1.0]], requires_grad=True)
    loss = multi_positive_contrastive_loss(scores, positives)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(scores.grad).all()


def test_global_bootstrap_has_no_historical_teacher_loss() -> None:
    profile = resolve_training_phase("global_bootstrap")
    assert "teacher_distillation" not in profile.active_losses
    assert profile.active_losses == ("global_multi_positive", "base_text_preservation")
