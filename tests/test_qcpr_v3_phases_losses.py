from __future__ import annotations

import pytest
import torch

from land_change_detection.models.qcpr_v3 import QCPRV3Config, QCPRV3GenericGrounding
from land_change_detection.models.qcpr_v3_losses import query_mask_loss, query_mask_metrics
from land_change_detection.models.qcpr_v3_phases import PHASES, apply_phase_to_model, resolve_training_phase


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = torch.nn.Linear(2, 2)
        self.text_encoder = torch.nn.Linear(2, 2)
        self.temporal_encoder = torch.nn.Linear(2, 2)
        self.retrieval_head = torch.nn.Linear(2, 2)
        self.text_adapter = torch.nn.Linear(2, 2)
        self.grounder = QCPRV3GenericGrounding(QCPRV3Config(input_dim=8, text_dim=8, hidden_dim=8, heads=2, decoder_layers=1, scales=(2,), output_size=(8, 8)))


def test_training_phases_are_complete_and_overrides_rejected() -> None:
    assert set(PHASES) == {"global_recovery", "late_interaction", "mask_grounding", "region_slots", "joint_finetune"}
    with pytest.raises(ValueError, match="immutable"):
        resolve_training_phase("late_interaction", {"mask_loss": 1.0})
    with pytest.raises(ValueError, match="Unknown"):
        resolve_training_phase("object_specific")


def test_phase_enables_only_declared_modules() -> None:
    model = Wrapper()
    audit = apply_phase_to_model(model, resolve_training_phase("late_interaction"))
    assert audit["trainable_parameters"]
    assert all(name.startswith(("grounder.temporal_field", "grounder.grounding_decoder", "grounder.local_projection", "grounder.rerank_logits")) for name in audit["trainable_parameters"])


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
