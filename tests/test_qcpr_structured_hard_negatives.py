from __future__ import annotations

import torch

from land_change_detection.models.qcpr import (
    conditional_instance_discrimination_loss,
    local_positive_negative_margin_loss,
    structured_hard_negative_masks,
    structured_auxiliary_evidence_loss,
)


CAPTIONS = [
    "two houses appeared at the top",
    "two houses disappeared at the top",
    "two houses appeared at the bottom",
    "one house appeared at the top",
    "two roads appeared at the top",
    "no change to the houses",
]
MAPPING = torch.arange(len(CAPTIONS))


def test_wrong_direction_hard_negative_is_detected() -> None:
    masks = structured_hard_negative_masks(CAPTIONS, MAPPING, len(CAPTIONS))
    assert masks["same_object_wrong_direction"][0, 1]


def test_wrong_location_hard_negative_is_detected() -> None:
    masks = structured_hard_negative_masks(CAPTIONS, MAPPING, len(CAPTIONS))
    assert masks["same_object_direction_wrong_location"][0, 2]


def test_wrong_count_hard_negative_is_detected() -> None:
    masks = structured_hard_negative_masks(CAPTIONS, MAPPING, len(CAPTIONS))
    assert masks["same_object_location_wrong_count"][0, 3]


def test_latent_positive_is_excluded_from_local_margin_loss() -> None:
    scores = torch.zeros(len(CAPTIONS), len(CAPTIONS), requires_grad=True)
    latent = torch.zeros_like(scores, dtype=torch.bool)
    latent[0, 1] = True
    _, diagnostics = local_positive_negative_margin_loss(scores, MAPPING, CAPTIONS, latent_positive_mask=latent)
    assert diagnostics["hard_negative_same_object_wrong_direction_count"] < int(
        structured_hard_negative_masks(CAPTIONS, MAPPING, len(CAPTIONS))["same_object_wrong_direction"].sum()
    )


def test_synthetic_optimization_step_improves_local_positive_over_hard_negative() -> None:
    scores = torch.nn.Parameter(torch.zeros(len(CAPTIONS), len(CAPTIONS)))
    optimizer = torch.optim.SGD([scores], lr=1.0)
    before = float((scores[0, 0] - scores[0, 1]).detach())
    loss, diagnostics = local_positive_negative_margin_loss(scores, MAPPING, CAPTIONS, margin=0.2)
    assert diagnostics["local_margin_valid_queries"] > 0
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = float((scores[0, 0] - scores[0, 1]).detach())
    assert after > before


def test_conditional_identity_loss_only_uses_broad_semantic_group() -> None:
    scores = torch.tensor([[0.2, 0.5, 0.9]], requires_grad=True)
    mapping = torch.tensor([0])
    broad = torch.tensor([[True, True, False]])
    loss = conditional_instance_discrimination_loss(scores, mapping, broad, margin=0.1)
    loss.backward()
    assert loss.item() > 0
    assert scores.grad[0, 2].item() == 0.0
    assert scores.grad[0, 0].item() < 0
    assert scores.grad[0, 1].item() > 0


def test_auxiliary_evidence_losses_only_use_available_attribute_labels() -> None:
    captions = ["two houses appeared at the top", "change occurred"]
    scores = {name: torch.full((2, 2), 0.75, requires_grad=True) for name in ("S_object", "S_direction", "S_location", "S_count", "S_relation")}
    loss, diagnostics = structured_auxiliary_evidence_loss(scores, torch.tensor([0, 1]), captions)
    assert torch.isfinite(loss)
    assert diagnostics["object_auxiliary_label_count"] == 1
    assert diagnostics["direction_auxiliary_label_count"] == 2
    assert diagnostics["location_auxiliary_label_count"] == 1
    assert diagnostics["count_auxiliary_label_count"] == 1
    assert diagnostics["relation_auxiliary_label_count"] == 0
