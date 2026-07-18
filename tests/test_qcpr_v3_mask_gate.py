from __future__ import annotations

import torch

from train_qcpr_v3 import (
    _direction_query_separation_loss,
    _frozen_parameter_fingerprint,
    _micro_overfit_gate,
    _validation_direction,
)


def _row(*, dice: float, iou: float, margin: float, empty: float, swap: float):
    return {
        "nonempty_soft_dice": dice,
        "nonempty_soft_iou": iou,
        "localization_margin": margin,
        "empty_mean_probability": empty,
        "soft_query_swap_iou_gap": swap,
        "query_swap_count": 2,
        "appeared_soft_query_swap_iou_gap": swap,
        "disappeared_soft_query_swap_iou_gap": swap,
        "predicted_area": 0.5,
    }


def test_frozen_fingerprint_supports_bfloat16_and_is_value_sensitive() -> None:
    model = torch.nn.Module()
    model.frozen = torch.nn.Linear(3, 2, bias=False).to(torch.bfloat16)
    first = _frozen_parameter_fingerprint(model, ("frozen",))
    assert first == _frozen_parameter_fingerprint(model, ("frozen",))
    with torch.no_grad():
        model.frozen.weight[0, 0] += 1
    assert first != _frozen_parameter_fingerprint(model, ("frozen",))


def test_micro_gate_uses_fixed_train_soft_metrics_without_arbitrary_hard_thresholds() -> None:
    history = [
        _row(dice=0.01, iou=0.005, margin=-0.01, empty=0.50, swap=-0.001),
        _row(dice=0.02, iou=0.010, margin=0.00, empty=0.49, swap=0.001),
    ]
    gate = _micro_overfit_gate(history, gradients_finite=True)
    assert gate["passed"]
    assert set(gate["checks"]) == {
        "train_soft_dice_increased", "train_soft_iou_increased",
        "train_localization_margin_increased", "train_empty_mean_probability_not_increased",
        "train_soft_query_swap_gap_positive", "train_appeared_soft_query_swap_gap_positive",
        "train_disappeared_soft_query_swap_gap_positive", "gradients_finite",
        "train_mask_not_empty_or_full",
    }


def test_validation_is_separate_generalization_signal() -> None:
    history = [
        _row(dice=0.01, iou=0.005, margin=-0.01, empty=0.50, swap=-0.001),
        _row(dice=0.011, iou=0.006, margin=-0.005, empty=0.49, swap=0.001),
    ]
    assert _validation_direction(history)["positive"]


def test_validation_rejects_query_independent_mask_improvement() -> None:
    history = [
        _row(dice=0.01, iou=0.005, margin=-0.01, empty=0.50, swap=-0.001),
        _row(dice=0.02, iou=0.01, margin=0.01, empty=0.49, swap=-0.0001),
    ]
    gate = _validation_direction(history)
    assert not gate["positive"]
    assert not gate["checks"]["validation_soft_query_swap_gap_positive"]


def test_direction_gates_reject_all_foreground_predictions() -> None:
    first = _row(dice=0.01, iou=0.005, margin=-0.01, empty=0.0, swap=-0.001)
    last = _row(dice=0.02, iou=0.010, margin=0.01, empty=0.0, swap=0.001)
    last["predicted_area"] = 1.0
    assert not _micro_overfit_gate([first, last], gradients_finite=True)["passed"]
    assert not _validation_direction([first, last])["positive"]


def test_direction_separation_compares_opposite_query_on_same_target() -> None:
    target_appeared = torch.zeros(4, 4); target_appeared[:2, :2] = 1
    target_disappeared = torch.zeros(4, 4); target_disappeared[2:, 2:] = 1
    targets = torch.stack((target_appeared, target_disappeared))
    logits = torch.full((2, 1, 4, 4), -4.0)
    logits[0, 0, :2, :2] = 4.0
    logits[1, 0, 2:, 2:] = 4.0
    loss, count = _direction_query_separation_loss(
        logits, torch.tensor([0, 0]), torch.tensor([0, 1]),
        ["appeared", "disappeared"], targets,
    )
    assert count == 2
    shared = logits[[0, 0]].clone()
    shared_loss, _ = _direction_query_separation_loss(
        shared, torch.tensor([0, 0]), torch.tensor([0, 1]),
        ["appeared", "disappeared"], targets,
    )
    assert float(loss) < float(shared_loss)


def test_direction_separation_optimizes_soft_query_swap_gap() -> None:
    target_appeared = torch.zeros(4, 4); target_appeared[:2, :2] = 1
    target_disappeared = torch.zeros(4, 4); target_disappeared[2:, 2:] = 1
    targets = torch.stack((target_appeared, target_disappeared))
    logits = torch.zeros((2, 1, 4, 4), requires_grad=True)
    loss, count = _direction_query_separation_loss(
        logits, torch.tensor([0, 0]), torch.tensor([0, 1]),
        ["appeared", "disappeared"], targets,
    )
    assert count == 2
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, :2, :2].mean() < 0
    assert logits.grad[0, 0, 2:, 2:].mean() > 0


def test_direction_separation_rejects_nonpositive_temperature() -> None:
    target = torch.zeros(2, 4, 4)
    target[0, :2, :2] = 1
    target[1, 2:, 2:] = 1
    try:
        _direction_query_separation_loss(
            torch.zeros(2, 1, 4, 4),
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
            ["appeared", "disappeared"],
            target,
            temperature=0.0,
        )
    except ValueError as error:
        assert "temperature" in str(error)
    else:
        raise AssertionError("nonpositive contrastive temperature must be rejected")
