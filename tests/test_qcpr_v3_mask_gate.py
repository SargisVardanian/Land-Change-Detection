from __future__ import annotations

import torch

from train_qcpr_v3 import _direction_query_separation_loss, _micro_overfit_gate, _validation_direction


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
    }


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
    assert count == 2 and float(loss) == 0.0
    shared = logits[[0, 0]].clone()
    shared_loss, _ = _direction_query_separation_loss(
        shared, torch.tensor([0, 0]), torch.tensor([0, 1]),
        ["appeared", "disappeared"], targets,
    )
    assert float(shared_loss) > 0.0
