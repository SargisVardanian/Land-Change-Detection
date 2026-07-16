from __future__ import annotations

from train_qcpr_v3 import _micro_overfit_gate, _validation_direction


def _row(*, dice: float, iou: float, margin: float, empty: float, swap: float):
    return {
        "nonempty_soft_dice": dice,
        "nonempty_soft_iou": iou,
        "localization_margin": margin,
        "empty_mean_probability": empty,
        "soft_query_swap_iou_gap": swap,
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
        "train_soft_query_swap_gap_positive", "gradients_finite",
    }


def test_validation_is_separate_generalization_signal() -> None:
    history = [
        _row(dice=0.01, iou=0.005, margin=-0.01, empty=0.50, swap=0.0),
        _row(dice=0.011, iou=0.006, margin=-0.005, empty=0.49, swap=0.0),
    ]
    assert _validation_direction(history)["positive"]
