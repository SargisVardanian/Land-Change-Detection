from __future__ import annotations

import pytest

from select_gradient_clip_threshold import select_threshold


def test_select_threshold_targets_requested_clipping_fraction() -> None:
    rows = [
        {"record_type": "train_step", "grad_norm_before_clip": float(index), "grad_norm_qcpr_interaction": index / 2}
        for index in range(1, 101)
    ]
    report = select_threshold(rows, target_clipping_fraction=0.2)
    assert report["step_count"] == 100
    assert report["resulting_clipping_fraction"] == pytest.approx(0.2)
    assert report["suggested_grad_clip_norm"] == pytest.approx(80.2)
    assert report["module_norms"]["qcpr_interaction"]["p90"] > 40


def test_select_threshold_rejects_missing_or_invalid_evidence() -> None:
    with pytest.raises(ValueError):
        select_threshold([], 0.2)
    with pytest.raises(ValueError):
        select_threshold([{"grad_norm_before_clip": 1.0}], 1.0)
