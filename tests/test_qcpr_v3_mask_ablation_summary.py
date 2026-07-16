from __future__ import annotations

import json

from summarize_qcpr_v3_mask_ablation import summarize


def _report(path, *, passed: bool, dice: float):
    path.mkdir(parents=True)
    payload = {
        "runtime_status": "PASS", "gradient_status": "PASS",
        "scientific_status": "MICRO_OVERFIT_PASS" if passed else "SCIENTIFIC_HOLD",
        "micro_overfit_gate": {"passed": passed},
        "fixed_validation_history": [{}, {
            "nonempty_dice": dice, "nonempty_iou": dice / 2,
            "nonempty_precision": 0.5, "foreground_background_margin": 0.1,
            "empty_false_positive_area": 0.0,
        }],
    }
    (path / "smoke_report.json").write_text(json.dumps(payload))


def test_ablation_selection_never_uses_training_loss(tmp_path):
    _report(tmp_path / "A", passed=True, dice=0.2)
    _report(tmp_path / "B", passed=False, dice=0.9)
    _report(tmp_path / "C", passed=True, dice=0.3)
    result = summarize(tmp_path)
    assert result["status"] == "MICRO_OVERFIT_PASS"
    assert result["selected_objective"] == "C"
