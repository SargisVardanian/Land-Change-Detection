from __future__ import annotations

import torch

from land_change_detection.models.qcpr_v3_losses import foreground_preserving_resize, separated_query_mask_losses


def test_foreground_preserving_resize_keeps_single_small_positive() -> None:
    target = torch.zeros(1, 8, 8)
    target[0, 7, 7] = 1
    assert foreground_preserving_resize(target, (2, 2)).sum().item() == 1


def test_separated_mask_contract_reports_provenance_and_mismatch_negative() -> None:
    logits = torch.zeros(3, 4, 4, requires_grad=True)
    target = torch.zeros_like(logits)
    target[1, 1, 1] = 1
    values = separated_query_mask_losses(logits, target, ["binary_generic", "query_specific", "query_specific"], [None, "appeared", "disappeared"])
    assert values["generic_changed_count"] == 1
    assert values["query_specific_count"] == 2
    assert values["appeared_count"] == 1
    assert values["disappeared_count"] == 1
    values["total"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
