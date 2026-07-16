from __future__ import annotations

import torch

from land_change_detection.models.qcpr_v3_losses import foreground_preserving_resize, query_mask_metrics, separated_query_mask_losses


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


def test_positive_predictions_are_never_reused_as_mismatch_negatives() -> None:
    logits = torch.full((2, 4, 4), 4.0, requires_grad=True)
    targets = torch.ones_like(logits)
    values = separated_query_mask_losses(
        logits, targets, ["query_specific", "query_specific"], ["appeared", "disappeared"]
    )
    assert values["verified_mismatch_count"] == 0
    assert values["mismatched_query_empty_loss"].item() == 0.0
    # appeared/disappeared are reported, but do not double-count the primary loss.
    torch.testing.assert_close(values["total"], values["query_specific_loss"])


def test_only_explicit_recomputed_mismatch_is_penalized() -> None:
    positives = torch.zeros(1, 4, 4, requires_grad=True)
    targets = torch.ones_like(positives)
    mismatched = torch.full((1, 4, 4), 3.0, requires_grad=True)
    values = separated_query_mask_losses(
        positives, targets, ["query_specific"], ["appeared"],
        verified_mismatched_logits=mismatched,
    )
    values["total"].backward()
    assert values["verified_mismatch_count"] == 1
    assert mismatched.grad is not None and torch.all(mismatched.grad > 0)


def test_sparse_positive_gradient_raises_foreground_and_lowers_background() -> None:
    from land_change_detection.models.qcpr_v3_losses import query_mask_loss
    logits = torch.zeros(1, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits); target[0, 3, 4] = 1
    query_mask_loss(logits, target).total.backward()
    assert logits.grad[0, 3, 4] < 0
    assert logits.grad[target == 0].mean() > 0


def test_generic_union_has_lower_primary_weight_than_verified_query_mask() -> None:
    generic = torch.zeros(1, 4, 4, requires_grad=True)
    query = generic.detach().clone().requires_grad_(True)
    target = torch.zeros_like(generic); target[0, 1, 1] = 1
    generic_loss = separated_query_mask_losses(generic, target, ["binary_generic"], [None])["total"]
    query_loss = separated_query_mask_losses(query, target, ["query_specific"], ["appeared"])["total"]
    torch.testing.assert_close(query_loss, 4.0 * generic_loss)


def test_scientific_mask_metrics_are_unsmoothed_and_report_pixel_average_precision() -> None:
    logits = torch.full((1, 2, 2), -8.0)
    targets = torch.zeros_like(logits); targets[0, 0, 0] = 1
    metrics = query_mask_metrics(logits, targets)
    assert metrics["nonempty_dice"] == 0.0
    assert metrics["nonempty_iou"] == 0.0
    assert metrics["nonempty_recall"] == 0.0
    assert 0.0 <= metrics["pixel_average_precision"] <= 1.0
    assert "pr_auc" not in metrics


def test_mask_objective_a_is_dice_plus_balanced_focal_with_separate_empty_bce() -> None:
    logits = torch.zeros(1, 4, 4, requires_grad=True)
    target = torch.zeros_like(logits); target[0, 1, 1] = 1
    value = separated_query_mask_losses(
        logits, target, ["query_specific"], ["appeared"], tversky_weight=0.0,
        empty_weight=0.25,
    )["total"]
    assert torch.isfinite(value)


def test_empty_mask_uses_quarter_weighted_bce_with_logits() -> None:
    from land_change_detection.models.qcpr_v3_losses import query_mask_loss
    logits = torch.zeros(1, 4, 4, requires_grad=True)
    target = torch.zeros_like(logits)
    loss = query_mask_loss(logits, target, dice_weight=1.0, focal_weight=1.0, tversky_weight=0.0)
    torch.testing.assert_close(loss.total, 0.25 * torch.log(torch.tensor(2.0)))
    loss.total.backward()
    assert torch.all(logits.grad > 0)


def test_localization_margin_excludes_empty_rows_from_background() -> None:
    logits = torch.tensor([
        [[4.0, -4.0], [-4.0, -4.0]],
        [[8.0, 8.0], [8.0, 8.0]],
    ])
    target = torch.zeros_like(logits); target[0, 0, 0] = 1
    metrics = query_mask_metrics(logits, target)
    assert metrics["localization_margin"] > 0.9
    assert metrics["empty_mean_probability"] > 0.99
