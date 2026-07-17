from __future__ import annotations

import torch

from land_change_detection.models.qcpr_v3_data import DirectionalM0BatchSampler
from land_change_detection.models.qcpr_v3_losses import (
    m0_balanced_mask_loss,
    m0_symmetric_query_swap_loss,
)


def test_m0_sampler_is_pair_unique_and_70_30() -> None:
    rows = [
        {
            "pair_id": f"pair-{index}",
            "dataset_name": "s2looking",
            "quality_tier": "core",
            "directional_targets": [{"direction": "appeared"}, {"direction": "disappeared"}],
        }
        for index in range(20)
    ]
    sampler = DirectionalM0BatchSampler(rows, batch_size=10, seed=13)
    batches = list(sampler)
    assert batches == list(DirectionalM0BatchSampler(rows, batch_size=10, seed=13))
    for batch in batches:
        assert [mode for _, mode, _ in batch].count("positive") == 7
        assert [mode for _, mode, _ in batch].count("hard_background") == 3
        assert len({index for index, _, _ in batch}) == 10


def test_m0_loss_favors_calibrated_mask_and_handles_empty() -> None:
    target = torch.zeros(2, 8, 8)
    target[0, 2:6, 2:6] = 1
    good = torch.full_like(target, -7.0)
    good[0, 2:6, 2:6] = 7.0
    bad = torch.zeros_like(target)
    good_loss = m0_balanced_mask_loss(good, target)["total"]
    bad_loss = m0_balanced_mask_loss(bad, target)["total"]
    assert torch.isfinite(good_loss)
    assert good_loss < bad_loss


def test_m0_symmetric_query_swap_uses_direct_opposite_query_logits() -> None:
    mapping = torch.tensor([0, 0])
    query_indices = torch.tensor([0, 1])
    targets = torch.zeros(2, 4, 4)
    targets[0, :2, :2] = 1
    targets[1, 2:, 2:] = 1
    logits = torch.full((2, 1, 4, 4), -5.0)
    logits[0, 0, :2, :2] = 5.0
    logits[1, 0, 2:, 2:] = 5.0
    loss, count = m0_symmetric_query_swap_loss(
        logits, mapping, query_indices, ["appeared", "disappeared"], targets,
    )
    assert count == 2
    assert torch.isfinite(loss)
    assert loss < 0.02

