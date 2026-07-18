from __future__ import annotations

import copy

import torch

from land_change_detection.models.qcpr_v3 import (
    GenericCrossModalDecoder,
    MultiScaleQueryMaskDecoder,
    QCPRV3Config,
    QCPRV3GenericGrounding,
)
from land_change_detection.models.qcpr_v3_data import DirectionalM0BatchSampler
from land_change_detection.models.qcpr_v3_losses import (
    m0_aligned_symmetric_query_swap_loss,
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


def _toy_decoder() -> tuple[QCPRV3Config, GenericCrossModalDecoder, MultiScaleQueryMaskDecoder]:
    config = QCPRV3Config(
        input_dim=8,
        text_dim=8,
        global_text_dim=8,
        hidden_dim=8,
        heads=2,
        output_size=(8, 8),
        mask_decoder_dim=8,
    )
    return config, GenericCrossModalDecoder(config), MultiScaleQueryMaskDecoder(config)


def test_aligned_masks_equal_cartesian_diagonal() -> None:
    torch.manual_seed(7)
    _, decoder, masks = _toy_decoder()
    patches = torch.randn(4, 16, 8)
    tokens = torch.randn(4, 5, 8)
    attention = torch.ones(4, 5, dtype=torch.bool)
    query = torch.randn(4, 8)
    cartesian, cartesian_logits, _ = decoder(patches, tokens, attention, query)
    aligned, aligned_logits, _ = decoder.forward_aligned(patches, tokens, attention, query)
    diagonal = torch.arange(4)
    torch.testing.assert_close(aligned, cartesian[diagonal, diagonal])
    torch.testing.assert_close(aligned_logits, cartesian_logits[diagonal, diagonal])
    cartesian_masks = masks(cartesian_logits, cartesian, ((0, 16, 4),))
    aligned_masks = masks(
        aligned_logits[:, None], aligned[:, None], ((0, 16, 4),)
    )[:, 0]
    torch.testing.assert_close(aligned_masks, cartesian_masks[diagonal, diagonal])


def test_aligned_and_diagonal_cartesian_gradients_match() -> None:
    torch.manual_seed(11)
    _, first, first_masks = _toy_decoder()
    second = copy.deepcopy(first)
    second_masks = copy.deepcopy(first_masks)
    patches = torch.randn(3, 16, 8)
    tokens = torch.randn(3, 5, 8)
    attention = torch.ones(3, 5, dtype=torch.bool)
    query = torch.randn(3, 8)
    cartesian, logits, _ = first(patches, tokens, attention, query)
    diagonal = torch.arange(3)
    first_masks(logits, cartesian, ((0, 16, 4),))[diagonal, diagonal].mean().backward()
    aligned, aligned_logits, _ = second.forward_aligned(patches, tokens, attention, query)
    second_masks(
        aligned_logits[:, None], aligned[:, None], ((0, 16, 4),)
    )[:, 0].mean().backward()
    for left, right in zip(
        list(first.parameters()) + list(first_masks.parameters()),
        list(second.parameters()) + list(second_masks.parameters()),
        strict=True,
    ):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-5)


def test_physical_batch_32_decodes_exactly_64_aligned_masks() -> None:
    config = QCPRV3Config(
        input_dim=8,
        text_dim=8,
        global_text_dim=8,
        hidden_dim=8,
        heads=2,
        output_size=(8, 8),
        mask_decoder_dim=8,
    )
    model = QCPRV3GenericGrounding(config)
    mapping = torch.arange(32).repeat_interleave(2)
    output = model.score_aligned_query_pairs(
        torch.randn(64, 8),
        torch.randn(64, 5, 8),
        torch.ones(64, 5, dtype=torch.bool),
        torch.randn(32, 2, 16, 8),
        mapping,
    )
    assert output.decoded_mask_logits.shape == (64, 8, 8)
    assert output.memory_diagnostics["decoded_mask_count"] == 64
    assert output.memory_diagnostics["cartesian_mask_count"] == 0
    assert output.matched_temporal_descriptors.shape[:2] == (64, 16)
    torch.testing.assert_close(
        output.matched_temporal_descriptors[0],
        output.matched_temporal_descriptors[1],
    )


def test_aligned_microbatch_accumulation_matches_full_gradient() -> None:
    torch.manual_seed(17)
    _, full, _ = _toy_decoder()
    accumulated = copy.deepcopy(full)
    patches = torch.randn(8, 16, 8)
    tokens = torch.randn(8, 5, 8)
    attention = torch.ones(8, 5, dtype=torch.bool)
    query = torch.randn(8, 8)
    full.forward_aligned(patches, tokens, attention, query)[1].sum().div(8).backward()
    for start in range(0, 8, 2):
        accumulated.forward_aligned(
            patches[start:start + 2],
            tokens[start:start + 2],
            attention[start:start + 2],
            query[start:start + 2],
        )[1].sum().div(8).backward()
    for left, right in zip(full.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-5)


def test_aligned_query_swap_loss_uses_pair_shared_directional_queries() -> None:
    logits = torch.full((4, 4, 4), -5.0)
    targets = torch.zeros_like(logits)
    targets[0, :2, :2] = targets[1, 2:, 2:] = 1
    targets[2, 1:3, 1:3] = targets[3, :2, 2:] = 1
    logits.copy_(targets * 10 - 5)
    loss, count = m0_aligned_symmetric_query_swap_loss(
        logits,
        torch.tensor([0, 0, 1, 1]),
        ["appeared", "disappeared", "appeared", "disappeared"],
        targets,
    )
    assert count == 4
    assert torch.isfinite(loss)


def test_production_global_512_and_local_256_contract() -> None:
    config = QCPRV3Config(
        input_dim=512,
        text_dim=512,
        global_text_dim=512,
        hidden_dim=256,
    )
    model = QCPRV3GenericGrounding(config)
    assert model.global_query_projection.in_features == 512
    assert model.global_query_projection.out_features == 256
    assert model.temporal_field.output_norm.normalized_shape == (256,)
