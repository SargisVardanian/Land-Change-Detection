from __future__ import annotations

from types import SimpleNamespace

import torch

from ucv2_stage1_next_core import FrequencyBalancedCaptionCollator


def test_frequency_balanced_collator_is_deterministic_per_epoch_and_pair():
    items = [
        SimpleNamespace(
            pair_id=f"p{index}",
            t1=torch.zeros(3, 4, 4),
            t2=torch.ones(3, 4, 4),
            captions=["common", "rare", "other"],
            mask=torch.zeros(4, 4),
        )
        for index in range(2)
    ]
    collator = FrequencyBalancedCaptionCollator(
        {"common": 100, "rare": 1, "other": 2},
        max_captions_per_pair=2,
        frequency_power=0.5,
        seed=11,
        epoch=3,
        training=True,
    )
    first = collator(items)
    second = collator(items)
    assert first["captions"] == second["captions"]
    assert first["images"].shape == (2, 2, 3, 4, 4)
    assert len(first["captions"]) == 4
    assert first["mask_fractions"].tolist() == [0.0, 0.0]


def test_caption_sampling_rotates_to_cover_all_captions():
    item = SimpleNamespace(
        pair_id="stable-pair",
        t1=torch.zeros(3, 4, 4),
        t2=torch.ones(3, 4, 4),
        captions=[f"caption {index}" for index in range(5)],
        mask=torch.zeros(4, 4),
    )
    seen: set[str] = set()
    for epoch in range(5):
        collator = FrequencyBalancedCaptionCollator(
            {caption: 1 for caption in item.captions},
            max_captions_per_pair=2,
            frequency_power=0.5,
            seed=11,
            epoch=epoch,
            training=True,
        )
        seen.update(collator([item])["captions"])
    assert seen == set(item.captions)


def test_validation_collator_keeps_all_captions():
    item = SimpleNamespace(
        pair_id="p",
        t1=torch.zeros(3, 4, 4),
        t2=torch.ones(3, 4, 4),
        captions=["a", "b", "c"],
        mask=torch.zeros(4, 4),
    )
    collator = FrequencyBalancedCaptionCollator({}, max_captions_per_pair=1, frequency_power=1.0, seed=1, epoch=0, training=False)
    assert collator([item])["captions"] == ["a", "b", "c"]
