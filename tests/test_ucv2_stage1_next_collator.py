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


def test_detail_aware_caption_sampling_keeps_most_detailed_slot():
    item = SimpleNamespace(
        pair_id="detail-pair",
        t1=torch.zeros(3, 4, 4),
        t2=torch.ones(3, 4, 4),
        captions=[
            "change",
            "new buildings",
            "Three new buildings appeared in the upper left corner",
        ],
        mask=torch.zeros(4, 4),
    )
    collator = FrequencyBalancedCaptionCollator(
        {caption: 1 for caption in item.captions},
        max_captions_per_pair=2,
        frequency_power=0.5,
        seed=7,
        epoch=0,
        training=True,
    )
    captions = collator([item])["captions"]
    assert "Three new buildings appeared in the upper left corner" in captions
    assert len(captions) == 2


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


def test_pair_level_directional_masks_follow_selected_queries():
    appeared = torch.zeros(4, 4); appeared[0, 0] = 1
    disappeared = torch.zeros(4, 4); disappeared[-1, -1] = 1
    item = SimpleNamespace(
        pair_id="paired", dataset_name="s2looking",
        t1=torch.zeros(3, 4, 4), t2=torch.ones(3, 4, 4),
        captions=["new buildings appeared", "buildings disappeared"],
        mask=torch.maximum(appeared, disappeared),
        query_masks=[appeared, disappeared],
        query_change_types=["appeared", "disappeared"],
        metadata={"segmentation_supervision": True},
    )
    collator = FrequencyBalancedCaptionCollator(
        {}, max_captions_per_pair=None, frequency_power=0.0,
        seed=1, epoch=0, training=False,
    )
    batch = collator([item])
    assert batch["caption_to_pair"].tolist() == [0, 0]
    assert batch["query_change_types"] == ["appeared", "disappeared"]
    assert torch.equal(batch["query_masks"][0], appeared)
    assert torch.equal(batch["query_masks"][1], disappeared)
    assert batch["query_segmentation_supervision"].tolist() == [True, True]
