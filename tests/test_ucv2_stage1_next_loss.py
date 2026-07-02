from __future__ import annotations

import torch

from land_change_detection.models.retrieval_heads import (
    RetrievalProjectionHead,
    build_caption_positive_mask,
    multi_positive_set_info_nce,
    multi_positive_symmetric_info_nce,
    stable_caption_group_ids,
)


def test_stable_caption_groups_ignore_case_and_punctuation():
    groups = stable_caption_group_ids(
        ["No change has occurred.", "no change has occurred", "A building appeared"]
    )
    assert groups[0].item() == groups[1].item()
    assert groups[0].item() != groups[2].item()


def test_positive_mask_expands_normalized_duplicate_groups():
    mapping = torch.tensor([0, 1, 1])
    groups = stable_caption_group_ids(["same.", "Same", "different"])
    mask = build_caption_positive_mask(mapping, pair_count=2, caption_group_ids=groups)
    assert mask.tolist() == [
        [True, True, False],
        [True, True, True],
    ]


def test_set_mass_loss_removes_positive_count_floor():
    pair = torch.eye(2)
    text = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    mapping = torch.tensor([0, 0, 1, 1])
    groups = stable_caption_group_ids(["p0a", "p0b", "p1a", "p1b"])
    old = multi_positive_symmetric_info_nce(
        pair,
        text,
        mapping,
        groups,
        temperature=0.01,
    )
    new = multi_positive_set_info_nce(
        pair,
        text,
        mapping,
        groups,
        temperature=0.01,
        text_to_pair_weight=0.5,
        pair_to_text_weight=0.5,
    )
    assert new.item() < 1e-4
    assert old.item() > 0.3


def test_trainable_temperature_is_bounded_and_differentiable():
    head = RetrievalProjectionHead(
        4,
        trainable_temperature=True,
        initial_temperature=0.07,
        max_logit_scale=20.0,
    )
    scale = head.similarity_scale()
    assert scale.requires_grad
    assert 14.0 < scale.item() < 15.0
    head.logit_scale.data.fill_(100.0)
    assert head.similarity_scale().item() <= 20.0001


def test_fixed_temperature_keeps_legacy_projection_state_dict_shape():
    head = RetrievalProjectionHead(4, trainable_temperature=False)
    assert "logit_scale" not in head.state_dict()
