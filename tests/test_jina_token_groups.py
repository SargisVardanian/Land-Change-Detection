from __future__ import annotations

import torch

from land_change_detection.backbones.jina_v5_text import _content_token_mask, _token_group_metadata


class _Tokenizer:
    vocabulary = ["Query", ":", "two", "houses", "appeared", "at", "the", "top", "replacing", "trees", "2", "disappeared", "demolished", "new"]
    all_special_ids = ()

    def convert_ids_to_tokens(self, ids):
        return [self.vocabulary[index] for index in ids]


def test_jina_token_metadata_builds_attribute_groups_and_targets() -> None:
    ids = torch.arange(10).reshape(1, 10)
    attention = torch.ones_like(ids)
    metadata = _token_group_metadata(
        _Tokenizer(), ids, attention, ["two houses appeared at the top replacing trees"]
    )
    groups = metadata["token_group_masks"]
    assert groups["object"].sum().item() == 2
    assert groups["direction"].sum().item() == 1
    assert groups["location"].sum().item() == 1
    assert groups["count"].sum().item() == 1
    assert groups["relation"].sum().item() == 1
    assert metadata["direction_targets"].tolist() == [1]
    assert metadata["location_targets"].tolist() == [[0.5, 0.0]]
    assert metadata["count_targets"].tolist() == [2.0]

    numeric = _token_group_metadata(_Tokenizer(), torch.tensor([[10]]), torch.ones(1, 1, dtype=torch.long), ["2 houses appeared"])
    assert numeric["token_group_masks"]["count"].item() is True
    assert numeric["count_targets"].tolist() == [2.0]


def test_direction_words_are_content_tokens_and_map_to_opposite_targets() -> None:
    tokenizer = _Tokenizer()
    ids = torch.tensor([[4, 13], [11, 12]])
    attention = torch.ones_like(ids)
    content = _content_token_mask(tokenizer, ids, attention)
    assert content.all()
    metadata = _token_group_metadata(tokenizer, ids, attention, ["new buildings appeared", "buildings were demolished"])
    assert metadata["direction_targets"].tolist() == [1, -1]
    assert metadata["token_group_masks"]["direction"].sum(dim=1).tolist() == [1, 2]
