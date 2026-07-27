from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from land_change_detection.models.retrieval_heads import (
    build_caption_positive_mask,
    multi_positive_symmetric_info_nce,
    stable_caption_group_ids,
)
from land_change_detection.models.retrieval_heads import RetrievalProjectionHead
from land_change_detection.models.unichange_v2_retrieval import UniChangeV2RetrievalModel


def test_duplicate_captions_share_positive_pairs():
    captions = ["There is no difference.", "there is no difference", "a new road appears"]
    groups = stable_caption_group_ids(captions)
    mapping = torch.tensor([0, 1, 2])
    positives = build_caption_positive_mask(mapping, pair_count=3, caption_group_ids=groups)

    assert positives.tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]


def test_loss_infers_duplicate_groups_from_identical_text_embeddings():
    pair_embeddings = torch.eye(3)
    text_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mapping = torch.tensor([0, 1, 2])
    explicit_groups = stable_caption_group_ids(
        ["no change", "no change", "new road"]
    )

    inferred = multi_positive_symmetric_info_nce(
        pair_embeddings,
        text_embeddings,
        mapping,
    )
    explicit = multi_positive_symmetric_info_nce(
        pair_embeddings,
        text_embeddings,
        mapping,
        explicit_groups,
    )

    assert torch.isfinite(inferred)
    assert torch.allclose(inferred, explicit, atol=1e-6)


class _VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(1, 1)


class _TemporalEncoder(nn.Module):
    pass


class _TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.last_role = None

    def forward(self, texts, role="document"):
        self.last_role = role
        return SimpleNamespace(global_embedding=torch.ones(len(texts), 512))


def test_text_to_pair_retrieval_uses_query_role():
    text_encoder = _TextEncoder()
    model = UniChangeV2RetrievalModel(
        visual_encoder=_VisualEncoder(),
        temporal_encoder=_TemporalEncoder(),
        text_encoder=text_encoder,
        retrieval_head=RetrievalProjectionHead(512),
    )

    embedding = model.encode_texts(["new buildings appeared"])

    assert embedding.shape == (1, 512)
    assert text_encoder.last_role == "query"
