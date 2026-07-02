from __future__ import annotations

import torch
from torch.nn import functional as F

from land_change_detection.models.retrieval_heads import stable_caption_group_ids
from ucv2_retrieval_metrics import RetrievalCorpus, compute_retrieval_metrics


def test_extended_metrics_report_exact_frequency_and_mask_strata():
    pair_embeddings = F.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]),
        dim=-1,
    )
    text_embeddings = F.normalize(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        dim=-1,
    )
    corpus = RetrievalCorpus(
        pair_embeddings=pair_embeddings,
        text_embeddings=text_embeddings,
        caption_to_pair=torch.tensor([0, 2, 1]),
        caption_group_ids=stable_caption_group_ids(["same", "same", "A new building appeared"]),
        pair_ids=["a", "b", "c"],
        captions=["same", "same", "A new building appeared"],
        pair_mask_fractions=torch.tensor([0.0, 0.1, 0.005]),
        encode_seconds=1.0,
        peak_allocated_vram_bytes=0,
        peak_reserved_vram_bytes=0,
    )
    metrics, similarities = compute_retrieval_metrics(corpus)
    assert similarities.shape == (3, 3)
    assert metrics["text_to_pair_R@1"] == 1.0
    assert metrics["exact_pair_R@1"] < 1.0
    assert metrics["rare_caption_count"] == 2
    assert metrics["unique_caption_count"] == 1
    assert metrics["mask_no_change_count"] == 1
    assert metrics["mask_small_change_count"] == 1
    assert metrics["mask_large_change_count"] == 1
    assert metrics["appeared_count"] == 1
    assert metrics["changed_count"] == 1
    assert metrics["disappeared_empty"] is True
    assert metrics["text_to_pair_R@1"] <= metrics["text_to_pair_R@5"] <= metrics["text_to_pair_R@10"]
