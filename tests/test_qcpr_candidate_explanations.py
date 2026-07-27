from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from land_change_detection.models.qcpr import QCPRPatchReranker, temporal_patch_descriptor
from render_unichange_v2_retrieval import _candidate_mask, _result_logits, _top_regions
from ucv2_retrieval_metrics import RetrievalCorpus, compute_retrieval_ranks


def test_candidates_receive_distinct_masks_and_mask_weights_define_local_score() -> None:
    torch.manual_seed(7)
    reranker = QCPRPatchReranker(hidden_dim=4, retrieval_dim=4, beta=0.25, architecture_version="v2")
    query = torch.randn(1, 4)
    pairs = torch.randn(2, 4)
    per_time = torch.randn(2, 2, 4, 4)
    tokens = torch.randn(1, 3, 4)
    output = reranker.score_v2(query, tokens, torch.ones(1, 3, dtype=torch.bool), pairs, per_time)
    logits = output["query_mask_logits"]
    assert not torch.allclose(logits[0, 0], logits[0, 1])
    patches = output["patch_tokens"]
    pooled = reranker.masked_local_embedding(patches, logits)
    expected = torch.einsum("qd,qbd->qb", torch.nn.functional.normalize(query, dim=-1), pooled)
    assert torch.allclose(output["local_score"], expected)


def test_temporal_descriptor_reversal_swaps_signed_direction() -> None:
    before = torch.zeros(1, 4, 1)
    after = torch.ones(1, 4, 1)
    forward = temporal_patch_descriptor(torch.stack((before, after), dim=1))
    reverse = temporal_patch_descriptor(torch.stack((after, before), dim=1))
    # For D=[V1,V2,V2-V1,abs(...),xy], channel 2 is the signed temporal evidence.
    assert torch.all(forward[..., 2] > 0)
    assert torch.all(reverse[..., 2] < 0)
    appeared_forward = torch.sigmoid(forward[..., 2])
    appeared_reverse = torch.sigmoid(reverse[..., 2])
    disappeared_forward = torch.sigmoid(-forward[..., 2])
    disappeared_reverse = torch.sigmoid(-reverse[..., 2])
    assert torch.all(appeared_forward > appeared_reverse)
    assert torch.all(disappeared_forward < disappeared_reverse)


def test_candidate_renderer_uses_candidate_logits_functionally() -> None:
    reranker = QCPRPatchReranker(hidden_dim=4, retrieval_dim=4)
    queries = torch.randn(1, 4)
    patches = reranker.project_patches(torch.randn(2, 4, 4))
    mask_queries = torch.nn.functional.normalize(reranker.query_mask_head(queries), dim=-1) / reranker.logit_scale
    corpus = RetrievalCorpus(pair_embeddings=torch.randn(2, 4), text_embeddings=queries, caption_to_pair=torch.tensor([0]), caption_group_ids=torch.tensor([0]), pair_ids=["a", "b"], captions=["query"], pair_mask_fractions=torch.zeros(2), encode_seconds=0.0, peak_allocated_vram_bytes=0, peak_reserved_vram_bytes=0, patch_tokens=patches, mask_query_embeddings=mask_queries, qcpr_beta=.25)
    logits_a = _result_logits(corpus, 0, 0)
    logits_b = _result_logits(corpus, 0, 1)
    assert not torch.allclose(logits_a, logits_b)
    soft = _candidate_mask(logits_b, (16, 16))
    assert soft.shape == (16, 16)
    assert np.isfinite(soft).all()
    assert isinstance(_top_regions(soft, .5), list)
    assert torch.allclose(logits_b.sigmoid().amax(), torch.tensor(float(soft.max())), atol=.1)


def test_v2_candidate_renderer_reproduces_changed_channel_mask_prior() -> None:
    torch.manual_seed(13)
    reranker = QCPRPatchReranker(hidden_dim=4, retrieval_dim=4, architecture_version="v2")
    query = torch.randn(1, 4)
    tokens = torch.randn(1, 3, 4)
    pairs = torch.randn(2, 4)
    per_time = torch.randn(2, 2, 4, 4)
    with torch.no_grad():
        for parameter in reranker.interaction_mlp.parameters():
            parameter.zero_()
        for parameter in reranker.temporal_channel_head.parameters():
            parameter.zero_()
        reranker.temporal_channel_head[-1].bias[0] = 1.5
    output = reranker.score_v2(query, tokens, torch.ones(1, 3, dtype=torch.bool), pairs, per_time)
    corpus = RetrievalCorpus(
        pair_embeddings=pairs, text_embeddings=torch.nn.functional.normalize(query, dim=-1),
        caption_to_pair=torch.tensor([0]), caption_group_ids=torch.tensor([0]),
        pair_ids=["a", "b"], captions=["query"], pair_mask_fractions=torch.zeros(2),
        encode_seconds=0.0, peak_allocated_vram_bytes=0, peak_reserved_vram_bytes=0,
        patch_tokens=output["patch_tokens"].detach(), qcpr_architecture_version="v2",
        text_token_embeddings=tokens, text_attention_mask=torch.ones(1, 3, dtype=torch.bool),
        temporal_explanation_logits=output["temporal_explanation_logits"].detach(), qcpr_reranker=reranker,
    )
    displayed = _result_logits(corpus, 0, 1)
    assert torch.allclose(displayed, output["query_mask_logits"][0, 1], atol=1e-6)


def test_chunked_and_unchunked_qcpr_rankings_match() -> None:
    torch.manual_seed(11)
    reranker = QCPRPatchReranker(hidden_dim=4, retrieval_dim=4)
    queries = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    pairs = torch.nn.functional.normalize(torch.randn(6, 4), dim=-1)
    patches = reranker.project_patches(torch.randn(6, 4, 4))
    mask_queries = torch.nn.functional.normalize(reranker.query_mask_head(queries), dim=-1) / reranker.logit_scale
    corpus = RetrievalCorpus(
        pair_embeddings=pairs, text_embeddings=queries,
        caption_to_pair=torch.tensor([0, 1, 2, 3, 4]), caption_group_ids=torch.arange(5),
        pair_ids=[f"p{i}" for i in range(6)], captions=[f"q{i}" for i in range(5)],
        pair_mask_fractions=torch.zeros(6), encode_seconds=0.0,
        peak_allocated_vram_bytes=0, peak_reserved_vram_bytes=0,
        patch_tokens=patches, mask_query_embeddings=mask_queries, qcpr_alpha=1.0, qcpr_beta=.25,
    )
    full = compute_retrieval_ranks(corpus)
    chunked = compute_retrieval_ranks(corpus, query_chunk_size=2, candidate_chunk_size=3)
    assert torch.allclose(full.similarities, chunked.similarities, atol=1e-6, rtol=1e-6)
    assert torch.equal(full.ranked_candidate_indices, chunked.ranked_candidate_indices)
