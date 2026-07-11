from __future__ import annotations

from pathlib import Path

import torch

from land_change_detection.models.qcpr import QCPRPatchReranker, temporal_patch_descriptor
from ucv2_retrieval_metrics import RetrievalCorpus, compute_retrieval_ranks


def test_candidates_receive_distinct_masks_and_mask_weights_define_local_score() -> None:
    torch.manual_seed(7)
    reranker = QCPRPatchReranker(hidden_dim=4, retrieval_dim=4, beta=0.25)
    query = torch.randn(1, 4)
    pairs = torch.randn(2, 4)
    candidate_changes = torch.randn(2, 4, 4)
    patches = reranker.project_patches(candidate_changes)
    output = reranker.score(query, pairs, patches)
    logits = output["query_mask_logits"]
    assert not torch.allclose(logits[0, 0], logits[0, 1])
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


def test_renderer_contract_uses_candidate_tokens_and_shows_both_times() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "render_unichange_v2_retrieval.py").read_text()
    assert "corpus.patch_tokens[int(retrieved_index)]" in source
    assert "corpus.patch_tokens[query_pair_index]" not in source
    assert 'set_title("Soft mask over T1")' in source
    assert 'set_title("Soft mask over T2")' in source


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
