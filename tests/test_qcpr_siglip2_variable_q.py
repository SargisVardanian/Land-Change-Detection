"""Scientific contracts for unique-query, variable-Q logical batches."""

from __future__ import annotations

import pytest
import torch

from qcpr_siglip2.data.loader import (
    make_unique_exact_batches,
    select_unique_epoch_rows,
)
from qcpr_siglip2.data.runtime import RawFeatureBatch, build_relevance_masks
from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel
from qcpr_siglip2.training.gradcache import logical_listwise_step
from qcpr_siglip2.training.optimizer import build_adamw
from qcpr_siglip2.training.objective import (
    pair_balanced_symmetric_multi_positive_listwise_loss,
    pair_balanced_text_to_pair_loss,
)


def _rows(counts: list[int]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for pair_index, count in enumerate(counts):
        for caption_index in range(count):
            result.append(
                {
                    "canonical_pair_id": f"p{pair_index}",
                    "caption_id": f"p{pair_index}q{caption_index}",
                }
            )
    return result


def _masks(query_pair_indices: list[int], pair_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    positive = torch.zeros((len(query_pair_indices), pair_count), dtype=torch.bool)
    for query_index, pair_index in enumerate(query_pair_indices):
        positive[query_index, pair_index] = True
    ignored = torch.zeros_like(positive)
    return positive, ignored


def test_unique_sampler_all_singletons_has_q_equal_p():
    batch = make_unique_exact_batches(
        _rows([1, 1, 1, 1]), physical_batch_size=4, max_unique_captions_per_pair=2
    )[0]
    assert len(batch.query_rows) == 4
    assert batch.query_pair_indices == (0, 1, 2, 3)
    assert len(set(batch.query_ids)) == 4


def test_unique_sampler_two_captions_per_pair_has_variable_q():
    batch = make_unique_exact_batches(
        _rows([2, 2, 2, 2]), physical_batch_size=4, max_unique_captions_per_pair=2
    )[0]
    assert len(batch.query_rows) == 8
    assert batch.query_pair_indices == (0, 0, 1, 1, 2, 2, 3, 3)
    assert len(set(batch.query_ids)) == 8


def test_unique_sampler_mixed_one_and_two_caption_pairs():
    batch = make_unique_exact_batches(
        _rows([1, 2, 1, 2]), physical_batch_size=4, max_unique_captions_per_pair=2
    )[0]
    assert len(batch.query_rows) == 6
    counts_by_pair = {
        batch.pair_rows[pair_index]["canonical_pair_id"]: batch.query_pair_indices.count(
            pair_index
        )
        for pair_index in range(len(batch.pair_rows))
    }
    assert sorted(counts_by_pair.values()) == [1, 1, 2, 2]
    assert len(batch.query_ids) == len(set(batch.query_ids))


def test_unique_sampler_caps_many_captions_without_duplicate_query_ids():
    selected = select_unique_epoch_rows(
        _rows([5]), max_unique_captions_per_pair=3, epoch=4, seed=19
    )
    assert len(selected) == 3
    assert len({row["caption_id"] for row in selected}) == 3


def test_duplicate_query_id_is_rejected_in_logical_batch():
    rows = [
        {"canonical_pair_id": "p0", "caption_id": "duplicate"},
        {"canonical_pair_id": "p0", "caption_id": "duplicate"},
    ]
    with pytest.raises(ValueError, match="duplicate caption_id"):
        select_unique_epoch_rows(rows, max_unique_captions_per_pair=2)


def test_pair_balanced_text_to_pair_equalizes_one_vs_two_caption_weight():
    one_scores = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    one_positive, one_ignored = _masks([0, 1], 2)
    two_scores = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    two_positive, two_ignored = _masks([0, 0, 1], 2)
    one = pair_balanced_text_to_pair_loss(
        one_scores, one_positive, one_ignored, torch.tensor([0, 1])
    )
    two = pair_balanced_text_to_pair_loss(
        two_scores, two_positive, two_ignored, torch.tensor([0, 0, 1])
    )
    assert torch.equal(one, two)


def test_pair_to_text_uses_one_singleton_positive_and_all_unique_positives():
    scores = torch.tensor([[3.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
    positive, ignored = _masks([0, 0, 1], 2)
    total, text_to_pair, pair_to_text = (
        pair_balanced_symmetric_multi_positive_listwise_loss(
            scores, positive, ignored, torch.tensor([0, 0, 1])
        )
    )
    assert torch.isfinite(total)
    assert torch.isfinite(text_to_pair)
    assert torch.isfinite(pair_to_text)
    assert int(positive[:, 0].sum()) == 2
    assert int(positive[:, 1].sum()) == 1


def test_cross_pair_cells_are_implicit_negatives_and_ignore_cells_stay_ignored():
    scores = torch.zeros(3, 2)
    positive, ignored = _masks([0, 0, 1], 2)
    ignored[0, 1] = True
    _, text_to_pair, _ = pair_balanced_symmetric_multi_positive_listwise_loss(
        scores, positive, ignored, torch.tensor([0, 0, 1])
    )
    assert not bool(positive[0, 1])
    assert bool(ignored[0, 1])
    assert torch.isfinite(text_to_pair)


def test_variable_q_objective_accepts_nonrectangular_query_count():
    scores = torch.tensor([[3.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
    positive, ignored = _masks([0, 0, 1], 2)
    loss, _, _ = pair_balanced_symmetric_multi_positive_listwise_loss(
        scores, positive, ignored, torch.tensor([0, 0, 1])
    )
    assert scores.shape == (3, 2)
    assert torch.isfinite(loss)


def test_deterministic_variable_q_selection_rotates_and_is_reproducible():
    rows = _rows([1, 2, 3])
    first = select_unique_epoch_rows(rows, max_unique_captions_per_pair=2, epoch=0, seed=7)
    same = select_unique_epoch_rows(rows, max_unique_captions_per_pair=2, epoch=0, seed=7)
    rotated = select_unique_epoch_rows(rows, max_unique_captions_per_pair=2, epoch=1, seed=7)
    assert [row["caption_id"] for row in first] == [row["caption_id"] for row in same]
    assert [row["caption_id"] for row in first] != [row["caption_id"] for row in rotated]
    assert len({row["caption_id"] for row in first}) == len(first)


def test_gradcache_recompute_contract_accepts_variable_q(monkeypatch):
    torch.manual_seed(23)
    features = (
        torch.randn(2, 2, 256, 768),
        torch.randn(2, 2, 768),
        torch.randn(3, 7, 768),
        torch.randn(3, 768),
        torch.ones(3, 7, dtype=torch.bool),
    )

    def fake_encode(_backbone, _processor, pair_rows, query_rows, _device, **_kwargs):
        pair_count = len(pair_rows)
        query_count = len(query_rows)
        return RawFeatureBatch(
            features[0][:pair_count].clone(),
            features[1][:pair_count].clone(),
            features[2][:query_count].clone(),
            features[3][:query_count].clone(),
            features[4][:query_count].clone(),
        )

    monkeypatch.setattr(
        "qcpr_siglip2.training.gradcache.encode_real_features", fake_encode
    )
    model = Siglip2TemporalRetrievalModel(
        None,
        Siglip2TemporalConfig(retrieval_score_mode="final_v1_primary"),
    )
    optimizer, _, scheduler = build_adamw(model, phase="A", total_steps=1)
    pairs = [
        {"canonical_pair_id": "p0"},
        {"canonical_pair_id": "p1"},
    ]
    queries = [
        {"caption_id": "q0", "positive_pair_ids": ["p0"]},
        {"caption_id": "q1", "positive_pair_ids": ["p0"]},
        {"caption_id": "q2", "positive_pair_ids": ["p1"]},
    ]
    positive, ignored, _ = build_relevance_masks(
        queries, pairs, torch.device("cpu")
    )
    result = logical_listwise_step(
        model,
        object(),
        object(),
        pairs,
        queries,
        positive,
        ignored,
        optimizer,
        device=torch.device("cpu"),
        physical_batch_size=1,
        query_pair_indices=[0, 0, 1],
        scheduler=scheduler,
        dtype=torch.float32,
    )
    assert result["score_shape"] == [3, 2]
    assert result["query_count"] == 3
    assert torch.isfinite(torch.tensor(result["loss"]))
