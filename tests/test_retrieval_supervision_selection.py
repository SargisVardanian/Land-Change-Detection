from __future__ import annotations

import pytest
import torch

from ucv2_stage1_next_core import retrieval_supervision_selection


def _batch(retrieval: list[bool], mapping: list[int]) -> dict[str, torch.Tensor]:
    return {
        "retrieval_supervision": torch.tensor(retrieval, dtype=torch.bool),
        "caption_to_pair": torch.tensor(mapping, dtype=torch.long),
    }


def _assert_device(result: dict[str, torch.Tensor], device: torch.device) -> None:
    assert all(value.device == device for value in result.values())


def test_all_retrieval_supervised_cpu_indices_are_compact_and_device_safe() -> None:
    result = retrieval_supervision_selection(_batch([True, True, True], [0, 1, 2, 0]), torch.device("cpu"))
    assert result["selected_pairs"].tolist() == [0, 1, 2]
    assert result["selected_queries"].tolist() == [0, 1, 2, 3]
    assert result["selected_mapping"].tolist() == [0, 1, 2, 0]
    _assert_device(result, torch.device("cpu"))


def test_mixed_supervision_uses_compact_pair_mapping() -> None:
    result = retrieval_supervision_selection(_batch([True, False, True], [0, 1, 2, 2, 0]), torch.device("cpu"))
    assert result["pair_mask"].tolist() == [True, False, True]
    assert result["caption_mask"].tolist() == [True, False, True, True, True]
    assert result["selected_pairs"].tolist() == [0, 2]
    assert result["selected_queries"].tolist() == [0, 2, 3, 4]
    assert result["selected_mapping"].tolist() == [0, 1, 1, 0]
    assert (result["selected_mapping"] >= 0).all()
    _assert_device(result, torch.device("cpu"))


def test_empty_retrieval_subset_returns_empty_device_safe_tensors() -> None:
    result = retrieval_supervision_selection(_batch([False, False], [0, 1]), torch.device("cpu"))
    assert result["selected_pairs"].numel() == 0
    assert result["selected_queries"].numel() == 0
    assert result["selected_mapping"].numel() == 0
    _assert_device(result, torch.device("cpu"))


@pytest.mark.parametrize("mapping", [[2], [-1]])
def test_out_of_range_caption_to_pair_is_rejected(mapping: list[int]) -> None:
    with pytest.raises(ValueError, match="caption_to_pair values"):
        retrieval_supervision_selection(_batch([True, True], mapping), torch.device("cpu"))


def test_rank_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="retrieval_supervision must be rank-1"):
        retrieval_supervision_selection({"retrieval_supervision": torch.ones(1, 1, dtype=torch.bool), "caption_to_pair": torch.tensor([0])}, torch.device("cpu"))
    with pytest.raises(ValueError, match="caption_to_pair must be rank-1"):
        retrieval_supervision_selection({"retrieval_supervision": torch.ones(1, dtype=torch.bool), "caption_to_pair": torch.tensor([[0]])}, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only regression: CPU collator indices must move before CUDA indexing")
def test_cpu_collator_indices_produce_cuda_selection() -> None:
    device = torch.device("cuda")
    result = retrieval_supervision_selection(_batch([True, False, True], [0, 1, 2, 0]), device)
    assert result["selected_pairs"].tolist() == [0, 2]
    assert result["selected_mapping"].tolist() == [0, 1, 0]
    _assert_device(result, device)
