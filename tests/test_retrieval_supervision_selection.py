from __future__ import annotations

import pytest
import torch

from ucv2_stage1_next_core import retrieval_supervision_selection
from land_change_detection.training.runtime_device import assert_runtime_tensor_devices, resolve_runtime_device


def _batch(retrieval: list[bool], mapping: list[int]) -> dict[str, torch.Tensor]:
    return {
        "retrieval_supervision": torch.tensor(retrieval, dtype=torch.bool),
        "caption_to_pair": torch.tensor(mapping, dtype=torch.long),
    }


def _assert_device(result: dict[str, torch.Tensor], device: torch.device) -> None:
    assert all(value.device == resolve_runtime_device(device) for value in result.values())


def test_cpu_resolution_never_calls_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("CUDA API called")))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: (_ for _ in ()).throw(AssertionError("CUDA API called")))
    assert resolve_runtime_device("cpu") == torch.device("cpu")


def test_explicit_cuda_index_is_preserved_without_cuda_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("CUDA probe not expected")))
    assert resolve_runtime_device("cuda:0") == torch.device("cuda:0")


def test_generic_cuda_alias_resolves_to_current_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert resolve_runtime_device("cuda") == torch.device("cuda:3")


def test_generic_cuda_alias_without_cuda_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA device was requested, but CUDA is unavailable"):
        resolve_runtime_device("cuda")


def test_explicit_device_mismatch_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="expected cuda:0"):
        assert_runtime_tensor_devices({"selection": torch.empty(0)}, "cuda:0")


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
