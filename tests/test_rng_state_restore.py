from __future__ import annotations

import torch

from land_change_detection.training.rng_state import restore_rng_state


def test_restore_cpu_rng_state_normalizes_to_cpu_byte_tensor(monkeypatch):
    captured = {}

    monkeypatch.setattr(torch, "set_rng_state", lambda value: captured.setdefault("cpu", value))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    restore_rng_state({"torch_cpu": [1, 2, 255]})

    restored = captured["cpu"]
    assert restored.device.type == "cpu"
    assert restored.dtype == torch.uint8
    assert restored.tolist() == [1, 2, 255]


def test_restore_cuda_rng_state_selects_current_device_and_normalizes(monkeypatch):
    captured = {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, device=None: captured.update(value=value, device=device),
    )

    restore_rng_state(
        {
            "torch_cuda": [
                torch.tensor([1, 2], dtype=torch.int64),
                torch.tensor([3, 4], dtype=torch.int16),
            ]
        }
    )

    assert captured["device"] == 1
    assert captured["value"].device.type == "cpu"
    assert captured["value"].dtype == torch.uint8
    assert captured["value"].tolist() == [3, 4]


def test_restore_rng_state_accepts_single_cuda_tensor(monkeypatch):
    captured = {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, device=None: captured.update(value=value, device=device),
    )

    restore_rng_state({"torch_cuda": torch.tensor([7, 8], dtype=torch.uint8)})

    assert captured["device"] == 0
    assert captured["value"].tolist() == [7, 8]
