from __future__ import annotations

import torch
from torch import nn

from land_change_detection.backbones.universat_frame_backend import UniverSatFrameBackend


class EncodeOnlyUniverSat(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.payload = None
        self.kwargs = None

    def encode(self, payload, **kwargs):
        self.payload = payload
        self.kwargs = kwargs
        batch = payload["spot"].shape[0]
        grid = kwargs["output_grid"]
        return torch.zeros(batch, grid * grid, 768), {"ok": True}


def test_frame_backend_uses_encode_without_dates():
    model = EncodeOnlyUniverSat()
    backend = UniverSatFrameBackend(model, output_grid=8)
    output = backend(torch.rand(6, 3, 32, 32))
    assert output.shape == (6, 64, 768)
    assert set(model.payload) == {"spot"}
    assert model.payload["spot"].shape == (6, 3, 32, 32)
    assert model.kwargs == {"patch_size": 10.0, "output_grid": 8}
    assert model.weight.requires_grad is False
