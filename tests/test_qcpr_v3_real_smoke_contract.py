from __future__ import annotations

import torch

from run_qcpr_v3_real_integration_smoke import remove_flat_token_indices


def test_evidence_deletion_uses_temporal_spatial_token_axis() -> None:
    mask = torch.ones((1, 2, 4), dtype=torch.bool)
    result = remove_flat_token_indices(mask, torch.tensor([0, 4, 7]))
    assert not bool(result[0, 0, 0])
    assert not bool(result[0, 1, 0])
    assert not bool(result[0, 1, 3])
    assert int(result.sum()) == 5
