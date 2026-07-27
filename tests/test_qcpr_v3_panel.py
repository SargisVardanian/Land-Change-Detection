from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from train_qcpr_v3 import _save_panel


def test_grounding_panel_is_labeled_multiview_not_unlabeled_strip(tmp_path: Path) -> None:
    images = torch.rand(1, 2, 3, 32, 32)
    logits = torch.full((32, 32), -4.0); logits[8:16, 10:18] = 4.0
    target = torch.zeros(32, 32); target[9:15, 11:17] = 1.0
    output = tmp_path / "panel.png"
    _save_panel(output, images, logits, target, query="new buildings appeared", pair_id="s2looking:val:1:appeared")
    with Image.open(output) as panel:
        assert panel.width > panel.height
        assert panel.width >= 1600
        assert panel.height >= 800
