from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from land_change_detection.data.unichange_mci import _load_mask


def test_rgb_direction_masks_decode_red_and_blue_without_luminance_loss(tmp_path: Path) -> None:
    red = np.zeros((8, 8, 3), dtype=np.uint8); red[1, 2, 0] = 255
    blue = np.zeros((8, 8, 3), dtype=np.uint8); blue[5, 6, 2] = 255
    red_path, blue_path = tmp_path / "red.png", tmp_path / "blue.png"
    Image.fromarray(red).save(red_path); Image.fromarray(blue).save(blue_path)
    red_mask = _load_mask(red_path, None)
    blue_mask = _load_mask(blue_path, None)
    assert red_mask.sum().item() == 1 and red_mask[1, 2].item() == 1
    assert blue_mask.sum().item() == 1 and blue_mask[5, 6].item() == 1


def test_mask_loader_downsampling_preserves_small_foreground(tmp_path: Path) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8); image[31, 31, 2] = 255
    path = tmp_path / "small.png"; Image.fromarray(image).save(path)
    mask = _load_mask(path, 4)
    assert mask.shape == (4, 4)
    assert mask.sum().item() == 1
