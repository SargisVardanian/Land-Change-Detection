from __future__ import annotations

import numpy as np
import pytest

from land_change_detection.visualization import mask_rgba_overlay, rgb_absolute_difference


def test_empty_mask_overlay_is_fully_transparent():
    overlay = mask_rgba_overlay(np.zeros((4, 5), dtype=np.uint8))
    assert overlay.shape == (4, 5, 4)
    assert np.all(overlay[..., 3] == 0.0)


def test_positive_mask_overlay_marks_only_changed_pixels():
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 2] = 255
    overlay = mask_rgba_overlay(mask, alpha=0.6)
    assert overlay[1, 2, 0] == 1.0
    assert overlay[1, 2, 3] == pytest.approx(0.6)
    assert np.count_nonzero(overlay[..., 3]) == 1


def test_rgb_absolute_difference_uses_channel_mean():
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.full((2, 2, 3), 9, dtype=np.uint8)
    difference = rgb_absolute_difference(first, second)
    assert difference.shape == (2, 2)
    assert np.all(difference == 9.0)
