from __future__ import annotations

import numpy as np


def rgb_absolute_difference(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    first = np.asarray(t1, dtype=np.float32)
    second = np.asarray(t2, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError(f"RGB inputs must share shape, got {first.shape} and {second.shape}")
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB arrays, got {first.shape}")
    return np.abs(second - first).mean(axis=2)


def mask_rgba_overlay(mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    binary = np.asarray(mask) > 0
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {binary.shape}")
    overlay = np.zeros((*binary.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = binary.astype(np.float32) * alpha
    return overlay
