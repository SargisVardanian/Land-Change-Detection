from __future__ import annotations

import numpy as np

from land_change_detection.semantic_transitions import (
    changed_area_ratio,
    compute_transition_histogram,
    compute_transition_map,
    dominant_transition,
    transition_similarity,
)


def test_transition_histogram_synthetic_case():
    before = np.asarray([[0, 0], [1, 1]], dtype=np.int64)
    after = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    transition_map = compute_transition_map(before, after, num_classes=3)
    hist = compute_transition_histogram(transition_map, num_classes=3, ignore_no_change=False)
    assert np.isclose(hist.sum(), 1.0)
    dominant = dominant_transition(hist)
    assert dominant[:2] in {(0, 0), (0, 1), (1, 1), (1, 2)}
    assert changed_area_ratio(before, after) == 0.5


def test_transition_similarity_cosine():
    hist_a = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    hist_b = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    hist_c = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert np.isclose(transition_similarity(hist_a, hist_b), 1.0)
    assert np.isclose(transition_similarity(hist_a, hist_c), 0.0)
