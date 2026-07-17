from __future__ import annotations

from build_s2looking_v31_manifest import (
    _caption,
    _direction_caption,
    radiometric_core_threshold,
    s2looking_chronological_paths,
    select_directional_sanity_pair,
)


def test_empty_directional_target_uses_explicit_negative_caption() -> None:
    assert _caption("appeared", 0, "") == "no buildings appeared"
    assert _caption("disappeared", 0, "") == "no buildings disappeared"
    assert _direction_caption("appeared", 0) == "no new buildings appeared"
    assert _direction_caption("disappeared", 0) == "no buildings were demolished"


def test_nonempty_directional_target_keeps_positive_caption() -> None:
    assert _caption("appeared", 1, "top left") == "one building appeared near the top left"
    assert _caption(
        "appeared", 0, "top left", has_visual_target=True
    ) == "buildings appeared near the top left"
    assert _direction_caption("appeared", 1) == "new buildings appeared"
    assert _direction_caption(
        "appeared", 0, has_visual_target=True
    ) == "new buildings appeared"
    assert _direction_caption("disappeared", 2) == "buildings were demolished"


def test_official_s2looking_image_order_is_reversed_to_chronological_order() -> None:
    before, after = s2looking_chronological_paths("Image1/42.png", "Image2/42.png")
    assert before == "Image2/42.png"
    assert after == "Image1/42.png"


def test_radiometric_core_threshold_is_fit_on_train_geometry_core_only() -> None:
    rows = [
        {"split": "train", "geometry_tier": "core", "radiometric_shift": value}
        for value in (0.01, 0.02, 0.03, 0.04)
    ] + [
        {"split": "val", "geometry_tier": "core", "radiometric_shift": 0.99},
        {"split": "train", "geometry_tier": "hard", "radiometric_shift": 0.98},
    ]
    threshold = radiometric_core_threshold(rows, quantile=0.5)
    assert threshold == 0.025


def test_directional_sanity_pair_selection_uses_quality_not_model_results() -> None:
    rows = [
        {
            "split": "val", "quality_tier": "core", "radiometric_shift": 0.04,
            "directional_overlap_iou": 0.02, "pair_id": "s2looking:val:b",
        },
        {
            "split": "val", "quality_tier": "core", "radiometric_shift": 0.04,
            "directional_overlap_iou": 0.01, "pair_id": "s2looking:val:a",
        },
        {
            "split": "val", "quality_tier": "hard", "radiometric_shift": 0.03,
            "directional_overlap_iou": 0.0, "pair_id": "s2looking:val:hard",
        },
    ]
    assert select_directional_sanity_pair(rows, "val", 0.04) == "s2looking:val:a"
