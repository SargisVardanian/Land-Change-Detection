from __future__ import annotations

from build_s2looking_v31_manifest import s2looking_chronological_paths


def test_official_s2looking_image_order_is_reversed_to_chronological_order() -> None:
    before, after = s2looking_chronological_paths("Image1/42.png", "Image2/42.png")
    assert before == "Image2/42.png"
    assert after == "Image1/42.png"
