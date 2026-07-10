from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from land_change_detection.temporal_caption_manifest import make_manifest_row
from land_change_detection.training.temporal_caption_dataset import TemporalCaptionManifestDataset


def _image(path: Path, mode: str, pixels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(mode, (2, 2))
    image.putdata(pixels if mode == "L" else [(value, value, value) for value in pixels])
    image.save(path)


def test_second_semantic_maps_create_real_change_union_target(tmp_path: Path):
    t1 = tmp_path / "t1.png"
    t2 = tmp_path / "t2.png"
    sem1 = tmp_path / "sem1.png"
    sem2 = tmp_path / "sem2.png"
    _image(t1, "RGB", [1, 1, 1, 1])
    _image(t2, "RGB", [2, 2, 2, 2])
    _image(sem1, "L", [1, 1, 2, 2])
    _image(sem2, "L", [1, 3, 2, 4])
    row = make_manifest_row(
        dataset_name="second_cc",
        split="train",
        original_id="sample",
        t1_path=t1,
        t2_path=t2,
        captions=["Buildings changed"],
        semantic_t1_path=sem1,
        semantic_t2_path=sem2,
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    item = TemporalCaptionManifestDataset(manifest, split="train", image_size=None, output_grid=2)[0]

    assert item.mask.tolist() == [[0.0, 1.0], [0.0, 1.0]]
    assert item.metadata["segmentation_supervision"] is True
    assert item.metadata["segmentation_target_source"] == "semantic_transition_union"


def test_missing_mask_and_semantics_are_not_treated_as_negative_supervision(tmp_path: Path):
    t1 = tmp_path / "t1.png"
    t2 = tmp_path / "t2.png"
    _image(t1, "RGB", [1, 1, 1, 1])
    _image(t2, "RGB", [2, 2, 2, 2])
    row = make_manifest_row(
        dataset_name="example",
        split="train",
        original_id="sample",
        t1_path=t1,
        t2_path=t2,
        captions=["A change occurred"],
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    item = TemporalCaptionManifestDataset(manifest, split="train", image_size=None, output_grid=2)[0]

    assert item.mask.sum().item() == 0.0
    assert item.metadata["segmentation_supervision"] is False
    assert item.metadata["segmentation_target_source"] == "none"
