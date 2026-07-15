from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from land_change_detection.models.qcpr_v3_data import CappedCompositionalBatchSampler, WeightingConfig, derive_qcpr_v3_manifests


def _write_manifest(tmp_path):
    image = tmp_path / "image.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image)
    validation_image = tmp_path / "validation.png"
    validation_pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    validation_pixels[:, 4:] = 255
    Image.fromarray(validation_pixels).save(validation_image)
    rows = [
        {"dataset_name": "tiny", "split": "train", "pair_id": "train-1", "t1_path": str(image), "t2_path": str(image), "captions": ["A building appeared.", "A building appeared."], "retrieval_supervision": True},
        {"dataset_name": "tiny", "split": "train", "pair_id": "train-2", "t1_path": str(image), "t2_path": str(image), "captions": ["No change."], "retrieval_supervision": True},
        {"dataset_name": "tiny", "split": "val", "pair_id": "val-1", "t1_path": str(validation_image), "t2_path": str(validation_image), "captions": ["A building appeared."], "retrieval_supervision": True},
    ]
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_derived_manifests_are_deterministic_capped_and_duplicate_aware(tmp_path) -> None:
    source = _write_manifest(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = WeightingConfig(alpha=0.4, tau=5, min_weight=0.2, max_weight=5)
    report = derive_qcpr_v3_manifests([source], first, config=config)
    derive_qcpr_v3_manifests([source], second, config=config)
    required = {
        "dataset_coverage_report.json", "dataset_coverage_report.csv", "dataset_weighting_report.json",
        "duplicate_audit.jsonl", "caption_quality_audit.jsonl", "natural_train_manifest.jsonl",
        "balanced_train_manifest.jsonl", "natural_validation_manifest.jsonl", "balanced_validation_manifest.jsonl",
    }
    assert required <= {path.name for path in first.iterdir()}
    for name in required:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert report["coverage"]["duplicate_caption_clusters"] == 1
    assert report["weighting"]["weight_min"] > 0
    assert report["weighting"]["weight_max"] <= config.max_weight / config.min_weight
    natural = [json.loads(line) for line in (first / "natural_train_manifest.jsonl").read_text().splitlines()]
    assert natural[0]["sampling_weight"] < natural[1]["sampling_weight"]


def test_cross_split_duplicate_caption_is_reported_not_silently_deleted(tmp_path) -> None:
    source = _write_manifest(tmp_path)
    output = tmp_path / "derived"
    derive_qcpr_v3_manifests([source], output)
    duplicate_rows = [json.loads(line) for line in (output / "duplicate_audit.jsonl").read_text().splitlines()]
    cluster = next(row for row in duplicate_rows if row["kind"] == "normalized_caption")
    assert any(member.startswith("train-1") for member in cluster["members"])
    assert any(member.startswith("val-1") for member in cluster["members"])


def test_cross_split_duplicate_pair_images_are_a_hard_leakage_failure(tmp_path) -> None:
    source = _write_manifest(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[2]["captions"] = ["A distinct validation caption."]
    rows[2]["t1_path"] = rows[0]["t1_path"]
    rows[2]["t2_path"] = rows[0]["t2_path"]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="perceptual pair leakage"):
        derive_qcpr_v3_manifests([source], tmp_path / "derived")


def test_capped_sampler_limits_no_change_and_never_repeats_pair_in_batch() -> None:
    rows = [
        {"pair_id": f"changed-{index}", "captions": ["a house appeared"], "sampling_weight": 1.0}
        for index in range(6)
    ] + [
        {"pair_id": f"nochange-{index}", "captions": ["there is no difference"], "sampling_weight": 1.0}
        for index in range(6)
    ]
    sampler = CappedCompositionalBatchSampler(rows, 4, seed=3, no_change_fraction_cap=0.25)
    batches = list(sampler)
    for batch in batches:
        no_change = sum(rows[index]["pair_id"].startswith("nochange") for index in batch)
        assert no_change <= 1
        assert len({rows[index]["pair_id"] for index in batch}) == len(batch)
    assert list(CappedCompositionalBatchSampler(rows, 4, seed=3, no_change_fraction_cap=0.25)) == batches
