from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from land_change_detection.models.qcpr_v3_data import (
    CappedCompositionalBatchSampler, DirectionalCurriculumBatchSampler,
    WeightingConfig, derive_qcpr_v3_manifests,
)
from land_change_detection.training.temporal_caption_dataset import _hard_background_crop_box


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
        "natural_train_retrieval_manifest.jsonl", "natural_validation_retrieval_manifest.jsonl",
        "natural_train_localization_manifest.jsonl", "natural_validation_localization_manifest.jsonl",
    }
    assert required <= {path.name for path in first.iterdir()}
    for name in required:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert report["coverage"]["duplicate_caption_clusters"] == 1
    assert report["weighting"]["weight_min"] > 0
    assert report["weighting"]["weight_max"] <= config.max_weight / config.min_weight
    natural = [json.loads(line) for line in (first / "natural_train_manifest.jsonl").read_text().splitlines()]
    assert natural[0]["sampling_weight"] < natural[1]["sampling_weight"]
    retrieval = [json.loads(line) for line in (first / "natural_train_retrieval_manifest.jsonl").read_text().splitlines()]
    assert retrieval and all(row.get("retrieval_supervision", True) for row in retrieval)


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


def test_capped_sampler_small_batch_terminates_and_consumes_no_change_rows() -> None:
    rows = [
        {"pair_id": "changed", "captions": ["a house appeared"], "sampling_weight": 1.0},
        {"pair_id": "nochange-1", "captions": ["no change"], "sampling_weight": 1.0},
        {"pair_id": "nochange-2", "captions": ["no change"], "sampling_weight": 1.0},
    ]
    batches = list(CappedCompositionalBatchSampler(rows, 2, seed=1, no_change_fraction_cap=0.25))
    observed = {rows[index]["pair_id"] for batch in batches for index in batch}
    assert {"nochange-1", "nochange-2"} <= observed
    assert all(sum(rows[index]["pair_id"].startswith("nochange") for index in batch) <= 1 for batch in batches)


def test_dataset_audit_reports_pair_progress(tmp_path) -> None:
    source = _write_manifest(tmp_path)
    progress = []
    derive_qcpr_v3_manifests(
        [source],
        tmp_path / "derived",
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_directional_curriculum_is_deterministic_pair_unique_and_60_20_20() -> None:
    rows = (
        [{"pair_id": f"core-{index}", "quality_tier": "core"} for index in range(30)]
        + [{"pair_id": f"hard-{index}", "quality_tier": "hard"} for index in range(30)]
        + [{"pair_id": "stress", "quality_tier": "stress"}]
    )
    sampler = DirectionalCurriculumBatchSampler(rows, 10, seed=9)
    batches = list(sampler)
    assert batches == list(DirectionalCurriculumBatchSampler(rows, 10, seed=9))
    for batch in batches:
        assert [mode for _, mode in batch].count("positive") == 6
        assert [mode for _, mode in batch].count("hard_background") == 2
        assert [mode for _, mode in batch].count("hard_directional") == 2
        indices = [index for index, _ in batch]
        assert len(indices) == len(set(indices))
        assert all(rows[index]["quality_tier"] != "stress" for index in indices)


def test_hard_background_crop_selects_a_foreground_free_native_tile() -> None:
    mask = torch.zeros(1024, 1024)
    mask[:256, :256] = 1
    box = _hard_background_crop_box(mask, crop_size=256)
    top, bottom, left, right = box
    assert mask[top:bottom, left:right].sum() == 0
    assert (bottom - top, right - left) == (256, 256)
