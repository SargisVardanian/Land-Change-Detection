from __future__ import annotations

import json
import subprocess
import sys
import hashlib
from pathlib import Path

from PIL import Image

from land_change_detection.training.temporal_caption_dataset import TemporalCaptionManifestDataset


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.blake2b(path.as_posix().encode("utf-8"), digest_size=3).digest()
    Image.new("RGB", (4, 4), tuple(digest)).save(path)


def _write_xbd_pair(root: Path, split: str, image_id: str) -> None:
    _image(root / split / "images" / f"{image_id}_pre_disaster.png")
    _image(root / split / "images" / f"{image_id}_post_disaster.png")


def _run_rscc(root: Path, annotations: Path, output: Path, audit: Path, expected: int, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/prepare_rscc_manifest.py",
            "--root",
            str(root),
            "--annotations",
            str(annotations),
            "--expected-pairs",
            str(expected),
            "--output",
            str(output),
            "--audit-report",
            str(audit),
            *extra,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_qvq_max_captions_are_model_generated_benchmark_ground_truth(tmp_path):
    root = tmp_path / "rscc"
    _write_xbd_pair(root, "test", "quake_001")
    annotations = tmp_path / "RSCC_qvq.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "image_id": "quake_001",
                "caption": "Several buildings were demolished in the north.",
                "disaster": "earthquake",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"
    audit = tmp_path / "audit.json"
    result = _run_rscc(root, annotations, output, audit, 1)
    assert result.returncode == 0, result.stderr
    row = _rows(output)[0]
    assert row["split"] == "test"
    assert row["caption_source"] == "model_generated"
    assert row["source_metadata"]["caption_generator"] == "QvQ-Max"
    assert row["source_metadata"]["benchmark_ground_truth"] is True
    assert row["source_metadata"]["benchmark_subset"] == "rscc_xbd_988"
    assert row["source_metadata"]["training_default_enabled"] is False
    assert row["source_metadata"]["license_family"] == "xBD"
    assert row["source_metadata"]["research_only"] is True


def test_rscc_qvq_subset_is_test_only_unless_authoritative_split_is_present(tmp_path):
    root = tmp_path / "rscc"
    _write_xbd_pair(root, "test", "default_001")
    _write_xbd_pair(root, "val", "explicit_001")
    annotations = tmp_path / "RSCC_qvq.jsonl"
    annotations.write_text(
        "\n".join(
            [
                json.dumps({"image_id": "default_001", "caption": "A road was removed."}),
                json.dumps({"image_id": "explicit_001", "split": "val", "caption": "Water expanded."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"
    audit = tmp_path / "audit.json"
    result = _run_rscc(root, annotations, output, audit, 2)
    assert result.returncode == 0, result.stderr
    rows = {row["original_id"]: row for row in _rows(output)}
    assert rows["default_001"]["split"] == "test"
    assert rows["explicit_001"]["split"] == "val"


def test_no_rscc_rows_enter_default_train_loader(tmp_path):
    root = tmp_path / "rscc"
    _write_xbd_pair(root, "test", "qvq_001")
    annotations = tmp_path / "RSCC_qvq.jsonl"
    annotations.write_text(json.dumps({"image_id": "qvq_001", "caption": "A building appeared."}) + "\n", encoding="utf-8")
    output = tmp_path / "manifest.jsonl"
    audit = tmp_path / "audit.json"
    result = _run_rscc(root, annotations, output, audit, 1)
    assert result.returncode == 0, result.stderr
    train = TemporalCaptionManifestDataset(output, split="train", image_size=4, output_grid=2)
    all_default = TemporalCaptionManifestDataset(output, split="all", image_size=4, output_grid=2)
    all_opted_in = TemporalCaptionManifestDataset(
        output,
        split="all",
        image_size=4,
        output_grid=2,
        allowed_caption_sources={"model_generated"},
        exclude_rscc_model_generated=False,
    )
    assert len(train) == 0
    assert len(all_default) == 0
    assert len(all_opted_in) == 1


def test_qvq_ground_truth_only_returns_expected_988_rows(tmp_path):
    root = tmp_path / "rscc"
    _image(root / "test" / "images" / "shared_pre_disaster.png")
    _image(root / "test" / "images" / "shared_post_disaster.png")
    annotations = tmp_path / "RSCC_qvq.jsonl"
    rows = []
    for index in range(988):
        rows.append(
            json.dumps(
                {
                    "image_id": f"pair_{index:04d}",
                    "before_path": "test/images/shared_pre_disaster.png",
                    "after_path": "test/images/shared_post_disaster.png",
                    "caption": f"QvQ-Max benchmark description {index}",
                }
            )
        )
    annotations.write_text("\n".join(rows) + "\n", encoding="utf-8")
    output = tmp_path / "manifest.jsonl"
    audit = tmp_path / "audit.json"
    result = _run_rscc(root, annotations, output, audit, 988)
    assert result.returncode == 0, result.stderr
    manifest_rows = _rows(output)
    assert len(manifest_rows) == 988
    assert {row["split"] for row in manifest_rows} == {"test"}
    assert {row["caption_source"] for row in manifest_rows} == {"model_generated"}


def test_generated_full_rscc_captions_remain_distinguishable_from_qvq_ground_truth(tmp_path):
    root = tmp_path / "rscc"
    _write_xbd_pair(root, "test", "qvq_001")
    _write_xbd_pair(root, "test", "full_001")
    qvq_annotations = tmp_path / "RSCC_qvq.jsonl"
    full_annotations = tmp_path / "RSCC_full.jsonl"
    qvq_annotations.write_text(json.dumps({"image_id": "qvq_001", "caption": "QvQ-Max road removal."}) + "\n", encoding="utf-8")
    full_annotations.write_text(
        json.dumps({"image_id": "full_001", "caption": "Generated full RSCC road removal.", "source": "generated", "caption_generator": "Other-VLM"})
        + "\n",
        encoding="utf-8",
    )
    qvq_output = tmp_path / "qvq.jsonl"
    full_output = tmp_path / "full.jsonl"
    qvq_result = _run_rscc(root, qvq_annotations, qvq_output, tmp_path / "qvq_audit.json", 1)
    full_result = _run_rscc(
        root,
        full_annotations,
        full_output,
        tmp_path / "full_audit.json",
        1,
        "--caption-policy",
        "model_generated_only",
    )
    assert qvq_result.returncode == 0, qvq_result.stderr
    assert full_result.returncode == 0, full_result.stderr
    qvq = _rows(qvq_output)[0]
    full = _rows(full_output)[0]
    assert qvq["caption_source"] == full["caption_source"] == "model_generated"
    assert qvq["source_metadata"]["benchmark_ground_truth"] is True
    assert qvq["source_metadata"]["caption_generator"] == "QvQ-Max"
    assert full["source_metadata"]["benchmark_ground_truth"] is False
    assert full["source_metadata"]["caption_generator"] == "Other-VLM"
