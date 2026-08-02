from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_structured_view_promotes_only_official_coarse_relation(tmp_path: Path) -> None:
    image_dir, mask_dir = tmp_path / "images", tmp_path / "masks"
    image_dir.mkdir(); mask_dir.mkdir()
    rows = []
    for index, split in enumerate(("train", "train", "val")):
        before, after = image_dir / f"{index}-before.png", image_dir / f"{index}-after.png"
        appeared, disappeared = mask_dir / f"{index}-appeared.png", mask_dir / f"{index}-disappeared.png"
        Image.new("RGB", (4, 4), color=(index, 0, 0)).save(before)
        Image.new("RGB", (4, 4), color=(0, index, 0)).save(after)
        Image.new("L", (4, 4), color=255).save(appeared)
        Image.new("L", (4, 4), color=0).save(disappeared)
        rows.append({
            "schema_version": "temporal-caption-manifest-v1", "dataset_name": "s2looking", "split": split,
            "pair_id": f"s2looking:{split}:{index}", "original_id": str(index), "t1_path": str(before), "t2_path": str(after),
            "source_metadata": {"official_source_order": "Image2_to_Image1", "official_label_mapping": {"label1": "appeared", "label2": "disappeared"}},
            "directional_targets": [
                {"direction": "appeared", "has_visual_target": True, "mask_path": str(appeared)},
                {"direction": "disappeared", "has_visual_target": False, "mask_path": str(disappeared)},
            ],
        })
    source, output = tmp_path / "s2looking.jsonl", tmp_path / "output"
    _write_jsonl(source, rows)
    subprocess.run([sys.executable, "scripts/build_qcpr_structured_semantic_view.py", "--input", str(source), "--output-dir", str(output)], check=True)
    train_rows = [json.loads(line) for line in (output / "retrieval_semantic_structured_train.jsonl").read_text().splitlines() if line.strip()]
    assert len(train_rows) == 2
    assert all(row["schema_version"] == "temporal-caption-manifest-v1" for row in train_rows)
    assert all(row["pair_id"] == row["canonical_pair_id"] for row in train_rows)
    assert all(row["captions"] == [row["text"]] for row in train_rows)
    assert all(Path(row["t1_path"]).is_file() and Path(row["t2_path"]).is_file() for row in train_rows)
    assert all(row["training_enabled"] is True for row in train_rows)
    assert all(row["verification_status"] == "structured_source_verified" for row in train_rows)
    assert all(row["semantic_group_pair_count"] == 2 for row in train_rows)
    assert all(set(row["positive_pair_ids"]) == {"s2looking:train:0", "s2looking:train:1"} for row in train_rows)
    assert all("mask_path" not in json.dumps(row) for row in train_rows)
    assert not list((output / "retrieval_semantic_structured_development.jsonl").read_text().splitlines())
    audit = json.loads((output / "structured_semantic_audit.json").read_text())
    assert audit["status"] == "STRUCTURED_SOURCE_SEMANTIC_READY"
    assert audit["excluded_empty_direction_counts"] == {"disappeared": 3}
    assert audit["mask_paths_in_output"] is False
    verification = output / "independent_verification.json"
    subprocess.run([
        sys.executable,
        "scripts/verify_qcpr_structured_semantic_view.py",
        "--structured-dir",
        str(output),
        "--output",
        str(verification),
    ], check=True)
    independent = json.loads(verification.read_text())
    assert independent["status"] == "INDEPENDENT_STRUCTURED_SEMANTIC_VERIFIED"
    assert independent["verified_rows"] == 2
    assert independent["mask_free_output"] is True


def test_structured_view_rejects_wrong_official_label_mapping(tmp_path: Path) -> None:
    mask = tmp_path / "mask.png"; Image.new("L", (2, 2), color=255).save(mask)
    source = tmp_path / "bad.jsonl"
    _write_jsonl(source, [{
        "schema_version": "temporal-caption-manifest-v1", "dataset_name": "s2looking", "split": "train", "pair_id": "s2looking:train:0",
        "source_metadata": {"official_source_order": "Image1_to_Image2", "official_label_mapping": {"label1": "disappeared", "label2": "appeared"}},
        "directional_targets": [{"direction": "appeared", "has_visual_target": True, "mask_path": str(mask)}, {"direction": "disappeared", "has_visual_target": True, "mask_path": str(mask)}],
    }])
    result = subprocess.run([sys.executable, "scripts/build_qcpr_structured_semantic_view.py", "--input", str(source), "--output-dir", str(tmp_path / "output")], capture_output=True, text=True)
    assert result.returncode != 0
