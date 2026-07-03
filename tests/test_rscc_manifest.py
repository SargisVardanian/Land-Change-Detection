from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(path)


def _write_xbd_pair(root: Path, split: str, image_id: str) -> None:
    _image(root / split / "images" / f"{image_id}_pre_disaster.png")
    _image(root / split / "images" / f"{image_id}_post_disaster.png")


def test_official_rscc_human_subset_parsing_excludes_generated_by_default(tmp_path):
    root = tmp_path / "rscc"
    _write_xbd_pair(root, "train", "quake_001")
    _write_xbd_pair(root, "train", "quake_002")
    annotations = tmp_path / "RSCC_qvq.jsonl"
    annotations.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "split": "train",
                        "image_id": "quake_001",
                        "caption": "Several buildings were demolished in the north.",
                        "caption_source": "human",
                        "disaster": "earthquake",
                        "license": "xBD",
                    }
                ),
                json.dumps(
                    {
                        "split": "train",
                        "image_id": "quake_002",
                        "caption": "GPT says buildings changed.",
                        "caption_source": "model_generated",
                        "disaster": "earthquake",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"
    audit = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_rscc_manifest.py",
            "--root",
            str(root),
            "--annotations",
            str(annotations),
            "--expected-pairs",
            "1",
            "--output",
            str(output),
            "--audit-report",
            str(audit),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["caption_source"] == "human"
    assert rows[0]["source_metadata"]["disaster"] == "earthquake"
    assert rows[0]["source_metadata"]["license_family"] == "xBD"


def test_rscc_model_generated_policy_keeps_generated_as_generated(tmp_path):
    root = tmp_path / "rscc"
    _write_xbd_pair(root, "val", "flood_001")
    annotations = tmp_path / "RSCC_qvq.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "split": "val",
                "image_id": "flood_001",
                "caption": "A road was removed by flooding.",
                "source": "generated",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"
    audit = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_rscc_manifest.py",
            "--root",
            str(root),
            "--annotations",
            str(annotations),
            "--caption-policy",
            "model_generated_only",
            "--expected-pairs",
            "1",
            "--output",
            str(output),
            "--audit-report",
            str(audit),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["caption_source"] == "model_generated"
