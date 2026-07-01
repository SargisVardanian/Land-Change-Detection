from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_unichange_v2_retrieval_smoke_cli_with_fake_backbones(tmp_path: Path):
    output_dir = tmp_path / "ucv2_smoke"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_unichange_v2_retrieval.py",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--synthetic-data",
            "--fake-backbones",
            "--smoke",
            "--image-size",
            "32",
            "--output-grid",
            "8",
            "--max-train-samples",
            "4",
            "--max-val-samples",
            "2",
            "--max-steps",
            "2",
            "--no-bf16",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "run_config.json").exists()
    assert (output_dir / "smoke_report.json").exists()
    assert (output_dir / "metrics_history.jsonl").exists()
    assert (output_dir / "best_retrieval.pt").exists()
    assert (output_dir / "last_retrieval.pt").exists()
    report = json.loads((output_dir / "smoke_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["fake_backbones"] is True
    assert report["train_val_disjoint"] is True
    assert report["finite_loss"] is True
    assert report["gradient_audit_passed"] is True
    assert report["frozen_grad_violations"] == []
    assert report["missing_gradients"] == []
    assert report["checkpoint_roundtrip"]["ok"] is True
    assert report["checkpoint_roundtrip_passed"] is True
    assert report["cluster_ready"] is False
    assert report["shape_report"]["images"] == [2, 2, 3, 32, 32]


def test_full_retrieval_sbatch_is_gated_on_real_smoke_report():
    script = (Path(__file__).resolve().parents[1] / "cluster" / "ysu" / "train_unichange_v2_retrieval.sbatch").read_text(
        encoding="utf-8"
    )

    assert "SMOKE_REPORT" in script
    assert "status" in script and "PASS" in script
    assert "fake_backbones" in script and "must be false" in script
    assert "frozen_grad_violations" in script
    assert "missing_gradients" in script
    assert "checkpoint_roundtrip_passed" in script


@pytest.mark.skipif(os.environ.get("RUN_REAL_UNICHANGE_V2_PROBE") != "1", reason="real UniverSat/Jina/LEVIR-MCI probe is opt-in")
def test_unichange_v2_retrieval_real_probe_command(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_unichange_v2_retrieval.py",
            "--output-dir",
            str(tmp_path / "real_probe"),
            "--device",
            "cuda",
            "--smoke",
            "--max-steps",
            "1",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src", **os.environ},
    )
    assert result.returncode == 0, result.stderr
