from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_diagnose_terratorch_env_cli(tmp_path: Path):
    output = tmp_path / "diag.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_terratorch_env.py",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "python_version" in payload
    assert "terratorch" in payload
    assert "ready_for_experimental_prithvi" in payload


def test_prithvi_runner_writes_diagnostics(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_prithvi_terratorch_experimental.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode != 0
    assert (output_dir / "terratorch_env_diagnostics.json").exists()
