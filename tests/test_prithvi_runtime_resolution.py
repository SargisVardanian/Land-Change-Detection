from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_resolve_prithvi_runtime_prefers_existing_candidates(tmp_path: Path):
    from land_change_detection.training.prithvi_experimental import resolve_prithvi_runtime

    model_dir = tmp_path / "Prithvi-EO-2.0-300M-TL"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "model.pt"
    config = model_dir / "terratorch_config.yaml"
    checkpoint.write_text("x", encoding="utf-8")
    config.write_text("model: prithvi", encoding="utf-8")

    resolution = resolve_prithvi_runtime(model_dir=model_dir)
    assert resolution.checkpoint_path == str(checkpoint)
    assert resolution.terratorch_config_path == str(config)
    assert resolution.runtime_ready is False


def test_inspect_prithvi_runtime_cli(tmp_path: Path):
    output = tmp_path / "runtime.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_prithvi_runtime.py",
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
    assert "checkpoint_path" in payload
    assert "terratorch_config_path" in payload
    assert "runtime_ready" in payload
