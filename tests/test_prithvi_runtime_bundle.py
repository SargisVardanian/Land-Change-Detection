from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_prithvi_runtime_bundle(tmp_path: Path):
    from land_change_detection.training.prithvi_experimental import build_prithvi_runtime_bundle

    checkpoint = tmp_path / "model.pt"
    config = tmp_path / "terratorch_config.yaml"
    checkpoint.write_text("x", encoding="utf-8")
    config.write_text(
        "\n".join(
            [
                "model_name: prithvi-eo-2.0-300m-tl",
                "task: semantic_change",
                "num_input_channels: 6",
                "num_classes: 12",
            ]
        ),
        encoding="utf-8",
    )
    bundle = build_prithvi_runtime_bundle(
        checkpoint_path=str(checkpoint),
        terratorch_config_path=str(config),
    )
    assert bundle.checkpoint_path == str(checkpoint)
    assert bundle.terratorch_config_path == str(config)
    assert bundle.config_valid is True
    assert "available" in bundle.module_summary


def test_validate_prithvi_runtime_bundle_cli(tmp_path: Path):
    output = tmp_path / "bundle.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_prithvi_runtime_bundle.py",
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
    assert "terratorch_available" in payload
    assert "runtime_ready" in payload
