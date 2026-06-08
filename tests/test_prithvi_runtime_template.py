from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_write_prithvi_terratorch_template_and_inspect(tmp_path: Path):
    template = tmp_path / "terratorch_config.yaml"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/write_prithvi_terratorch_template.py",
            "--output",
            str(template),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    assert template.exists()

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("x", encoding="utf-8")
    inspect_output = tmp_path / "inspect.json"
    inspect = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_prithvi_runtime.py",
            "--output",
            str(inspect_output),
            "--checkpoint-path",
            str(checkpoint),
            "--terratorch-config-path",
            str(template),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert inspect.returncode == 0, inspect.stderr
    payload = json.loads(inspect_output.read_text(encoding="utf-8"))
    assert payload["config_valid"] is True
    assert payload["config_summary"]["num_input_channels"] == 6
