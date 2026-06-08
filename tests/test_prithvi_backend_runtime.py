from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_prithvi_backend_runtime_and_feature_export(tmp_path: Path):
    from land_change_detection.segmentation_backends.prithvi_terratorch import PrithviTerratorchBackend

    model_dir = tmp_path / "Prithvi-EO-2.0-300M-TL"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pt").write_text("x", encoding="utf-8")
    (model_dir / "terratorch_config.yaml").write_text(
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
    backend = PrithviTerratorchBackend(model_dir=model_dir, device="cpu")
    runtime = backend.inspect_runtime()
    assert runtime.checkpoint_path is not None
    loaded = backend.load_runtime_object()
    assert loaded.loader_stage in {"loadable", "loaded_placeholder"}
    artifact = backend.export_features(np.random.rand(13, 8, 8).astype(np.float32))
    assert artifact.features is not None
    assert artifact.features.shape == (6, 8, 8)


def test_inspect_prithvi_backend_cli(tmp_path: Path):
    model_dir = tmp_path / "Prithvi-EO-2.0-300M-TL"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "terratorch_config.yaml").write_text(
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
    output = tmp_path / "backend.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_prithvi_backend.py",
            "--model-dir",
            str(model_dir),
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
    assert payload["model_dir"] == str(model_dir)
    assert "runtime_ready" in payload


def test_load_prithvi_backend_runtime_cli(tmp_path: Path):
    model_dir = tmp_path / "Prithvi-EO-2.0-300M-TL"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "terratorch_config.yaml").write_text(
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
    (model_dir / "model.pt").write_text("x", encoding="utf-8")
    output = tmp_path / "loaded.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/load_prithvi_backend_runtime.py",
            "--model-dir",
            str(model_dir),
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
    assert payload["loader_stage"] in {"loadable", "loaded_placeholder"}
