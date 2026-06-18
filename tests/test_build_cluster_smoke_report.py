from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "build_cluster_smoke_report.py"
    spec = importlib.util.spec_from_file_location("build_cluster_smoke_report", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_project_assets_report_not_ready(tmp_path: Path):
    module = _load_module()
    project_root = tmp_path / "rs_change_project"
    project_root.mkdir(parents=True)
    report = module._build_project_assets_report(project_root)
    assert report["ready"] is False


def test_build_project_assets_report_ready(tmp_path: Path):
    module = _load_module()
    project_root = tmp_path / "rs_change_project"
    for rel in [
        "datasets/raw/LEVIR-CC",
        "datasets/raw/LEVIR-MCI-unpacked/LEVIR-MCI-dataset",
        "datasets/raw/SECOND-CC",
        "checkpoints/models/semantic/mask2former-satellite",
        "checkpoints/models/semantic/Prithvi-EO-2.0-300M-TL",
        "checkpoints/models/semantic/Prithvi-EO-2.0-600M-TL",
    ]:
        (project_root / rel).mkdir(parents=True, exist_ok=True)
    (project_root / "datasets" / "dataset_manifest.md").parent.mkdir(parents=True, exist_ok=True)
    (project_root / "datasets" / "dataset_manifest.md").write_text("# Dataset Manifest\n", encoding="utf-8")
    for rel in [
        "indexes/levir_mci_samples.jsonl",
        "indexes/levir_mci_validation.json",
        "indexes/second_cc_samples.jsonl",
        "indexes/levir_cc_text_manifest.jsonl",
        "indexes/preview_samples.json",
        "runs/levir_mci_preview.png",
        "runs/levir_mci_grid.png",
    ]:
        path = project_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    report = module._build_project_assets_report(project_root)
    assert report["ready"] is True
    assert report["datasets"]["LEVIR-MCI"]["canonical_path"].endswith("LEVIR-MCI-unpacked/LEVIR-MCI-dataset")
    assert report["benchmark_curriculum"]["stage_2_grounded"]["ready"] is True
