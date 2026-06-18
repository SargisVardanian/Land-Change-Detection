from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "verify_rs_change_project.py"
    spec = importlib.util.spec_from_file_location("verify_rs_change_project", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_report_incomplete(tmp_path: Path):
    module = _load_module()
    project_root = tmp_path / "rs_change_project"
    (project_root / "datasets" / "raw").mkdir(parents=True)
    report = module.build_report(project_root)
    assert report["bootstrap_ready"] is False
    assert report["complete"] is False
    assert report["datasets"]["LEVIR-MCI"]["exists"] is False


def test_build_report_ready(tmp_path: Path):
    module = _load_module()
    project_root = tmp_path / "rs_change_project"
    for rel in [
        "datasets/raw/LEVIR-CC",
        "datasets/raw/LEVIR-MCI-unpacked/LEVIR-MCI-dataset",
        "datasets/raw/SECOND-CC",
    ]:
        (project_root / rel).mkdir(parents=True, exist_ok=True)
    (project_root / "datasets" / "dataset_manifest.md").parent.mkdir(parents=True, exist_ok=True)
    (project_root / "datasets" / "dataset_manifest.md").write_text("# Dataset Manifest\n", encoding="utf-8")
    for rel in [
        "indexes/levir_mci_samples.jsonl",
        "indexes/levir_mci_validation.json",
        "indexes/preview_samples.json",
        "runs/levir_mci_preview.png",
        "runs/levir_mci_grid.png",
        "runs/dino_pair_retrieval_simple_patch/train_summary.json",
        "runs/dino_pair_retrieval_simple_patch/eval_metrics.json",
        "runs/dino_pair_retrieval_simple_patch/eval_summary.json",
    ]:
        path = project_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    report = module.build_report(project_root)
    assert report["bootstrap_ready"] is True
    assert report["complete"] is False
    assert report["optional_pair_retrieval_runs"]["dino_pair_retrieval_simple_patch/train_summary.json"]["exists"] is True
