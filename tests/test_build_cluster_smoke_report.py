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
        "indexes/levir_cc_pairs.jsonl",
        "indexes/levir_cc_caption_queries.jsonl",
        "indexes/levir_cc_caption_queries_train.jsonl",
        "indexes/levir_cc_caption_queries_val.jsonl",
        "indexes/levir_cc_caption_queries_test.jsonl",
        "indexes/levir_cc_caption_queries_overfit_100.jsonl",
        "indexes/preview_samples.json",
        "runs/levir_mci_preview.png",
        "runs/levir_mci_grid.png",
        "runs/levir_cc_manifest_validation.json",
        "runs/levir_cc_random_retrieval_eval.json",
        "runs/dino_pair_retrieval_simple_patch/eval_metrics.json",
    ]:
        path = project_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    (project_root / "runs" / "dino_pair_retrieval_simple_patch" / "train_summary.json").write_text(
        '{"best_epoch": 2}\n', encoding="utf-8"
    )
    (project_root / "runs" / "dino_pair_retrieval_simple_patch" / "eval_summary.json").write_text(
        '{"overall_highlights": {"transition_recall@5": 0.9}}\n', encoding="utf-8"
    )
    overfit_dir = project_root / "runs" / "levir_cc_simple_patch_smoke_overfit100"
    train_dir = project_root / "runs" / "levir_cc_simple_patch_smoke_train"
    overfit_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    for rel, payload in [
        (overfit_dir / "overfit_report.json", '{"gate_passed": true}\n'),
        (overfit_dir / "train_summary.json", '{"best_epoch": 2}\n'),
        (overfit_dir / "eval_summary.json", '{"overall_highlights": {"recall@5": 1.0}}\n'),
        (overfit_dir / "eval_metrics.json", '{"recall@5": 1.0}\n'),
        (overfit_dir / "metrics_history.json", '{"history": []}\n'),
        (overfit_dir / "best.pt", "x"),
        (overfit_dir / "last.pt", "x"),
        (overfit_dir / "text_query_top5_grid.png", "x"),
        (overfit_dir / "environment_fingerprint.json", '{"python": "3.12"}\n'),
        (train_dir / "train_summary.json", '{"best_epoch": 5}\n'),
        (train_dir / "eval_summary.json", '{"overall_highlights": {"recall@5": 0.8}}\n'),
        (train_dir / "eval_metrics.json", '{"recall@5": 0.8}\n'),
        (train_dir / "metrics_history.json", '{"history": []}\n'),
        (train_dir / "best.pt", "x"),
        (train_dir / "last.pt", "x"),
        (train_dir / "text_query_top5_grid.png", "x"),
        (train_dir / "environment_fingerprint.json", '{"python": "3.12"}\n'),
    ]:
        rel.write_text(payload, encoding="utf-8")
    report = module._build_project_assets_report(project_root)
    assert report["ready"] is True
    assert report["datasets"]["LEVIR-MCI"]["canonical_path"].endswith("LEVIR-MCI-unpacked/LEVIR-MCI-dataset")
    assert report["benchmark_curriculum"]["stage_2_grounded"]["ready"] is True
    assert report["benchmark_curriculum"]["stage_1_text_to_pair"]["pair_manifest_ready"] is True
    assert report["benchmark_curriculum"]["stage_1_text_to_pair"]["caption_query_manifest_ready"] is True
    assert report["benchmark_curriculum"]["stage_1_text_to_pair"]["required_files_complete"] is True
    assert report["benchmark_curriculum"]["stage_1_text_to_pair"]["any_milestone_ready"] is True
    assert report["benchmark_curriculum"]["stage_1_text_to_pair"]["ready"] is True
    assert report["levir_cc_required"]["levir_cc_manifest_validation.json"]["exists"] is True
    assert report["levir_cc_runs"]["simple_patch_smoke"]["milestone_ready"] is True
    assert report["optional_pair_retrieval_runs"]["dino_pair_retrieval_simple_patch/eval_summary.json"]["exists"] is True
    assert report["pair_retrieval_summary_highlights"]["train_summary"]["best_epoch"] == 2
    assert report["pair_retrieval_summary_highlights"]["eval_summary"]["overall_highlights"]["transition_recall@5"] == 0.9


def test_main_report_includes_dataset_curriculum(tmp_path: Path):
    module = _load_module()
    project_root = tmp_path / "rs_change_project"
    (project_root / "datasets" / "raw" / "LEVIR-MCI-unpacked" / "LEVIR-MCI-dataset").mkdir(parents=True, exist_ok=True)
    curriculum = module.curriculum_summary(project_root)
    assert any(row["name"] == "LEVIR-MCI" for row in curriculum)
