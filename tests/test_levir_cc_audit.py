from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "levir_cc_audit.py"
    spec = importlib.util.spec_from_file_location("levir_cc_audit", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_run_marks_missing_artifacts_incomplete(tmp_path: Path):
    audit = _load_module()
    runs_root = tmp_path / "runs"
    overfit_dir = runs_root / "levir_cc_simple_patch_smoke_overfit100"
    train_dir = runs_root / "levir_cc_simple_patch_smoke_train"
    overfit_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    (overfit_dir / "overfit_report.json").write_text(json.dumps({"gate_passed": True}), encoding="utf-8")
    (train_dir / "train_summary.json").write_text(json.dumps({"best_epoch": 4}), encoding="utf-8")

    report = audit.summarize_run(runs_root, "simple_patch_smoke")
    assert report["overfit_gate_passed"] is True
    assert report["overfit_artifacts_complete"] is False
    assert report["train_artifacts_complete"] is False
    assert report["milestone_ready"] is False
    assert report["train_best_epoch"] == 4


def test_summarize_run_marks_complete_milestone_when_gate_and_artifacts_exist(tmp_path: Path):
    audit = _load_module()
    runs_root = tmp_path / "runs"
    overfit_dir = runs_root / "levir_cc_simple_patch_smoke_overfit100"
    train_dir = runs_root / "levir_cc_simple_patch_smoke_train"
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

    report = audit.summarize_run(runs_root, "simple_patch_smoke")
    assert report["overfit_gate_passed"] is True
    assert report["overfit_artifacts_complete"] is True
    assert report["train_artifacts_complete"] is True
    assert report["milestone_ready"] is True
    assert report["train_best_epoch"] == 5


def test_summarize_required_files_and_runs(tmp_path: Path):
    audit = _load_module()
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "levir_cc_manifest_validation.json").write_text('{"valid": true}\n', encoding="utf-8")
    required = audit.summarize_required_files(runs_root)
    assert required["levir_cc_manifest_validation.json"]["exists"] is True
    assert required["levir_cc_random_retrieval_eval.json"]["exists"] is False

    runs = audit.summarize_runs(runs_root, presets=("simple_patch_smoke",))
    assert set(runs) == {"simple_patch_smoke"}
    assert runs["simple_patch_smoke"]["preset"] == "simple_patch_smoke"
