from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "audit_levir_cc_milestone.py"
    spec = importlib.util.spec_from_file_location("audit_levir_cc_milestone", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_report_not_ready_without_required_files(tmp_path: Path):
    module = _load_module()
    project_root = tmp_path / "rs_change_project"
    (project_root / "runs").mkdir(parents=True, exist_ok=True)
    report = module.build_report(project_root)
    assert report["required_files_complete"] is False
    assert report["ready_presets"] == []
    assert report["milestone_ready"] is False


def test_build_report_ready_when_required_files_and_simple_preset_exist(tmp_path: Path):
    module = _load_module()
    project_root = tmp_path / "rs_change_project"
    runs_root = project_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "levir_cc_manifest_validation.json").write_text('{"valid": true}\n', encoding="utf-8")
    (runs_root / "levir_cc_random_retrieval_eval.json").write_text('{"recall@5": 0.2}\n', encoding="utf-8")

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

    report = module.build_report(project_root)
    assert report["required_files_complete"] is True
    assert report["ready_presets"] == ["simple_patch_smoke"]
    assert report["milestone_ready"] is True


def test_cli_exits_nonzero_until_milestone_ready(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "rs_change_project"
    (project_root / "runs").mkdir(parents=True, exist_ok=True)
    output = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_levir_cc_milestone.py",
            "--project-root",
            str(project_root),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["milestone_ready"] is False
