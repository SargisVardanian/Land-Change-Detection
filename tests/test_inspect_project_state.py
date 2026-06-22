from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_inspect_project_state_includes_pair_retrieval_summaries(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "rs_change_project"
    repo_clone = project_root / "code" / "project"
    dataset_root = project_root / "datasets" / "raw" / "LEVIR-MCI-unpacked" / "LEVIR-MCI-dataset" / "images" / "train" / "A"
    run_root = project_root / "runs" / "dino_pair_retrieval_simple_patch"
    levir_cc_overfit = project_root / "runs" / "levir_cc_simple_patch_smoke_overfit100"
    levir_cc_train = project_root / "runs" / "levir_cc_simple_patch_smoke_train"

    repo_clone.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "sample.png").write_text("x", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=repo_clone, check=True, capture_output=True, text=True)
    subprocess.run(["git", "checkout", "-b", "unit-test"], cwd=repo_clone, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "unit@test.local"], cwd=repo_clone, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Unit Test"], cwd=repo_clone, check=True, capture_output=True, text=True)
    (repo_clone / "README.md").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_clone, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_clone, check=True, capture_output=True, text=True)

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "train_summary.json").write_text(json.dumps({"best_epoch": 2}), encoding="utf-8")
    (run_root / "eval_summary.json").write_text(json.dumps({"overall_highlights": {"transition_recall@5": 0.9}}), encoding="utf-8")
    (project_root / "runs" / "levir_cc_random_retrieval_eval.json").write_text(json.dumps({"recall@5": 0.2}), encoding="utf-8")
    levir_cc_overfit.mkdir(parents=True, exist_ok=True)
    levir_cc_train.mkdir(parents=True, exist_ok=True)
    (levir_cc_overfit / "overfit_report.json").write_text(json.dumps({"gate_passed": True}), encoding="utf-8")
    (levir_cc_overfit / "eval_summary.json").write_text(json.dumps({"overall_highlights": {"recall@5": 1.0}}), encoding="utf-8")
    (levir_cc_train / "train_summary.json").write_text(json.dumps({"best_epoch": 5}), encoding="utf-8")
    (levir_cc_train / "eval_summary.json").write_text(json.dumps({"overall_highlights": {"recall@5": 0.8}}), encoding="utf-8")

    output = tmp_path / "inspect.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_project_state.py",
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
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["pair_retrieval_train_summary"]["best_epoch"] == 2
    assert payload["pair_retrieval_eval_summary"]["overall_highlights"]["transition_recall@5"] == 0.9
    assert payload["levir_cc_random_eval"]["recall@5"] == 0.2
    assert payload["levir_cc_simple_overfit_summary"]["gate_passed"] is True
    assert payload["levir_cc_simple_train_summary"]["best_epoch"] == 5
