from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_summarize_pair_retrieval_train_script(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "train_summary.json"
    (run_dir / "best.pt").write_text("x", encoding="utf-8")
    (run_dir / "last.pt").write_text("x", encoding="utf-8")
    (run_dir / "metrics_history.json").write_text(
        json.dumps(
            {
                "history": [
                    {
                        "epoch": 1,
                        "train": {"loss": 0.8, "recall@5": 0.4, "transition_recall@5": 0.6},
                        "eval": {
                            "loss": 0.7,
                            "recall@5": 0.5,
                            "mAP": 0.4,
                            "transition_recall@5": 0.7,
                            "transition_top1_hit_rate": 0.6,
                        },
                    },
                    {
                        "epoch": 2,
                        "train": {"loss": 0.6, "recall@5": 0.6, "transition_recall@5": 0.8},
                        "eval": {
                            "loss": 0.5,
                            "recall@5": 0.7,
                            "mAP": 0.5,
                            "transition_recall@5": 0.9,
                            "transition_top1_hit_rate": 0.8,
                        },
                    },
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_pair_retrieval_train.py",
            "--run-dir",
            str(run_dir),
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
    assert payload["best_epoch"] == 2
    assert payload["best_eval_highlights"]["recall@5"] == 0.7
    assert "transition_recall@5" not in payload["best_eval_highlights"]
    assert payload["best_eval_auxiliary_diagnostics"]["transition_recall@5"] == 0.9
    assert payload["best_checkpoint"].endswith("best.pt")


def test_summarize_pair_retrieval_train_omits_missing_transition_fields(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "train_summary.json"
    (run_dir / "best.pt").write_text("x", encoding="utf-8")
    (run_dir / "last.pt").write_text("x", encoding="utf-8")
    (run_dir / "metrics_history.json").write_text(
        json.dumps(
            {
                "history": [
                    {
                        "epoch": 1,
                        "train": {"loss": 0.8, "recall@5": 0.4},
                        "eval": {
                            "loss": 0.7,
                            "recall@5": 0.5,
                            "mAP": 0.4,
                            "pair_to_text_recall@5": 0.6,
                            "pair_to_text_MRR": 0.5,
                        },
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_pair_retrieval_train.py",
            "--run-dir",
            str(run_dir),
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
    assert "transition_recall@5" not in payload["best_eval_highlights"]
    assert payload["best_eval_auxiliary_diagnostics"] == {}
