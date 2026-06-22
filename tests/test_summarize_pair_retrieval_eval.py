from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_summarize_pair_retrieval_eval_script(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    eval_json = tmp_path / "eval_metrics.json"
    output = tmp_path / "eval_summary.json"
    eval_json.write_text(
        json.dumps(
            {
                "recall@5": 0.7,
                "mAP": 0.6,
                "transition_recall@5": 1.0,
                "overall": {
                    "recall@1": 0.5,
                    "recall@5": 0.7,
                    "recall@10": 0.8,
                    "mAP": 0.6,
                    "MRR": 0.55,
                    "transition_recall@1": 0.9,
                    "transition_recall@5": 1.0,
                    "transition_recall@10": 1.0,
                    "transition_MRR": 0.95,
                    "transition_top1_hit_rate": 0.9,
                    "mean_transition_similarity_top5": 0.88,
                    "pair_sample_fraction": 0.5,
                },
                "by_source": {
                    "SECOND-CC": {
                        "recall@5": 0.8,
                        "mAP": 0.7,
                        "transition_recall@5": 1.0,
                        "transition_top1_hit_rate": 1.0,
                    }
                },
                "dataset_summary": {
                    "num_samples": 4,
                    "num_pair_samples": 2,
                    "num_caption_samples": 2,
                    "source_counts": {"levir": 2, "SECOND-CC": 2},
                    "dominant_transition_counts": {"1->2": 2},
                },
                "transition_summary": {
                    "pair_only": {"dominant_transition_counts": {"1->2": 2}},
                    "caption_or_grounded_text": {"dominant_transition_counts": {"urban_growth": 2}},
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_pair_retrieval_eval.py",
            "--eval-json",
            str(eval_json),
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
    assert payload["overall_highlights"]["transition_recall@5"] == 1.0
    assert payload["source_highlights"]["SECOND-CC"]["mAP"] == 0.7
    assert payload["dataset_summary"]["top_dominant_transitions"][0]["transition"] == "1->2"


def test_summarize_pair_retrieval_eval_omits_missing_transition_fields(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    eval_json = tmp_path / "eval_metrics.json"
    output = tmp_path / "eval_summary.json"
    eval_json.write_text(
        json.dumps(
            {
                "overall": {
                    "recall@1": 0.5,
                    "recall@5": 0.7,
                    "recall@10": 0.8,
                    "mAP": 0.6,
                    "MRR": 0.55,
                    "pair_to_text_recall@5": 0.9,
                    "pair_to_text_MRR": 0.8,
                },
                "by_source": {"levir": {"recall@5": 0.7, "mAP": 0.6}},
                "dataset_summary": {
                    "num_samples": 4,
                    "num_unique_pairs": 2,
                    "num_caption_rows": 4,
                    "num_pair_samples": 0,
                    "num_caption_samples": 4,
                    "source_counts": {"levir": 4},
                    "dominant_transition_counts": {"urban_growth": 4},
                },
                "transition_summary": {
                    "pair_only": {"dominant_transition_counts": {}},
                    "caption_or_grounded_text": {"dominant_transition_counts": {"urban_growth": 4}},
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_pair_retrieval_eval.py",
            "--eval-json",
            str(eval_json),
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
    assert "transition_recall@5" not in payload["overall_highlights"]
    assert "transition_top1_hit_rate" not in payload["source_highlights"]["levir"]
