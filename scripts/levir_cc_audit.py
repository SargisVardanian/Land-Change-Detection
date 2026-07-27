from __future__ import annotations

import json
from pathlib import Path


LEVIR_CC_REQUIRED_FILES = (
    "levir_cc_manifest_validation.json",
    "levir_cc_random_retrieval_eval.json",
)
LEVIR_CC_PRESETS = ("simple_patch_smoke", "retired_visual_baseline_t2_only", "retired_visual_baseline_signed_delta", "retired_visual_baseline_change_fusion")
OVERFIT_REQUIRED_FILENAMES = (
    "overfit_report.json",
    "train_summary.json",
    "eval_summary.json",
    "text_query_top5_grid.png",
    "environment_fingerprint.json",
    "best.pt",
    "last.pt",
    "metrics_history.json",
    "eval_metrics.json",
)
TRAIN_REQUIRED_FILENAMES = (
    "train_summary.json",
    "eval_summary.json",
    "text_query_top5_grid.png",
    "environment_fingerprint.json",
    "best.pt",
    "last.pt",
    "metrics_history.json",
    "eval_metrics.json",
)


def summarize_file(path: Path) -> dict:
    exists = path.exists()
    payload = {"exists": exists}
    if exists and path.is_file():
        payload["bytes"] = path.stat().st_size
    return payload


def load_json(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_exists(summary: dict) -> bool:
    return bool(summary.get("exists"))


def all_exist(items: dict[str, dict]) -> bool:
    return all(artifact_exists(summary) for summary in items.values())


def summarize_required_files(runs_root: Path) -> dict[str, dict]:
    return {name: summarize_file(runs_root / name) for name in LEVIR_CC_REQUIRED_FILES}


def summarize_run(runs_root: Path, preset: str) -> dict:
    overfit_dir = runs_root / f"levir_cc_{preset}_overfit100"
    train_dir = runs_root / f"levir_cc_{preset}_train"
    overfit_report_json = load_json(overfit_dir / "overfit_report.json") or {}
    train_summary_json = load_json(train_dir / "train_summary.json") or {}

    overfit_required = {name: summarize_file(overfit_dir / name) for name in OVERFIT_REQUIRED_FILENAMES}
    train_required = {name: summarize_file(train_dir / name) for name in TRAIN_REQUIRED_FILENAMES}

    overfit_artifacts_complete = all_exist(overfit_required)
    train_artifacts_complete = all_exist(train_required)
    overfit_gate_passed = bool(overfit_report_json.get("gate_passed", False))

    return {
        "preset": preset,
        "overfit_dir": str(overfit_dir),
        "train_dir": str(train_dir),
        "overfit_report": summarize_file(overfit_dir / "overfit_report.json"),
        "overfit_train_summary": summarize_file(overfit_dir / "train_summary.json"),
        "overfit_eval_summary": summarize_file(overfit_dir / "eval_summary.json"),
        "overfit_grid": summarize_file(overfit_dir / "text_query_top5_grid.png"),
        "overfit_fingerprint": summarize_file(overfit_dir / "environment_fingerprint.json"),
        "train_train_summary": summarize_file(train_dir / "train_summary.json"),
        "train_eval_summary": summarize_file(train_dir / "eval_summary.json"),
        "train_grid": summarize_file(train_dir / "text_query_top5_grid.png"),
        "train_fingerprint": summarize_file(train_dir / "environment_fingerprint.json"),
        "overfit_required": overfit_required,
        "train_required": train_required,
        "overfit_report_json": overfit_report_json,
        "train_summary_json": train_summary_json,
        "overfit_gate_passed": overfit_gate_passed,
        "overfit_artifacts_complete": overfit_artifacts_complete,
        "train_artifacts_complete": train_artifacts_complete,
        "train_best_epoch": train_summary_json.get("best_epoch"),
        "milestone_ready": overfit_gate_passed and overfit_artifacts_complete and train_artifacts_complete,
    }


def summarize_runs(runs_root: Path, presets: tuple[str, ...] = LEVIR_CC_PRESETS) -> dict[str, dict]:
    return {preset: summarize_run(runs_root, preset) for preset in presets}

