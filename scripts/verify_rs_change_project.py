from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REQUIRED_DATASETS = ("LEVIR-MCI",)
OPTIONAL_DATASETS = ("LEVIR-CC", "SECOND-CC")
REQUIRED_INDEXES = (
    "levir_mci_samples.jsonl",
    "levir_mci_validation.json",
    "preview_samples.json",
)
OPTIONAL_INDEXES = ("levir_cc_text_manifest.jsonl", "second_cc_samples.jsonl", "levir_mci_train_manifest.jsonl")
REQUIRED_PREVIEWS = ("levir_mci_preview.png", "levir_mci_grid.png")
OPTIONAL_PAIR_RETRIEVAL_RUN_FILES = (
    "dino_pair_retrieval_simple_patch/train_summary.json",
    "dino_pair_retrieval_simple_patch/eval_metrics.json",
    "dino_pair_retrieval_simple_patch/eval_summary.json",
)
LEVIR_CC_REQUIRED_FILES = (
    "levir_cc_manifest_validation.json",
    "levir_cc_random_retrieval_eval.json",
)
LEVIR_CC_PRESETS = ("simple_patch_smoke", "dinov2_t2_only", "dinov2_signed_delta", "dinov2_change_fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that the YSU-HPC rs_change_project layout contains required datasets, models, and bootstrap artifacts.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def _du_size(path: Path) -> str:
    result = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return "unknown"
    return result.stdout.split()[0]


def _summarize_dir(path: Path) -> dict:
    exists = path.exists()
    payload = {"exists": exists}
    if not exists:
        return payload
    files = 0
    dirs = 0
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
        elif item.is_dir():
            dirs += 1
    payload["num_files"] = files
    payload["num_dirs"] = dirs
    payload["size"] = _du_size(path)
    return payload


def _summarize_file(path: Path) -> dict:
    exists = path.exists()
    payload = {"exists": exists}
    if exists and path.is_file():
        payload["bytes"] = path.stat().st_size
    return payload


def _load_json(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_exists(summary: dict) -> bool:
    return bool(summary.get("exists"))


def _all_exist(items: dict[str, dict]) -> bool:
    return all(_artifact_exists(summary) for summary in items.values())


def _levir_cc_run_report(runs_root: Path, preset: str) -> dict:
    overfit_dir = runs_root / f"levir_cc_{preset}_overfit100"
    train_dir = runs_root / f"levir_cc_{preset}_train"
    report = {
        "overfit_dir": _summarize_dir(overfit_dir),
        "train_dir": _summarize_dir(train_dir),
        "overfit_report": _summarize_file(overfit_dir / "overfit_report.json"),
        "overfit_train_summary": _summarize_file(overfit_dir / "train_summary.json"),
        "overfit_eval_summary": _summarize_file(overfit_dir / "eval_summary.json"),
        "overfit_grid": _summarize_file(overfit_dir / "text_query_top5_grid.png"),
        "overfit_fingerprint": _summarize_file(overfit_dir / "environment_fingerprint.json"),
        "train_train_summary": _summarize_file(train_dir / "train_summary.json"),
        "train_eval_summary": _summarize_file(train_dir / "eval_summary.json"),
        "train_grid": _summarize_file(train_dir / "text_query_top5_grid.png"),
        "train_fingerprint": _summarize_file(train_dir / "environment_fingerprint.json"),
    }
    overfit_report_json = _load_json(overfit_dir / "overfit_report.json") or {}
    train_summary_json = _load_json(train_dir / "train_summary.json") or {}
    overfit_required = {
        "overfit_report": report["overfit_report"],
        "train_summary": report["overfit_train_summary"],
        "eval_summary": report["overfit_eval_summary"],
        "grid": report["overfit_grid"],
        "fingerprint": report["overfit_fingerprint"],
        "best_checkpoint": _summarize_file(overfit_dir / "best.pt"),
        "last_checkpoint": _summarize_file(overfit_dir / "last.pt"),
        "metrics_history": _summarize_file(overfit_dir / "metrics_history.json"),
        "eval_metrics": _summarize_file(overfit_dir / "eval_metrics.json"),
    }
    train_required = {
        "train_summary": report["train_train_summary"],
        "eval_summary": report["train_eval_summary"],
        "grid": report["train_grid"],
        "fingerprint": report["train_fingerprint"],
        "best_checkpoint": _summarize_file(train_dir / "best.pt"),
        "last_checkpoint": _summarize_file(train_dir / "last.pt"),
        "metrics_history": _summarize_file(train_dir / "metrics_history.json"),
        "eval_metrics": _summarize_file(train_dir / "eval_metrics.json"),
    }
    report["overfit_required"] = overfit_required
    report["train_required"] = train_required
    report["overfit_gate_passed"] = bool(overfit_report_json.get("gate_passed", False))
    report["overfit_artifacts_complete"] = _all_exist(overfit_required)
    report["train_artifacts_complete"] = _all_exist(train_required)
    report["train_best_epoch"] = train_summary_json.get("best_epoch")
    report["milestone_ready"] = (
        report["overfit_gate_passed"] and report["overfit_artifacts_complete"] and report["train_artifacts_complete"]
    )
    return report

def build_report(project_root: Path) -> dict:
    raw_root = project_root / "datasets" / "raw"
    indexes_root = project_root / "indexes"
    runs_root = project_root / "runs"
    datasets_manifest = project_root / "datasets" / "dataset_manifest.md"
    overfit_dir = runs_root / "levir_mci_overfit"
    smoke_report = runs_root / "cluster_smoke_report.json"
    verification_copy = project_root / "rs_change_project_verification.json"

    levir_unpacked = raw_root / "LEVIR-MCI-unpacked" / "LEVIR-MCI-dataset"
    levir_fallback = raw_root / "LEVIR-MCI"
    levir_payload = _summarize_dir(levir_unpacked)
    if not levir_payload["exists"]:
        levir_payload = _summarize_dir(levir_fallback)
    levir_payload["canonical_path"] = str(levir_unpacked)
    levir_payload["fallback_path"] = str(levir_fallback)
    datasets = {"LEVIR-MCI": levir_payload}
    optional_datasets = {name: _summarize_dir(raw_root / name) for name in OPTIONAL_DATASETS}
    indexes = {name: _summarize_file(indexes_root / name) for name in REQUIRED_INDEXES}
    optional_indexes = {name: _summarize_file(indexes_root / name) for name in OPTIONAL_INDEXES}
    previews = {name: _summarize_file(runs_root / name) for name in REQUIRED_PREVIEWS}
    optional_pair_retrieval_runs = {name: _summarize_file(runs_root / name) for name in OPTIONAL_PAIR_RETRIEVAL_RUN_FILES}
    pair_run_dir = runs_root / "dino_pair_retrieval_simple_patch"
    levir_cc_required = {name: _summarize_file(runs_root / name) for name in LEVIR_CC_REQUIRED_FILES}
    levir_cc_runs = {preset: _levir_cc_run_report(runs_root, preset) for preset in LEVIR_CC_PRESETS}
    levir_cc_required_complete = _all_exist(levir_cc_required)
    any_levir_cc_milestone_ready = any(run["milestone_ready"] for run in levir_cc_runs.values())

    all_required_datasets = all(item["exists"] for item in datasets.values())
    all_required_indexes = all(item["exists"] for item in indexes.values())
    all_required_previews = all(item["exists"] for item in previews.values())

    bootstrap_ready = all(
        [
            datasets_manifest.exists(),
            all_required_datasets,
            all_required_indexes,
            all_required_previews,
        ]
    )

    return {
        "project_root": str(project_root),
        "datasets_manifest": _summarize_file(datasets_manifest),
        "datasets": datasets,
        "optional_datasets": optional_datasets,
        "indexes": indexes,
        "optional_indexes": optional_indexes,
        "previews": previews,
        "optional_pair_retrieval_runs": optional_pair_retrieval_runs,
        "levir_cc_required": levir_cc_required,
        "levir_cc_runs": levir_cc_runs,
        "levir_cc_required_complete": levir_cc_required_complete,
        "levir_cc_any_milestone_ready": any_levir_cc_milestone_ready,
        "pair_retrieval_summary_highlights": {
            "train_summary": _load_json(pair_run_dir / "train_summary.json"),
            "eval_summary": _load_json(pair_run_dir / "eval_summary.json"),
        },
        "overfit_run": _summarize_dir(overfit_dir),
        "smoke_report": _summarize_file(smoke_report),
        "root_verification_copy": _summarize_file(verification_copy),
        "bootstrap_ready": bootstrap_ready,
        "complete": all(
            [
                bootstrap_ready,
                overfit_dir.exists(),
                smoke_report.exists(),
            ]
        ),
    }


def main() -> int:
    args = parse_args()
    report = build_report(args.project_root)
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["bootstrap_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
