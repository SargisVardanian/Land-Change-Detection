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


def build_report(project_root: Path) -> dict:
    raw_root = project_root / "datasets" / "raw"
    indexes_root = project_root / "indexes"
    runs_root = project_root / "runs"
    datasets_manifest = project_root / "datasets" / "dataset_manifest.md"
    overfit_dir = runs_root / "levir_mci_overfit"
    smoke_report = runs_root / "cluster_smoke_report.json"
    verification_copy = project_root / "rs_change_project_verification.json"

    datasets = {name: _summarize_dir(raw_root / name) for name in REQUIRED_DATASETS}
    optional_datasets = {name: _summarize_dir(raw_root / name) for name in OPTIONAL_DATASETS}
    indexes = {name: _summarize_file(indexes_root / name) for name in REQUIRED_INDEXES}
    optional_indexes = {name: _summarize_file(indexes_root / name) for name in OPTIONAL_INDEXES}
    previews = {name: _summarize_file(runs_root / name) for name in REQUIRED_PREVIEWS}

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
