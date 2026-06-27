from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from land_change_detection.dataset_curriculum import curriculum_summary
from scripts.levir_cc_audit import LEVIR_CC_PRESETS, all_exist, summarize_required_files, summarize_runs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified smoke-test report for the YSU-HPC change project.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _torch_summary() -> dict:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    payload = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        payload["device_name"] = torch.cuda.get_device_name(0)
    return payload


def _resolve_host(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    return sorted({info[4][0] for info in infos})


def _du_size(path: Path) -> str:
    import subprocess

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


def _build_project_assets_report(project_root: Path) -> dict:
    required_datasets = ("LEVIR-MCI",)
    optional_datasets = ("LEVIR-CC", "SECOND-CC", "Hi-UCD")
    required_indexes = ("levir_mci_samples.jsonl", "levir_mci_validation.json", "preview_samples.json")
    optional_indexes = (
        "levir_cc_pairs.jsonl",
        "levir_cc_caption_queries.jsonl",
        "levir_cc_caption_queries_train.jsonl",
        "levir_cc_caption_queries_val.jsonl",
        "levir_cc_caption_queries_test.jsonl",
        "levir_cc_caption_queries_overfit_100.jsonl",
        "second_cc_samples.jsonl",
        "levir_mci_train_manifest.jsonl",
    )
    required_previews = ("levir_mci_preview.png", "levir_mci_grid.png")
    optional_pair_retrieval_runs = (
        "dino_pair_retrieval_simple_patch/train_summary.json",
        "dino_pair_retrieval_simple_patch/eval_metrics.json",
        "dino_pair_retrieval_simple_patch/eval_summary.json",
    )

    raw_root = project_root / "datasets" / "raw"
    indexes_root = project_root / "indexes"
    runs_root = project_root / "runs"
    datasets_manifest = project_root / "datasets" / "dataset_manifest.md"
    overfit_dir = runs_root / "levir_mci_overfit"

    levir_root = raw_root / "LEVIR-MCI-unpacked" / "LEVIR-MCI-dataset"
    levir_fallback = raw_root / "LEVIR-MCI"
    levir_payload = _summarize_dir(levir_root)
    if not levir_payload["exists"]:
        levir_payload = _summarize_dir(levir_fallback)
    levir_payload["canonical_path"] = str(levir_root)
    levir_payload["fallback_path"] = str(levir_fallback)

    datasets = {"LEVIR-MCI": levir_payload}
    optional_dataset_payload = {name: _summarize_dir(raw_root / name) for name in optional_datasets}
    indexes = {name: _summarize_file(indexes_root / name) for name in required_indexes}
    optional_index_payload = {name: _summarize_file(indexes_root / name) for name in optional_indexes}
    previews = {name: _summarize_file(runs_root / name) for name in required_previews}
    optional_pair_retrieval_payload = {name: _summarize_file(runs_root / name) for name in optional_pair_retrieval_runs}
    pair_run_dir = runs_root / "dino_pair_retrieval_simple_patch"
    levir_cc_required = summarize_required_files(runs_root)
    levir_cc_runs = summarize_runs(runs_root, LEVIR_CC_PRESETS)
    levir_cc_required_complete = all_exist(levir_cc_required)
    levir_cc_any_milestone_ready = any(run["milestone_ready"] for run in levir_cc_runs.values())
    train_summary = json.loads((pair_run_dir / "train_summary.json").read_text(encoding="utf-8")) if (pair_run_dir / "train_summary.json").exists() else None
    eval_summary = json.loads((pair_run_dir / "eval_summary.json").read_text(encoding="utf-8")) if (pair_run_dir / "eval_summary.json").exists() else None
    benchmark_curriculum = {
        "stage_1_text_to_pair": {
            "dataset": "LEVIR-CC",
            "pair_manifest_ready": optional_index_payload["levir_cc_pairs.jsonl"]["exists"],
            "caption_query_manifest_ready": optional_index_payload["levir_cc_caption_queries.jsonl"]["exists"],
            "required_files_complete": levir_cc_required_complete,
            "any_milestone_ready": levir_cc_any_milestone_ready,
            "ready": (
                optional_dataset_payload["LEVIR-CC"]["exists"]
                and optional_index_payload["levir_cc_pairs.jsonl"]["exists"]
                and optional_index_payload["levir_cc_caption_queries.jsonl"]["exists"]
                and levir_cc_required_complete
            ),
        },
        "stage_2_grounded": {
            "dataset": "LEVIR-MCI",
            "ready": levir_payload["exists"] and indexes["levir_mci_validation.json"]["exists"] and previews["levir_mci_grid.png"]["exists"],
        },
        "stage_3_transition_aware": {
            "dataset": "SECOND-CC",
            "ready": optional_dataset_payload["SECOND-CC"]["exists"] and optional_index_payload["second_cc_samples.jsonl"]["exists"],
        },
    }

    return {
        "project_root": str(project_root),
        "datasets_manifest": _summarize_file(datasets_manifest),
        "datasets": datasets,
        "optional_datasets": optional_dataset_payload,
        "indexes": indexes,
        "optional_indexes": optional_index_payload,
        "previews": previews,
        "optional_pair_retrieval_runs": optional_pair_retrieval_payload,
        "levir_cc_required": levir_cc_required,
        "levir_cc_runs": levir_cc_runs,
        "pair_retrieval_summary_highlights": {
            "train_summary": train_summary,
            "eval_summary": eval_summary,
        },
        "benchmark_curriculum": benchmark_curriculum,
        "overfit_run": _summarize_dir(overfit_dir),
        "ready": all(
            [
                datasets_manifest.exists(),
                all(item["exists"] for item in datasets.values()),
                indexes_root.exists(),
            ]
        ),
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root

    verification = _build_project_assets_report(project_root)

    report = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "project_root": str(project_root),
        "torch": _torch_summary(),
        "cluster_portal_dns_hint": {
            "host": "cluster.ysu.am",
            "resolved_addresses": _resolve_host("cluster.ysu.am"),
        },
        "dataset_curriculum": curriculum_summary(project_root),
        "project_assets": verification,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
