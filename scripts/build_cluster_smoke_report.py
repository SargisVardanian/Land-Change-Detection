from __future__ import annotations

import argparse
import json
import platform
import socket
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified smoke-test report for the YSU-HPC change project.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional Prithvi model directory. Defaults to <project-root>/checkpoints/models/semantic/Prithvi-EO-2.0-300M-TL.",
    )
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
    required_datasets = ("LEVIR-CC", "LEVIR-MCI", "SECOND-CC")
    optional_datasets = ("BigEarthNet-v2", "DynamicEarthNet")
    required_models = (
        "mask2former-satellite",
        "Prithvi-EO-2.0-300M-TL",
        "Prithvi-EO-2.0-600M-TL",
    )
    required_indexes = (
        "levir_mci_samples.jsonl",
        "second_cc_samples.jsonl",
        "levir_cc_text_manifest.jsonl",
        "preview_samples.json",
    )
    optional_indexes = ("levir_mci_train_manifest.jsonl", "second_cc_train_manifest.jsonl")
    required_previews = ("levir_mci_preview.png", "second_cc_preview.png")

    raw_root = project_root / "datasets" / "raw"
    model_root = project_root / "checkpoints" / "models" / "semantic"
    indexes_root = project_root / "indexes"
    runs_root = project_root / "runs"
    datasets_manifest = project_root / "datasets" / "dataset_manifest.md"

    datasets = {name: _summarize_dir(raw_root / name) for name in required_datasets}
    optional_dataset_payload = {name: _summarize_dir(raw_root / name) for name in optional_datasets}
    models = {name: _summarize_dir(model_root / name) for name in required_models}
    indexes = {name: _summarize_file(indexes_root / name) for name in required_indexes}
    optional_index_payload = {name: _summarize_file(indexes_root / name) for name in optional_indexes}
    previews = {name: _summarize_file(runs_root / name) for name in required_previews}

    return {
        "project_root": str(project_root),
        "datasets_manifest": _summarize_file(datasets_manifest),
        "datasets": datasets,
        "optional_datasets": optional_dataset_payload,
        "models": models,
        "indexes": indexes,
        "optional_indexes": optional_index_payload,
        "previews": previews,
        "ready": all(
            [
                datasets_manifest.exists(),
                all(item["exists"] for item in datasets.values()),
                all(item["exists"] for item in models.values()),
                all(item["exists"] for item in indexes.values()),
                all(item["exists"] for item in previews.values()),
            ]
        ),
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root

    from land_change_detection.training.prithvi_experimental import build_prithvi_runtime_bundle

    runtime_model_dir = args.model_dir or (
        project_root / "checkpoints" / "models" / "semantic" / "Prithvi-EO-2.0-300M-TL"
    )
    verification = _build_project_assets_report(project_root)
    runtime_bundle = build_prithvi_runtime_bundle(model_dir=str(runtime_model_dir))

    report = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "project_root": str(project_root),
        "torch": _torch_summary(),
        "cluster_portal_dns_hint": {
            "host": "cluster.ysu.am",
            "resolved_addresses": _resolve_host("cluster.ysu.am"),
        },
        "project_assets": verification,
        "prithvi_runtime_bundle": runtime_bundle.to_dict(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if verification["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
