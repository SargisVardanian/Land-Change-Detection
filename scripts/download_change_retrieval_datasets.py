from __future__ import annotations

import argparse
import json
import os
import sys
import shutil
import subprocess
from pathlib import Path

from land_change_detection.data.dataset_registry import (
    build_download_plan,
    current_hf_home,
    render_human_plan,
    registry_entry_by_name,
    write_download_checksums,
    write_download_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or prepare the first YSU-HPC change-retrieval datasets.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--skip-hf", action="store_true", help="Skip Hugging Face dataset downloads.")
    parser.add_argument("--force-hf", action="store_true", help="Download even if LEVIR-MCI already appears unpacked locally.")
    parser.add_argument("--skip-second-cc", action="store_true", help="Skip SECOND-CC preparation.")
    parser.add_argument("--include-levir-cc", action="store_true", help="Deprecated compatibility flag; LEVIR-CC is included by default.")
    parser.add_argument("--skip-reference-repos", action="store_true", help="Skip cloning dataset reference repositories.")
    parser.add_argument("--phase", choices=("baseline", "manual", "deferred", "all"), default="baseline")
    parser.add_argument("--reserve-gb", type=float, default=120.0)
    parser.add_argument("--max-download-gb", type=float, default=50.0)
    parser.add_argument("--allow-unknown-size", action="store_true")
    parser.add_argument(
        "--second-cc-zenodo-doi",
        default="10.5281/zenodo.16937571",
        help="Zenodo DOI for SECOND-CC.",
    )
    parser.add_argument(
        "--second-cc-gdrive-url",
        default="",
        help="Optional Google Drive URL for SECOND-CC fallback. If provided, a gdown command script is written.",
    )
    return parser.parse_args()


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _maybe_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _download_hf_dataset(raw_root: Path, dataset_name: str, repo_id: str) -> None:
    if not _maybe_tool("huggingface-cli"):
        raise RuntimeError("huggingface-cli is not installed or not on PATH.")
    out_dir = raw_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "huggingface-cli",
            "download",
            repo_id,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(out_dir),
        ]
    )


def _download_hf_model(model_root: Path, repo_id: str) -> None:
    if not _maybe_tool("huggingface-cli"):
        raise RuntimeError("huggingface-cli is not installed or not on PATH.")
    model_root.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "huggingface-cli",
            "download",
            repo_id,
            "--repo-type",
            "model",
            "--local-dir",
            str(model_root),
        ]
    )


def _prepare_second_cc(raw_root: Path, doi: str, gdrive_url: str) -> None:
    second_root = raw_root / "SECOND-CC"
    second_root.mkdir(parents=True, exist_ok=True)
    if _maybe_tool("zenodo_get"):
        _run(["zenodo_get", doi], cwd=second_root)
        return

    fallback_script = second_root / "download_second_cc_fallback.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'cd "{second_root}"',
        'echo "zenodo_get was not found. Use gdown or manual download here."',
    ]
    if gdrive_url:
        lines.append(f'gdown "{gdrive_url}"')
    fallback_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fallback_script.chmod(0o755)
def _download_from_entry(project_root: Path, entry: dict[str, object], args: argparse.Namespace) -> None:
    name = str(entry["canonical_name"])
    spec = registry_entry_by_name(name)
    if spec.download_handler == "hf_dataset":
        raw_root = project_root / "datasets" / "raw"
        repo_id = spec.source_identifier
        _download_hf_dataset(raw_root, spec.canonical_name, repo_id)
    elif spec.download_handler == "hf_model":
        model_root = project_root / spec.expected_directory
        _download_hf_model(model_root, spec.source_identifier)
    elif spec.download_handler == "zenodo_dataset":
        raw_root = project_root / "datasets" / "raw"
        _prepare_second_cc(raw_root, args.second_cc_zenodo_doi, args.second_cc_gdrive_url)
    elif spec.download_handler == "git_repo":
        target = project_root / spec.expected_directory
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            _run(["git", "clone", spec.source_identifier, str(target)])
    elif spec.download_handler == "none":
        return
    else:
        raise RuntimeError(f"Unsupported download handler for {name}: {spec.download_handler}")


def _recalculate_plan_after_filter(plan: dict[str, object]) -> dict[str, object]:
    entries = list(plan["entries"])
    totals = {
        "planned_download_bytes": sum(int(row["planned_download_bytes"] or 0) for row in entries if row["allowed"]),
        "planned_extracted_bytes": sum(int(row["planned_extracted_bytes"] or 0) for row in entries if row["allowed"]),
    }
    projected_free_after_download = int(plan["free_bytes"]) - totals["planned_download_bytes"]
    projected_free_after_extract = projected_free_after_download - totals["planned_extracted_bytes"]
    policy_violations: list[str] = []
    if totals["planned_download_bytes"] > int(plan["max_download_bytes"]):
        policy_violations.append("planned_download_bytes_exceed_cap")
    if totals["planned_download_bytes"] > 0 and projected_free_after_extract < int(plan["reserve_bytes"]):
        policy_violations.append("reserve_space_violated")
    plan["totals"] = totals
    plan["projected_free_after_download_bytes"] = projected_free_after_download
    plan["projected_free_after_extract_bytes"] = projected_free_after_extract
    plan["policy_violations"] = policy_violations
    return plan


def main() -> int:
    args = parse_args()
    raw_root = args.project_root / "datasets" / "raw"
    code_root = args.project_root / "code"
    reports_root = args.project_root / "reports"
    raw_root.mkdir(parents=True, exist_ok=True)
    code_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    hf_home = current_hf_home(args.project_root)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    plan = build_download_plan(
        args.project_root,
        phase=args.phase,
        reserve_gb=args.reserve_gb,
        max_download_gb=args.max_download_gb,
        allow_unknown_size=args.allow_unknown_size,
    )
    filtered_entries: list[dict[str, object]] = []
    for row in plan["entries"]:
        name = str(row["canonical_name"])
        if args.skip_hf and str(row["source_type"]) == "huggingface":
            continue
        if args.skip_second_cc and name == "SECOND-CC":
            continue
        if args.skip_reference_repos and str(row["category"]) == "reference_repo":
            continue
        if not args.force_hf and name == "LEVIR-MCI" and row["exists"]:
            continue
        filtered_entries.append(row)

    plan["entries"] = filtered_entries
    _recalculate_plan_after_filter(plan)
    print(render_human_plan(plan), end="")
    if plan["policy_violations"]:
        print("Refusing download because the filtered plan violates storage policy.", file=sys.stderr)
        inventory_path = write_download_inventory(args.project_root, plan)
        print(f"Wrote inventory to {inventory_path}", file=sys.stderr)
        return 1

    inventory_path = write_download_inventory(args.project_root, plan)
    (reports_root / "download_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    for row in filtered_entries:
        if not row["allowed"]:
            continue
        print(f"Downloading/preparing {row['canonical_name']}...")
        _download_from_entry(args.project_root, row, args)

    checksum_path = write_download_checksums(args.project_root, plan)
    print(f"Dataset download/prepare stage complete. Inventory: {inventory_path}. Checksums: {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
