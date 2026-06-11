from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


HF_DATASETS = {
    "LEVIR-MCI": "lcybuaa/LEVIR-MCI",
}

REFERENCE_REPOS = {
    "LEVIR-CC-Dataset": "https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset.git",
    "SecondCC": "https://github.com/ChangeCapsInRS/SecondCC.git",
    "ChangeChat": "https://github.com/hanlinwu/ChangeChat.git",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or prepare the first YSU-HPC change-retrieval datasets.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--skip-hf", action="store_true", help="Skip Hugging Face dataset downloads.")
    parser.add_argument("--force-hf", action="store_true", help="Download even if LEVIR-MCI already appears unpacked locally.")
    parser.add_argument("--skip-second-cc", action="store_true", help="Skip SECOND-CC preparation.")
    parser.add_argument("--include-levir-cc", action="store_true", help="Also download LEVIR-CC.")
    parser.add_argument("--skip-reference-repos", action="store_true", help="Skip cloning dataset reference repositories.")
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


def _clone_reference_repos(code_root: Path) -> None:
    code_root.mkdir(parents=True, exist_ok=True)
    for folder_name, repo_url in REFERENCE_REPOS.items():
        target = code_root / folder_name
        if target.exists():
            continue
        _run(["git", "clone", repo_url, str(target)])


def main() -> int:
    args = parse_args()
    raw_root = args.project_root / "datasets" / "raw"
    code_root = args.project_root / "code"
    raw_root.mkdir(parents=True, exist_ok=True)
    code_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_hf:
        hf_datasets = dict(HF_DATASETS)
        if args.include_levir_cc:
            hf_datasets["LEVIR-CC"] = "lcybuaa/LEVIR-CC"
        for dataset_name, repo_id in hf_datasets.items():
            if dataset_name == "LEVIR-MCI" and not args.force_hf:
                unpacked = raw_root / "LEVIR-MCI-unpacked" / "LEVIR-MCI-dataset" / "images"
                existing = raw_root / "LEVIR-MCI"
                if unpacked.exists() or existing.exists():
                    print(f"Skipping {dataset_name}: existing local copy detected under {raw_root}.")
                    continue
            print(f"Downloading {dataset_name} from Hugging Face...")
            _download_hf_dataset(raw_root, dataset_name, repo_id)

    if not args.skip_second_cc:
        print("Preparing SECOND-CC...")
        _prepare_second_cc(raw_root, args.second_cc_zenodo_doi, args.second_cc_gdrive_url)

    if not args.skip_reference_repos:
        print("Cloning reference repositories...")
        _clone_reference_repos(code_root)

    print("Dataset download/prepare stage complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
