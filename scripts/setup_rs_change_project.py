from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DATASET_MANIFEST = """# Dataset Manifest

## LEVIR-MCI
Path: {root}/datasets/raw/LEVIR-MCI
Use:
- first real binary change segmentation experiment
- paired text-to-change retrieval
- overfit-100 verification run
- sample-grid QA

## SECOND-CC
Path: {root}/datasets/raw/SECOND-CC
Use:
- semantic maps
- change captions
- transition segmentation
- semantic transition follow-on work

## LEVIR-CC
Path: {root}/datasets/raw/LEVIR-CC
Use:
- optional caption-only auxiliary retrieval experiments
- not required for the first LEVIR-MCI binary+retrieval run
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the rs_change_project folder layout on YSU-HPC style storage.")
    default_root = (
        Path("/mnt/weka") / os.environ.get("USER", "user") / "rs_change_project"
        if (Path("/mnt/weka") / os.environ.get("USER", "user")).exists()
        else Path("/data") / os.environ.get("USER", "user") / "rs_change_project"
    )
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--write-manifest", action="store_true", help="Also write datasets/dataset_manifest.md.")
    return parser.parse_args()


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def ensure_layout(root: Path) -> list[Path]:
    paths = [
        root,
        root / "datasets" / "raw",
        root / "datasets" / "processed",
        root / "datasets" / "cache",
        root / "code",
        root / "logs",
        root / "runs",
        root / "checkpoints",
        root / "checkpoints" / "models",
        root / "indexes",
        root / "envs",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def main() -> int:
    args = parse_args()
    paths = ensure_layout(args.root)
    print(f"Prepared project root: {args.root}")
    for path in paths[1:]:
        print(f" - {path}")

    probe_path = args.root if args.root.exists() else args.root.parent
    free_gb = disk_free_gb(probe_path)
    print(f"Approx free space at {probe_path}: {free_gb:.1f} GB")
    if free_gb < 250:
        print("Storage warning: keep the immediate scope to LEVIR-MCI and lightweight verification artifacts.")
    else:
        print("Storage note: maintain LEVIR-MCI as the active first experiment unless scope changes explicitly.")

    if args.write_manifest:
        manifest_path = args.root / "datasets" / "dataset_manifest.md"
        manifest_path.write_text(DATASET_MANIFEST.format(root=args.root), encoding="utf-8")
        print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
