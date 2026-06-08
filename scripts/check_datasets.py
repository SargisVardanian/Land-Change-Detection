from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_DATASETS = [
    "LEVIR-CC",
    "LEVIR-MCI",
    "SECOND-CC",
    "BigEarthNet-v2",
    "DynamicEarthNet",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a quick file and size summary for change-retrieval datasets.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data") / Path.home().name / "rs_change_project" / "datasets" / "raw",
        help="Dataset root directory.",
    )
    return parser.parse_args()


def dir_size(path: Path) -> str:
    result = subprocess.run(["du", "-sh", str(path)], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return "unknown"
    return result.stdout.split()[0] if result.stdout else "unknown"


def main() -> int:
    args = parse_args()
    root = args.root.expanduser()
    print(f"Dataset root: {root}")
    for name in DEFAULT_DATASETS:
        path = root / name
        print("=" * 80)
        print(name)
        print("exists:", path.exists())
        if not path.exists():
            continue
        files = 0
        dirs = 0
        for item in path.rglob("*"):
            if item.is_file():
                files += 1
            elif item.is_dir():
                dirs += 1
        print("num files:", files)
        print("num dirs:", dirs)
        print("size:", dir_size(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
