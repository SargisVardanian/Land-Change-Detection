from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


DATASET_GROUPS = [
    ("stage_1_text_to_pair", [("LEVIR-CC", "LEVIR-CC")]),
    ("stage_2_grounded", [("LEVIR-MCI", "LEVIR-MCI-unpacked/LEVIR-MCI-dataset")]),
    ("stage_3_transition_aware", [("SECOND-CC", "SECOND-CC"), ("Hi-UCD", "Hi-UCD")]),
    ("stage_4_scale_up", [("RSCC", "RSCC"), ("RSRCC", "RSRCC"), ("ChangeIMTI", "ChangeIMTI"), ("CC-Foundation", "CC-Foundation"), ("UCCD", "UCCD")]),
    ("stage_5_temporal_later", [("DynamicEarthNet", "DynamicEarthNet"), ("SpaceNet7", "SpaceNet7")]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a quick file and size summary for change-retrieval datasets.")
    default_project_root = (
        Path("/mnt/weka") / os.environ.get("USER", "user") / "rs_change_project"
        if (Path("/mnt/weka") / os.environ.get("USER", "user")).exists()
        else Path("/data") / os.environ.get("USER", "user") / "rs_change_project"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_project_root / "datasets" / "raw",
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
    for group_name, datasets in DATASET_GROUPS:
        print("=" * 80)
        print(group_name)
        for label, rel_path in datasets:
            path = root / rel_path
            print("-" * 40)
            print(label)
            print("path:", path)
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
