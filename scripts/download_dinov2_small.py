from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    default_root = Path(os.environ.get("RS_PROJECT_ROOT", "/mnt/weka/shared/rs_change_project"))
    parser = argparse.ArgumentParser(description="Download DINOv2-small into RS project storage.")
    parser.add_argument("--output-dir", type=Path, default=default_root / "models" / "dinov2-small")
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument("--cache-dir", type=Path, default=default_root / ".cache" / "huggingface")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def _du(path: Path) -> str:
    import subprocess

    result = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True, check=False)
    return result.stdout.split()[0] if result.returncode == 0 and result.stdout else "unknown"


def main() -> int:
    args = parse_args()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(args.output_dir),
        local_dir_use_symlinks=False,
        cache_dir=str(args.cache_dir),
        token=args.token or None,
        max_workers=args.max_workers,
    )
    config_path = Path(local_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json is missing under {local_dir}")
    files = sorted(str(path.relative_to(local_dir)) for path in Path(local_dir).rglob("*") if path.is_file())
    print(f"Downloaded model to: {local_dir}")
    print(f"Disk usage: {_du(Path(local_dir))}")
    print("Files:")
    for file_name in files:
        print(f" - {file_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
