from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from land_change_detection.dataset_curriculum import curriculum_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect retrieval-first project state for YSU-HPC.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=cwd)
    return result.stdout.strip() if result.stdout.strip() else result.stderr.strip()


def _recent_paths(path: Path, limit: int = 10) -> list[str]:
    if not path.exists():
        return []
    items = sorted(path.rglob("*"), key=lambda item: item.stat().st_mtime, reverse=True)
    return [str(item) for item in items[:limit]]


def _count_pngs(path: Path) -> int:
    return sum(1 for item in path.rglob("*.png")) if path.exists() else 0


def _load_json(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    repo_root = args.project_root / "code" / "project"
    dataset_root = args.project_root / "datasets" / "raw" / "LEVIR-MCI-unpacked" / "LEVIR-MCI-dataset"
    indexes_root = args.project_root / "indexes"
    runs_root = args.project_root / "runs"
    pair_run_root = runs_root / "dino_pair_retrieval_simple_patch"
    logs_root = args.project_root / "logs"
    models_root = args.project_root / "models"
    cache_root = args.project_root / ".cache"
    payload = {
        "git_branch": _run(["git", "branch", "--show-current"], cwd=repo_root),
        "git_commit": _run(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "project_root": str(args.project_root),
        "repo_root": str(repo_root),
        "dataset_exists": dataset_root.exists(),
        "dataset_root": str(dataset_root),
        "dataset_counts": {
            "train_A_png": _count_pngs(dataset_root / "images" / "train" / "A"),
            "val_A_png": _count_pngs(dataset_root / "images" / "val" / "A"),
            "test_A_png": _count_pngs(dataset_root / "images" / "test" / "A"),
        },
        "manifest_counts": {
            path.name: sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            for path in sorted(indexes_root.glob("*.jsonl"))
        } if indexes_root.exists() else {},
        "recent_runs": _recent_paths(runs_root),
        "recent_logs": _recent_paths(logs_root),
        "pair_retrieval_run_dir": str(pair_run_root),
        "pair_retrieval_train_summary": _load_json(pair_run_root / "train_summary.json"),
        "pair_retrieval_eval_summary": _load_json(pair_run_root / "eval_summary.json"),
        "model_sizes": _run(["du", "-sh", str(models_root)], cwd=repo_root) if models_root.exists() else "missing",
        "cache_sizes": _run(["du", "-sh", str(cache_root)], cwd=repo_root) if cache_root.exists() else "missing",
        "dataset_curriculum": curriculum_summary(args.project_root),
    }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
