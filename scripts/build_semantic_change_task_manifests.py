from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic-change task manifests from indexed change dataset JSONL files.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=("LEVIR-MCI", "SECOND-CC"), required=True)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def _build_change_mask_rows(rows: list[dict], dataset_name: str) -> list[dict]:
    output = []
    for row in rows:
        if not row.get("mask_path"):
            continue
        output.append(
            {
                "sample_id": row["sample_id"],
                "dataset_name": dataset_name,
                "task": "change_mask_segmentation",
                "before_path": row["before_path"],
                "after_path": row["after_path"],
                "change_mask_path": row["mask_path"],
                "caption": row.get("caption"),
                "split": row.get("split"),
                "metadata": dict(row.get("metadata", {})),
            }
        )
    return output


def _build_transition_rows(rows: list[dict], dataset_name: str) -> list[dict]:
    output = []
    for row in rows:
        if not row.get("semantic_before_path") or not row.get("semantic_after_path"):
            continue
        output.append(
            {
                "sample_id": row["sample_id"],
                "dataset_name": dataset_name,
                "task": "semantic_transition_segmentation",
                "before_path": row["before_path"],
                "after_path": row["after_path"],
                "semantic_before_path": row["semantic_before_path"],
                "semantic_after_path": row["semantic_after_path"],
                "change_mask_path": row.get("mask_path"),
                "caption": row.get("caption"),
                "split": row.get("split"),
                "metadata": dict(row.get("metadata", {})),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    rows = _read_rows(args.index)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mask_rows = _build_change_mask_rows(rows, args.dataset_name)
    _write_jsonl(args.output_dir / f"{args.dataset_name.lower()}_change_mask_manifest.jsonl".replace("-", "_"), mask_rows)
    print(f"Built change-mask manifest rows: {len(mask_rows)}")

    transition_rows = _build_transition_rows(rows, args.dataset_name)
    _write_jsonl(args.output_dir / f"{args.dataset_name.lower()}_transition_manifest.jsonl".replace("-", "_"), transition_rows)
    print(f"Built transition manifest rows: {len(transition_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
