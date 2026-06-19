from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.change_retrieval_datasets import ChangeRetrievalSample
from land_change_detection.manual_dataset_imports import (
    build_transition_manifest_rows,
    canonical_import_outputs,
    directory_inventory,
    ensure_symlink,
    samples_to_rows,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and import a manually downloaded Hi-UCD dataset.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--before-dir", type=Path, default=None)
    parser.add_argument("--after-dir", type=Path, default=None)
    parser.add_argument("--semantic-before-dir", type=Path, default=None)
    parser.add_argument("--semantic-after-dir", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, required=True)
    return parser.parse_args()


def _index_dir(path: Path | None) -> dict[str, Path]:
    if path is None or not path.exists():
        return {}
    return {item.stem.lower(): item for item in sorted(path.rglob("*")) if item.is_file()}


def _recursive_discover(root: Path) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path], dict[str, Path]]:
    before: dict[str, Path] = {}
    after: dict[str, Path] = {}
    sem_before: dict[str, Path] = {}
    sem_after: dict[str, Path] = {}
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        stem = item.stem.lower()
        parent = item.parent.name.lower()
        if parent == "before" or stem.endswith(("_a", "_before", "_t1")):
            before[stem.rsplit("_", 1)[0] if "_" in stem else stem] = item
        elif parent == "after" or stem.endswith(("_b", "_after", "_t2")):
            after[stem.rsplit("_", 1)[0] if "_" in stem else stem] = item
        elif "sem_before" in parent or stem.endswith(("_sem1", "_label1")):
            sem_before[stem.rsplit("_", 1)[0] if "_" in stem else stem] = item
        elif "sem_after" in parent or stem.endswith(("_sem2", "_label2")):
            sem_after[stem.rsplit("_", 1)[0] if "_" in stem else stem] = item
    return before, after, sem_before, sem_after


def main() -> int:
    args = parse_args()
    root = args.root or args.project_root / "datasets" / "manual" / "Hi-UCD"
    if not root.exists():
        raise SystemExit(f"Manual Hi-UCD root not found: {root}")
    if all(path is not None for path in (args.before_dir, args.after_dir, args.semantic_before_dir, args.semantic_after_dir)):
        before = _index_dir(args.before_dir)
        after = _index_dir(args.after_dir)
        sem_before = _index_dir(args.semantic_before_dir)
        sem_after = _index_dir(args.semantic_after_dir)
    else:
        before, after, sem_before, sem_after = _recursive_discover(root)
    shared = sorted(set(before) & set(after) & set(sem_before) & set(sem_after))
    if not shared:
        raise SystemExit(f"No matched Hi-UCD samples discovered under {root}")
    samples = [
        ChangeRetrievalSample(
            sample_id=key,
            dataset_name="Hi-UCD",
            before_path=str(before[key]),
            after_path=str(after[key]),
            semantic_before_path=str(sem_before[key]),
            semantic_after_path=str(sem_after[key]),
            split=None,
            metadata={"source_root": str(root)},
        )
        for key in shared
    ]
    outputs = canonical_import_outputs(args.project_root, "hi_ucd")
    sample_manifest = write_jsonl(outputs["sample_manifest"], samples_to_rows(samples))
    pair_rows = build_transition_manifest_rows(samples, dataset_name="Hi-UCD", num_classes=args.num_classes)
    pair_manifest = write_jsonl(outputs["pair_manifest"], pair_rows)
    raw_link = ensure_symlink(args.project_root / "datasets" / "raw" / "Hi-UCD", root)
    payload = {
        "dataset_name": "Hi-UCD",
        "manual_root": str(root),
        "raw_link": str(raw_link),
        "inventory": directory_inventory(root),
        "sample_manifest": str(sample_manifest),
        "pair_manifest": str(pair_manifest),
        "num_samples": len(samples),
        "num_pair_rows": len(pair_rows),
    }
    write_json(root / "IMPORT_PROVENANCE.json", payload)
    write_json(outputs["report"], payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
