from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.temporal_caption_manifest import (
    IMAGE_EXTENSIONS,
    annotation_rows_to_manifest,
    audit_manifest_rows,
    make_manifest_row,
    write_jsonl,
)


def _caption_map(root: Path) -> dict[str, list[str]]:
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and all(isinstance(key, str) for key in payload):
            mapped: dict[str, list[str]] = {}
            for key, value in payload.items():
                if isinstance(value, list):
                    captions = [str(item).strip() for item in value if str(item).strip()]
                else:
                    captions = [str(value).strip()] if str(value).strip() else []
                if captions:
                    mapped[Path(key).stem] = captions
                    mapped[key] = captions
            if mapped:
                return mapped
    return {}


def _find_split_dirs(root: Path, split: str) -> tuple[Path, Path, Path | None, Path | None, Path | None] | None:
    base = root / split
    candidates = [
        ("A", "B", "label", "semantic_A", "semantic_B"),
        ("before", "after", "mask", "semantic_before", "semantic_after"),
        ("im1", "im2", "label", "label1", "label2"),
        ("T1", "T2", "change", "semantic_t1", "semantic_t2"),
    ]
    for before_name, after_name, mask_name, sem1_name, sem2_name in candidates:
        before = base / before_name
        after = base / after_name
        if before.exists() and after.exists():
            return (
                before,
                after,
                base / mask_name if (base / mask_name).exists() else None,
                base / sem1_name if (base / sem1_name).exists() else None,
                base / sem2_name if (base / sem2_name).exists() else None,
            )
    return None


def discover_raw_rows(root: Path, split: str) -> list[dict]:
    captions = _caption_map(root)
    rows: list[dict] = []
    for selected_split in ("train", "val", "test"):
        if split != "all" and selected_split != split:
            continue
        dirs = _find_split_dirs(root, selected_split)
        if dirs is None:
            continue
        before_dir, after_dir, mask_dir, sem1_dir, sem2_dir = dirs
        for t1 in sorted(path for path in before_dir.iterdir() if path.suffix.casefold() in IMAGE_EXTENSIONS):
            t2 = after_dir / t1.name
            if not t2.exists():
                continue
            original_id = t1.stem
            caption_values = captions.get(original_id) or captions.get(t1.name) or []
            rows.append(
                make_manifest_row(
                    dataset_name="second_cc",
                    split=selected_split,
                    original_id=original_id,
                    t1_path=t1,
                    t2_path=t2,
                    captions=caption_values,
                    caption_source="human",
                    mask_path=(mask_dir / t1.name) if mask_dir and (mask_dir / t1.name).exists() else None,
                    semantic_t1_path=(sem1_dir / t1.name) if sem1_dir and (sem1_dir / t1.name).exists() else None,
                    semantic_t2_path=(sem2_dir / t1.name) if sem2_dir and (sem2_dir / t1.name).exists() else None,
                    source_metadata={"dataset": "SECOND-CC", "raw_image_layout": True},
                )
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical temporal-caption manifest for SECOND-CC raw files.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=None, help="Optional JSON/JSONL/CSV annotations. HDF5 is intentionally unsupported.")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--audit-report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.annotations and args.annotations.suffix.casefold() in {".h5", ".hdf5"}:
        raise SystemExit("SECOND-CC HDF5 annotations are not supported; use raw image/caption metadata.")
    rows = (
        annotation_rows_to_manifest(dataset_name="second_cc", root=args.root, annotations=args.annotations)
        if args.annotations
        else discover_raw_rows(args.root, args.split)
    )
    if args.split != "all":
        rows = [row for row in rows if row["split"] == args.split]
    write_jsonl(args.output, rows)
    report = audit_manifest_rows(rows)
    if args.audit_report:
        args.audit_report.parent.mkdir(parents=True, exist_ok=True)
        args.audit_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(f"Manifest audit failed: {report['errors'][:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
