from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.temporal_caption_manifest import IMAGE_EXTENSIONS, make_manifest_row, write_jsonl


DIRECTION_SPECS = (
    ("appeared", "new buildings appeared", "label1"),
    ("disappeared", "buildings were demolished", "label2"),
)


def _resolve_directory(split_root: Path, requested: str) -> Path:
    direct = split_root / requested
    if direct.is_dir():
        return direct
    matches = [path for path in split_root.iterdir() if path.is_dir() and path.name.casefold() == requested.casefold()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Missing S2Looking directory {requested!r} under {split_root}")


def _image_map(directory: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS:
            key = path.stem.casefold()
            if key in mapping:
                raise ValueError(f"Duplicate S2Looking sample stem {path.stem!r} in {directory}")
            mapping[key] = path.resolve()
    return mapping


def build_rows(
    root: Path,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    image1_dir: str = "Image1",
    image2_dir: str = "Image2",
    new_label_dir: str = "label1",
    demolished_label_dir: str = "label2",
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    audit: dict[str, object] = {"dataset_name": "s2looking", "splits": {}, "errors": []}
    for split in splits:
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing official S2Looking split directory: {split_root}")
        directories = {
            "image1": _resolve_directory(split_root, image1_dir),
            "image2": _resolve_directory(split_root, image2_dir),
            "label1": _resolve_directory(split_root, new_label_dir),
            "label2": _resolve_directory(split_root, demolished_label_dir),
        }
        maps = {name: _image_map(path) for name, path in directories.items()}
        expected = set(maps["image1"])
        mismatch = {name: sorted(expected ^ set(mapping)) for name, mapping in maps.items() if set(mapping) != expected}
        if mismatch:
            raise ValueError(f"S2Looking file alignment mismatch in split {split}: {mismatch}")
        for stem in sorted(expected):
            for direction, query, label_key in DIRECTION_SPECS:
                original_id = f"{stem}:{direction}"
                row = make_manifest_row(
                    dataset_name="s2looking",
                    split=split,
                    original_id=original_id,
                    t1_path=maps["image1"][stem],
                    t2_path=maps["image2"][stem],
                    captions=[query],
                    caption_source="semantic_template",
                    mask_path=maps[label_key][stem],
                    sensor="side_looking_vhr",
                    spatial_resolution="0.5-0.8m",
                    source_metadata={
                        "base_pair_id": stem,
                        "object_category": "building",
                        "change_type": direction,
                        "query_mask_path": str(maps[label_key][stem]),
                        "seg_supervision_mode": "query_specific",
                        "retrieval_supervision": False,
                        "official_split": True,
                        "label_semantics": "newly_built" if direction == "appeared" else "demolished",
                    },
                )
                row["query_mask_path"] = row["mask_path"]
                row["seg_supervision_mode"] = "query_specific"
                row["retrieval_supervision"] = False
                rows.append(row)
        audit["splits"][split] = {  # type: ignore[index]
            "base_pairs": len(expected),
            "query_rows": 2 * len(expected),
            "newly_built_rows": len(expected),
            "demolished_rows": len(expected),
        }
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("S2Looking QCPR manifest produced duplicate pair IDs")
    audit["total_query_rows"] = len(rows)
    audit["total_base_pairs"] = len(rows) // 2
    audit["retrieval_supervision"] = False
    return rows, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build query-specific S2Looking QCPR manifest rows")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--split", action="append", choices=("train", "val", "test"), default=[])
    parser.add_argument("--image1-dir", default="Image1")
    parser.add_argument("--image2-dir", default="Image2")
    parser.add_argument("--new-label-dir", default="label1")
    parser.add_argument("--demolished-label-dir", default="label2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, audit = build_rows(
        args.root,
        splits=tuple(args.split or ("train", "val", "test")),
        image1_dir=args.image1_dir,
        image2_dir=args.image2_dir,
        new_label_dir=args.new_label_dir,
        demolished_label_dir=args.demolished_label_dir,
    )
    write_jsonl(args.output, rows)
    args.audit_report.parent.mkdir(parents=True, exist_ok=True)
    args.audit_report.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
