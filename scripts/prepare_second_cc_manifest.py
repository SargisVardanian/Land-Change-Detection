from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.temporal_caption_manifest import (
    IMAGE_EXTENSIONS,
    audit_manifest_rows,
    make_manifest_row,
    write_jsonl,
)


OFFICIAL_SPLITS = ("train", "val", "test")


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


def _sentence_text(sentence: dict) -> str:
    for key in ("raw", "caption", "text", "sentence"):
        value = sentence.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.strip().split())
    tokens = sentence.get("tokens")
    if isinstance(tokens, list):
        return " ".join(str(token).strip() for token in tokens if str(token).strip())
    return ""


def _row_split(raw_split: str, restval_policy: str) -> str:
    value = raw_split.strip()
    if value in OFFICIAL_SPLITS:
        return value
    if value == "restval":
        if restval_policy == "error":
            raise ValueError("SECOND-CC JSON contains restval split and --restval-policy=error")
        return restval_policy
    raise ValueError(f"Unsupported SECOND-CC split: {raw_split!r}")


def build_karpathy_rows(root: Path, annotations: Path, split: str, restval_policy: str) -> list[dict]:
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        raise ValueError("SECOND-CC Karpathy annotations must contain payload['images']")
    rows: list[dict] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        filename = str(image.get("filename") or "").strip()
        raw_split = str(image.get("split") or "").strip()
        if not filename or not raw_split:
            raise ValueError("SECOND-CC image rows must include filename and split")
        selected_split = _row_split(raw_split, restval_policy)
        if split != "all" and selected_split != split:
            continue
        sentences = image.get("sentences")
        if not isinstance(sentences, list):
            raise ValueError(f"SECOND-CC image row {filename!r} is missing sentences")
        captions = [_sentence_text(sentence) for sentence in sentences if isinstance(sentence, dict)]
        captions = [caption for caption in captions if caption]
        rows.append(
            make_manifest_row(
                dataset_name="second_cc",
                split=selected_split,
                original_id=Path(filename).stem,
                t1_path=root / selected_split / "rgb" / "A" / filename,
                t2_path=root / selected_split / "rgb" / "B" / filename,
                captions=captions,
                caption_source="human",
                mask_path=None,
                semantic_t1_path=root / selected_split / "sem" / "A" / filename,
                semantic_t2_path=root / selected_split / "sem" / "B" / filename,
                source_metadata={
                    "dataset": "SECOND-CC",
                    "annotation_file": str(annotations),
                    "official_karpathy_schema": True,
                    "official_split": raw_split,
                    "restval_policy": restval_policy if raw_split == "restval" else None,
                    "sentids": [sentence.get("sentid") for sentence in sentences if isinstance(sentence, dict) and sentence.get("sentid") is not None],
                },
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical temporal-caption manifest for SECOND-CC raw files.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=None, help="SECOND-CC-AUG.json Karpathy annotations. HDF5 is intentionally unsupported.")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--restval-policy", choices=("train", "val", "test", "error"), default="train")
    parser.add_argument("--expected-pairs", type=int, default=None)
    parser.add_argument("--expected-captions", type=int, default=None)
    parser.add_argument("--audit-report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.annotations and args.annotations.suffix.casefold() in {".h5", ".hdf5"}:
        raise SystemExit("SECOND-CC HDF5 annotations are not supported; use raw image/caption metadata.")
    rows = build_karpathy_rows(args.root, args.annotations, args.split, args.restval_policy) if args.annotations else discover_raw_rows(args.root, args.split)
    caption_count = sum(len(row.get("captions", [])) for row in rows)
    if args.expected_pairs is not None and len(rows) != args.expected_pairs:
        raise SystemExit(f"Expected {args.expected_pairs} SECOND-CC pairs, found {len(rows)}")
    if args.expected_captions is not None and caption_count != args.expected_captions:
        raise SystemExit(f"Expected {args.expected_captions} SECOND-CC captions, found {caption_count}")
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
