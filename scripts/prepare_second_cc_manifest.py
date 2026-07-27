from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from land_change_detection.temporal_caption_manifest import (
    IMAGE_EXTENSIONS,
    audit_manifest_rows,
    make_manifest_row,
)


OFFICIAL_SPLITS = ("train", "val", "test")
AUGMENTED_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_(?P<kind>random_augment(?:_[A-Za-z0-9]+)*)$")


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _base_pair_info(filename: str) -> tuple[str, bool, str | None, str]:
    stem = Path(filename).stem
    match = AUGMENTED_SUFFIX_RE.match(stem)
    if not match:
        return stem, False, None, stem
    return match.group("base"), True, match.group("kind"), stem


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
    rows, _ = build_karpathy_rows_with_audit(root, annotations, split, restval_policy, "canonical_only")
    return rows


def _pre_filter_summary(images: list[dict[str, Any]], restval_policy: str) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    caption_counts: Counter[str] = Counter()
    base_counts: defaultdict[str, set[str]] = defaultdict(set)
    augmented_counts: Counter[str] = Counter()
    for image in images:
        if not isinstance(image, dict):
            continue
        filename = str(image.get("filename") or "")
        raw_split = str(image.get("split") or "")
        try:
            selected_split = _row_split(raw_split, restval_policy)
        except ValueError:
            selected_split = raw_split
        base_pair_id, is_augmented, augmentation_kind, _ = _base_pair_info(filename)
        split_counts[selected_split] += 1
        if is_augmented:
            augmented_counts[selected_split] += 1
        sentences = image.get("sentences")
        if isinstance(sentences, list):
            caption_counts[selected_split] += sum(1 for sentence in sentences if isinstance(sentence, dict) and _sentence_text(sentence))
        base_counts[selected_split].add(base_pair_id)
    return {
        "row_count": sum(split_counts.values()),
        "rows_by_split": dict(sorted(split_counts.items())),
        "captions_by_split": dict(sorted(caption_counts.items())),
        "underlying_pairs_by_split": {key: len(value) for key, value in sorted(base_counts.items())},
        "augmented_rows_by_split": dict(sorted(augmented_counts.items())),
    }


def _select_karpathy_images(
    images: list[dict[str, Any]],
    *,
    split: str,
    restval_policy: str,
    augmentation_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []
    base_splits: defaultdict[str, set[str]] = defaultdict(set)
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            continue
        filename = str(image.get("filename") or "").strip()
        raw_split = str(image.get("split") or "").strip()
        selected_split = _row_split(raw_split, restval_policy)
        if split != "all" and selected_split != split:
            continue
        base_pair_id, is_augmented, augmentation_kind, view_id = _base_pair_info(filename)
        enriched = dict(image)
        enriched["_selected_split"] = selected_split
        enriched["_base_pair_id"] = base_pair_id
        enriched["_is_augmented"] = is_augmented
        enriched["_augmentation_kind"] = augmentation_kind
        enriched["_view_id"] = view_id
        enriched["_annotation_index"] = index
        base_splits[base_pair_id].add(selected_split)
        if augmentation_policy == "all_rows":
            all_rows.append(enriched)
        else:
            candidates[(selected_split, base_pair_id)].append(enriched)
    if augmentation_policy == "all_rows":
        selected = sorted(all_rows, key=lambda row: (row["_selected_split"], row["_base_pair_id"], row["_view_id"], row["_annotation_index"]))
    else:
        selected = []
        duplicate_canonical: list[dict[str, Any]] = []
        for (selected_split, base_pair_id), views in sorted(candidates.items()):
            canonical = [row for row in views if not row["_is_augmented"]]
            if len(canonical) > 1:
                duplicate_canonical.append({"split": selected_split, "base_pair_id": base_pair_id, "view_ids": [row["_view_id"] for row in canonical]})
            if augmentation_policy == "train_views" and selected_split == "train":
                selected.extend(sorted(views, key=lambda row: (row["_is_augmented"], row["_view_id"], row["_annotation_index"])))
            else:
                selected.append(sorted(canonical or views, key=lambda row: (row["_is_augmented"], row["_view_id"], row["_annotation_index"]))[0])
        if duplicate_canonical:
            raise ValueError(f"SECOND-CC has multiple canonical rows for one base pair/split: {duplicate_canonical[:10]}")
    augmented_val_test = [
        {"split": row["_selected_split"], "base_pair_id": row["_base_pair_id"], "view_id": row["_view_id"]}
        for row in selected
        if row["_is_augmented"] and row["_selected_split"] in {"val", "test"} and augmentation_policy in {"canonical_only", "train_views"}
    ]
    if augmented_val_test:
        raise ValueError(f"Augmented validation/test rows survived {augmentation_policy}: {augmented_val_test[:10]}")
    leakage = {base: sorted(splits) for base, splits in base_splits.items() if len(splits) > 1}
    if leakage:
        raise ValueError(f"SECOND-CC base-pair leakage across splits: {dict(list(leakage.items())[:10])}")
    return selected, {
        "augmentation_policy": augmentation_policy,
        "selected_row_count": len(selected),
        "selected_augmented_rows_by_split": dict(sorted(Counter(row["_selected_split"] for row in selected if row["_is_augmented"]).items())),
        "selected_underlying_pairs_by_split": {
            split_name: len({row["_base_pair_id"] for row in selected if row["_selected_split"] == split_name})
            for split_name in OFFICIAL_SPLITS
        },
        "warning": "all_rows is diagnostic-only; augmented validation/test views may inflate metrics." if augmentation_policy == "all_rows" else None,
    }


def build_karpathy_rows_with_audit(
    root: Path,
    annotations: Path,
    split: str,
    restval_policy: str,
    augmentation_policy: str,
) -> tuple[list[dict], dict[str, Any]]:
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        raise ValueError("SECOND-CC Karpathy annotations must contain payload['images']")
    selected_images, selection_report = _select_karpathy_images(
        images,
        split=split,
        restval_policy=restval_policy,
        augmentation_policy=augmentation_policy,
    )
    rows: list[dict] = []
    for image in selected_images:
        if not isinstance(image, dict):
            continue
        filename = str(image.get("filename") or "").strip()
        raw_split = str(image.get("split") or "").strip()
        if not filename or not raw_split:
            raise ValueError("SECOND-CC image rows must include filename and split")
        selected_split = str(image["_selected_split"])
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
                    "base_pair_id": image["_base_pair_id"],
                    "is_augmented": bool(image["_is_augmented"]),
                    "augmentation_kind": image["_augmentation_kind"],
                    "view_id": image["_view_id"],
                    "sentids": [sentence.get("sentid") for sentence in sentences if isinstance(sentence, dict) and sentence.get("sentid") is not None],
                },
            )
        )
    return rows, {"pre_filter": _pre_filter_summary(images, restval_policy), "post_filter": selection_report}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical temporal-caption manifest for SECOND-CC raw files.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=None, help="SECOND-CC-AUG.json Karpathy annotations. HDF5 is intentionally unsupported.")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--restval-policy", choices=("train", "val", "test", "error"), default="train")
    parser.add_argument("--augmentation-policy", choices=("canonical_only", "train_views", "all_rows"), default="canonical_only")
    parser.add_argument("--expected-pairs", type=int, default=None)
    parser.add_argument("--expected-captions", type=int, default=None)
    parser.add_argument("--audit-report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.annotations and args.annotations.suffix.casefold() in {".h5", ".hdf5"}:
        raise SystemExit("SECOND-CC HDF5 annotations are not supported; use raw image/caption metadata.")
    adapter_report: dict[str, Any] = {}
    rows: list[dict]
    if args.annotations:
        try:
            rows, adapter_report = build_karpathy_rows_with_audit(args.root, args.annotations, args.split, args.restval_policy, args.augmentation_policy)
        except ValueError as exc:
            if args.audit_report:
                args.audit_report.parent.mkdir(parents=True, exist_ok=True)
                args.audit_report.write_text(json.dumps({"valid": False, "error": str(exc)}, indent=2, sort_keys=True), encoding="utf-8")
            raise SystemExit(str(exc)) from exc
    else:
        rows = discover_raw_rows(args.root, args.split)
    caption_count = sum(len(row.get("captions", [])) for row in rows)
    report = audit_manifest_rows(rows)
    report = {
        **report,
        "second_cc_adapter": adapter_report,
        "post_filter_pair_count": len(rows),
        "post_filter_caption_count": caption_count,
    }
    if args.audit_report:
        args.audit_report.parent.mkdir(parents=True, exist_ok=True)
        args.audit_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(f"Manifest audit failed: {report['errors'][:3]}")
    if args.expected_pairs is not None and len(rows) != args.expected_pairs:
        summary = {
            "expected_pairs": args.expected_pairs,
            "found_pairs": len(rows),
            "expected_captions": args.expected_captions,
            "found_captions": caption_count,
            "adapter_summary": adapter_report,
        }
        raise SystemExit(f"SECOND-CC expected pair count mismatch: {json.dumps(summary, sort_keys=True)}")
    if args.expected_captions is not None and caption_count != args.expected_captions:
        summary = {
            "expected_captions": args.expected_captions,
            "found_captions": caption_count,
            "expected_pairs": args.expected_pairs,
            "found_pairs": len(rows),
            "adapter_summary": adapter_report,
        }
        raise SystemExit(f"SECOND-CC expected caption count mismatch: {json.dumps(summary, sort_keys=True)}")
    _write_jsonl_atomic(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
