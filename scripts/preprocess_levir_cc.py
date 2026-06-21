from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from land_change_detection.run_metadata import file_sha256


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
BEFORE_HINTS = {"a", "t1", "before"}
AFTER_HINTS = {"b", "t2", "after"}
SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    split: str
    before_path: Path
    after_path: Path
    width: int
    height: int
    captions: tuple[dict[str, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministically preprocess LEVIR-CC into pair-safe retrieval manifests.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("\\", "/")


def _strip_known_suffixes(stem: str) -> str:
    lowered = stem.lower()
    for suffix in ("_before", "_after", "_t1", "_t2"):
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _infer_role(path: Path) -> str | None:
    parent_tokens = {_normalize_token(part) for part in path.parts}
    if parent_tokens & BEFORE_HINTS:
        return "before"
    if parent_tokens & AFTER_HINTS:
        return "after"
    lowered = path.stem.lower()
    if lowered.endswith(("_before", "_a", "_t1")):
        return "before"
    if lowered.endswith(("_after", "_b", "_t2")):
        return "after"
    return None


def _canonical_pair_id(path: Path) -> str:
    parent_parts = [part.lower() for part in path.parts]
    for anchor in ("a", "b", "t1", "t2", "before", "after"):
        if anchor in parent_parts:
            return _strip_known_suffixes(path.stem)
    lowered = path.stem.lower()
    if lowered.endswith(("_a", "_b")):
        return path.stem.rsplit("_", 1)[0]
    return _strip_known_suffixes(path.stem)


def _infer_split_from_path(path: Path) -> str:
    normalized = [part.lower() for part in path.parts]
    if "train" in normalized:
        return "train"
    if "val" in normalized or "valid" in normalized or "validation" in normalized:
        return "val"
    if "test" in normalized:
        return "test"
    return "unknown"


def _deterministic_split(pair_id: str) -> str:
    bucket = int(hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _load_caption_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if "images" in payload and isinstance(payload["images"], list):
            return [dict(row) for row in payload["images"] if isinstance(row, dict)]
        if "items" in payload and isinstance(payload["items"], list):
            return [dict(row) for row in payload["items"] if isinstance(row, dict)]
        if all(isinstance(value, str) for value in payload.values()):
            return [{"id": key, "caption": value} for key, value in payload.items()]
    raise ValueError(f"Unsupported caption JSON format: {path}")


def _find_caption_json(root: Path) -> Path:
    candidates = sorted(root.rglob("*.json"))
    ranked = [
        path
        for path in candidates
        if any(token in path.name.lower() for token in ("caption", "change", "label", "levir"))
    ]
    chosen = ranked[0] if ranked else (candidates[0] if candidates else None)
    if chosen is None:
        raise FileNotFoundError(f"No caption JSON found under {root}")
    return chosen


def _caption_key_variants(raw: str) -> set[str]:
    value = raw.strip()
    if not value:
        return set()
    normalized = value.replace("\\", "/")
    candidates = {normalized.lower()}
    candidates.add(Path(normalized).stem.lower())
    candidates.add(_strip_known_suffixes(Path(normalized).stem).lower())
    return {candidate for candidate in candidates if candidate}


def _build_caption_map(path: Path) -> dict[str, list[dict[str, str]]]:
    mapping: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _load_caption_rows(path):
        keys = set()
        for field in ("id", "image_id", "sample_id", "filename", "image", "name"):
            keys |= _caption_key_variants(str(row.get(field) or ""))
        caption = str(row.get("caption") or row.get("text") or row.get("change_caption") or "").strip()
        transition_label = str(row.get("transition_label") or row.get("class") or row.get("change_type") or "").strip()
        split = str(row.get("split") or row.get("partition") or row.get("subset") or "").strip().lower()
        payload = {
            "caption": caption,
            "transition_label": transition_label,
            "split": split if split in {"train", "val", "test"} else "",
        }
        for key in keys:
            mapping[key].append(payload)
    return mapping


def _discover_image_pairs(root: Path) -> tuple[dict[str, dict[str, Path]], list[str]]:
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    issues: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        role = _infer_role(path)
        if role is None:
            continue
        pair_id = _canonical_pair_id(path)
        if role in grouped[pair_id] and grouped[pair_id][role] != path:
            issues.append(f"duplicate_{role}:{pair_id}:{path}")
            continue
        grouped[pair_id][role] = path
    return grouped, issues


def _verify_image(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return width, height


def _relative(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def _sample_id(pair_id: str, index: int, total: int) -> str:
    return pair_id if total == 1 else f"{pair_id}#cap{index:03d}"


def _hash_outputs(paths: list[Path]) -> dict[str, str]:
    return {path.name: file_sha256(path) for path in paths if path.exists()}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_text_report(path: Path, report: dict[str, Any]) -> None:
    histogram = report.get("captions_per_pair_histogram", {})
    lines = [
        f"root: {report['root']}",
        f"caption_json: {report['caption_json']}",
        f"unique_pair_count: {report.get('unique_pair_count', 0)}",
        f"caption_row_count: {report.get('caption_row_count', 0)}",
        f"dataset_bytes: {report['dataset_bytes']}",
        f"pairs_by_split: {json.dumps(report.get('pairs_by_split', {}), sort_keys=True)}",
        f"captions_by_split: {json.dumps(report.get('captions_by_split', {}), sort_keys=True)}",
        f"captions_per_pair_histogram: {json.dumps(histogram, sort_keys=True)}",
        f"missing_files: {len(report['missing_files'])}",
        f"corrupt_files: {len(report['corrupt_files'])}",
        f"dimension_mismatches: {len(report['dimension_mismatches'])}",
        f"split_leakage: {len(report['split_leakage'])}",
        f"manifest_hashes: {json.dumps(report.get('manifest_hashes', {}), sort_keys=True)}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_dataset_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _build_records(root: Path, max_pairs: int | None) -> tuple[list[PairRecord], dict[str, Any]]:
    caption_json = _find_caption_json(root)
    caption_map = _build_caption_map(caption_json)
    discovered, discovery_issues = _discover_image_pairs(root)
    pair_ids = sorted(discovered)
    if max_pairs is not None:
        pair_ids = pair_ids[: max(0, max_pairs)]

    pair_records: list[PairRecord] = []
    missing_files: list[str] = list(discovery_issues)
    corrupt_files: list[str] = []
    dimension_mismatches: list[str] = []
    split_assignments: dict[str, set[str]] = defaultdict(set)
    caption_histogram: Counter[int] = Counter()

    for pair_id in pair_ids:
        row = discovered[pair_id]
        if "before" not in row or "after" not in row:
            missing_files.append(f"missing_pair_half:{pair_id}")
            continue
        before_path = row["before"]
        after_path = row["after"]
        if not before_path.exists() or not after_path.exists():
            missing_files.append(f"missing_image:{pair_id}")
            continue
        try:
            before_size = _verify_image(before_path)
            after_size = _verify_image(after_path)
        except ValueError as exc:
            corrupt_files.append(str(exc))
            continue
        if before_size != after_size:
            dimension_mismatches.append(f"{pair_id}:{before_size}!={after_size}")
            continue
        inferred_splits = {split for split in (_infer_split_from_path(before_path), _infer_split_from_path(after_path)) if split != "unknown"}
        caption_entries = list(caption_map.get(pair_id.lower(), []))
        caption_splits = {entry["split"] for entry in caption_entries if entry["split"]}
        split_assignments[pair_id].update(inferred_splits | caption_splits)
        if len(split_assignments[pair_id]) > 1:
            continue
        split = next(iter(split_assignments[pair_id]), _deterministic_split(pair_id))
        normalized_captions = tuple(
            {
                "caption": entry["caption"],
                "transition_label": entry["transition_label"] or pair_id,
            }
            for entry in caption_entries
        ) or ({"caption": "", "transition_label": pair_id},)
        caption_histogram[len(normalized_captions)] += 1
        pair_records.append(
            PairRecord(
                pair_id=pair_id,
                split=split,
                before_path=before_path.resolve(),
                after_path=after_path.resolve(),
                width=before_size[0],
                height=before_size[1],
                captions=tuple(normalized_captions),
            )
        )

    split_leakage = sorted(pair_id for pair_id, splits in split_assignments.items() if len(splits) > 1)
    report = {
        "root": str(root.resolve()),
        "caption_json": str(caption_json.resolve()),
        "dataset_bytes": _collect_dataset_bytes(root),
        "missing_files": missing_files,
        "corrupt_files": corrupt_files,
        "dimension_mismatches": dimension_mismatches,
        "split_leakage": split_leakage,
        "captions_per_pair_histogram": {str(key): value for key, value in sorted(caption_histogram.items())},
    }
    return pair_records, report


def _records_to_rows(records: list[PairRecord], project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    caption_rows: list[dict[str, Any]] = []
    for record in records:
        before_rel = _relative(record.before_path, project_root)
        after_rel = _relative(record.after_path, project_root)
        sample_ids = [_sample_id(record.pair_id, index, len(record.captions)) for index in range(len(record.captions))]
        pair_rows.append(
            {
                "pair_id": record.pair_id,
                "split": record.split,
                "before_path": before_rel,
                "after_path": after_rel,
                "width": record.width,
                "height": record.height,
                "sample_ids": sample_ids,
                "captions": [entry["caption"] for entry in record.captions],
                "transition_labels": [entry["transition_label"] for entry in record.captions],
                "dataset_name": "LEVIR-CC",
            }
        )
        for index, entry in enumerate(record.captions):
            caption_rows.append(
                {
                    "sample_id": sample_ids[index],
                    "pair_id": record.pair_id,
                    "dataset_name": "LEVIR-CC",
                    "before_path": before_rel,
                    "after_path": after_rel,
                    "caption": entry["caption"],
                    "split": record.split,
                    "width": record.width,
                    "height": record.height,
                    "metadata": {
                        "transition_label": entry["transition_label"],
                        "retrieval_role": "text_to_pair_retrieval",
                        "curriculum_stage": "stage_1_text_to_pair",
                    },
                }
            )
    return pair_rows, caption_rows


def _split_rows(caption_rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in caption_rows if row["split"] == split]


def _overfit_rows(caption_rows: list[dict[str, Any]], limit_pairs: int = 100) -> list[dict[str, Any]]:
    selected_pair_ids: list[str] = []
    seen: set[str] = set()
    for row in caption_rows:
        if row["split"] != "train":
            continue
        pair_id = str(row["pair_id"])
        if pair_id in seen:
            continue
        seen.add(pair_id)
        selected_pair_ids.append(pair_id)
        if len(selected_pair_ids) >= limit_pairs:
            break
    selected = set(selected_pair_ids)
    return [row for row in caption_rows if row["pair_id"] in selected]


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    project_root = (args.project_root or root.parents[2]).resolve()
    indexes_dir = project_root / "indexes"
    reports_dir = project_root / "reports"
    outputs = {
        "pairs": indexes_dir / "levir_cc_pairs.jsonl",
        "train": indexes_dir / "levir_cc_train.jsonl",
        "val": indexes_dir / "levir_cc_val.jsonl",
        "test": indexes_dir / "levir_cc_test.jsonl",
        "overfit": indexes_dir / "levir_cc_overfit_100.jsonl",
        "report_json": reports_dir / "levir_cc_preprocess_report.json",
        "report_txt": reports_dir / "levir_cc_preprocess_report.txt",
    }

    if args.force and not args.verify_only:
        for path in outputs.values():
            if path.exists():
                path.unlink()

    pair_records, report = _build_records(root, args.max_pairs)
    report.update(
        {
            "unique_pair_count": len(pair_records),
            "caption_row_count": sum(len(record.captions) for record in pair_records),
            "pairs_by_split": dict(Counter(record.split for record in pair_records)),
            "captions_by_split": dict(
                Counter(record.split for record in pair_records for _ in record.captions)
            ),
            "manifest_hashes": {},
        }
    )
    if report["missing_files"] or report["corrupt_files"] or report["dimension_mismatches"] or report["split_leakage"]:
        rendered = json.dumps(report, indent=2)
        if not args.verify_only:
            reports_dir.mkdir(parents=True, exist_ok=True)
            outputs["report_json"].write_text(rendered + "\n", encoding="utf-8")
            _write_text_report(outputs["report_txt"], report)
        print(rendered)
        return 1

    pair_rows, caption_rows = _records_to_rows(pair_records, project_root)
    train_rows = _split_rows(caption_rows, "train")
    val_rows = _split_rows(caption_rows, "val")
    test_rows = _split_rows(caption_rows, "test")
    overfit_rows = _overfit_rows(caption_rows)

    if not args.verify_only:
        _write_jsonl(outputs["pairs"], pair_rows)
        _write_jsonl(outputs["train"], train_rows)
        _write_jsonl(outputs["val"], val_rows)
        _write_jsonl(outputs["test"], test_rows)
        _write_jsonl(outputs["overfit"], overfit_rows)

    report.update(
        {
            "unique_pair_count": len(pair_rows),
            "caption_row_count": len(caption_rows),
            "pairs_by_split": {split: len({row["pair_id"] for row in _split_rows(caption_rows, split)}) for split in SPLIT_ORDER},
            "captions_by_split": {split: len(_split_rows(caption_rows, split)) for split in SPLIT_ORDER},
            "manifest_hashes": _hash_outputs([outputs["pairs"], outputs["train"], outputs["val"], outputs["test"], outputs["overfit"]]),
        }
    )

    rendered = json.dumps(report, indent=2)
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs["report_json"].write_text(rendered + "\n", encoding="utf-8")
    _write_text_report(outputs["report_txt"], report)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
