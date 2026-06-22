from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

from land_change_detection.run_metadata import file_sha256


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tar.gz", ".tgz")
BEFORE_HINTS = {"a", "t1", "before"}
AFTER_HINTS = {"b", "t2", "after"}
SPLIT_ORDER = ("train", "val", "test")
EXTRACTION_MANIFEST_NAME = "levir_cc_extraction_manifest.json"
EXTRACTION_SENTINEL_NAME = ".extraction_complete.json"


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    split: str
    before_path: Path
    after_path: Path
    width: int
    height: int
    captions: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class ArchiveMember:
    archive_path: Path
    archive_type: str
    member_name: str
    size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive-aware deterministic LEVIR-CC preprocessing.")
    parser.add_argument("--root", type=Path, required=True, help="Raw LEVIR-CC root under datasets/raw/LEVIR-CC")
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
        path for path in candidates if any(token in path.name.lower() for token in ("caption", "change", "label", "levir"))
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
    candidates = {normalized.lower(), Path(normalized).stem.lower(), _strip_known_suffixes(Path(normalized).stem).lower()}
    return {candidate for candidate in candidates if candidate}


def _build_caption_map(path: Path) -> dict[str, list[dict[str, str]]]:
    mapping: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _load_caption_rows(path):
        keys = set()
        for field in ("id", "image_id", "sample_id", "filename", "image", "name"):
            keys |= _caption_key_variants(str(row.get(field) or ""))
        caption = str(row.get("caption") or row.get("text") or row.get("change_caption") or "")
        transition_label = str(row.get("transition_label") or row.get("class") or row.get("change_type") or "")
        split = str(row.get("split") or row.get("partition") or row.get("subset") or "").strip().lower()
        payload = {"caption": caption, "transition_label": transition_label, "split": split if split in {"train", "val", "test"} else ""}
        for key in keys:
            mapping[key].append(payload)
    return mapping


def _detect_archives(root: Path) -> list[Path]:
    archives: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(ARCHIVE_EXTENSIONS):
            archives.append(path)
    return archives


def _member_is_safe(member_name: str) -> bool:
    member = PurePosixPath(member_name)
    if member.is_absolute():
        return False
    parts = [part for part in member.parts if part not in ("", ".")]
    return all(part != ".." for part in parts)


def _zip_members(path: Path) -> list[ArchiveMember]:
    with zipfile.ZipFile(path) as archive:
        return [
            ArchiveMember(path, "zip", info.filename, 0 if info.is_dir() else int(info.file_size))
            for info in archive.infolist()
        ]


def _tar_members(path: Path) -> list[ArchiveMember]:
    mode = "r:gz" if path.name.lower().endswith((".tar.gz", ".tgz")) else "r:"
    with tarfile.open(path, mode) as archive:
        members = []
        for member in archive.getmembers():
            if member.isdir():
                size = 0
            else:
                size = int(member.size)
            members.append(ArchiveMember(path, "tar", member.name, size))
        return members


def _inspect_archives(archives: list[Path]) -> tuple[list[ArchiveMember], list[str], dict[str, Any]]:
    members: list[ArchiveMember] = []
    issues: list[str] = []
    sha_payload: list[dict[str, Any]] = []
    for archive in archives:
        if archive.name.lower().endswith(".zip"):
            archive_members = _zip_members(archive)
            archive_type = "zip"
        else:
            archive_members = _tar_members(archive)
            archive_type = "tar"
        for member in archive_members:
            if not _member_is_safe(member.member_name):
                issues.append(f"path_traversal:{archive.name}:{member.member_name}")
        members.extend(archive_members)
        sha_payload.append(
            {
                "path": str(archive),
                "archive_type": archive_type,
                "sha256": file_sha256(archive),
                "bytes": archive.stat().st_size,
                "member_count": len(archive_members),
            }
        )
    summary = {
        "archive_count": len(archives),
        "member_count": len(members),
        "raw_archives": sha_payload,
    }
    return members, issues, summary


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_extract_zip(path: Path, destination: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not _member_is_safe(info.filename):
                raise ValueError(f"Archive path traversal detected: {path.name}:{info.filename}")
            member_path = destination / PurePosixPath(info.filename)
            if info.is_dir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue
            member_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, member_path.open("wb") as target:
                data = source.read()
                target.write(data)
            file_count += 1
            total_bytes += len(data)
    return file_count, total_bytes


def _safe_extract_tar(path: Path, destination: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    mode = "r:gz" if path.name.lower().endswith((".tar.gz", ".tgz")) else "r:"
    with tarfile.open(path, mode) as archive:
        for member in archive.getmembers():
            if not _member_is_safe(member.name):
                raise ValueError(f"Archive path traversal detected: {path.name}:{member.name}")
            member_path = destination / PurePosixPath(member.name)
            if member.isdir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            member_path.parent.mkdir(parents=True, exist_ok=True)
            data = extracted.read()
            with member_path.open("wb") as target:
                target.write(data)
            file_count += 1
            total_bytes += len(data)
    return file_count, total_bytes


def _collect_tree_stats(root: Path) -> tuple[int, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def _ensure_extracted(raw_root: Path, processed_root: Path, verify_only: bool, force: bool) -> dict[str, Any]:
    archives = _detect_archives(raw_root)
    manifest_path = processed_root / EXTRACTION_MANIFEST_NAME
    sentinel_path = processed_root / EXTRACTION_SENTINEL_NAME
    extracted_layout_root = processed_root / "extracted"
    members, issues, archive_summary = _inspect_archives(archives)
    if issues:
        raise ValueError("; ".join(issues))

    expected_manifest = {
        "raw_root": str(raw_root.resolve()),
        "processed_root": str(processed_root.resolve()),
        "raw_archives": archive_summary["raw_archives"],
        "archive_member_count": archive_summary["member_count"],
    }
    current_manifest = _load_json(manifest_path)
    extraction_complete = (
        current_manifest is not None
        and all(current_manifest.get(key) == value for key, value in expected_manifest.items())
        and sentinel_path.exists()
        and extracted_layout_root.exists()
    )
    if verify_only:
        if archives and not extraction_complete:
            raise ValueError("Archive extraction has not been completed yet for verify-only mode.")
        if not archives and not raw_root.exists():
            raise ValueError(f"Raw root does not exist: {raw_root}")
    if archives and (force or not extraction_complete):
        if verify_only:
            raise ValueError("verify-only refuses to perform archive extraction.")
        processed_root.mkdir(parents=True, exist_ok=True)
        extracted_layout_root.mkdir(parents=True, exist_ok=True)
        extracted_files_before, _ = _collect_tree_stats(extracted_layout_root)
        if extracted_files_before and force:
            for item in sorted(extracted_layout_root.rglob("*"), reverse=True):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            extracted_layout_root.mkdir(parents=True, exist_ok=True)
        extracted_file_count = 0
        extracted_byte_size = 0
        for archive in archives:
            if archive.name.lower().endswith(".zip"):
                count, byte_size = _safe_extract_zip(archive, extracted_layout_root)
            else:
                count, byte_size = _safe_extract_tar(archive, extracted_layout_root)
            extracted_file_count += count
            extracted_byte_size += byte_size
        manifest_payload = dict(expected_manifest)
        manifest_payload["extracted_file_count"] = extracted_file_count
        manifest_payload["extracted_byte_size"] = extracted_byte_size
        manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
        sentinel_path.write_text(json.dumps({"complete": True, "manifest_sha256": file_sha256(manifest_path)}, indent=2) + "\n", encoding="utf-8")
    active_root = extracted_layout_root if archives else raw_root
    extracted_file_count, extracted_byte_size = _collect_tree_stats(active_root) if active_root.exists() else (0, 0)
    return {
        **archive_summary,
        "used_archives": bool(archives),
        "processed_root": str(processed_root.resolve()),
        "active_data_root": str(active_root.resolve()),
        "extraction_manifest": str(manifest_path.resolve()) if manifest_path.exists() else None,
        "extraction_sentinel": str(sentinel_path.resolve()) if sentinel_path.exists() else None,
        "extracted_file_count": extracted_file_count,
        "extracted_byte_size": extracted_byte_size,
        "active_root": active_root,
    }


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _hash_outputs(paths: Iterable[Path]) -> dict[str, str]:
    return {path.name: file_sha256(path) for path in paths if path.exists()}


def _write_text_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"raw_root: {report['raw_root']}",
        f"active_data_root: {report['active_data_root']}",
        f"caption_json: {report['caption_json']}",
        f"used_archives: {report['used_archives']}",
        f"unique_pair_count: {report['unique_pair_count']}",
        f"caption_row_count: {report['caption_row_count']}",
        f"pairs_by_split: {json.dumps(report['pairs_by_split'], sort_keys=True)}",
        f"captions_by_split: {json.dumps(report['captions_by_split'], sort_keys=True)}",
        f"captions_per_pair_histogram: {json.dumps(report['captions_per_pair_histogram'], sort_keys=True)}",
        f"raw_archive_sha256: {json.dumps(report['raw_archive_sha256'], sort_keys=True)}",
        f"extracted_file_count: {report['extracted_file_count']}",
        f"extracted_byte_size: {report['extracted_byte_size']}",
        f"missing_files: {len(report['missing_files'])}",
        f"corrupt_files: {len(report['corrupt_files'])}",
        f"dimension_mismatches: {len(report['dimension_mismatches'])}",
        f"split_leakage: {len(report['split_leakage'])}",
        f"manifest_hashes: {json.dumps(report['manifest_hashes'], sort_keys=True)}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_records(data_root: Path, max_pairs: int | None) -> tuple[list[PairRecord], dict[str, Any]]:
    caption_json = _find_caption_json(data_root)
    caption_map = _build_caption_map(caption_json)
    discovered, discovery_issues = _discover_image_pairs(data_root)
    if not discovered:
        raise ValueError(f"Could not map extracted layout under {data_root} to T1/T2 image pairs.")
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
            {"caption": entry["caption"], "transition_label": entry["transition_label"] or pair_id}
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
                captions=normalized_captions,
            )
        )

    split_leakage = sorted(pair_id for pair_id, splits in split_assignments.items() if len(splits) > 1)
    report = {
        "caption_json": str(caption_json.resolve()),
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
                    "caption_index": index,
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
    raw_root = args.root.resolve()
    project_root = (args.project_root or raw_root.parents[2]).resolve()
    processed_root = project_root / "datasets" / "processed" / "LEVIR-CC"
    indexes_dir = project_root / "indexes"
    reports_dir = project_root / "reports"
    outputs = {
        "pairs": indexes_dir / "levir_cc_pairs.jsonl",
        "caption_queries": indexes_dir / "levir_cc_caption_queries.jsonl",
        "train": indexes_dir / "levir_cc_caption_queries_train.jsonl",
        "val": indexes_dir / "levir_cc_caption_queries_val.jsonl",
        "test": indexes_dir / "levir_cc_caption_queries_test.jsonl",
        "overfit": indexes_dir / "levir_cc_caption_queries_overfit_100.jsonl",
        "legacy_train": indexes_dir / "levir_cc_train.jsonl",
        "legacy_val": indexes_dir / "levir_cc_val.jsonl",
        "legacy_test": indexes_dir / "levir_cc_test.jsonl",
        "legacy_overfit": indexes_dir / "levir_cc_overfit_100.jsonl",
        "report_json": reports_dir / "levir_cc_preprocess_report.json",
        "report_txt": reports_dir / "levir_cc_preprocess_report.txt",
    }

    extraction = _ensure_extracted(raw_root, processed_root, verify_only=args.verify_only, force=args.force)
    pair_records, record_report = _build_records(extraction["active_root"], args.max_pairs)
    report = {
        "raw_root": str(raw_root),
        "active_data_root": extraction["active_data_root"],
        "processed_root": extraction["processed_root"],
        "used_archives": extraction["used_archives"],
        "raw_archive_sha256": {entry["path"]: entry["sha256"] for entry in extraction["raw_archives"]},
        "archive_member_count": extraction["member_count"],
        "extracted_file_count": extraction["extracted_file_count"],
        "extracted_byte_size": extraction["extracted_byte_size"],
        "extraction_manifest": extraction["extraction_manifest"],
        "extraction_sentinel": extraction["extraction_sentinel"],
        **record_report,
    }

    if report["missing_files"] or report["corrupt_files"] or report["dimension_mismatches"] or report["split_leakage"]:
        report.update(
            {
                "unique_pair_count": len(pair_records),
                "caption_row_count": sum(len(record.captions) for record in pair_records),
                "pairs_by_split": dict(Counter(record.split for record in pair_records)),
                "captions_by_split": dict(Counter(record.split for record in pair_records for _ in record.captions)),
                "manifest_hashes": {},
            }
        )
        rendered = json.dumps(report, indent=2)
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
        _write_jsonl(outputs["caption_queries"], caption_rows)
        _write_jsonl(outputs["train"], train_rows)
        _write_jsonl(outputs["val"], val_rows)
        _write_jsonl(outputs["test"], test_rows)
        _write_jsonl(outputs["overfit"], overfit_rows)
        _write_jsonl(outputs["legacy_train"], train_rows)
        _write_jsonl(outputs["legacy_val"], val_rows)
        _write_jsonl(outputs["legacy_test"], test_rows)
        _write_jsonl(outputs["legacy_overfit"], overfit_rows)

    report.update(
        {
            "unique_pair_count": len(pair_rows),
            "caption_row_count": len(caption_rows),
            "pairs_by_split": {split: len({row["pair_id"] for row in _split_rows(caption_rows, split)}) for split in SPLIT_ORDER},
            "captions_by_split": {split: len(_split_rows(caption_rows, split)) for split in SPLIT_ORDER},
            "manifest_hashes": _hash_outputs(
                [
                    outputs["pairs"],
                    outputs["caption_queries"],
                    outputs["train"],
                    outputs["val"],
                    outputs["test"],
                    outputs["overfit"],
                ]
            ),
        }
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2)
    outputs["report_json"].write_text(rendered + "\n", encoding="utf-8")
    _write_text_report(outputs["report_txt"], report)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
