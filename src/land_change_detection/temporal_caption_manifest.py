from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from land_change_detection.models.retrieval_heads import classify_caption_semantics, normalize_caption_text


SCHEMA_VERSION = "temporal-caption-manifest-v1"
VALID_CAPTION_SOURCES = {"human", "model_generated", "semantic_template"}
VALID_TIME_ORDER = ("before", "after")
OFFICIAL_SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_caption_groups(captions: Iterable[str]) -> list[str]:
    return [normalize_caption_text(str(caption)) for caption in captions]


def namespaced_pair_id(dataset_name: str, split: str, original_id: str) -> str:
    dataset_key = dataset_name.strip().casefold().replace("-", "_")
    return f"{dataset_key}:{split}:{original_id}"


def image_dimensions(path: str | Path | None) -> dict[str, int | None]:
    if not path:
        return {"width": None, "height": None}
    with Image.open(path) as image:
        return {"width": int(image.width), "height": int(image.height)}


def file_fingerprint(path: str | Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = hashlib.blake2b(digest_size=16)
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_fingerprint(payload: dict[str, Any]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(stable_json_dumps(payload).encode("utf-8"))
    return digest.hexdigest()


def make_manifest_row(
    *,
    dataset_name: str,
    split: str,
    original_id: str,
    t1_path: str | Path,
    t2_path: str | Path,
    captions: Iterable[str],
    caption_source: str = "human",
    mask_path: str | Path | None = None,
    semantic_t1_path: str | Path | None = None,
    semantic_t2_path: str | Path | None = None,
    sensor: str | None = None,
    spatial_resolution: str | float | int | None = None,
    source_metadata: dict[str, Any] | None = None,
    time_order: tuple[str, str] = VALID_TIME_ORDER,
) -> dict[str, Any]:
    cleaned_captions = [str(caption).strip() for caption in captions if str(caption).strip()]
    if caption_source not in VALID_CAPTION_SOURCES:
        raise ValueError(f"caption_source must be one of {sorted(VALID_CAPTION_SOURCES)}")
    t1 = str(Path(t1_path))
    t2 = str(Path(t2_path))
    mask = str(Path(mask_path)) if mask_path else None
    semantic_t1 = str(Path(semantic_t1_path)) if semantic_t1_path else None
    semantic_t2 = str(Path(semantic_t2_path)) if semantic_t2_path else None
    dims = image_dimensions(t1) if Path(t1).is_file() else {"width": None, "height": None}
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "split": split,
        "original_id": original_id,
        "t1_sha": file_fingerprint(t1),
        "t2_sha": file_fingerprint(t2),
        "mask_sha": file_fingerprint(mask),
        "caption_source": caption_source,
        "time_order": list(time_order),
        "runtime_resize_only": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "pair_id": namespaced_pair_id(dataset_name, split, str(original_id)),
        "original_id": str(original_id),
        "split": split,
        "t1_path": t1,
        "t2_path": t2,
        "captions": cleaned_captions,
        "normalized_caption_groups": normalize_caption_groups(cleaned_captions),
        "caption_source": caption_source,
        "time_order": list(time_order),
        "mask_path": mask,
        "semantic_t1_path": semantic_t1,
        "semantic_t2_path": semantic_t2,
        "sensor": sensor,
        "spatial_resolution": spatial_resolution,
        "image_width": dims["width"],
        "image_height": dims["height"],
        "source_metadata": source_metadata or {},
        "preprocessing_fingerprint": preprocessing_fingerprint(fingerprint_payload),
    }


def load_json_rows(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path)
    if manifest.suffix.casefold() == ".jsonl":
        return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    for key in ("rows", "items", "annotations", "samples", "images", "data"):
        rows = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"Unsupported JSON manifest shape: {manifest}")


def load_table_rows(path: str | Path) -> list[dict[str, Any]]:
    table = Path(path)
    if table.suffix.casefold() in {".json", ".jsonl"}:
        return load_json_rows(table)
    with table.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def manifest_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for row in sorted(rows, key=lambda item: str(item.get("pair_id", ""))):
        digest.update(stable_json_dumps(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def manifest_file_fingerprint(path: str | Path) -> str:
    return file_fingerprint(path) or ""


def summarize_manifest_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    pairs_by_dataset_split: Counter[str] = Counter()
    captions_by_dataset_split: Counter[str] = Counter()
    captions_per_pair: Counter[int] = Counter()
    semantic_coverage: Counter[str] = Counter()
    dataset_names: set[str] = set()
    for row in rows:
        dataset = str(row.get("dataset_name", ""))
        split = str(row.get("split", ""))
        key = f"{dataset}:{split}"
        captions = [caption for caption in row.get("captions", []) if str(caption).strip()]
        dataset_names.add(dataset)
        pairs_by_dataset_split[key] += 1
        captions_by_dataset_split[key] += len(captions)
        captions_per_pair[len(captions)] += 1
        if row.get("semantic_t1_path"):
            semantic_coverage[f"{key}:semantic_t1"] += 1
        if row.get("semantic_t2_path"):
            semantic_coverage[f"{key}:semantic_t2"] += 1
    audit = audit_manifest_rows(rows)
    return {
        "row_count": len(rows),
        "dataset_names": sorted(dataset_names),
        "pairs_by_dataset_split": dict(sorted(pairs_by_dataset_split.items())),
        "captions_by_dataset_split": dict(sorted(captions_by_dataset_split.items())),
        "captions_per_pair_distribution": {str(key): value for key, value in sorted(captions_per_pair.items())},
        "semantic_map_coverage": dict(sorted(semantic_coverage.items())),
        "missing_files": [error for error in audit["errors"] if error.get("kind") == "missing_file"],
        "image_dimension_mismatches": [error for error in audit["errors"] if error.get("kind") == "inconsistent_image_dimensions"],
        "split_leakage": audit["split_leakage"],
        "fingerprint": manifest_fingerprint(rows),
        "valid": audit["valid"],
    }


def _coerce_captions(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                payload = json.loads(stripped)
                if isinstance(payload, list):
                    return [str(item).strip() for item in payload if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in stripped.split("|") if part.strip()]
    return []


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def annotation_rows_to_manifest(
    *,
    dataset_name: str,
    root: str | Path,
    annotations: str | Path,
    default_caption_source: str = "human",
    include_model_generated: bool = True,
    sensor: str | None = None,
    spatial_resolution: str | float | int | None = None,
) -> list[dict[str, Any]]:
    root_path = Path(root)
    rows: list[dict[str, Any]] = []
    for raw in load_table_rows(annotations):
        split = str(_pick(raw, "split", "subset") or "").strip()
        if split not in OFFICIAL_SPLITS:
            raise ValueError(f"{dataset_name} annotation row has invalid split: {split!r}")
        source = str(_pick(raw, "caption_source", "source") or default_caption_source).strip()
        if source in {"model", "generated", "model-generated"}:
            source = "model_generated"
        if source == "model_generated" and not include_model_generated:
            continue
        original_id = str(_pick(raw, "original_id", "pair_id", "sample_id", "id", "filename") or "").strip()
        if not original_id:
            raise ValueError(f"{dataset_name} annotation row is missing an id")
        captions = _coerce_captions(_pick(raw, "captions", "caption", "sentences", "text"))
        def resolve(value: Any) -> str | None:
            if value in (None, ""):
                return None
            path = Path(str(value))
            return str(path if path.is_absolute() else root_path / path)
        metadata = {key: value for key, value in raw.items() if key not in {"captions", "caption", "sentences", "text"}}
        return_row = make_manifest_row(
            dataset_name=dataset_name,
            split=split,
            original_id=original_id,
            t1_path=resolve(_pick(raw, "t1_path", "before_path", "image_before", "A", "im1", "image1")) or "",
            t2_path=resolve(_pick(raw, "t2_path", "after_path", "image_after", "B", "im2", "image2")) or "",
            captions=captions,
            caption_source=source,
            mask_path=resolve(_pick(raw, "mask_path", "label_path", "change_mask", "mask")),
            semantic_t1_path=resolve(_pick(raw, "semantic_t1_path", "semantic_before_path", "label1_path")),
            semantic_t2_path=resolve(_pick(raw, "semantic_t2_path", "semantic_after_path", "label2_path")),
            sensor=str(_pick(raw, "sensor") or sensor) if (_pick(raw, "sensor") or sensor) is not None else None,
            spatial_resolution=_pick(raw, "spatial_resolution", "resolution") or spatial_resolution,
            source_metadata=metadata,
        )
        rows.append(return_row)
    return rows


def audit_manifest_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    pair_ids = [str(row.get("pair_id", "")) for row in rows]
    pair_counts = Counter(pair_ids)
    duplicate_pair_ids = sorted(pair_id for pair_id, count in pair_counts.items() if pair_id and count > 1)
    if duplicate_pair_ids:
        errors.append({"kind": "duplicate_pair_ids", "pair_ids": duplicate_pair_ids})
    dataset_counts = Counter(str(row.get("dataset_name", "")) for row in rows)
    split_counts = Counter(str(row.get("split", "")) for row in rows)
    caption_source_counts = Counter(str(row.get("caption_source", "")) for row in rows)
    normalized_counts: Counter[str] = Counter()
    image_hashes: defaultdict[str, list[str]] = defaultdict(list)
    pair_hashes: defaultdict[tuple[str | None, str | None], list[tuple[str, str]]] = defaultdict(list)
    original_by_dataset: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    conflict_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        pair_id = str(row.get("pair_id", ""))
        split = str(row.get("split", ""))
        dataset = str(row.get("dataset_name", ""))
        if row.get("schema_version") != SCHEMA_VERSION:
            errors.append({"kind": "invalid_schema_version", "row": index, "pair_id": pair_id})
        if split not in OFFICIAL_SPLITS:
            errors.append({"kind": "invalid_split", "row": index, "pair_id": pair_id, "split": split})
        if row.get("time_order") != list(VALID_TIME_ORDER):
            errors.append({"kind": "invalid_temporal_order", "row": index, "pair_id": pair_id, "time_order": row.get("time_order")})
        captions = [str(caption) for caption in row.get("captions", []) if str(caption).strip()]
        if not captions:
            errors.append({"kind": "empty_captions", "row": index, "pair_id": pair_id})
        groups = row.get("normalized_caption_groups") or []
        expected_groups = normalize_caption_groups(captions)
        if list(groups) != expected_groups:
            errors.append({"kind": "normalized_caption_group_mismatch", "row": index, "pair_id": pair_id})
        normalized_counts.update(expected_groups)
        for key in ("t1_path", "t2_path"):
            path = row.get(key)
            if not path or not Path(str(path)).exists():
                errors.append({"kind": "missing_file", "row": index, "pair_id": pair_id, "field": key, "path": path})
        for key in ("mask_path", "semantic_t1_path", "semantic_t2_path"):
            path = row.get(key)
            if path and not Path(str(path)).exists():
                errors.append({"kind": "missing_file", "row": index, "pair_id": pair_id, "field": key, "path": path})
        t1_hash = file_fingerprint(row.get("t1_path"))
        t2_hash = file_fingerprint(row.get("t2_path"))
        if t1_hash:
            image_hashes[t1_hash].append(f"{pair_id}:t1")
        if t2_hash:
            image_hashes[t2_hash].append(f"{pair_id}:t2")
        if t1_hash and t2_hash:
            pair_hashes[(t1_hash, t2_hash)].append((pair_id, split))
        if row.get("image_width") is not None and row.get("image_height") is not None:
            try:
                t1_dims = image_dimensions(row.get("t1_path"))
                t2_dims = image_dimensions(row.get("t2_path"))
                if t1_dims != t2_dims or int(row["image_width"]) != t1_dims["width"] or int(row["image_height"]) != t1_dims["height"]:
                    errors.append({"kind": "inconsistent_image_dimensions", "row": index, "pair_id": pair_id})
            except Exception as exc:
                warnings.append({"kind": "dimension_probe_failed", "row": index, "pair_id": pair_id, "error": str(exc)})
        original_by_dataset[(dataset, str(row.get("original_id", "")))].add(split)
        semantics = [classify_caption_semantics(caption) for caption in captions]
        if semantics and all(item["no_change"] for item in semantics) and row.get("mask_path"):
            conflict_counts["all_no_change_captions_with_mask"] += 1
        if semantics and any(item["appeared"] for item in semantics) and any(item["disappeared"] for item in semantics):
            conflict_counts["appeared_disappeared_caption_contradiction"] += 1

    duplicate_images = {digest: refs for digest, refs in image_hashes.items() if len(refs) > 1}
    duplicated_pairs_across_splits = {
        "|".join(digest for digest in hashes if digest): refs
        for hashes, refs in pair_hashes.items()
        if len({split for _, split in refs}) > 1
    }
    leakage = {
        f"{dataset}:{original_id}": sorted(splits)
        for (dataset, original_id), splits in original_by_dataset.items()
        if original_id and len(splits) > 1
    }
    if duplicated_pairs_across_splits:
        errors.append({"kind": "duplicated_t1_t2_pairs_across_splits", "pairs": duplicated_pairs_across_splits})
    if leakage:
        errors.append({"kind": "train_validation_test_leakage", "items": leakage})
    return {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(rows),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "caption_source_counts": dict(sorted(caption_source_counts.items())),
        "duplicate_pair_ids": duplicate_pair_ids,
        "exact_duplicate_images": duplicate_images,
        "duplicated_t1_t2_pairs_across_splits": duplicated_pairs_across_splits,
        "split_leakage": leakage,
        "caption_mask_semantic_disagreement_counts": dict(sorted(conflict_counts.items())),
        "normalized_caption_frequency": dict(sorted(normalized_counts.items())),
        "normalized_caption_unique": len(normalized_counts),
    }
