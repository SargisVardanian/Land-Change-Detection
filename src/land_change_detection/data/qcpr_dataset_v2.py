"""Canonical, provenance-preserving Dataset-v2 utilities for QCPR.

The registry is deliberately independent from the training loaders.  In
particular, dense labels never appear in the mask-free retrieval records.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

MASK_KEY_RE = re.compile(r"(?:^|_)(?:mask|masks|semantic|target|label)(?:_|$)", re.I)
GENERIC_NO_CHANGE = {
    "there is no difference", "the two scenes seem identical",
    "the scene is the same as before", "no change has occurred",
    "almost nothing has changed",
}


def normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w ]+", " ", text.casefold()).split())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: str | Path) -> str | None:
    """Stable small-image digest; returns None if PIL cannot read the asset."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            image = image.convert("L").resize((16, 16))
            return hashlib.sha256(image.tobytes()).hexdigest()
    except Exception:
        return None


def jsonl_read(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def jsonl_write(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))
    with target.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return sha256_file(target)


def source_scene_id(row: dict[str, Any]) -> str:
    meta = row.get("source_metadata") or {}
    return str(meta.get("source_scene_group_id") or meta.get("scene_id") or row.get("pair_id"))


def change_status(row: dict[str, Any]) -> str:
    value = (row.get("source_metadata") or {}).get("changeflag")
    return "no_change" if value in (0, "0", False) else "changed"


def query_scope(caption: str, status: str) -> str:
    normalized = normalize_text(caption)
    if status == "no_change" and normalized in GENERIC_NO_CHANGE:
        return "generic_no_change"
    return "exact_pair"


def frame(path: str | None, timestamp: str, root: Path | None = None) -> dict[str, Any]:
    candidate = Path(path) if path else None
    exists = bool(candidate and candidate.exists())
    return {
        "path": str(candidate) if candidate else None,
        "timestamp": timestamp,
        "modality": "rgb",
        "sensor": None,
        "sha256": sha256_file(candidate) if exists else None,
        "perceptual_hash": perceptual_hash(candidate) if exists else None,
    }


def legacy_pair_to_v2(row: dict[str, Any], *, dataset_version: str = "existing") -> dict[str, Any]:
    pair_id = str(row["pair_id"])
    frames = [frame(row.get("t1_path"), "t1"), frame(row.get("t2_path"), "t2")]
    return {
        "canonical_pair_id": pair_id,
        "source_dataset": str(row.get("dataset_name", "unknown")),
        "source_version": dataset_version,
        "source_pair_id": pair_id,
        "source_scene_group_id": source_scene_id(row),
        "frames": frames,
        "timestamps": [x["timestamp"] for x in frames],
        "modalities": ["rgb", "rgb"],
        "sensor": None,
        "spatial_resolution": None,
        "height": None,
        "width": None,
        "is_synthetic": False,
        "synthetic_generator": None,
        "split": str(row.get("split", "unknown")),
        "license": None,
        # Compatibility views only; source of truth is frames.
        "t1_path": row.get("t1_path"),
        "t2_path": row.get("t2_path"),
    }


def captions_for_pair(row: dict[str, Any]) -> list[dict[str, Any]]:
    status = change_status(row)
    pair_id = str(row["pair_id"])
    result: list[dict[str, Any]] = []
    for index, text in enumerate(row.get("captions") or []):
        text = str(text).strip()
        if not text:
            continue
        scope = query_scope(text, status)
        normalized = normalize_text(text)
        result.append({
            "caption_id": f"{pair_id}:caption:{index}",
            "canonical_pair_id": pair_id,
            "text": text,
            "normalized_text": normalized,
            "caption_source": "human",
            "task_type": "temporal_retrieval",
            "query_scope": scope,
            "semantic_group_id": f"caption:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}",
            "equivalent_caption_group_id": None,
            "quality_score": 1.0,
            "identifiability_score": 0.0 if scope == "generic_no_change" else 1.0,
            "language": "en",
            "is_generated": False,
            "generator": None,
            "verification_status": "human",
            "change_status": status,
            "dataset_name": row.get("dataset_name"),
        })
    return result


def build_relevance(captions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    captions = list(captions)
    collisions: dict[str, set[str]] = defaultdict(set)
    for caption in captions:
        collisions[caption["normalized_text"]].add(caption["canonical_pair_id"])
    out = []
    for caption in captions:
        pair = caption["canonical_pair_id"]
        # A generic no-change query deliberately has no exact physical-pair
        # target.  Materialising every stable scene in ignored_pair_ids would
        # create a quadratic registry while adding no usable exact supervision.
        # Its semantic-group policy is enough to keep it out of ordinary
        # contrastive negatives.
        shared = [] if caption["query_scope"] == "generic_no_change" else sorted(
            collisions[caption["normalized_text"]] - {pair}
        )
        out.append({
            "caption_id": caption["caption_id"],
            "positive_pair_ids": [] if caption["query_scope"] == "generic_no_change" else [pair],
            "ignored_pair_ids": shared,
            "valid_negative_policy": "all_other_valid" if caption["query_scope"] != "generic_no_change" else "semantic_group_only",
        })
    return out


def mask_free(row: dict[str, Any]) -> dict[str, Any]:
    forbidden = [key for key in row if MASK_KEY_RE.search(key)]
    if forbidden:
        raise ValueError(f"mask-bearing keys forbidden in mask-free manifest: {forbidden}")
    return row


def build_legacy_v2(manifests: Iterable[str | Path], output_root: str | Path) -> dict[str, Any]:
    rows = [row for manifest in manifests for row in jsonl_read(manifest)]
    return build_rows_v2(rows, output_root)


def build_rows_v2(rows: Iterable[dict[str, Any]], output_root: str | Path) -> dict[str, Any]:
    """Write registries from already audited legacy rows."""
    rows = list(rows)
    pairs = {str(row["pair_id"]): legacy_pair_to_v2(row) for row in rows}
    captions = [caption for row in rows for caption in captions_for_pair(row)]
    relevance = build_relevance(captions)
    root = Path(output_root)
    hashes = {
        "pair_registry": jsonl_write(root / "pair_registry.jsonl", pairs.values()),
        "caption_registry": jsonl_write(root / "caption_registry.jsonl", captions),
        "relevance_registry": jsonl_write(root / "relevance_registry.jsonl", relevance),
        "dense_label_registry": jsonl_write(root / "dense_label_registry.jsonl", []),
    }
    return {"pairs": len(pairs), "captions": len(captions), "hashes": hashes}


def split_role(split: str) -> str:
    """Normalize source split labels without inventing a test split."""
    value = str(split).casefold()
    if value in {"val", "validation", "dev", "development"}:
        return "development"
    if value == "test":
        return "test"
    return "train"


def exclude_cross_split_image_conflicts(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep held-out examples and exclude training examples sharing an image.

    Source metadata must never be relabelled to hide a split collision.  For a
    connected image component that spans splits, ``test`` wins over
    ``development``, which wins over ``train``; every member in a lower-priority
    split is excluded.  The returned report is an immutable accounting record.
    """
    rows = list(rows)
    pairs = [legacy_pair_to_v2(row) for row in rows]
    roles_by_hash: dict[str, set[str]] = defaultdict(set)
    roles_by_scene: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        role = split_role(pair["split"])
        roles_by_scene[str(pair["source_scene_group_id"])].add(role)
        for item in pair["frames"]:
            if item.get("sha256"):
                roles_by_hash[str(item["sha256"])].add(role)
    priority = {"train": 0, "development": 1, "test": 2}
    excluded: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        pair_id = str(pair["canonical_pair_id"])
        role = split_role(pair["split"])
        competing_roles = set(roles_by_scene[str(pair["source_scene_group_id"])])
        for item in pair["frames"]:
            if item.get("sha256"):
                competing_roles.update(roles_by_hash[str(item["sha256"])])
        retained_role = max(competing_roles, key=lambda item: priority[item])
        if role != retained_role:
            excluded[role].add(pair_id)
    kept = [row for row in rows if str(row["pair_id"]) not in set().union(*excluded.values())]
    report = {
        "policy": "exclude lower-priority split members; never relabel a source split",
        "priority": ["test", "development", "train"],
        "cross_split_image_hashes_before": sum(len(roles) > 1 for roles in roles_by_hash.values()),
        "cross_split_scene_groups_before": sum(len(roles) > 1 for roles in roles_by_scene.values()),
        "excluded_pair_ids_by_source_role": {role: sorted(values) for role, values in sorted(excluded.items())},
        "excluded_pairs": sum(map(len, excluded.values())),
        "kept_pairs": len(kept),
    }
    return kept, report


def leakage_audit(pairs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_scene: dict[str, set[str]] = defaultdict(set)
    by_hash: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        split = str(pair["split"])
        by_scene[str(pair["source_scene_group_id"])].add(split)
        for item in pair.get("frames", []):
            if item.get("sha256"):
                by_hash[str(item["sha256"])].add(split)
    scene_leaks = {key: sorted(value) for key, value in by_scene.items() if len(value) > 1}
    image_leaks = {key: sorted(value) for key, value in by_hash.items() if len(value) > 1}
    return {"passed": not scene_leaks and not image_leaks, "scene_leaks": scene_leaks, "image_leaks": image_leaks}


def inventory(paths: dict[str, Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in paths.items():
        exists = path.exists()
        files = [p for p in path.rglob("*") if p.is_file()] if exists else []
        result[name] = {"path": str(path), "exists": exists, "file_count": len(files),
                        "bytes": sum(p.stat().st_size for p in files)}
    return result


def stable_manifest_hashes(root: str | Path) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(Path(root).glob("*.jsonl"))}
