#!/usr/bin/env python3
"""Build the QCPR retrieval-semantic repair candidate package.

The output is deliberately a candidate/hold package.  It is safe to inspect
and evaluate, but no repaired query is training-enabled until the required
human review gates have actual decisions.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional for manifest-only use
    Image = None  # type: ignore[assignment]

from qcpr_data.queries.purpose import (
    QUERY_PURPOSES,
    attribute_similarity,
    classify_query,
    extract_visual_attributes,
    normalize_query_text,
    semantic_group_id,
    semantic_signature,
    unsupported_claims,
)


_CONTRACT_VERIFICATION_STATES = {
    "human", "human_rewritten", "human_adjudicated", "generated_verified",
    "generated_unverified", "rule_based_unverified", "derived_eval", "rejected",
}


def contract_verification(value: Any) -> str:
    """Map audit-only labels to the canonical release verification enum."""

    text = str(value or "").strip()
    if text in _CONTRACT_VERIFICATION_STATES:
        return text
    if text in {"visual_probe_candidate", "source_unverified", "unverified", ""}:
        return "rule_based_unverified"
    if text in {"derived_not_human_reviewed", "official_dense_label_semantics", "structured_source_verified"}:
        return "derived_eval"
    return "generated_unverified"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_name(value: Any) -> str:
    value = str(value or "")
    return {"val": "development", "validation": "development", "dev": "development"}.get(value, value or "unknown")


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def distribution(values: Iterable[float | int]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": pick(0.25),
        "median": pick(0.5),
        "p75": pick(0.75),
        "p90": pick(0.9),
        "p99": pick(0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def entropy(values: Iterable[str]) -> float:
    counter = collections.Counter(values)
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-release", type=Path, required=True)
    parser.add_argument("--caption-registry", type=Path, required=True)
    parser.add_argument("--pair-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stable-probe", type=Path)
    parser.add_argument("--forest-dir", type=Path)
    parser.add_argument("--review-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def item_frames(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames = item.get("frames")
    return list(frames) if isinstance(frames, list) else []


def item_paths(item: Mapping[str, Any]) -> tuple[str, str]:
    frames = item_frames(item)
    paths = [str(frame.get("path") or "") for frame in frames]
    return (paths[0] if paths else "", paths[1] if len(paths) > 1 else "")


def item_source(item: Mapping[str, Any]) -> str:
    return str(item.get("source") or item.get("provenance", {}).get("source_dataset") or "unknown")


def resolve_item_id(value: Any, items: Mapping[str, Mapping[str, Any]]) -> str:
    """Resolve source-review aliases to the canonical physical item ID."""

    raw = str(value or "")
    if raw in items:
        return raw
    if raw.startswith("tamms:"):
        suffix = raw.split(":", 2)[-1]
        candidates = [
            item_id for item_id in items
            if item_id.startswith("tamms:") and item_id.split(":", 2)[-1] == suffix
        ]
        if len(candidates) == 1:
            return candidates[0]
    return raw


def item_record(item: Mapping[str, Any], source_key: str) -> dict[str, Any] | None:
    if not item:
        return None
    return {
        "item_id": str(item.get("item_id") or source_key),
        "source": item_source(item),
        "split": split_name(item.get("split")),
        "item_type": str(item.get("item_type") or "pair"),
        "physical_group_id": str(item.get("physical_group_id") or item.get("item_id") or source_key),
        "training_enabled": bool(item.get("training_enabled", False)),
        "t1_path": item_paths(item)[0],
        "t2_path": item_paths(item)[1],
        "frames": item_frames(item),
    }


def canonical_forest_item(row: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Convert the Forest pilot row into the canonical physical-item schema."""

    pair_id = str(row.get("pair_id") or row.get("canonical_pair_id") or "")
    metadata = row.get("source_metadata") if isinstance(row.get("source_metadata"), Mapping) else {}
    scene = str(metadata.get("source_scene_group_id") or row.get("source_scene_group_id") or pair_id)
    source_revision = str(metadata.get("source_version") or "forest-change-pilot")
    sensor = row.get("sensor") or metadata.get("sensor")
    width = row.get("image_width")
    height = row.get("image_height")
    frames = [
        {
            "frame_id": f"{pair_id}:frame:0",
            "path": str(row.get("t1_path") or ""),
            "sha256": str(metadata.get("t1_sha256") or ""),
            "timestamp": "t1",
            "sensor": sensor,
            "gsd": None,
            "width": int(width) if width is not None else None,
            "height": int(height) if height is not None else None,
        },
        {
            "frame_id": f"{pair_id}:frame:1",
            "path": str(row.get("t2_path") or ""),
            "sha256": str(metadata.get("t2_sha256") or ""),
            "timestamp": "t2",
            "sensor": sensor,
            "gsd": None,
            "width": int(width) if width is not None else None,
            "height": int(height) if height is not None else None,
        },
    ]
    return {
        "item_id": pair_id,
        "item_type": "pair",
        "source": "forest_change",
        "source_revision": source_revision,
        "physical_group_id": scene,
        "scene_id": scene,
        "event_id": None,
        "frames": frames,
        "split": split_name(split),
        "training_enabled": False,
        "quality_status": "PHYSICAL_READY_TEXT_HOLD",
        "provenance": {
            "source_dataset": "Forest-Change",
            "source_pair_id": metadata.get("source_pair_id") or row.get("original_id") or pair_id,
            "source_scene_group_id": scene,
            "source_split": row.get("split"),
            "split_policy": "deterministic_scene_component_disjoint_repair_candidate",
            "spatial_resolution": row.get("spatial_resolution") or metadata.get("gsd"),
            "license": metadata.get("license"),
            "training_enabled": False,
        },
    }


def source_caption_row(caption: Mapping[str, Any], item: Mapping[str, Any] | None) -> dict[str, Any]:
    source_key = str(caption.get("canonical_pair_id") or "")
    record = item_record(item or {}, source_key)
    return {
        "record_type": "source_caption",
        "query_id": str(caption.get("caption_id") or f"caption:{source_key}"),
        "source_item_id": str((record or {}).get("item_id") or source_key),
        "source_dataset": str(caption.get("dataset_name") or (record or {}).get("source") or "unknown"),
        "split": split_name(caption.get("split") or (record or {}).get("split")),
        "text": str(caption.get("text") or caption.get("normalized_text") or "").strip(),
        "query_scope": str(caption.get("query_scope") or ""),
        "verification": str(caption.get("verification_status") or ""),
        "change_status": str(caption.get("change_status") or ""),
        "is_generated": bool(caption.get("is_generated", False)),
        "caption_source": caption.get("caption_source"),
        "source_caption_id": str(caption.get("caption_id") or ""),
        "source_event_id": caption.get("source_event_id"),
        "item_type": (record or {}).get("item_type", "pair"),
        "physical_group_id": (record or {}).get("physical_group_id"),
        "t1_path": (record or {}).get("t1_path", ""),
        "t2_path": (record or {}).get("t2_path", ""),
        "physical_pair_available": bool((record or {}).get("t1_path") and (record or {}).get("t2_path")),
    }


def current_query_row(query: Mapping[str, Any], caption_by_id: Mapping[str, Mapping[str, Any]], items: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    provenance = query.get("provenance") if isinstance(query.get("provenance"), Mapping) else {}
    caption_id = str(provenance.get("source_caption_id") or "")
    caption = caption_by_id.get(caption_id, {})
    item = items.get(str(query.get("source_item_id") or ""), {})
    t1_path, t2_path = item_paths(item)
    return {
        "record_type": "current_query",
        "query_id": str(query.get("query_id") or ""),
        "source_item_id": str(query.get("source_item_id") or ""),
        "source_dataset": str(provenance.get("source_dataset") or item_source(item)),
        "split": split_name(query.get("split") or item.get("split")),
        "text": str(query.get("text") or "").strip(),
        "query_scope": str(query.get("query_scope") or ""),
        "verification": str(query.get("verification") or ""),
        "change_status": str(caption.get("change_status") or ""),
        "positive_item_ids": list(query.get("positive_item_ids") or []),
        "positive_set_size": len(query.get("positive_item_ids") or []),
        "localized_relation": query.get("localized_relation"),
        "temporal_direction": query.get("temporal_direction"),
        "source_caption_id": caption_id,
        "item_type": str(item.get("item_type") or "pair"),
        "physical_group_id": str(item.get("physical_group_id") or ""),
        "t1_path": t1_path,
        "t2_path": t2_path,
        "current_training_enabled": bool(query.get("training_enabled", False)),
    }


def distinct_items(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], set[str]]:
    output: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in rows:
        split = split_name(row.get("split"))
        signature = str(row.get("semantic_signature") or "")
        item_id = str(row.get("source_item_id") or row.get("canonical_pair_id") or "")
        if signature and item_id:
            output[(split, signature)].add(item_id)
    return output


def confidence_score(
    *,
    purpose: str,
    text: str,
    collision_count: int,
    neighbour_count: int,
    positive_set_size: int,
    attributes: Mapping[str, Any],
    stable_identifiability: float = 0.0,
) -> float:
    if purpose == "stable_scene_specific":
        return max(0.0, min(1.0, stable_identifiability))
    if purpose != "exact_discriminative":
        return max(0.0, min(1.0, 0.35 * (1.0 / max(collision_count, 1)) + 0.35 * (1.0 / max(neighbour_count, 1)) + 0.30 * bool(attributes.get("has_specific_visual_claim"))))
    length_score = min(1.0, len(normalize_query_text(text).split()) / 12.0)
    detail_score = min(1.0, (len(attributes.get("objects", [])) + len(attributes.get("directions", [])) + len(attributes.get("spatial_relations", []))) / 3.0)
    return max(0.0, min(1.0, 0.35 * (1.0 / max(collision_count, 1)) + 0.25 * (1.0 / max(neighbour_count, 1)) + 0.20 * length_score + 0.20 * detail_score))


def annotate_purpose(
    row: Mapping[str, Any],
    *,
    collision_map: Mapping[tuple[str, str], int],
    neighbour_map: Mapping[tuple[str, str], int],
    stable_signature_counts: Mapping[tuple[str, str], int] | None = None,
    stable_anchor_count: int = 0,
    stable_identifiability: float = 0.0,
) -> dict[str, Any]:
    text = str(row.get("text") or "")
    attributes = extract_visual_attributes(text)
    norm = normalize_query_text(text)
    signature = semantic_signature(attributes)
    split = split_name(row.get("split"))
    item_id = str(row.get("source_item_id") or "")
    positive_set_size = int(row.get("positive_set_size") or len(row.get("positive_item_ids") or []) or 1)
    collision = int(collision_map.get((split, norm), 1)) if norm else 0
    neighbour = int(neighbour_map.get((split, signature), 1)) if signature else 0
    if stable_signature_counts is not None and stable_anchor_count >= 2:
        stable_key = (split, "anchors=" + ",".join(sorted(str(v) for v in row.get("stable_anchors", []))))
        positive_set_size = stable_signature_counts.get(stable_key, 1)
    result = classify_query(
        {**row, "text": text, "source_dataset": row.get("source_dataset")},
        collision_count=collision,
        neighbour_item_count=neighbour,
        positive_set_size=positive_set_size,
        stable_anchor_count=stable_anchor_count,
        stable_identifiability=stable_identifiability,
        has_physical_pair=bool(row.get("t1_path") and row.get("t2_path")),
    )
    row_out = dict(row)
    row_out.update({
        "purpose": result["purpose"],
        "normalized_text": result["normalized_text"],
        "semantic_signature": result["semantic_signature"],
        "semantic_group_id": result["semantic_group_id"],
        "attributes": result["attributes"],
        "unsupported_claims": result["unsupported_claims"],
        "classifier_reasons": result["reasons"],
        "generic_evidence": result["generic_evidence"],
        "generic_classification_evidence": {
            "normalized_collision_item_count": collision,
            "semantic_neighbour_item_count": neighbour,
            "specific_visual_attribute_count": sum(len(result["attributes"].get(field, [])) for field in ("objects", "directions", "spatial_relations", "surfaces")),
            "source_change_status": row.get("change_status"),
            "source_scope": row.get("query_scope"),
            "physical_pair_available": bool(row.get("t1_path") and row.get("t2_path")),
            "phrase_match_is_supporting_signal_only": bool(result["generic_evidence"].get("phrase_signal")),
        },
        "collision_count": collision,
        "semantic_neighbour_item_count": neighbour,
        "positive_set_size": positive_set_size,
        "stable_anchor_count": stable_anchor_count,
        "identifiability_score": confidence_score(
            purpose=result["purpose"], text=text, collision_count=max(collision, 1),
            neighbour_count=max(neighbour, 1), positive_set_size=positive_set_size,
            attributes=result["attributes"], stable_identifiability=stable_identifiability,
        ),
        "training_enabled_after_repair": False,
        "physical_pair_evidence": {
            "available": bool(row.get("t1_path") and row.get("t2_path")),
            "t1_path": row.get("t1_path") or None,
            "t2_path": row.get("t2_path") or None,
            "masks_used_for_text": False,
        },
    })
    return row_out


def assign_scene_disjoint(rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        metadata = row.get("source_metadata") if isinstance(row.get("source_metadata"), Mapping) else {}
        group_id = str(metadata.get("source_scene_group_id") or row.get("source_scene_group_id") or row.get("pair_id") or row.get("canonical_pair_id"))
        groups[group_id].append(row)
    total = sum(len(group) for group in groups.values())
    targets = {"train": round(total * 0.70), "development": round(total * 0.15), "test": total - round(total * 0.70) - round(total * 0.15)}
    assigned = collections.Counter()
    group_split: dict[str, str] = {}
    ordered = sorted(groups.items(), key=lambda value: (-len(value[1]), hashlib.sha256(value[0].encode()).hexdigest()))
    for group_id, values in ordered:
        split = min(targets, key=lambda name: (assigned[name] - targets[name], name))
        group_split[group_id] = split
        assigned[split] += len(values)
    split_groups = collections.defaultdict(set)
    for group_id, split in group_split.items():
        split_groups[split].add(group_id)
    overlap = sum(len(split_groups[left] & split_groups[right]) for left in split_groups for right in split_groups if left < right)
    return group_split, {
        "status": "MATERIALIZED_SCENE_COMPONENT_DISJOINT_CANDIDATE",
        "pair_count": total,
        "component_count": len(groups),
        "split_pair_counts": dict(sorted(assigned.items())),
        "split_component_counts": {key: len(value) for key, value in sorted(split_groups.items())},
        "cross_split_component_overlap": overlap,
        "training_enabled": False,
        "promotion_status": "HOLD_CAPTION_PROVENANCE_AND_HUMAN_REVIEW",
    }


def region_from_text(text: str) -> dict[str, Any]:
    lowered = normalize_query_text(text)
    regions = []
    for token, value in (("top", "upper"), ("bottom", "lower"), ("left", "left"), ("right", "right"), ("center", "center"), ("centre", "center"), ("north", "north"), ("south", "south")):
        if token in lowered.split():
            regions.append(value)
    return {"regions": sorted(set(regions)), "source": "caption_text_candidate", "evaluation_only": True}


def sidecar_for_query(query_id: str, item_id: str, mask_path: str, *, source: str, evaluation_only: bool = True) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "mask_paths": [mask_path] if mask_path else [],
        "label_paths": [],
        "derived_attributes": {"query_id": query_id, "source": source},
        "evaluation_only": evaluation_only,
    }


def review_neighbours(row: Mapping[str, Any], by_group: Mapping[tuple[str, str], set[str]], by_item: Mapping[str, Mapping[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    split = split_name(row.get("split"))
    signature = str(row.get("semantic_signature") or "")
    target = str(row.get("source_item_id") or "")
    ids = sorted(by_group.get((split, signature), set()) - {target})[:limit]
    return [
        {"item_id": item_id, "t1_path": item_paths(by_item.get(item_id, {}))[0], "t2_path": item_paths(by_item.get(item_id, {}))[1]}
        for item_id in ids
    ]


def stratified_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= size:
        return sorted(rows, key=lambda row: str(row.get("query_id") or row.get("candidate_id")))
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        attrs = row.get("attributes") if isinstance(row.get("attributes"), Mapping) else {}
        bucket = (str(row.get("source_dataset") or "unknown"), split_name(row.get("split")), str(attrs.get("objects", ["none"])[0] if attrs.get("objects") else "none"))
        buckets[bucket].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: stable_digest(row.get("query_id") or row.get("candidate_id")))
    keys = sorted(buckets)
    output: list[dict[str, Any]] = []
    cursor = 0
    while len(output) < size and keys:
        key = keys[cursor % len(keys)]
        values = buckets[key]
        if values:
            output.append(values.pop(0))
        else:
            keys.remove(key)
            cursor -= 1
        cursor += 1
    return output


def make_review_packet(rows: list[dict[str, Any]], purpose: str, size: int, by_group: Mapping[tuple[str, str], set[str]], items: Mapping[str, Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected = stratified_sample(rows, size, seed)
    packet = []
    for index, row in enumerate(selected):
        true_item_id = str(row.get("source_item_id") or row.get("canonical_pair_id") or "")
        item = items.get(true_item_id, {})
        item_t1, item_t2 = item_paths(item)
        t1_path = str(row.get("t1_path") or item_t1 or "")
        t2_path = str(row.get("t2_path") or item_t2 or "")
        text = str(row.get("text") or row.get("query_text") or "")
        provenance = row.get("provenance") if isinstance(row.get("provenance"), Mapping) else {}
        semantic_sig = str(
            row.get("semantic_signature")
            or provenance.get("semantic_signature")
            or semantic_signature(extract_visual_attributes(text))
        )
        review_row = dict(row)
        review_row.update({
            "source_item_id": true_item_id,
            "split": split_name(row.get("split") or item.get("split")),
            "semantic_signature": semantic_sig,
        })
        neighbours = review_neighbours(review_row, by_group, items)
        source_dataset = str(row.get("source_dataset") or row.get("source") or item_source(item))
        split = split_name(row.get("split") or item.get("split"))
        positive_ids = list(row.get("positive_item_ids") or [])
        neighbour_count = row.get("semantic_neighbour_item_count")
        if neighbour_count is None:
            neighbour_count = len(by_group.get((split, semantic_sig), set()))
        positive_set_size = row.get("positive_set_size")
        if positive_set_size is None:
            positive_set_size = len(positive_ids) or 1
        physical_pair_available = bool(t1_path and t2_path)
        neighbour_evidence_available = bool(neighbours)
        packet.append({
            "review_id": f"RETRIEVAL-{purpose.upper()}-{index:04d}",
            "purpose": purpose,
            "query_id": row.get("query_id") or row.get("candidate_id"),
            "text": text,
            "source_dataset": source_dataset,
            "split": split,
            "true_item_id": true_item_id,
            "t1_path": t1_path,
            "t2_path": t2_path,
            "semantic_signature": semantic_sig,
            "semantic_neighbour_pairs": neighbours,
            "candidate_attributes": row.get("attributes") or {"common_atomic_anchors": row.get("common_atomic_anchors", [])},
            "classifier_evidence": {
                "collision_count": row.get("collision_count"),
                "semantic_neighbour_item_count": neighbour_count,
                "positive_set_size": positive_set_size,
                "identifiability_score": row.get("identifiability_score"),
            },
            "review_evidence": {
                "physical_pair_available": physical_pair_available,
                "semantic_neighbour_evidence_available": neighbour_evidence_available,
                "paths_from_canonical_physical_registry": bool(item_t1 and item_t2),
                "masks_used_for_text": False,
            },
            "review_fields": [
                "true_pair_matches_scope",
                "semantic_neighbour_matches_scope",
                "caption_factuality",
                "generic_no_change_contamination",
                "accept_or_reject",
                "reviewer_identity",
                "reviewed_at",
                "independence_attestation",
            ],
            "reviewer_a": None,
            "reviewer_b": None,
            "review_decision": None,
            "adjudication": None,
            "training_enabled": False,
        })
    return packet

def physical_stats(items: Mapping[str, Mapping[str, Any]], sample_limit: int = 256) -> dict[str, Any]:
    sources: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    frame_sha: dict[str, list[str]] = collections.defaultdict(list)
    for item_id, item in items.items():
        source = item_source(item)
        sources[source].append(item)
        for frame in item_frames(item):
            sha = str(frame.get("sha256") or "")
            if sha:
                frame_sha[sha].append(item_id)
    resolutions: dict[str, list[list[int]]] = collections.defaultdict(list)
    gsd_counts: dict[str, int] = collections.Counter()
    for source, values in sources.items():
        for item in sorted(values, key=lambda value: str(value.get("item_id")))[:sample_limit]:
            for frame in item_frames(item)[:2]:
                if frame.get("gsd") is not None:
                    gsd_counts[source] += 1
                path = str(frame.get("path") or "")
                if Image is not None and path and Path(path).exists():
                    try:
                        with Image.open(path) as image:
                            resolutions[source].append([int(image.width), int(image.height)])
                    except Exception:
                        pass
    duplicate_shas = sum(len(ids) for ids in frame_sha.values() if len(set(ids)) > 1)
    frame_count = sum(len(item_frames(item)) for item in items.values())
    return {
        "source_item_counts": dict(sorted(collections.Counter(item_source(item) for item in items.values()).items())),
        "frame_count": frame_count,
        "cross_item_exact_frame_hash_duplicate_rows": duplicate_shas,
        "cross_item_exact_frame_hash_duplicate_rate": duplicate_shas / max(frame_count, 1),
        "sampled_image_resolution_by_source": {source: distribution([width * height for width, height in values]) for source, values in sorted(resolutions.items())},
        "sampled_width_height_by_source": {source: values[:20] for source, values in sorted(resolutions.items())},
        "gsd_present_frame_counts_by_source": dict(sorted(gsd_counts.items())),
    }


def random_baseline(gallery_size: int, average_positive_set: float = 1.0) -> dict[str, Any]:
    n = max(1, int(gallery_size))
    harmonic = sum(1.0 / rank for rank in range(1, n + 1)) / n
    positive = max(1.0, min(float(n), average_positive_set))
    return {
        "gallery_size": n,
        "expected_MRR_single_positive": harmonic,
        "expected_R@1_single_positive": 1.0 / n,
        "expected_R@5_single_positive": min(5, n) / n,
        "expected_R@10_single_positive": min(10, n) / n,
        "expected_candidate_recall_at_10_for_average_positive_set": min(1.0, min(10, n) * positive / n),
        "average_positive_set_size": average_positive_set,
    }


def build_difficulty(rows: list[dict[str, Any]], items: Mapping[str, Mapping[str, Any]], physical: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        source_item = items.get(str(row.get("source_item_id") or row.get("canonical_pair_id") or ""), {})
        source = str(
            row.get("source_dataset")
            or row.get("provenance", {}).get("source_dataset")
            or item_source(source_item)
            or "unknown"
        )
        source = {
            "s2looking": "s2looking",
            "second-cc": "second_cc",
            "forest-change": "forest_change",
            "tamms": "tamms",
        }.get(source.casefold(), source)
        population = str(row.get("_difficulty_population") or "audit")
        grouped[(population, source, str(row.get("purpose") or "unsupported_or_reject"))].append(row)
    report: dict[str, Any] = {
        "schema_version": "qcpr-retrieval-data-difficulty-v1",
        "physical_summary": physical,
        "frozen_anchor_baseline": {
            "status": "NOT_RUN",
            "reason": "same-frozen-model D0-D3 evaluation is a separate post-freeze operation; no proxy is reported as a model result",
        },
        "groups": {},
    }
    for (population, source, purpose), values in sorted(grouped.items()):
        norms = [str(row.get("normalized_text") or normalize_query_text(row.get("text") or row.get("query_text"))) for row in values]
        pair_ids = {str(row.get("source_item_id") or row.get("canonical_pair_id") or "") for row in values}
        positive_sizes = [int(row.get("positive_set_size") or len(row.get("positive_item_ids") or []) or 1) for row in values]
        ident = [float(row.get("identifiability_score") or 0.0) for row in values]
        attrs = [
            row.get("attributes")
            if isinstance(row.get("attributes"), Mapping)
            else extract_visual_attributes(str(row.get("text") or row.get("query_text") or ""))
            for row in values
        ]
        spatial = sum(bool(value.get("spatial_relations")) for value in attrs)
        direction = sum(bool(value.get("directions")) for value in attrs)
        changed_pairs = {pid for pid in pair_ids if str(next((row.get("change_status") for row in values if str(row.get("source_item_id") or row.get("canonical_pair_id") or "") == pid), "")) == "changed"}
        unchanged_pairs = {pid for pid in pair_ids if str(next((row.get("change_status") for row in values if str(row.get("source_item_id") or row.get("canonical_pair_id") or "") == pid), "")) in {"no_change", "stable"}}
        collision_counter = collections.Counter(norms)
        duplicate_rows = sum(value for value in collision_counter.values() if value > 1)
        report["groups"][f"{population}:{source}:{purpose}"] = {
            "population": population,
            "source": source,
            "query_purpose": purpose,
            "pair_or_sequence_count": len(pair_ids),
            "query_count": len(values),
            "caption_entropy_bits": entropy(norms),
            "normalized_duplicate_rate": (len(values) - len(collision_counter)) / max(len(values), 1),
            "normalized_duplicate_rows_in_duplicate_clusters": duplicate_rows,
            "generic_no_change_rate": sum(row.get("purpose") == "generic_no_change" for row in values) / max(len(values), 1),
            "identifiability_distribution": distribution(ident),
            "positive_set_size_distribution": distribution(positive_sizes),
            "visual_near_duplicate_rate": physical.get("cross_item_exact_frame_hash_duplicate_rate", 0.0),
            "image_resolution": physical.get("sampled_image_resolution_by_source", {}).get(source, {"count": 0}),
            "gsd_present_frame_count": physical.get("gsd_present_frame_counts_by_source", {}).get(source, 0),
            "change_no_change_balance": {"changed_pairs": len(changed_pairs), "no_change_pairs": len(unchanged_pairs)},
            "spatial_detail_coverage": spatial / max(len(values), 1),
            "direction_coverage": direction / max(len(values), 1),
            "source_event_imbalance": {"event_ids_used_for_semantics": False, "event_id_counts": {}},
            "random_retrieval_baseline": random_baseline(len(pair_ids), sum(positive_sizes) / max(len(positive_sizes), 1)),
        }
    return report


def main() -> int:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    current_release = args.current_release
    items = {str(row.get("item_id")): row for row in read_jsonl(current_release / "registries/physical_items.jsonl")}
    forest_dir = args.forest_dir or (current_release / "source_reports/forest")
    forest_physical_input = read_jsonl(forest_dir / "forest_physical_registry.jsonl")
    forest_caption_input = read_jsonl(forest_dir / "forest_caption_registry.jsonl")
    forest_group_split_input, forest_split_audit_input = assign_scene_disjoint(forest_physical_input) if forest_physical_input else ({}, {"status": "NOT_AVAILABLE", "training_enabled": False})
    for forest_row in forest_physical_input:
        metadata = forest_row.get("source_metadata") if isinstance(forest_row.get("source_metadata"), Mapping) else {}
        component = str(metadata.get("source_scene_group_id") or forest_row.get("source_scene_group_id") or forest_row.get("pair_id") or forest_row.get("canonical_pair_id"))
        forest_item = canonical_forest_item(forest_row, forest_group_split_input.get(component, split_name(forest_row.get("split"))))
        items[str(forest_item["item_id"])] = forest_item
    captions = read_jsonl(args.caption_registry)
    pairs = {str(row.get("canonical_pair_id")): row for row in read_jsonl(args.pair_registry)}
    current_queries = read_jsonl(current_release / "registries/queries.jsonl")
    caption_by_id = {str(row.get("caption_id")): row for row in captions}
    pair_to_item = {str(row.get("item_id")): row for row in items.values()}

    base_source_rows = [source_caption_row(caption, items.get(str(caption.get("canonical_pair_id")))) for caption in captions]
    base_current_rows = [current_query_row(query, caption_by_id, items) for query in current_queries]
    all_base = base_source_rows + base_current_rows
    feature_rows = []
    for row in all_base:
        attributes = extract_visual_attributes(row.get("text"))
        row = dict(row)
        row["normalized_text"] = normalize_query_text(row.get("text"))
        row["semantic_signature"] = semantic_signature(attributes)
        row["attributes"] = attributes
        feature_rows.append(row)
    norm_items: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    sig_items: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in feature_rows:
        item_id = str(row.get("source_item_id") or "")
        if not item_id:
            continue
        norm = str(row.get("normalized_text") or "")
        signature = str(row.get("semantic_signature") or "")
        split = split_name(row.get("split"))
        if norm:
            norm_items[(split, norm)].add(item_id)
        if signature and row.get("attributes", {}).get("has_specific_visual_claim"):
            sig_items[(split, signature)].add(item_id)
    collision_map = {key: len(value) for key, value in norm_items.items()}
    neighbour_map = {key: len(value) for key, value in sig_items.items()}

    # Audit every source caption and every current query.  The source captions
    # include diagnostic generic rows that were correctly absent from b19 exact.
    audit_rows: list[dict[str, Any]] = []
    for row in feature_rows:
        if row["record_type"] == "source_caption":
            audit_rows.append(annotate_purpose(row, collision_map=collision_map, neighbour_map=neighbour_map))
        else:
            audit_rows.append(annotate_purpose(row, collision_map=collision_map, neighbour_map=neighbour_map))

    # Stable candidates are generated from independent T1/T2 anchor probes.
    stable_probe = read_jsonl(args.stable_probe) if args.stable_probe else []
    stable_by_id = {str(row.get("canonical_pair_id")): row for row in stable_probe}
    stable_sig_counts: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in stable_probe:
        anchors = sorted(str(value) for value in row.get("common_atomic_anchors", []))
        if len(anchors) >= 2:
            source_item = items.get(str(row.get("canonical_pair_id")))
            split = split_name((source_item or {}).get("split"))
            stable_sig_counts[(split, "anchors=" + ",".join(anchors))].add(str(row.get("canonical_pair_id")))
    stable_sig_sizes = {key: len(value) for key, value in stable_sig_counts.items()}
    stable_audit: list[dict[str, Any]] = []
    for probe in stable_probe:
        pair_id = str(probe.get("canonical_pair_id") or "")
        item = items.get(pair_id, {})
        anchors = sorted(str(value) for value in probe.get("common_atomic_anchors", []))
        stable_row = {
            "record_type": "stable_candidate",
            "candidate_id": probe.get("candidate_id"),
            "query_id": str(probe.get("candidate_id") or ""),
            "source_item_id": pair_id,
            "source_dataset": item_source(item),
            "split": split_name(item.get("split")),
            "text": str(probe.get("query_text") or ""),
            "query_scope": "stable_scene_candidate",
            "verification": probe.get("verification"),
            "change_status": "no_change",
            "item_type": item.get("item_type", "pair"),
            "t1_path": probe.get("t1_path") or item_paths(item)[0],
            "t2_path": probe.get("t2_path") or item_paths(item)[1],
            "stable_anchors": anchors,
            "independent_t1_claims": probe.get("independent_t1_claims", []),
            "independent_t2_claims": probe.get("independent_t2_claims", []),
            "visual_evidence": probe.get("visual_evidence", {}),
            "review_decision": probe.get("review_decision"),
        }
        stable_key = (stable_row["split"], "anchors=" + ",".join(anchors))
        annotated = annotate_purpose(
            stable_row,
            collision_map=collision_map,
            neighbour_map=neighbour_map,
            stable_signature_counts=stable_sig_sizes,
            stable_anchor_count=len(anchors),
            stable_identifiability=float(probe.get("identifiability_score") or 0.0),
        )
        annotated["stable_positive_item_ids"] = sorted(stable_sig_counts.get(stable_key, set()))
        annotated["stable_candidate_pair_discriminative"] = len(annotated["stable_positive_item_ids"]) == 1 and len(anchors) >= 2
        stable_audit.append(annotated)
    audit_rows.extend(stable_audit)

    # Forest-Change has a separate five-level caption registry.  Add every
    # caption to the purpose audit, but keep its unresolved source_unverified
    # provenance out of promoted exact/semantic training candidates.
    forest_audit_rows = []
    for caption in forest_caption_input:
        pair_id = str(caption.get("canonical_pair_id") or "")
        forest_item = items.get(pair_id)
        forest_caption_row = dict(caption)
        forest_caption_row.setdefault("dataset_name", "forest_change")
        forest_caption_row.setdefault("change_status", "changed")
        forest_caption_row.setdefault("verification_status", "source_unverified")
        row = source_caption_row(forest_caption_row, forest_item)
        row["record_type"] = "forest_source_caption"
        row["source_caption_id"] = str(caption.get("caption_id") or "")
        forest_audit_rows.append(annotate_purpose(row, collision_map=collision_map, neighbour_map=neighbour_map))
    audit_rows.extend(forest_audit_rows)

    # Generic rows become a separate diagnostic registry; never exact.
    generic_rows = []
    for row in audit_rows:
        if row.get("purpose") != "generic_no_change":
            continue
        generic_rows.append({
            "query_id": f"{row.get('source_item_id')}:generic_no_change:{row.get('source_caption_id') or row.get('query_id')}",
            "text": row.get("text", ""),
            "query_scope": "generic_no_change",
            "source_item_id": row.get("source_item_id"),
            "positive_item_ids": [row.get("source_item_id")],
            "graded_relevance": {str(row.get("source_item_id")): 3},
            "verification": row.get("verification") or "human",
            "training_enabled": False,
            "diagnostic_only": True,
            "split": split_name(row.get("split")),
            "purpose": "generic_no_change",
            "provenance": {"source_caption_id": row.get("source_caption_id"), "source_dataset": row.get("source_dataset"), "excluded_from_exact": True},
        })
    write_jsonl(out / "registries/generic_no_change_diagnostic.jsonl", generic_rows)

    # Repaired exact candidates are built from verified human changed captions.
    exact_candidates: list[dict[str, Any]] = []
    for row in audit_rows:
        if row.get("record_type") != "source_caption" or row.get("query_scope") != "exact_pair" or row.get("change_status") != "changed":
            continue
        if row.get("verification") not in {"human", "human_rewritten", "independently_source_verified"}:
            continue
        if row.get("purpose") != "exact_discriminative":
            continue
        item_id = str(row.get("source_item_id") or "")
        if item_id not in items:
            continue
        exact_candidates.append({
            "query_id": f"{item_id}:exact_repaired:{row.get('source_caption_id')}",
            "text": row.get("text", ""),
            "query_scope": "exact",
            "purpose": "exact_discriminative",
            "source_item_id": item_id,
            "positive_item_ids": [item_id],
            "graded_relevance": {item_id: 3},
            "temporal_direction": "forward",
            "localized_relation": None,
            "verification": row.get("verification"),
            "training_enabled": False,
            "split": split_name(row.get("split")),
            "candidate_status": "HUMAN_REVIEW_REQUIRED_EXACT_GATE",
            "identifiability_score": row.get("identifiability_score"),
            "provenance": {"source_caption_id": row.get("source_caption_id"), "source_dataset": row.get("source_dataset"), "source_query_scope": "exact_pair", "collision_count": row.get("collision_count"), "semantic_neighbour_item_count": row.get("semantic_neighbour_item_count")},
        })

    # Semantic groups are attribute-only and partitioned by split.  Source and
    # event IDs are kept as provenance and never enter the group key.
    semantic_source_rows = [
        row for row in audit_rows
        if row.get("record_type") == "source_caption"
        and row.get("purpose") in {"semantic_multi_positive", "exact_discriminative"}
        and row.get("change_status") == "changed"
        and row.get("semantic_signature")
        and str(row.get("source_item_id") or "") in items
        and bool((row.get("attributes") or {}).get("has_specific_visual_claim"))
        and row.get("verification") in {"human", "human_rewritten", "independently_source_verified", "derived_not_human_reviewed"}
    ]
    semantic_members: dict[tuple[str, str], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in semantic_source_rows:
        key = (split_name(row.get("split")), str(row.get("semantic_signature")))
        item_id = str(row.get("source_item_id") or "")
        if item_id:
            semantic_members[key][item_id] = row
    semantic_queries: list[dict[str, Any]] = []
    semantic_groups: list[dict[str, Any]] = []
    for (split, signature), members in sorted(semantic_members.items()):
        if len(members) < 2:
            continue
        item_ids = sorted(members)
        group_id = "semantic:" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
        grades = {}
        for query_item, query_row in sorted(members.items()):
            qattrs = query_row.get("attributes") or {}
            qgrades = {}
            for candidate_item, candidate_row in sorted(members.items()):
                sim = attribute_similarity(qattrs, candidate_row.get("attributes") or {})
                qgrades[candidate_item] = 3 if candidate_item == query_item else int(sim["grade"])
            grades[query_item] = qgrades
            positive_ids = sorted(candidate_item for candidate_item, grade in qgrades.items() if int(grade) >= 1)
            if len(positive_ids) < 2:
                continue
            semantic_queries.append({
                "query_id": f"{query_item}:semantic_repaired:{group_id}:{query_row.get('source_caption_id')}",
                "text": query_row.get("text", ""),
                "query_scope": "semantic",
                "purpose": "semantic_multi_positive",
                "source_item_id": query_item,
                "positive_item_ids": positive_ids,
                "graded_relevance": {candidate_item: int(grade) for candidate_item, grade in qgrades.items() if int(grade) >= 1},
                "temporal_direction": "forward",
                "localized_relation": None,
                "verification": contract_verification(query_row.get("verification")),
                "training_enabled": False,
                "split": split,
                "candidate_status": "EVAL_ONLY_ATTRIBUTE_GROUP_HUMAN_REVIEW_REQUIRED",
                "provenance": {"semantic_group_id": group_id, "semantic_signature": signature, "source_caption_id": query_row.get("source_caption_id"), "source_dataset": query_row.get("source_dataset"), "event_ids_are_provenance_only": True, "group_key_excludes_source_and_event": True},
            })
        semantic_groups.append({
            "group_id": group_id,
            "split": split,
            "semantic_signature": signature,
            "item_ids": item_ids,
            "grades_by_query_item": grades,
            "positive_count": len(item_ids),
            "source_counts": dict(collections.Counter(str(row.get("source_dataset")) for row in members.values())),
            "event_ids_are_provenance_only": True,
            "training_enabled": False,
            "evaluation_only": True,
        })

    # Stable candidates that are not unique are semantic positives, with the
    # anchor signature as an attribute-only group.
    stable_semantic_rows = []
    stable_exact_rows = []
    for row in stable_audit:
        if row.get("purpose") == "stable_scene_specific" and row.get("text"):
            stable_exact_rows.append({
                "query_id": f"{row.get('source_item_id')}:stable_repaired:{row.get('candidate_id')}",
                "text": row.get("text"),
                "query_scope": "stable",
                "purpose": "stable_scene_specific",
                "source_item_id": row.get("source_item_id"),
                "positive_item_ids": [row.get("source_item_id")],
                "graded_relevance": {str(row.get("source_item_id")): 3},
                "temporal_direction": "none",
                "localized_relation": None,
                "verification": contract_verification(row.get("verification")),
                "training_enabled": False,
                "split": split_name(row.get("split")),
                "candidate_status": "HUMAN_REVIEW_REQUIRED_STABLE_SCENE_GATE",
                "identifiability_score": row.get("identifiability_score"),
                "provenance": {"stable_probe_id": row.get("candidate_id"), "independent_t1_claims": row.get("independent_t1_claims"), "independent_t2_claims": row.get("independent_t2_claims"), "masks_used_for_text": False},
            })
        elif len(row.get("stable_anchors", [])) >= 2 and row.get("text"):
            stable_semantic_rows.append(row)
    stable_semantic_by_key: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in stable_semantic_rows:
        stable_semantic_by_key[(split_name(row.get("split")), ",".join(sorted(row.get("stable_anchors", []))))].append(row)
    for (split, anchor_signature), members in sorted(stable_semantic_by_key.items()):
        if len(members) < 2:
            continue
        ids = sorted(str(row.get("source_item_id")) for row in members)
        group_id = "semantic:stable_anchors:" + hashlib.sha256(anchor_signature.encode()).hexdigest()[:20]
        for row in members:
            stable_semantic_queries_row = {
                "query_id": f"{row.get('source_item_id')}:stable_semantic:{row.get('candidate_id')}",
                "text": row.get("text"),
                "query_scope": "semantic",
                "purpose": "semantic_multi_positive",
                "source_item_id": row.get("source_item_id"),
                "positive_item_ids": ids,
                "graded_relevance": {item_id: (3 if item_id == str(row.get("source_item_id")) else 2) for item_id in ids},
                "temporal_direction": "none",
                "localized_relation": None,
                "verification": contract_verification(row.get("verification")),
                "training_enabled": False,
                "split": split,
                "candidate_status": "EVAL_ONLY_STABLE_ANCHOR_GROUP_HOLD",
                "provenance": {"semantic_group_id": group_id, "stable_anchor_signature": anchor_signature, "source_dataset": row.get("source_dataset"), "masks_used_for_text": False},
            }
            semantic_queries.append(stable_semantic_queries_row)
        semantic_groups.append({"group_id": group_id, "split": split, "semantic_signature": "stable_anchors=" + anchor_signature, "item_ids": ids, "positive_count": len(ids), "event_ids_are_provenance_only": True, "training_enabled": False, "evaluation_only": True})

    # Direction view is derived from the current direction queries but remains
    # training-disabled until the same text is reviewed against reverse pairs.
    direction_queries = []
    for row in base_current_rows:
        if row.get("query_scope") != "direction":
            continue
        item_id = str(row.get("source_item_id") or "")
        if item_id not in items:
            continue
        direction_queries.append({
            "query_id": f"{row.get('query_id')}:repaired_direction",
            "text": row.get("text", ""),
            "query_scope": "direction",
            "purpose": "direction_sensitive",
            "source_item_id": item_id,
            "positive_item_ids": [item_id],
            "graded_relevance": {item_id: 3},
            "temporal_direction": row.get("temporal_direction") or "none",
            "localized_relation": None,
            "verification": contract_verification(row.get("verification")),
            "training_enabled": False,
            "split": split_name(row.get("split")),
            "candidate_status": "HUMAN_REVIEW_REQUIRED_DIRECTION_GATE",
            "provenance": {"current_query_id": row.get("query_id"), "source_caption_id": row.get("source_caption_id"), "source_dataset": row.get("source_dataset")},
        })

    # Localized evaluation rows: mask paths are only sidecars.
    current_sidecars = read_jsonl(current_release / "evaluation_sidecars/dense.jsonl")
    sidecar_by_query = {str((row.get("derived_attributes") or {}).get("query_id")): row for row in current_sidecars}
    localized_queries = []
    localized_sidecars = []
    for row in base_current_rows:
        if row.get("query_scope") != "localized":
            continue
        item_id = str(row.get("source_item_id") or "")
        qid = f"{row.get('query_id')}:repaired_localized"
        localized_queries.append({
            "query_id": qid,
            "text": row.get("text", ""),
            "query_scope": "localized",
            "purpose": "localized",
            "source_item_id": item_id,
            "positive_item_ids": [item_id],
            "graded_relevance": {item_id: 3},
            "temporal_direction": row.get("temporal_direction") or "none",
            "localized_relation": row.get("localized_relation") or region_from_text(str(row.get("text") or "")),
            "verification": contract_verification(row.get("verification") or "derived_eval"),
            "training_enabled": False,
            "split": split_name(row.get("split")),
            "candidate_status": "EVAL_ONLY_MASK_SIDECAR",
            "provenance": {"source_caption_id": row.get("source_caption_id"), "source_dataset": row.get("source_dataset"), "mask_text_training_disabled": True},
        })
        source_sidecar = sidecar_by_query.get(str(row.get("query_id")))
        if source_sidecar:
            for mask_path in source_sidecar.get("mask_paths", []):
                localized_sidecars.append(sidecar_for_query(qid, item_id, str(mask_path), source="current_release", evaluation_only=True))

    # Forest is materialized as a scene-disjoint physical/text candidate, but
    # no caption is promoted until its provenance and reviewer decisions exist.
    forest_dir = args.forest_dir or (current_release / "source_reports/forest")
    forest_physical = read_jsonl(forest_dir / "forest_physical_registry.jsonl")
    forest_captions = forest_caption_input
    forest_packet = read_jsonl(current_release / "source_reports/human_review_packets/forest_human_review_packet.jsonl")
    forest_group_split, forest_split_audit = assign_scene_disjoint(forest_physical) if forest_physical else ({}, {"status": "NOT_AVAILABLE", "training_enabled": False})
    forest_by_pair = {str(row.get("pair_id") or row.get("canonical_pair_id")): row for row in forest_physical}
    forest_scene_rows = []
    for physical in forest_physical:
        pair_id = str(physical.get("pair_id") or physical.get("canonical_pair_id"))
        metadata = physical.get("source_metadata") if isinstance(physical.get("source_metadata"), Mapping) else {}
        component = str(metadata.get("source_scene_group_id") or physical.get("source_scene_group_id") or pair_id)
        split = forest_group_split.get(component, split_name(physical.get("split")))
        copied = dict(physical)
        copied["split"] = split
        copied["training_enabled"] = False
        copied["source_metadata"] = {**metadata, "official_split": physical.get("split"), "split_policy": "deterministic_scene_component_disjoint_repair_candidate", "training_enabled": False}
        forest_scene_rows.append(copied)
    forest_localized = []
    packet_by_query = {str(row.get("query_id")): row for row in forest_packet}
    for caption in forest_captions:
        pair_id = str(caption.get("canonical_pair_id") or "")
        physical = forest_by_pair.get(pair_id, {})
        metadata = physical.get("source_metadata") if isinstance(physical.get("source_metadata"), Mapping) else {}
        component = str(metadata.get("source_scene_group_id") or pair_id)
        split = forest_group_split.get(component, split_name(caption.get("split")))
        text_value = str(caption.get("text") or "").strip()
        if not text_value or not region_from_text(text_value)["regions"]:
            continue
        qid = str(caption.get("query_id") or caption.get("caption_id") or f"{pair_id}:localized_candidate")
        forest_localized.append({
            "query_id": f"{qid}:repair_localized",
            "text": text_value,
            "query_scope": "localized",
            "purpose": "localized",
            "source_item_id": pair_id,
            "source_dataset": "forest_change",
            "positive_item_ids": [pair_id],
            "graded_relevance": {pair_id: 3},
            "temporal_direction": "forward",
            "localized_relation": region_from_text(text_value),
            "verification": contract_verification("source_unverified"),
            "training_enabled": False,
            "split": split,
            "candidate_status": "HOLD_FOREST_HUMAN_REWRITE_AND_ADJUDICATION",
            "provenance": {"source_caption_id": caption.get("caption_id"), "source_dataset": "forest_change", "review_id": packet_by_query.get(str(caption.get("query_id")), {}).get("review_id"), "caption_provenance": caption.get("caption_provenance"), "masks_as_sidecar_only": True},
        })
        label_path = ""
        for sidecar in read_jsonl(forest_dir / "forest_dense_eval_sidecar.jsonl"):
            if str(sidecar.get("canonical_pair_id")) == pair_id:
                label_path = str(sidecar.get("label_path") or "")
                break
        if label_path:
            localized_sidecars.append(sidecar_for_query(forest_localized[-1]["query_id"], pair_id, label_path, source="forest_change", evaluation_only=True))

    localized_queries.extend(forest_localized)

    # TAMMs: retain all physical sequences and make a non-empty but disabled
    # temporal-annotation candidate view from the existing review packet.
    tamm_packet = read_jsonl(current_release / "source_reports/human_review_packets/tamms_human_review_packet.jsonl")
    tamm_rows = []
    for index, row in enumerate(tamm_packet):
        raw_sequence_id = str(row.get("sequence_id") or "")
        sequence_id = resolve_item_id(raw_sequence_id, items)
        item = items.get(sequence_id, {})
        if not item:
            continue
        split = split_name(item.get("split"))
        if split == "unknown":
            split = ["train", "development", "test"][index % 3]
        text_value = str(row.get("text") or "").strip()
        if not text_value:
            continue
        tamm_rows.append({
            "query_id": f"{row.get('query_id')}:repair_long_series",
            "text": text_value,
            "query_scope": "long_series",
            "purpose": "long_series",
            "source_item_id": sequence_id,
            "source_dataset": "TAMMs",
            "positive_item_ids": [sequence_id],
            "graded_relevance": {sequence_id: 3},
            "temporal_direction": None,
            "localized_relation": None,
            "verification": row.get("verification_status") or "generated_unverified",
            "training_enabled": False,
            "split": split,
            "candidate_status": "HOLD_TAMMS_REVIEW_REQUIRED",
            "temporal_annotations": {"onset": None, "duration": None, "gradual_or_abrupt": None, "relevant_frame_range": None},
            "provenance": {
                "source": "TAMMs",
                "original_sequence_id": raw_sequence_id,
                "review_id": row.get("review_id"),
                "text_provenance": row.get("text_provenance"),
                "physical_inventory_preserved": True,
                "query_temporal_extent": {"start": "frame_0", "end": f"frame_{max(0, len(item_frames(item)) - 1)}"},
                "relevant_frame_range": {"start": 0, "end": max(0, len(item_frames(item)) - 1)},
                "temporal_annotation_status": "unverified_placeholder_for_schema_only",
            },
        })
    tamm_physical = [row for row in items.values() if item_source(row).casefold() == "tamms" and str(row.get("item_type")) in {"sequence", "long_series"}]
    tamm_annotation_audit = []
    for item in tamm_physical:
        tamm_annotation_audit.append({
            "sequence_id": item.get("item_id"),
            "frame_count": len(item_frames(item)),
            "split": split_name(item.get("split")),
            "onset": None,
            "duration": None,
            "gradual_or_abrupt": None,
            "review_status": "REVIEW_REQUIRED",
            "training_enabled": False,
        })

    # Temporal review candidates are also part of the singular-purpose audit.
    # Their physical IDs are canonicalized above, while their generated text
    # remains disabled until license and temporal review are complete.
    long_series_audit_rows = []
    for row in tamm_rows:
        item = items.get(str(row.get("source_item_id") or ""), {})
        t1_path, t2_path = item_paths(item)
        long_series_audit_rows.append(annotate_purpose({
            "record_type": "long_series_candidate",
            "query_id": row.get("query_id"),
            "source_item_id": row.get("source_item_id"),
            "source_dataset": "TAMMs",
            "split": row.get("split"),
            "text": row.get("text"),
            "query_scope": "long_series",
            "verification": row.get("verification"),
            "item_type": "sequence",
            "physical_group_id": item.get("physical_group_id"),
            "t1_path": t1_path,
            "t2_path": t2_path,
            "positive_item_ids": row.get("positive_item_ids"),
            "positive_set_size": len(row.get("positive_item_ids") or []),
        }, collision_map=collision_map, neighbour_map=neighbour_map))
    audit_rows.extend(long_series_audit_rows)

    all_candidate_rows = exact_candidates + semantic_queries + localized_queries + direction_queries + stable_exact_rows + tamm_rows + generic_rows
    # Query purpose registry is the authoritative audit index.  It includes
    # source captions, current queries, and stable candidates with exactly one
    # purpose each; final manifests contain only candidate query records.
    audit_by_id: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        key = f"{row.get('record_type')}:{row.get('query_id') or row.get('candidate_id')}"
        audit_by_id[key] = row
    write_jsonl(out / "registries/query_purpose_registry.jsonl", audit_rows)
    write_jsonl(out / "registries/semantic_groups.jsonl", semantic_groups)
    write_jsonl(out / "registries/repaired_queries.jsonl", all_candidate_rows)
    write_jsonl(out / "source_reports/forest/scene_disjoint_train.jsonl", [row for row in forest_scene_rows if row.get("split") == "train"])
    write_jsonl(out / "source_reports/forest/scene_disjoint_development.jsonl", [row for row in forest_scene_rows if row.get("split") == "development"])
    write_jsonl(out / "source_reports/forest/scene_disjoint_test.jsonl", [row for row in forest_scene_rows if row.get("split") == "test"])
    forest_canonical_items = [items[str(row.get("pair_id") or row.get("canonical_pair_id"))] for row in forest_physical if str(row.get("pair_id") or row.get("canonical_pair_id")) in items]
    write_jsonl(out / "registries/forest_physical_items.jsonl", forest_canonical_items)
    write_jsonl(out / "registries/forest_frames.jsonl", [frame | {"item_id": item["item_id"], "source": "forest_change"} for item in forest_canonical_items for frame in item.get("frames", [])])
    write_jsonl(out / "source_reports/tamms_long_series_annotation_audit.jsonl", tamm_annotation_audit)
    write_jsonl(out / "evaluation_sidecars/dense.jsonl", localized_sidecars)

    # Build the five required stratified review packets.  Review decisions are
    # intentionally null; no numerical precision is fabricated.
    by_group = sig_items
    purpose_pools = {
        "exact_discriminative": exact_candidates,
        "semantic_multi_positive": semantic_queries,
        "generic_no_change": generic_rows,
        "stable_scene_specific": [row for row in stable_audit if len(row.get("stable_anchors", [])) >= 2],
        "localized_direction": localized_queries + direction_queries,
    }
    review_counts = {}
    for purpose, pool in purpose_pools.items():
        packet = make_review_packet(pool, purpose, args.review_size, by_group, items, args.seed)
        review_counts[purpose] = {"available": len(pool), "sampled": len(packet), "required": args.review_size, "decisions_completed": sum(row.get("review_decision") is not None for row in packet)}
        write_jsonl(out / f"source_reports/review_samples/{purpose}_{args.review_size}.jsonl", packet)

    exact_review = review_counts["exact_discriminative"]
    calibration = {
        "schema_version": "qcpr-retrieval-identifiability-calibration-v2",
        "required_sample_counts": {purpose: args.review_size for purpose in purpose_pools},
        "actual_sample_counts": {purpose: value["sampled"] for purpose, value in review_counts.items()},
        "sample_size_gate": all(value["sampled"] >= args.review_size for value in review_counts.values()),
        "decision_counts": {purpose: value["decisions_completed"] for purpose, value in review_counts.items()},
        "exact_scope_precision": None,
        "exact_scope_precision_95_ci": None,
        "false_exact_rate": None,
        "generic_no_change_contamination": None,
        "semantic_false_negative_rate": None,
        "average_positive_set_size": distribution([len(row.get("positive_item_ids") or []) for row in exact_candidates]),
        "collision_rate": sum(int(row.get("collision_count", 1)) > 1 for row in audit_rows) / max(len(audit_rows), 1),
        "exact_gate": {"status": "HOLD_REVIEW_REQUIRED", "passes": False, "reason": "all reviewer decisions are null; exact view cannot be enabled"},
        "reviewer_independence_required": True,
        "human_review_not_replaced_by_phrase_blacklist": True,
    }
    write_json(out / "audits/retrieval_identifiability_calibration.json", calibration)
    write_json(out / "source_reports/human_review_packet_status.json", {"schema_version": "qcpr-retrieval-human-review-status-v2", "packets": review_counts, "all_decisions_null": True, "training_enabled": False, "status": "HOLD_HUMAN_REVIEW_REQUIRED"})

    physical = physical_stats(items)
    difficulty_rows = list(audit_rows)
    difficulty_rows.extend({**row, "_difficulty_population": "materialized_view"} for row in all_candidate_rows)
    difficulty = build_difficulty(difficulty_rows, items, physical)
    write_json(out / "source_reports/retrieval_data_difficulty_report.json", difficulty)
    markdown = [
        "# QCPR retrieval data difficulty report",
        "",
        "The JSON file is authoritative. All frozen-anchor model metrics remain explicitly `NOT_RUN` until the same frozen checkpoint is evaluated against frozen D0-D3 manifests.",
        "",
        f"- Difficulty groups: {len(difficulty['groups'])}",
        f"- Physical items: {sum(physical.get('source_item_counts', {}).values())}",
        f"- Exact frame-hash duplicate rate: {physical.get('cross_item_exact_frame_hash_duplicate_rate', 0.0):.6f}",
        "",
        "| population | source | purpose | pairs/sequences | queries | entropy (bits) | normalized duplicate rate | generic rate | mean positive-set size | spatial coverage | direction coverage |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in difficulty["groups"].items():
        markdown.append(
            f"| {value['population']} | {value['source']} | {value['query_purpose']} | {value['pair_or_sequence_count']} | {value['query_count']} | {value['caption_entropy_bits']:.4f} | {value['normalized_duplicate_rate']:.4f} | {value['generic_no_change_rate']:.4f} | {value['positive_set_size_distribution'].get('mean', 0.0):.2f} | {value['spatial_detail_coverage']:.4f} | {value['direction_coverage']:.4f} |"
        )
    (out / "source_reports/retrieval_data_difficulty_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    # Source integration reports.
    forest_caption_counts = collections.Counter(str(row.get("caption_id", "")).rsplit(":", 1)[-1] for row in forest_captions)
    forest_pair_counts = collections.Counter(str(row.get("canonical_pair_id")) for row in forest_captions)
    write_json(out / "source_reports/forest/scene_disjoint_split_audit.json", forest_split_audit)
    write_json(out / "source_reports/forest/caption_level_audit.json", {
        "pair_count": len(forest_pair_counts),
        "caption_count": len(forest_captions),
        "caption_levels_per_pair_distribution": distribution(list(collections.Counter(forest_pair_counts.values()).values())),
        "caption_level_suffix_counts": dict(sorted(forest_caption_counts.items())),
        "caption_provenance_counts": dict(collections.Counter(str(row.get("caption_provenance")) for row in forest_captions)),
        "human_rows": sum(str(row.get("caption_source")) in {"human", "human_rewritten"} for row in forest_captions),
        "rule_based_or_generated_rows": sum(str(row.get("caption_source")) not in {"human", "human_rewritten"} for row in forest_captions),
        "promoted_training_rows": 0,
        "status": "HOLD_PROVENANCE_AND_REVIEW",
    })
    write_json(out / "source_reports/tamms_long_series_audit.json", {
        "physical_sequence_count": len(tamm_physical),
        "review_candidate_count": len(tamm_rows),
        "annotation_audit_count": len(tamm_annotation_audit),
        "verified_long_series_query_count": 0,
        "license_status": "REQUIRES_RESOLUTION",
        "sequence_split_status": "PRESERVED_PHYSICAL_SPLIT_CANDIDATE",
        "fields": ["onset", "duration", "gradual_or_abrupt", "relevant_frame_range"],
        "training_enabled": False,
        "status": "HOLD_LICENSE_AND_HUMAN_REVIEW",
    })
    write_json(out / "source_reports/rscc_verified_text_audit.json", {
        "verified_query_count": 0,
        "reviewed_rows": 0,
        "reviewer_a_completed": 0,
        "reviewer_b_completed": 0,
        "adjudicated_rows": 0,
        "training_enabled": False,
        "status": "HOLD_INPUT_OR_REVIEW_NOT_AVAILABLE",
    })
    official = read_json(current_release / "source_reports/official_source_audit.json", {})
    write_json(out / "source_reports/official_source_acquisition_audit.json", {
        "sources": {name: {"status": "NOT_INTEGRATED_UNTIL_PHYSICAL_HASH_LICENSE_LOADER_MANIFEST", "physical_assets": False, "hashes": False, "license": False, "loader": False, "manifest": False, "training_enabled": False} for name in ["DUBAI-CC", "RSRCC", "DynamicEarthNet", "SpaceNet 7", "TERRA-CD"]},
        "prior_audit": official,
        "status": "ACQUISITION_AUDIT_CONTINUES",
    })

    # Materialized view manifests.  Rows are non-empty candidate records where
    # possible, but every view carries an explicit READY/EVAL_ONLY/HOLD state.
    view_rows: dict[str, list[dict[str, Any]]] = {}
    for scope, values in (("exact", exact_candidates), ("semantic", semantic_queries), ("localized", localized_queries), ("direction", direction_queries), ("stable", stable_exact_rows), ("long_series", tamm_rows)):
        for split in ("train", "development", "test"):
            view_rows[f"{scope}_{split}"] = [row for row in values if split_name(row.get("split")) == split]
            write_jsonl(out / f"manifests/{scope}_{split}.jsonl", view_rows[f"{scope}_{split}"])

    view_status = {
        "exact": {"status": "HOLD_EXACT_REVIEW_GATE", "training_enabled": False, "counts": {split: len(view_rows[f"exact_{split}"]) for split in ("train", "development", "test")}, "evidence": "300-row exact review packet exists only if source pool permits; all decisions null"},
        "semantic": {"status": "EVAL_ONLY_ATTRIBUTE_GROUPS_HOLD_PRIMARY_TRAINING", "training_enabled": False, "counts": {split: len(view_rows[f"semantic_{split}"]) for split in ("train", "development", "test")}, "evidence": "groups keyed only by verified visual attributes and split; no event/source key"},
        "localized": {"status": "EVAL_ONLY_AND_HOLD_UNVERIFIED_FOREST", "training_enabled": False, "counts": {split: len(view_rows[f"localized_{split}"]) for split in ("train", "development", "test")}, "evidence": "masks are sidecars; current S2Looking rows are eval-only; Forest rows await human rewrite"},
        "direction": {"status": "HOLD_DIRECTION_REVIEW_GATE", "training_enabled": False, "counts": {split: len(view_rows[f"direction_{split}"]) for split in ("train", "development", "test")}, "evidence": "separate direction view retained; decisions null"},
        "stable": {"status": "HOLD_STABLE_SCENE_REVIEW_GATE", "training_enabled": False, "counts": {split: len(view_rows[f"stable_{split}"]) for split in ("train", "development", "test")}, "evidence": "independent T1/T2 anchor probes, common anchors >=2, masks not used; reviewers still required"},
        "long_series": {"status": "HOLD_TAMMS_LICENSE_AND_TEMPORAL_REVIEW", "training_enabled": False, "counts": {split: len(view_rows[f"long_series_{split}"]) for split in ("train", "development", "test")}, "evidence": "489 physical sequences retained; text candidates unverified"},
    }
    write_json(out / "audits/view_readiness.json", {"views": view_status, "empty_files_are_not_success": True, "all_training_authorized": False})

    # D0-D3 is a manifest/frozen-model comparison contract.  Existing frozen
    # rankings are not silently reused for changed galleries; metrics remain
    # null until the same checkpoint is run against these exact manifests.
    current_exact = [row for row in current_queries if row.get("query_scope") == "exact"]
    comparison = {
        "schema_version": "qcpr-retrieval-data-only-comparison-v1",
        "same_frozen_model_required": True,
        "D0": {"description": "current exact core", "query_count": len(current_exact), "manifest_source": str(current_release / "manifests"), "metrics": None},
        "D1": {"description": "repaired exact plus stable-scene candidates", "query_count": len(exact_candidates) + len(stable_exact_rows), "manifest_source": str(out / "manifests"), "metrics": None},
        "D2": {"description": "D1 plus verified Forest/RSCC", "verified_forest_query_count": 0, "verified_rscc_query_count": 0, "query_count": len(exact_candidates) + len(stable_exact_rows), "metrics": None},
        "D3": {"description": "D2 plus verified long-series/localized", "verified_long_series_query_count": 0, "verified_localized_query_count": 0, "query_count": len(exact_candidates) + len(stable_exact_rows), "metrics": None},
        "metric_contract": ["exact MRR", "exact R@K", "semantic mAP/nDCG", "stable-scene retrieval", "localized retrieval", "candidate recall", "per-source metrics", "paired bootstrap intervals"],
        "status": "PENDING_MANIFEST_FREEZE_AND_SAME_FROZEN_MODEL_EVALUATION",
        "improvement_claim": False,
        "reason_metrics_null": "This package is still HOLD because exact/stable review gates are not passed; generated-only or size/loss comparisons are prohibited.",
    }
    write_json(out / "audits/data_only_D0_D3_comparison.json", comparison)

    # Handoff counts and hashes are generated before packaging and are copied
    # into the immutable release by the release wrapper.
    purpose_counts = collections.Counter(str(row.get("purpose")) for row in audit_rows)
    purpose_counts_by_record_type = {
        record_type: dict(sorted(collections.Counter(str(row.get("purpose")) for row in audit_rows if row.get("record_type") == record_type).items()))
        for record_type in sorted({str(row.get("record_type")) for row in audit_rows})
    }
    manifest_counts = {name: len(values) for name, values in sorted(view_rows.items())}
    handoff = {
        "schema_version": "qcpr-dataset-to-model-retrieval-semantic-repair-v1",
        "decision": "DO_NOT_AUTHORIZE_MAIN_EXPANDED_TRAINING",
        "training_authorized": False,
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "purpose_counts_by_record_type": purpose_counts_by_record_type,
        "verified_exact_query_count": sum(row.get("verification") in {"human", "human_rewritten", "independently_source_verified"} for row in exact_candidates),
        "semantic_group_count": len(semantic_groups),
        "semantic_query_count": len(semantic_queries),
        "localized_query_count": len(localized_queries),
        "stable_query_count": len(stable_exact_rows),
        "direction_query_count": len(direction_queries),
        "long_series_query_count": len(tamm_rows),
        "disabled_generic_no_change_row_count": len(generic_rows),
        "disabled_generic_no_change_source_caption_count": purpose_counts_by_record_type.get("source_caption", {}).get("generic_no_change", 0),
        "disabled_generic_no_change_current_query_count": purpose_counts_by_record_type.get("current_query", {}).get("generic_no_change", 0),
        "mask_sidecar_count": len(localized_sidecars),
        "manifest_counts": manifest_counts,
        "hashes": {
            "query_purpose_registry_sha256": sha256_file(out / "registries/query_purpose_registry.jsonl"),
            "repaired_queries_sha256": sha256_file(out / "registries/repaired_queries.jsonl"),
            "semantic_groups_sha256": sha256_file(out / "registries/semantic_groups.jsonl"),
            "dense_sidecars_sha256": sha256_file(out / "evaluation_sidecars/dense.jsonl"),
        },
        "exact_gate": calibration["exact_gate"],
        "stable_gate": view_status["stable"],
        "localized_gate": view_status["localized"],
        "long_series_gate": view_status["long_series"],
    }
    write_json(out / "handoff/model_agent_handoff.json", handoff)
    write_json(out / "REPAIR_PACKAGE.json", {
        "schema_version": "qcpr-retrieval-semantic-repair-package-v1",
        "status": "DATA_QUALITY_HOLD",
        "source_release": str(current_release),
        "source_release_sha256s": sha256_file(current_release / "SHA256SUMS") if (current_release / "SHA256SUMS").exists() else None,
        "query_purpose_classes": list(QUERY_PURPOSES),
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "purpose_counts_by_record_type": purpose_counts_by_record_type,
        "manifest_counts": manifest_counts,
        "all_training_enabled": False,
        "exact_gate_passed": False,
        "stable_gate_passed": False,
        "human_review_required": True,
        "frozen_model_comparison_status": comparison["status"],
    })
    print(json.dumps({"output_dir": str(out), "audit_rows": len(audit_rows), "current_queries": len(current_queries), "source_captions": len(captions), "generic_rows": len(generic_rows), "exact_candidates": len(exact_candidates), "semantic_queries": len(semantic_queries), "localized_queries": len(localized_queries), "stable_candidates": len(stable_audit), "stable_exact": len(stable_exact_rows), "long_series_candidates": len(tamm_rows), "views": manifest_counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
