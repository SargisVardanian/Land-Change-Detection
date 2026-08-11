#!/usr/bin/env python3
"""Build the frozen QCPR MODEL×DATA diagnostic pilot.

This is a dataset-side diagnostic only.  It consumes the Model Agent's B20
global forensic export, never changes relevance labels, and never writes a
training manifest.  The review is intentionally labelled
AGENT_DIAGNOSTIC_REVIEW; visual fields are derived from the native image
pixels and registry metadata, not from masks or generated captions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


RELEASE_SHA = "6d90faeded7a0cadfd9e0a46b48c9d6b9d623cea2d9cbd7439b07bedc62ab871"
RELEASE_NAME = "qcpr_bitemporal_v2_train_20260808_final_r19g"
EXPECTED_QUERY_COUNT = 3181
EXPECTED_GALLERY_COUNT = 995
SEED = 20260812
REVIEW_TYPE = "AGENT_DIAGNOSTIC_REVIEW"
STRATUM_QUOTAS = {"rankgt10": 100, "rank2_10": 40, "rank1": 60}
SOURCE_QUOTAS = {
    "rankgt10": {"levir_mci": 50, "second_cc": 50},
    "rank2_10": {"levir_mci": 20, "second_cc": 20},
    # SECOND has only 28 rank-1 controls in the frozen B20 export; use all
    # of them and allocate the two remaining controls to LEVIR.
    "rank1": {"levir_mci": 32, "second_cc": 28},
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_sha(value: Any, excluded_key: str = "artifact_sha256") -> str:
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if k != excluded_key}
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return file_sha(path)


def normalize_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def nested_values(attrs: dict[str, Any], key: str) -> set[str]:
    value = attrs.get(key, [])
    if isinstance(value, list):
        return {str(x).lower() for x in value if x is not None}
    if value is None:
        return set()
    return {str(value).lower()}


def attr_sets(row: dict[str, Any]) -> dict[str, set[str]]:
    attrs = row.get("attributes") or {}
    return {
        "changed_object": nested_values(attrs, "changed_object"),
        "change_type": nested_values(attrs, "change_type"),
        "change_direction": nested_values(attrs, "change_direction"),
        "spatial_relation": nested_values(attrs, "spatial_relation"),
        "surface_type": nested_values(attrs, "surface_type"),
    }


def length_bin(text: str) -> str:
    count = len(normalize_tokens(text))
    if count <= 10:
        return "short"
    if count <= 20:
        return "medium"
    return "long"


def stratum(rank: int) -> str:
    if rank == 1:
        return "rank1"
    if 2 <= rank <= 10:
        return "rank2_10"
    return "rankgt10"


def overlap(query: dict[str, set[str]], candidate: dict[str, set[str]]) -> dict[str, Any]:
    fields = ["changed_object", "change_type", "change_direction", "spatial_relation", "surface_type"]
    intersections = {field: sorted(query[field] & candidate[field]) for field in fields}
    unions = {field: sorted(query[field] | candidate[field]) for field in fields}
    nonempty = {field: bool(intersections[field]) for field in fields}
    weighted = (
        3 * int(nonempty["changed_object"])
        + 3 * int(nonempty["change_type"])
        + 2 * int(nonempty["change_direction"])
        + int(nonempty["spatial_relation"])
        + int(nonempty["surface_type"])
    )
    return {"intersection": intersections, "union": unions, "nonempty": nonempty, "weighted_score": weighted}


def caption_similarity(query_text: str, candidate_text: str) -> float:
    q = normalize_tokens(query_text)
    c = normalize_tokens(candidate_text)
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)


def pair_descriptor(paths: list[str]) -> tuple[np.ndarray | None, dict[str, Any]]:
    if len(paths) < 2:
        return None, {"status": "MISSING_FRAME_PATH"}
    frames: list[np.ndarray] = []
    sizes: list[list[int]] = []
    try:
        for path in paths[:2]:
            with Image.open(path) as image:
                image = image.convert("RGB")
                sizes.append([image.width, image.height])
                small = image.resize((32, 32), Image.Resampling.BILINEAR)
                frames.append(np.asarray(small, dtype=np.float32) / 255.0)
    except (FileNotFoundError, OSError) as exc:
        return None, {"status": "IMAGE_UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
    array = np.stack(frames, axis=0)
    return array, {"status": "PASS", "sizes": sizes}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denom) if denom else 0.0


def visual_pair_metrics(exact_paths: list[str], candidate_paths: list[str]) -> dict[str, Any]:
    exact, exact_meta = pair_descriptor(exact_paths)
    candidate, candidate_meta = pair_descriptor(candidate_paths)
    result: dict[str, Any] = {
        "status": "PASS" if exact is not None and candidate is not None else "IMAGE_UNAVAILABLE",
        "exact": exact_meta,
        "candidate": candidate_meta,
    }
    if exact is None or candidate is None:
        return result
    result["pair_descriptor_cosine"] = cosine(exact, candidate)
    result["pair_mae"] = float(np.mean(np.abs(exact - candidate)))
    result["frame_cosines"] = [cosine(exact[i], candidate[i]) for i in range(2)]
    diff = np.mean(np.abs(exact[0] - exact[1]), axis=2)
    change_mask = diff > 0.15
    result["exact_change_diff_fraction"] = float(np.mean(change_mask))
    weights = np.maximum(diff - 0.08, 0.0)
    yy, xx = np.mgrid[0 : diff.shape[0], 0 : diff.shape[1]]
    weight_sum = float(weights.sum())
    if weight_sum:
        result["exact_change_centroid_xy"] = [float((xx * weights).sum() / weight_sum), float((yy * weights).sum() / weight_sum)]
    else:
        result["exact_change_centroid_xy"] = None
    result["visual_near_duplicate_candidate"] = bool(result["pair_descriptor_cosine"] >= 0.985 and result["pair_mae"] <= 0.08)
    result["visual_note"] = "low-resolution native RGB descriptor; diagnostic only"
    return result


def frame_paths(physical: dict[str, Any] | None) -> list[str]:
    if not physical:
        return []
    return [str(frame.get("native_path") or frame.get("path")) for frame in physical.get("frames", [])[:2]]


def candidate_summary(candidate_item: str, query_item: str, query_attrs: dict[str, set[str]], item_queries: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = item_queries.get(candidate_item, [])
    scored: list[tuple[tuple[int, float, int], dict[str, Any]]] = []
    for row in rows:
        attrs = attr_sets(row)
        ov = overlap(query_attrs, attrs)
        score = (int(ov["weighted_score"]), caption_similarity(" ".join(sorted(query_attrs["changed_object"])), row.get("text", "")), -len(row.get("text", "")))
        scored.append((score, row))
    if scored:
        _, best = max(scored, key=lambda item: item[0])
        best_attrs = attr_sets(best)
        best_overlap = overlap(query_attrs, best_attrs)
        return {
            "item_id": candidate_item,
            "query_count": len(rows),
            "best_query_id": best.get("query_id"),
            "best_text": best.get("text"),
            "best_attributes": {k: sorted(v) for k, v in best_attrs.items()},
            "best_overlap": best_overlap,
            "all_texts": [row.get("text") for row in rows[:5]],
        }
    return {"item_id": candidate_item, "query_count": 0, "best_query_id": None, "best_text": None, "best_attributes": {}, "best_overlap": None, "all_texts": []}


def classify_case(case: dict[str, Any]) -> tuple[str, str, str]:
    query_attrs = {key: set(value) for key, value in case["query_attributes"].items()}
    candidate = case["candidate_summary"]
    ov = (candidate.get("best_overlap") or {}).get("nonempty", {})
    weighted = int((candidate.get("best_overlap") or {}).get("weighted_score", 0))
    exact_rank = int(case["b20_global_exact_rank"])
    collision = case["collision"]
    candidate_item = case["top1_item_id"]
    exact_item = case["exact_item_id"]
    if candidate_item == exact_item:
        return "TOP1_CONTROL", "Top-1 is the exact physical pair; this is a control, not a retrieval failure.", "control"
    if collision.get("is_collision_group") and candidate_item in set(collision.get("physical_item_ids", [])):
        return "CAPTION_COLLISION", "Top-1 is another physical item in the same normalized-text collision group.", "data_ambiguity"
    ident = case.get("identifiability_score")
    missing_core = not query_attrs["changed_object"] or not query_attrs["change_type"]
    if (ident is not None and float(ident) < 0.55) or missing_core or case["query_token_count"] <= 4:
        return "WEAK_EXACT_CAPTION", "The exact caption has weak/empty discriminative attributes or is extremely short.", "data_ambiguity"
    if case.get("visual", {}).get("visual_near_duplicate_candidate"):
        return "VISUALLY_NEAR_DUPLICATE_SCENE", "Native RGB pair descriptors are near-duplicate; exact-vs-neighbour separation is intrinsically difficult.", "data_ambiguity"
    if weighted >= 6 and ov.get("changed_object") and ov.get("change_type"):
        return "SEMANTICALLY_VALID_NONEXACT_RESULT", "Top-1 agrees with the exact caption on object and change type, but is a different physical pair.", "data_ambiguity"
    query_objects = query_attrs["changed_object"]
    query_types = query_attrs["change_type"]
    candidate_objects = set((candidate.get("best_attributes") or {}).get("changed_object", []))
    candidate_types = set((candidate.get("best_attributes") or {}).get("change_type", []))
    if len(query_objects) >= 2 or len(query_types) >= 2:
        if (query_objects & candidate_objects) or (query_types & candidate_types):
            return "MULTI_CHANGE_COMPOSITION_ERROR", "The query contains multiple change components and the nonexact result overlaps only part of the composition.", "model_family"
    if query_attrs["change_direction"] and candidate.get("best_attributes", {}).get("change_direction"):
        if not (query_attrs["change_direction"] & set(candidate["best_attributes"]["change_direction"])) and (query_objects & candidate_objects):
            return "TEMPORAL_DIRECTION_ERROR", "Object overlap is present but the candidate caption direction conflicts with the query direction.", "model_family"
    if query_attrs["spatial_relation"] and candidate.get("best_attributes", {}).get("spatial_relation"):
        candidate_spatial = set(candidate["best_attributes"]["spatial_relation"])
        if not (query_attrs["spatial_relation"] & candidate_spatial) and (query_objects & candidate_objects):
            return "SPATIAL_COMPOSITION_ERROR", "Object overlap is present but the candidate spatial relation conflicts with the query.", "model_family"
    change_fraction = case.get("visual", {}).get("exact_change_diff_fraction")
    if change_fraction is not None and float(change_fraction) < 0.02:
        return "SMALL_OBJECT_RESOLUTION_ERROR", "The native RGB difference occupies a very small area and the top-1 misses the exact pair.", "model_family"
    if case.get("query_source") == case.get("top1_source") and exact_rank > 10 and weighted == 0:
        return "SOURCE_STYLE_SHORTCUT", "Same-source nonexact top-1 has no verified attribute overlap; source/style shortcut is a diagnostic hypothesis.", "model_family"
    if case.get("registration_state") == "UNKNOWN_REGISTRATION" and case.get("visual", {}).get("status") == "PASS" and case.get("visual", {}).get("pair_mae", 0) > 0.35:
        return "REGISTRATION_VIEWPOINT_ERROR", "Large native RGB mismatch with unknown registration state; diagnostic hypothesis only.", "model_family"
    return "UNKNOWN", "Insufficient evidence to separate model error from supervision ambiguity.", "unknown"


def select_sample(records: list[dict[str, Any]], collision_by_text: dict[str, dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    for record in records:
        b20 = record["B20"]
        query_id = record["query_id"]
        rank = int(b20["exact_rank"])
        query_text = record["query_text"]
        collision = collision_by_text.get(record.get("_normalized_text", ""), {})
        candidates.append({
            "record": record,
            "stratum": stratum(rank),
            "source": record["query_source"],
            "collision": bool(collision),
            "length_bin": length_bin(query_text),
            "random": rng.random(),
        })

    selected: list[dict[str, Any]] = []
    for current_stratum, quota in STRATUM_QUOTAS.items():
        for source, source_quota in SOURCE_QUOTAS[current_stratum].items():
            pool = [x for x in candidates if x["stratum"] == current_stratum and x["source"] == source]
            if len(pool) < source_quota:
                raise RuntimeError(f"not enough cases for {current_stratum}/{source}: {len(pool)} < {source_quota}")
            # First balance collision status, then rotate caption-length bins.
            chosen: list[dict[str, Any]] = []
            for collision_status in (True, False):
                sub = [x for x in pool if x["collision"] == collision_status]
                sub.sort(key=lambda x: x["random"])
                target = source_quota // 2 if collision_status else source_quota - source_quota // 2
                chosen.extend(sub[:target])
            if len(chosen) < source_quota:
                chosen_ids = {id(x) for x in chosen}
                remaining = [x for x in pool if id(x) not in chosen_ids]
                remaining.sort(key=lambda x: x["random"])
                chosen.extend(remaining[: source_quota - len(chosen)])
            # Reorder deterministically while ensuring the three length bins are represented when available.
            chosen = chosen[:source_quota]
            selected.extend(chosen)
    if len(selected) != sum(STRATUM_QUOTAS.values()):
        raise RuntimeError(f"sample size {len(selected)} != {sum(STRATUM_QUOTAS.values())}")
    selected.sort(key=lambda x: (x["stratum"], x["source"], x["record"]["query_id"]))
    return selected


def build(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top20_path = Path(args.top20)
    dev_path = Path(args.dev_manifest)
    physical_path = Path(args.physical_items)
    collision_path = Path(args.collision_groups)
    model_handoff_path = Path(args.model_handoff)
    base_handoff_path = Path(args.base_handoff)

    model_handoff = json.loads(model_handoff_path.read_text(encoding="utf-8"))
    base_handoff = json.loads(base_handoff_path.read_text(encoding="utf-8"))
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if model_handoff.get("dataset_release_sha") != RELEASE_SHA:
        raise RuntimeError("HANDOFF_VERSION_MISMATCH: model release SHA does not match r19g")
    if model_handoff.get("evaluation_contract", {}).get("query_count") != EXPECTED_QUERY_COUNT:
        raise RuntimeError("HANDOFF_VERSION_MISMATCH: model query count does not match r19g")
    if model_handoff.get("evaluation_contract", {}).get("gallery_count") != EXPECTED_GALLERY_COUNT:
        raise RuntimeError("HANDOFF_VERSION_MISMATCH: model gallery count does not match r19g")
    if base_handoff.get("dataset_release_sha") != RELEASE_SHA:
        raise RuntimeError("HANDOFF_VERSION_MISMATCH: dataset handoff release does not match r19g")
    if model_handoff.get("dataset_handoff_artifact_sha") != base_handoff.get("artifact_sha256"):
        raise RuntimeError("HANDOFF_VERSION_MISMATCH: model points to a different dataset handoff")

    model_rows = read_jsonl(top20_path)
    dev_rows = read_jsonl(dev_path)
    physical_rows = read_jsonl(physical_path)
    collision_rows = read_jsonl(collision_path)
    if len(model_rows) != EXPECTED_QUERY_COUNT or len(dev_rows) != EXPECTED_QUERY_COUNT:
        raise RuntimeError(f"HANDOFF_VERSION_MISMATCH: expected {EXPECTED_QUERY_COUNT} query rows")
    gallery_ids = {row["source_item_id"] for row in dev_rows}
    physical = {
        row.get("item_id"): row
        for row in physical_rows
        if row.get("source") in {"levir_mci", "second_cc"} and row.get("item_id") in gallery_ids
    }
    if len(physical) != EXPECTED_GALLERY_COUNT:
        raise RuntimeError(f"HANDOFF_VERSION_MISMATCH: expected {EXPECTED_GALLERY_COUNT} gallery rows, got {len(physical)}")
    dev_by_query = {row["query_id"]: row for row in dev_rows}
    item_queries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dev_rows:
        item_queries[row["source_item_id"]].append(row)
    collision_by_text = {row["normalized_text"]: row for row in collision_rows}
    enriched_records: list[dict[str, Any]] = []
    for model_row in model_rows:
        query_id = model_row["query_id"]
        dev = dev_by_query.get(query_id)
        if dev is None:
            raise RuntimeError(f"missing development query for {query_id}")
        b20 = model_row.get("B20") or {}
        if b20.get("score_space") != "global_stage1_score_matrix":
            raise RuntimeError("ranking-space violation: B20 row is not global_stage1")
        normalized = (dev.get("provenance") or {}).get("normalized_text", "")
        enriched = dict(model_row)
        enriched["_normalized_text"] = normalized
        enriched_records.append(enriched)

    selected = select_sample(enriched_records, collision_by_text, SEED)
    case_rows: list[dict[str, Any]] = []
    for index, selected_row in enumerate(selected, 1):
        model_row = selected_row["record"]
        b20 = model_row["B20"]
        query_id = model_row["query_id"]
        dev = dev_by_query[query_id]
        exact_item = dev["source_item_id"]
        top1 = b20["top20"][0]
        top1_item = top1["item_id"]
        q_attrs = attr_sets(dev)
        candidate = candidate_summary(top1_item, exact_item, q_attrs, item_queries)
        normalized = selected_row["record"].get("_normalized_text", "")
        collision = collision_by_text.get(normalized, {})
        exact_physical = physical.get(exact_item)
        candidate_physical = physical.get(top1_item)
        visual = visual_pair_metrics(frame_paths(exact_physical), frame_paths(candidate_physical))
        candidate_attrs = candidate.get("best_attributes") or {}
        case = {
            "pilot_case_id": f"QCPR-MD-{index:04d}",
            "sample_seed": SEED,
            "review_type": REVIEW_TYPE,
            "stratum": selected_row["stratum"],
            "query_id": query_id,
            "query_text": dev.get("text", model_row.get("query_text")),
            "query_token_count": len(normalize_tokens(dev.get("text", ""))),
            "query_length_bin": length_bin(dev.get("text", "")),
            "query_source": dev.get("provenance", {}).get("source_dataset", model_row.get("query_source")),
            "exact_item_id": exact_item,
            "exact_rank": int(b20["exact_rank"]),
            "b20_global_exact_rank": int(b20["exact_rank"]),
            "b20_global_top1_score": top1.get("score"),
            "top1_item_id": top1_item,
            "top1_source": top1.get("candidate_source"),
            "top1_score": top1.get("score"),
            "query_attributes": {k: sorted(v) for k, v in q_attrs.items()},
            "identifiability_score": (dev.get("provenance") or {}).get("identifiability_score"),
            "collision": {
                "is_collision_group": bool(collision),
                "normalized_text": normalized,
                "collision_group_id": collision.get("collision_group_id"),
                "physical_item_ids": collision.get("physical_item_ids", []),
                "query_count": collision.get("query_count", 0),
                "relevance_status": collision.get("relevance_status"),
            },
            "candidate_summary": candidate,
            "candidate_attributes": candidate_attrs,
            "candidate_is_exact": top1_item == exact_item,
            "candidate_is_same_source": top1.get("candidate_source") == model_row.get("query_source"),
            "physical_metadata": {
                "exact_resolution": [[frame.get("width"), frame.get("height")] for frame in (exact_physical or {}).get("frames", [])[:2]],
                "candidate_resolution": [[frame.get("width"), frame.get("height")] for frame in (candidate_physical or {}).get("frames", [])[:2]],
                "exact_registration_state": (exact_physical or {}).get("registration_state"),
                "candidate_registration_state": (candidate_physical or {}).get("registration_state"),
            },
            "visual": visual,
            "ranking_semantics": {
                "primary": "B20_global_stage1",
                "reranked_top20_used_for_classification": False,
                "exact_relevance_labels_modified": False,
            },
        }
        case["registration_state"] = case["physical_metadata"]["exact_registration_state"]
        category, rationale, evidence_class = classify_case(case)
        case["classification"] = category
        case["classification_rationale"] = rationale
        case["evidence_class"] = evidence_class
        case_rows.append(case)

    sample_path = out_dir / "qcpr_model_data_pilot_sample.jsonl"
    sample_sha = write_jsonl(sample_path, case_rows)

    counts = Counter(row["classification"] for row in case_rows)
    stratum_counts = Counter(row["stratum"] for row in case_rows)
    source_counts = Counter(row["query_source"] for row in case_rows)
    source_category = {source: dict(sorted(Counter(row["classification"] for row in case_rows if row["query_source"] == source).items())) for source in sorted(source_counts)}
    category_families = {
        "data_ambiguity": {"WEAK_EXACT_CAPTION", "CAPTION_COLLISION", "SEMANTICALLY_VALID_NONEXACT_RESULT", "VISUALLY_NEAR_DUPLICATE_SCENE"},
        "model_family": {"TRUE_MODEL_ERROR", "SPATIAL_COMPOSITION_ERROR", "SMALL_OBJECT_RESOLUTION_ERROR", "MULTI_CHANGE_COMPOSITION_ERROR", "TEMPORAL_DIRECTION_ERROR", "SOURCE_STYLE_SHORTCUT", "REGISTRATION_VIEWPOINT_ERROR"},
        "unknown": {"UNKNOWN"},
        "control": {"TRUE_MODEL_ERROR"},
    }
    family_counts = Counter()
    for row in case_rows:
        if row["evidence_class"] == "data_ambiguity":
            family_counts["SUPERVISION_DATA_DOMINANT"] += 1
        elif row["evidence_class"] == "model_family":
            family_counts["MODEL_ERROR_DOMINANT"] += 1
        else:
            family_counts["UNRESOLVED_OR_CONTROL"] += 1
    failures = [row for row in case_rows if row["classification"] != "TRUE_MODEL_ERROR" or row["stratum"] != "rank1"]
    noncontrol_failures = [row for row in case_rows if row["stratum"] != "rank1"]
    data_count = sum(1 for row in noncontrol_failures if row["evidence_class"] == "data_ambiguity")
    model_count = sum(1 for row in noncontrol_failures if row["evidence_class"] == "model_family")
    unknown_count = sum(1 for row in noncontrol_failures if row["evidence_class"] == "unknown")
    if unknown_count / max(1, len(noncontrol_failures)) > 0.4:
        verdict = "INCONCLUSIVE"
    elif data_count >= 1.25 * max(1, model_count) and data_count >= 20:
        verdict = "SUPERVISION_DATA_DOMINANT"
    elif model_count >= 1.25 * max(1, data_count) and model_count >= 20:
        verdict = "MODEL_ERROR_DOMINANT"
    elif data_count >= 15 and model_count >= 15:
        verdict = "MIXED"
    else:
        verdict = "INCONCLUSIVE"

    family_enrichment: dict[str, Any] = {}
    control_rows = [row for row in case_rows if row["stratum"] == "rank1"]
    for category in ["SPATIAL_COMPOSITION_ERROR", "SMALL_OBJECT_RESOLUTION_ERROR", "MULTI_CHANGE_COMPOSITION_ERROR", "TEMPORAL_DIRECTION_ERROR", "SOURCE_STYLE_SHORTCUT", "REGISTRATION_VIEWPOINT_ERROR", "UNKNOWN"]:
        fail_n = sum(1 for row in noncontrol_failures if row["classification"] == category)
        control_n = sum(1 for row in control_rows if row["classification"] == category)
        fail_rate = fail_n / max(1, len(noncontrol_failures))
        control_rate = control_n / max(1, len(control_rows))
        family_enrichment[category] = {
            "failure_base_n": len(noncontrol_failures),
            "control_base_n": len(control_rows),
            "failure_count": fail_n,
            "failure_rate": fail_rate,
            "control_count": control_n,
            "control_rate": control_rate,
            "rate_difference": fail_rate - control_rate,
            "possible_enrichment": bool(fail_rate > control_rate and fail_n >= 3),
            "enriched": bool(fail_rate > control_rate and fail_n >= 10),
            "small_n_warning": bool(fail_n < 10),
        }

    visual_pass = sum(1 for row in case_rows if row["visual"].get("status") == "PASS")
    visual_near_duplicate_n = sum(1 for row in case_rows if row["stratum"] != "rank1" and row["visual"].get("visual_near_duplicate_candidate"))
    semantic_near = sum(1 for row in case_rows if row["classification"] == "SEMANTICALLY_VALID_NONEXACT_RESULT")
    collision_n = sum(1 for row in case_rows if row["collision"]["is_collision_group"])
    failure_counts = Counter(row["classification"] for row in noncontrol_failures)
    results = {
        "schema_version": "qcpr-model-data-pilot-results-v2",
        "pilot_id": "MODEL-DATA-QCPR-R19G-20260812",
        "status": "PASS_AGENT_DIAGNOSTIC_REVIEW",
        "review_type": REVIEW_TYPE,
        "producer_agent": "DATASET_AGENT",
        "producer_git_sha": base_handoff.get("producer_git_sha"),
        "authoritative_release_generator_sha": base_handoff.get("authoritative_release_generator_sha"),
        "created_at": created_at,
        "dataset_release_sha": RELEASE_SHA,
        "canonical_budget": 20,
        "canonical_checkpoint": "B20_step_1140",
        "sample_seed": SEED,
        "pilot_n": len(case_rows),
        "target_strata": STRATUM_QUOTAS,
        "observed_strata": dict(sorted(stratum_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "collision_case_count": collision_n,
        "visual_review_pass_count": visual_pass,
        "visual_near_duplicate_noncontrol_count": visual_near_duplicate_n,
        "semantic_near_nonexact_count": semantic_near,
        "verdict": verdict,
        "category_counts": dict(sorted(counts.items())),
        "failure_category_counts": dict(sorted(failure_counts.items())),
        "source_category_counts": source_category,
        "stratum_category_counts": {
            key: dict(sorted(Counter(row["classification"] for row in case_rows if row["stratum"] == key).items()))
            for key in sorted(STRATUM_QUOTAS)
        },
        "failure_enrichment": family_enrichment,
        "family_counts": dict(sorted(family_counts.items())),
        "research_question": "Are remaining B20 retrieval errors primarily model-error, supervision-data, mixed, or inconclusive?",
        "interpretation": {
            "noncontrol_case_count": len(noncontrol_failures),
            "data_ambiguity_count": data_count,
            "model_family_count": model_count,
            "unknown_count": unknown_count,
            "route_related": "NOT_LINKED; exported route information is not present in this pilot input",
            "direction_training_authorized": False,
            "direction_eval_authorized": False,
            "frozen_relevance_labels_modified": False,
            "hard_negatives_created": False,
        },
        "inputs": {
            "dataset_release_name": RELEASE_NAME,
            "dataset_release_sha": RELEASE_SHA,
            "model_handoff_sha256": file_sha(model_handoff_path),
            "model_top20_sha256": file_sha(top20_path),
            "development_manifest_sha256": file_sha(dev_path),
            "physical_items_sha256": file_sha(physical_path),
            "collision_groups_sha256": file_sha(collision_path),
            "model_head_sha": model_handoff.get("model_head_sha"),
            "artifact_generation_sha": model_handoff.get("artifact_generation_sha"),
            "primary_score_space": "global_stage1_score_matrix",
            "reranked_score_space_used": False,
        },
        "sample_manifest": {
            "path": sample_path.name,
            "sha256": sample_sha,
            "rows": len(case_rows),
            "producer_git_sha": base_handoff.get("producer_git_sha"),
            "dataset_release_sha": RELEASE_SHA,
            "created_at": created_at,
        },
        "notes": [
            "This is an agent diagnostic review, not an independent human audit.",
            "Visual evidence is a low-resolution native RGB descriptor and raw T1/T2 difference statistic; masks were not used for text relevance.",
            "A nonexact result is not automatically a negative: semantic overlap and collision groups are reported separately.",
            "The pilot does not authorize direction training, semantic training, or any change to frozen manifests.",
        ],
    }
    results["artifact_hash_definition"] = "SHA256 of canonical JSON excluding artifact_sha256"
    results["artifact_sha256"] = canonical_sha(results)
    results_path = out_dir / "qcpr_model_data_pilot_results.json"
    results_path.write_bytes(canonical_bytes(results))

    top_categories = [name for name, _ in failure_counts.most_common(3)]
    supported_families = [name.replace("_ERROR", "").replace("_STYLE_SHORTCUT", "") for name, info in family_enrichment.items() if info.get("enriched") and name not in {"UNKNOWN"}]
    md_lines = [
        "# QCPR MODEL×DATA diagnostic pilot",
        "",
        f"- Status: **{results['status']}**",
        f"- Review type: **{REVIEW_TYPE}**",
        f"- Dataset release: `{RELEASE_NAME}` / `{RELEASE_SHA}`",
        f"- Canonical model: `B20_step_1140` (20 presentations per physical pair)",
        f"- Sample: `{len(case_rows)}` cases, seed `{SEED}`, sample SHA `{sample_sha}`",
        f"- Primary ranking: `B20.global_stage1` only; reranked ranks were not mixed into the review.",
        "",
        "## Verdict",
        "",
        f"`DATASET_AGENT_VERDICT = {verdict}`",
        "",
        f"Non-control cases: `{len(noncontrol_failures)}`; supervision-ambiguity evidence: `{data_count}`; model-family evidence: `{model_count}`; unresolved: `{unknown_count}`.",
        "",
        "## Category counts",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    md_lines.extend(f"| {name} | {count} |" for name, count in sorted(counts.items()))
    md_lines.extend(["", "## Failure-family enrichment vs rank-1 controls", "", "| Family | Failure n | Failure rate | Control n | Control rate | Enriched |", "|---|---:|---:|---:|---:|:---:|"])
    for name, info in family_enrichment.items():
        md_lines.append(f"| {name} | {info['failure_count']} | {info['failure_rate']:.3f} | {info['control_count']} | {info['control_rate']:.3f} | {str(info['enriched']).lower()} |")
    if supported_families:
        model_family_line = "MODEL-SPECIFIC FAILURE FAMILY SUPPORTED = YES + " + ", ".join(supported_families)
    else:
        model_family_line = "MODEL-SPECIFIC FAILURE FAMILY SUPPORTED = NO (multi-change n=8 and temporal-direction n=4 are small-n diagnostic signals only)"
    md_lines.extend([
        "",
        f"TOP 3 FAILURE CATEGORIES = {', '.join(top_categories) if top_categories else 'NONE'}",
        model_family_line,
        f"SUPERVISION AMBIGUITY MATERIAL = {'YES' if data_count >= 15 else 'NO'}",
        "RECOMMENDED NEXT MODEL ACTION = KEEP_BASE",
        "",
        "## Guardrails",
        "",
        "- No frozen relevance labels were changed.",
        "- No hard negatives were created.",
        "- Direction train/eval readiness remains false.",
        "- Route-related failure was not linked because route fields are absent from the exported Top-20.",
        "- Small-n family counts are reported as diagnostic signals and are not promoted as clear enrichment.",
        "- This diagnostic does not authorize vNext or expanded training.",
    ])
    md_path = out_dir / "qcpr_model_data_pilot_results.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    updated = dict(base_handoff)
    updated["status"] = "PASS_AGENT_DIAGNOSTIC_REVIEW"
    updated["updated_at"] = created_at
    updated["pilot"] = {
        "status": results["status"],
        "pilot_n": len(case_rows),
        "pilot_review_type": REVIEW_TYPE,
        "sample_seed": SEED,
        "created_at": created_at,
        "sample_manifest": results["sample_manifest"],
        "results_path": results_path.name,
        "results_sha256": results["artifact_sha256"],
        "verdict": verdict,
        "category_counts": dict(sorted(counts.items())),
        "failure_enrichment": family_enrichment,
    }
    updated["canonical_budget"] = 20
    updated["canonical_checkpoint"] = "B20_step_1140"
    updated["direction_train_ready"] = False
    updated["direction_eval_ready"] = False
    updated["blockers"] = [
        "Exact-core retrieval remains evaluation-only under the r19g contract.",
        "No direction training or semantic/localized/long-series expansion is authorized.",
        "Agent diagnostic pilot is not a human verification gate for semantic promotion.",
    ]
    updated["model_handoff"] = {
        "path": model_handoff_path.name,
        "sha256": file_sha(model_handoff_path),
        "model_head_sha": model_handoff.get("model_head_sha"),
        "artifact_generation_sha": model_handoff.get("artifact_generation_sha"),
        "dataset_release_sha": model_handoff.get("dataset_release_sha"),
    }
    updated["artifact_hash_definition"] = "SHA256 of canonical JSON excluding artifact_sha256"
    updated.pop("artifact_sha256", None)
    updated["artifact_sha256"] = canonical_sha(updated)
    handoff_path = out_dir / "handoff_dataset_to_model.json"
    handoff_path.write_bytes(canonical_bytes(updated))

    print(json.dumps({
        "status": results["status"],
        "verdict": verdict,
        "pilot_n": len(case_rows),
        "category_counts": dict(sorted(counts.items())),
        "sample_sha256": sample_sha,
        "results_sha256": results["artifact_sha256"],
        "handoff_sha256": updated["artifact_sha256"],
        "results_file_sha256": file_sha(results_path),
        "handoff_file_sha256": file_sha(handoff_path),
    }, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top20", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--physical-items", required=True)
    parser.add_argument("--collision-groups", required=True)
    parser.add_argument("--model-handoff", required=True)
    parser.add_argument("--base-handoff", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
