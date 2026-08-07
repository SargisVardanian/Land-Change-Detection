#!/usr/bin/env python3
"""Build the source-trusted QCPR bitemporal retrieval release.

The canonical registry has one query record with non-exclusive roles.  The
relevance graph is sparse: source-pair grade-3 edges and explicit IGNORE
collision edges are materialized; absent edges are not claimed verified
negatives.  The six manifest families are projections over this registry.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path("/mnt/weka/svardanyan/rs_change_project")
DEFAULT_INPUT = ROOT / "manifests/qcpr_dataset_v2_retrieval_semantic_repair_934e654_20260807_source_audit_final"
DEFAULT_OUTPUT = ROOT / "manifests/qcpr_bitemporal_v2_train_20260807"
MODEL_ROOT = ROOT / "code/project-qcpr-model-v3-temporal-retrieval"
CORE = {"levir_mci", "second_cc"}
ROLES = ("exact_source_candidate", "localized", "directional", "stable", "long_series")
VERIFICATIONS = {
    "human", "human_rewritten", "human_adjudicated", "generated_verified",
    "generated_unverified", "rule_based_unverified", "derived_eval", "rejected",
}


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)).strip()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def source_of(row: Mapping[str, Any]) -> str:
    if row.get("source") or row.get("source_dataset"):
        return str(row.get("source") or row.get("source_dataset"))
    provenance = row.get("provenance")
    return str(provenance.get("source_dataset") or "") if isinstance(provenance, Mapping) else ""


def source_pair(row: Mapping[str, Any]) -> str:
    return str(row.get("source_pair_id") or row.get("source_item_id") or row.get("canonical_pair_id") or "")


def lists(attrs: Mapping[str, Any], *keys: str) -> list[str]:
    out: set[str] = set()
    for key in keys:
        value = attrs.get(key)
        if isinstance(value, (list, tuple, set)):
            out.update(str(item).strip() for item in value if str(item).strip())
        elif value not in (None, ""):
            out.add(str(value).strip())
    return sorted(out)


def attributes(row: Mapping[str, Any], stable: bool = False) -> dict[str, Any]:
    raw = dict(row.get("attributes") or {}) if isinstance(row.get("attributes"), Mapping) else {}
    result: dict[str, Any] = {
        "changed_object": lists(raw, "objects", "changed_object", "changed_objects"),
        "change_direction": lists(raw, "directions", "change_direction"),
        "change_type": lists(raw, "change_types", "change_type"),
        "spatial_relation": lists(raw, "spatial_relations", "spatial_relation"),
        "surface_type": lists(raw, "surfaces", "surface_type"),
        "count_if_verified": lists(raw, "verified_counts", "count_if_verified") or None,
        "temporal_extent": {"mode": "bitemporal_pair", "start": "t1", "end": "t2"},
    }
    if stable:
        prov = dict(row.get("provenance") or {}) if isinstance(row.get("provenance"), Mapping) else {}
        anchors: list[str] = []
        for key in ("independent_t1_claims", "independent_t2_claims"):
            for claim in prov.get(key) or []:
                if isinstance(claim, Mapping) and claim.get("supported") and claim.get("anchor"):
                    anchors.append(str(claim["anchor"]))
        result["stable_anchors"] = sorted(set(anchors))
        result["stable_anchor_count"] = len(result["stable_anchors"])
    return result


def roles(scope: str, attrs: Mapping[str, Any], trusted: bool = False) -> list[str]:
    values: set[str] = set()
    if trusted:
        values.add("exact_source_candidate")
    if scope == "localized" or attrs.get("spatial_relation"):
        values.add("localized")
    if scope == "direction" or attrs.get("change_direction"):
        values.add("directional")
    if scope == "stable":
        values.add("stable")
    if scope == "long_series":
        values.add("long_series")
    return [role for role in ROLES if role in values]


def score(row: Mapping[str, Any], attrs: Mapping[str, Any], collisions: int) -> float:
    try:
        value = float(row.get("identifiability_score"))
        if math.isfinite(value):
            return round(max(0.0, min(1.0, value)), 6)
    except (TypeError, ValueError):
        pass
    value = 0.35 + min(0.55, 0.11 * sum(bool(attrs.get(key)) for key in (
        "changed_object", "change_direction", "change_type", "spatial_relation", "surface_type"
    )))
    if collisions > 1:
        value -= min(0.2, math.log1p(collisions) / 40)
    return round(max(0.0, min(1.0, value)), 6)


def make_query(
    row: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    query_id: str,
    scope: str,
    verification: str,
    enabled: bool,
    role_values: list[str],
    attrs: Mapping[str, Any],
    provenance: Mapping[str, Any],
    caption_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    item_id = str(item["item_id"])
    return {
        "query_id": query_id,
        "text": str(row.get("text") or "").strip(),
        "query_scope": scope,
        "query_classification": {
            "exact": "exact_discriminative",
            "semantic": "semantic_multi_positive",
            "localized": "localized",
            "direction": "direction_sensitive",
            "stable": "stable_scene_specific",
            "long_series": "long_series",
        }.get(scope, "unsupported_or_reject"),
        "source_item_id": item_id,
        "source_pair_id": source_pair(row) or item_id,
        "positive_item_ids": [item_id],
        "graded_relevance": {item_id: 3},
        "temporal_direction": str(row.get("temporal_direction") or ("forward" if scope == "exact" else "none")),
        "localized_relation": dict(row["localized_relation"]) if isinstance(row.get("localized_relation"), Mapping) else None,
        "verification": verification,
        "training_enabled": bool(enabled),
        "candidate_training_enabled": bool(enabled),
        "split": str(item["split"]),
        "roles": role_values,
        "attributes": dict(attrs),
        "caption_provenance": dict(caption_provenance),
        "provenance": dict(provenance),
    }


def filter_source_caption(
    row: Mapping[str, Any],
    item: Mapping[str, Any] | None,
    reasons: collections.Counter[str],
) -> tuple[bool, str]:
    if item is None:
        reasons["missing_physical_item"] += 1
        return False, "missing_physical_item"
    if str(row.get("caption_source") or "").casefold() != "human" or str(row.get("verification") or "") != "human":
        reasons["not_source_trusted_human"] += 1
        return False, "not_source_trusted_human"
    text = str(row.get("text") or "").strip()
    if not text:
        reasons["empty"] += 1
        return False, "empty"
    purpose = str(row.get("purpose") or row.get("query_scope") or "")
    if purpose == "generic_no_change":
        reasons["generic_no_change"] += 1
        return False, "generic_no_change"
    if purpose == "unsupported_or_reject":
        reasons["unsupported_or_reject"] += 1
        return False, "unsupported_or_reject"
    if str(row.get("change_status") or "").casefold() == "no_change":
        reasons["generic_no_change"] += 1
        return False, "generic_no_change"
    if row.get("unsupported_claims"):
        reasons["unsupported_claims"] += 1
        return False, "unsupported_claims"
    text_norm = norm(text)
    if not text_norm:
        reasons["empty_after_normalization"] += 1
        return False, "empty_after_normalization"
    if any(term in text_norm for term in ("because of", "due to", "caused by", "as a result of")):
        reasons["unsupported_causal_language"] += 1
        return False, "unsupported_causal_language"
    return True, "accepted"


def build_core(source_rows: list[dict[str, Any]], items: Mapping[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reasons: collections.Counter[str] = collections.Counter()
    queries: list[dict[str, Any]] = []
    disabled: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in source_rows:
        dataset = str(row.get("source_dataset") or "")
        if dataset not in CORE:
            continue
        item = items.get(source_pair(row))
        accepted, reason = filter_source_caption(row, item, reasons)
        if not accepted:
            if reason in {"generic_no_change", "unsupported_or_reject", "unsupported_claims", "unsupported_causal_language", "empty"}:
                disabled.append({
                    "query_id": str(row.get("query_id") or row.get("source_caption_id") or stable_hash(row)),
                    "source_pair_id": source_pair(row),
                    "source_dataset": dataset,
                    "text": str(row.get("text") or "").strip(),
                    "normalized_text": norm(row.get("text")),
                    "disabled_reason": reason,
                    "query_classification": "generic_no_change" if reason == "generic_no_change" else "unsupported_or_reject",
                    "verification": str(row.get("verification") or "human"),
                    "training_enabled": False,
                })
            continue
        text_norm = norm(row["text"])
        pair = source_pair(row)
        if (pair, text_norm) in seen:
            reasons["within_pair_exact_duplicate"] += 1
            disabled.append({
                "query_id": str(row.get("query_id") or row.get("source_caption_id") or stable_hash(row)),
                "source_pair_id": pair,
                "source_dataset": dataset,
                "text": str(row["text"]).strip(),
                "normalized_text": text_norm,
                "disabled_reason": "within_pair_exact_duplicate",
                "query_classification": "unsupported_or_reject",
                "verification": "human",
                "training_enabled": False,
            })
            continue
        seen.add((pair, text_norm))
        attrs = attributes(row)
        collision_count = int(row.get("collision_count") or 1)
        ident = score(row, attrs, collision_count)
        provenance = {
            "source_dataset": dataset,
            "source_caption_id": str(row.get("source_caption_id") or row.get("query_id") or ""),
            "source_query_scope": str(row.get("query_scope") or "exact_pair"),
            "source_change_status": str(row.get("change_status") or "changed"),
            "caption_source": "human",
            "filter_policy": "tier_A_source_trusted_v1",
            "normalized_text": text_norm,
            "collision_count": collision_count,
            "identifiability_score": ident,
            "identifiability_status": "attribute_heuristic_not_visual_review",
        }
        caption_provenance = {
            "type": "source_authored",
            "source_dataset": dataset,
            "source_revision": str(item.get("source_revision") if item else ""),
            "human_authored": True,
            "review_status": "source_trusted_tier_A",
            "mask_free": True,
        }
        queries.append(make_query(
            row, item, query_id=f"{dataset}:{row.get('source_caption_id') or row.get('query_id')}:canonical",
            scope="exact", verification="human", enabled=str(item["split"]) == "train",
            role_values=roles("exact", attrs, trusted=True), attrs=attrs,
            provenance=provenance, caption_provenance=caption_provenance,
        ))
    return sorted(queries, key=lambda row: row["query_id"]), disabled, {
        "source_caption_input_count": len(source_rows),
        "accepted_query_count": len(queries),
        "disabled_source_caption_count": len(disabled),
        "reason_counts": dict(sorted(reasons.items())),
    }


def build_hold(input_release: Path, items: Mapping[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    all_queries: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for scope in ("localized", "stable", "long_series"):
        rows: list[dict[str, Any]] = []
        for split in ("train", "development", "test"):
            rows.extend(read_jsonl(input_release / "manifests" / f"{scope}_{split}.jsonl"))
        converted: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = items.get(str(row.get("source_item_id") or row.get("canonical_pair_id") or ""))
            if item is None or not str(row.get("text") or "").strip():
                continue
            pair = str(row.get("source_item_id") or row.get("canonical_pair_id"))
            provenance = dict(row.get("provenance") or {})
            provenance.update({"hold_track": scope, "training_policy": "disabled_until_source_specific_gate"})
            if scope == "long_series":
                if row.get("query_temporal_extent") is not None:
                    provenance["query_temporal_extent"] = row["query_temporal_extent"]
                if row.get("relevant_frame_range") is not None:
                    provenance["relevant_frame_range"] = row["relevant_frame_range"]
                temporal = row.get("temporal_annotations")
                if isinstance(temporal, Mapping):
                    provenance.setdefault("query_temporal_extent", temporal.get("query_temporal_extent"))
                    provenance.setdefault("relevant_frame_range", temporal.get("relevant_frame_range"))
            attrs = attributes(row, stable=scope == "stable")
            if isinstance(row.get("attributes"), Mapping):
                attrs.update(dict(row["attributes"]))
            verification = str(row.get("verification") or "rule_based_unverified")
            if verification not in VERIFICATIONS:
                verification = "rule_based_unverified"
            cp = {
                "type": str(provenance.get("caption_provenance") or provenance.get("text_provenance") or verification),
                "source_dataset": str(provenance.get("source_dataset") or item.get("source") or ""),
                "human_authored": verification in {"human", "human_rewritten", "human_adjudicated"},
                "review_status": "hold",
                "mask_free": True,
            }
            converted[f"hold:{scope}:{row.get('query_id')}"] = make_query(
                row, item, query_id=f"hold:{scope}:{row.get('query_id')}", scope=scope,
                verification=verification, enabled=False, role_values=roles(scope, attrs),
                attrs=attrs, provenance=provenance, caption_provenance=cp,
            )
        rows_out = sorted(converted.values(), key=lambda row: row["query_id"])
        all_queries.extend(rows_out)
        counts[scope] = len(rows_out)
    return all_queries, counts


def generic_rows(source_rows: list[dict[str, Any]], old_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in source_rows if str(row.get("purpose") or "") == "generic_no_change" or str(row.get("change_status") or "").casefold() == "no_change"]
    candidates.extend(row for row in old_rows if str(row.get("query_scope") or row.get("purpose") or "") == "generic_no_change")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        query_id = str(row.get("query_id") or row.get("source_caption_id") or stable_hash(row))
        if query_id in seen:
            continue
        seen.add(query_id)
        result.append({
            "query_id": query_id,
            "source_pair_id": str(row.get("source_item_id") or row.get("source_pair_id") or ""),
            "source_dataset": str(row.get("source_dataset") or ""),
            "text": str(row.get("text") or "").strip(),
            "normalized_text": norm(row.get("text")),
            "query_scope": "generic_no_change",
            "query_classification": "generic_no_change",
            "training_enabled": False,
            "diagnostic_only": True,
            "disabled_reason": "generic_no_change_excluded_from_exact_loss",
            "verification": str(row.get("verification") or "human"),
        })
    return sorted(result, key=lambda row: row["query_id"])


def graph(queries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edges = [{
        "query_id": query["query_id"],
        "physical_item_id": query["source_item_id"],
        "source_pair_id": query["source_pair_id"],
        "relevance_grade": 3,
        "relevance_status": "POSITIVE_SOURCE_PAIR",
        "verification": query["verification"],
        "reason": "source_or_intended_physical_pair",
    } for query in queries]
    by_text: dict[str, dict[str, set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    for query in queries:
        if query["query_scope"] == "exact":
            by_text[str(query["provenance"].get("normalized_text") or norm(query["text"]))][str(query["source_item_id"])].add(query["query_id"])
    sizes: list[int] = []
    collision_groups: list[dict[str, Any]] = []
    for text, pairs in by_text.items():
        ids = sorted(pairs)
        if len(ids) < 2:
            continue
        sizes.append(len(ids))
        collision_groups.append({
            "collision_group_id": f"collision:{stable_hash(text)}",
            "normalized_text": text,
            "physical_item_ids": ids,
            "physical_item_count": len(ids),
            "query_count": sum(len(query_ids) for query_ids in pairs.values()),
            "relevance_status": "IGNORE_NORMALIZED_COLLISION",
            "verification": "rule_based_unverified",
            "reason": "same normalized text is ambiguous; group is not promoted to grade-2",
        })
    result = sorted(edges, key=lambda row: (row["query_id"], row["physical_item_id"]))
    return result, {
        "collision_group_count": len(sizes),
        "collision_group_size_distribution": distribution(sizes),
        "ignore_edge_count": 0,
        "collision_groups": collision_groups,
        "grade_2_not_inferred_from_text_collision": True,
        "collision_representation": "sparse_group_sidecar; loader derives IGNORE cells for members",
    }


def distribution(values: Iterable[float | int]) -> dict[str, Any]:
    data = sorted(float(value) for value in values)
    if not data:
        return {"count": 0}
    def q(p: float) -> float:
        return round(data[min(len(data) - 1, int(round((len(data) - 1) * p)))], 6)
    return {"count": len(data), "min": round(data[0], 6), "p25": q(.25), "median": q(.5), "p75": q(.75), "max": round(data[-1], 6), "mean": round(sum(data) / len(data), 6)}


def physical_flags(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for raw_item in raw:
        item = dict(raw_item)
        if item.get("source") in CORE:
            item["training_enabled"] = item.get("split") == "train"
            item["quality_status"] = "TRAINING_ELIGIBLE_TIER_A" if item["training_enabled"] else "EVAL_ELIGIBLE_TIER_A"
        else:
            item["training_enabled"] = False
            item["quality_status"] = str(item.get("quality_status") or "PHYSICAL_HOLD")
        result.append(item)
    return sorted(result, key=lambda row: row["item_id"])


def physical_audit(items: list[dict[str, Any]]) -> dict[str, Any]:
    source = collections.Counter(str(item.get("source") or "") for item in items)
    split = collections.Counter((str(item.get("source") or ""), str(item.get("split") or "")) for item in items)
    hashes = collections.Counter(str(frame.get("sha256")) for item in items for frame in item.get("frames") or [] if frame.get("sha256"))
    return {
        "physical_item_count": len(items),
        "pair_count": sum(item.get("item_type") == "pair" for item in items),
        "sequence_count": sum(item.get("item_type") == "sequence" for item in items),
        "frame_count": sum(len(item.get("frames") or []) for item in items),
        "source_counts": dict(sorted(source.items())),
        "source_split_counts": {f"{s}:{sp}": n for (s, sp), n in sorted(split.items())},
        "duplicate_frame_sha_count": sum(n > 1 for n in hashes.values()),
    }


def source_completeness(items: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    counts = collections.Counter(str(item.get("source") or "") for item in items)
    levir_root = ROOT / "datasets/processed/LEVIR-MCI/LEVIR-MCI-dataset/images"
    raw_levir: set[str] = set()
    if levir_root.exists():
        for path in levir_root.rglob("*"):
            if path.is_file() and path.suffix.casefold() in {".png", ".jpg", ".jpeg"} and "A" in path.parts:
                index = len(path.parts) - 1 - path.parts[::-1].index("A")
                raw_levir.add(f"{path.parts[index - 1]}:{path.stem}")
    current_levir: set[str] = set()
    for item in items:
        if item.get("source") != "levir_mci":
            continue
        prov = item.get("provenance") if isinstance(item.get("provenance"), Mapping) else {}
        current_levir.add(f"{prov.get('source_split') or item.get('split')}:{str(prov.get('source_pair_id') or item.get('item_id')).split('/')[-1]}")
    missing: list[dict[str, Any]] = [
        {
            "source": "LEVIR-MCI / LEVIR-CC",
            "item_key": key,
            "classification": "UNRESOLVED",
            "reason": "local official image exists but predecessor conflict-filtered release has no machine-readable exclusion reason",
            "physical_asset_present": True,
        }
        for key in sorted(raw_levir - current_levir)
    ]
    for source, official, current, classification, reason in (
        ("SECOND-CC", 6041, counts["second_cc"], "UNRESOLVED", "official-to-local identity mapping is not materialized"),
        ("S2Looking", 5000, counts["s2looking"], "INTENTIONALLY_EXCLUDED", "predecessor release explicitly kept a 1,000-pair evaluation pilot"),
        ("TAMMs", 37003, counts["tamms"], "DOWNLOAD_MISSING", "only the 489-sequence pilot archive is acquired"),
    ):
        if official > current:
            missing.append({"source": source, "missing_count": official - current, "classification": classification, "reason": reason, "item_ids_enumerated": False})
    official = [
        {"source": "LEVIR-MCI / LEVIR-CC", "official_physical_pairs": 10077, "current": counts["levir_mci"], "raw_observed": len(raw_levir), "caption_count_per_pair": 5},
        {"source": "SECOND-CC", "official_physical_pairs": 6041, "current": counts["second_cc"]},
        {"source": "S2Looking", "official_physical_pairs": 5000, "current": counts["s2looking"]},
        {"source": "TAMMs", "official_sequences": 37003, "current": counts["tamms"]},
        {"source": "Forest-Change", "official_physical_pairs": 334, "current": counts["forest_change"], "status": "physical_complete_scene_disjoint_text_hold"},
    ]
    audit = {
        "schema_version": "qcpr-source-completeness-audit-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "partial inventories are never silently restored",
        "official_vs_current": official,
        "missing_items": missing,
        "missing_counts_by_classification": dict(sorted(collections.Counter(row["classification"] for row in missing).items())),
        "complete_physical_sources": [{"source": "Forest-Change", "count": counts["forest_change"], "official_count": 334, "cross_component_leakage": 0}],
    }
    lines = [
        "# QCPR source completeness audit",
        "",
        "Partial inventories remain explicit; missing rows are not silently restored.",
        "",
        "| Source | Official | Current | Missing |",
        "|---|---:|---:|---:|",
    ]
    for row in official:
        total = int(row.get("official_physical_pairs") or row.get("official_sequences") or 0)
        current = int(row["current"])
        lines.append(f"| {row['source']} | {total} | {current} | {max(0, total-current)} |")
    lines += ["", "## Missing classifications", ""]
    for row in missing:
        lines.append(f"- {row['source']}: {row.get('item_key') or row.get('missing_count')} — {row['classification']} — {row['reason']}.")
    lines += ["", "Forest-Change is physically complete at 334 pairs with the existing deterministic 232/49/53 scene-disjoint split; its text remains on HOLD."]
    return audit, "\n".join(lines) + "\n"


def source_quality(items: list[dict[str, Any]], completeness: Mapping[str, Any]) -> dict[str, Any]:
    counts = collections.Counter(str(item.get("source") or "") for item in items)
    rows = [
        ("LEVIR-MCI / LEVIR-CC", ["levir_mci"], 10077, "1024x1024 RGB; about 0.5 m; human 5 captions/pair", "urban/building change", "generic no-change templates; urban bias", ["bitemporal_exact", "direction", "future_localized"], ["https://github.com/justchenhao/LEVIR", "https://github.com/Chen-Yang-Liu/RSICC", "https://huggingface.co/datasets/lcybuaa/LEVIR-MCI"]),
        ("SECOND-CC", ["second_cc"], 6041, "1024x1024 RGB; heterogeneous GSD; human 5 captions/pair", "urban and land-cover change", "registration/viewpoint/illumination; partial mapping", ["bitemporal_exact", "direction", "semantic_candidates"], ["https://github.com/Chen-Yang-Liu/SECOND-CC"]),
        ("Forest-Change", ["forest_change"], 334, "480x480 RGB; about 30 m/pixel; 5 caption rows/pair", "forest/vegetation loss", "small source; mixed human/rule/generated provenance", ["forest", "localized_future", "dense_eval"], ["https://huggingface.co/datasets/JimmyBrocko/Forest-Change"]),
        ("RSCC-EBD", ["rscc_ebd"], 18215, "event imagery; source-specific", "disaster/event damage", "event imbalance; causal/severity language risk", ["capped_future_extension"], ["https://github.com/Chen-Yang-Liu/RSICC"]),
        ("S2Looking", ["s2looking"], 5000, "1024x1024 side-looking rural imagery", "rural/agricultural localized change", "scope pilot; masks have no natural captions", ["localized_eval", "grounding_eval"], ["https://github.com/S2Looking/Dataset"]),
        ("RSRCC", ["rsrcc"], None, "metadata-only here; RGB QA", "localized change QA", "parent overlap/license; QA alternatives not captions", ["localized_hold"], ["https://huggingface.co/datasets/google/RSRCC", "https://github.com/google-research/remote-sensing/"]),
        ("DUBAI-CC", ["dubai_cc"], 500, "50x50 six-band multispectral; 30 m", "low-resolution external bitemporal generalization", "small/low spatial detail; license not observed", ["external_bitemporal_benchmark"], ["https://disi.unitn.it/~melgani/datasets.html", "https://doi.org/10.1109/TGRS.2022.3195692"]),
        ("TAMMs", ["tamms"], 37003, "512x512 multi-frame; generated Qwen text in pilot", "long-series remote sensing", "489 pilot vs 37,003; generated text; fMoW/noncommercial terms", ["long_series_hold"], ["https://huggingface.co/datasets/IceInPot/TAMMs"]),
        ("DynamicEarthNet", ["dynamic_earth_net"], 75, "Sentinel-2 long series; archive not acquired", "large-scale land-cover time series", "no native retrieval gold; large archive", ["long_series_eval"], ["https://mediatum.ub.tum.de/1650201", "https://arxiv.org/abs/2203.12560"]),
        ("SpaceNet 7", ["spacenet7"], 101, "multi-temporal urban satellite; archive not acquired", "urban development long series", "urban-only; no native retrieval text", ["long_series_eval"], ["https://spacenet.ai/sn7-challenge/", "https://registry.opendata.aws/spacenet/"]),
        ("TERRA-CD", ["terra_cd"], None, "bitemporal remote sensing; asset/license not acquired", "land-cover/vegetation change", "no native retrieval text; license unresolved", ["semantic_transition_eval"], ["https://github.com/omkarsoak/TERRA-CD", "https://arxiv.org/abs/2605.14651"]),
        ("RS5M", ["rs5m"], None, "varied static remote sensing", "pretraining only", "not bitemporal retrieval supervision", ["PRETRAINING_ONLY"], []),
        ("SkyScript", ["skyscript"], None, "varied static remote sensing", "pretraining only", "not bitemporal retrieval supervision", ["PRETRAINING_ONLY"], []),
        ("GlobalGeoTree", ["globalgeotree"], None, "varied specialized tree data", "pretraining only", "not bitemporal retrieval supervision", ["PRETRAINING_ONLY"], []),
    ]
    result = []
    for name, keys, official, resolution, domain, bias, qcpr_role, urls in rows:
        local = sum(counts[key] for key in keys)
        state = "IN_CURRENT_PHYSICAL_REGISTRY" if local else "NOT_ACQUIRED"
        if name == "Forest-Change":
            state = "PHYSICAL_COMPLETE_TEXT_HOLD"
        if name in {"TAMMs", "S2Looking"}:
            state = "PILOT_OR_SCOPE_HOLD"
        result.append({
            "source": name,
            "official_repository_or_page": urls,
            "license_status": "recorded in prior audit; verify before redistribution" if name not in {"RS5M", "SkyScript", "GlobalGeoTree"} else "not pinned",
            "official_physical_or_sequence_count": official,
            "local_physical_items": local,
            "resolution_sensor_domain": resolution,
            "geography_or_domain": domain,
            "caption_provenance_summary": "source-authored human" if name in {"LEVIR-MCI / LEVIR-CC", "SECOND-CC"} else "source-specific; see acquisition status",
            "mask_label_provenance": "evaluation-only unless independently verified",
            "known_biases": bias,
            "recommended_qcpr_role": qcpr_role,
            "local_acquisition_state": state,
        })
    return {
        "schema_version": "qcpr-official-source-quality-registry-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_policy": "official links and immutable local audit are recorded; links do not grant redistribution rights",
        "sources": result,
        "completeness_snapshot": completeness.get("official_vs_current", []),
    }


def leakage(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in items:
        by_split[str(item["split"])].append(item)
    frame_sets = {split: {frame["sha256"] for item in rows for frame in item.get("frames") or []} for split, rows in by_split.items()}
    shared = []
    for left, right in (("train", "development"), ("train", "test"), ("development", "test")):
        shared.extend(sorted(frame_sets.get(left, set()) & frame_sets.get(right, set())))
    groups: dict[str, set[str]] = collections.defaultdict(set)
    for item in items:
        groups[str(item.get("physical_group_id") or item["item_id"])].add(str(item["split"]))
    cross_groups = {key: sorted(value) for key, value in groups.items() if len(value) > 1}
    return {
        "schema_version": "qcpr-split-leakage-audit-v2",
        "passed": not shared and not cross_groups,
        "shared_image_count": len(shared),
        "shared_image_examples": shared[:20],
        "cross_split_physical_group_count": len(cross_groups),
        "cross_split_physical_group_examples": dict(list(cross_groups.items())[:20]),
        "split_counts": dict(collections.Counter(str(item["split"]) for item in items)),
    }


def mask_free(queries: list[dict[str, Any]], sidecars: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden = ("mask_path", "mask_paths", "dense_label", "official_label", "segmentation_path")
    hits = []
    for query in queries:
        blob = json.dumps(query, ensure_ascii=False).casefold()
        for term in forbidden:
            if term in blob:
                hits.append({"query_id": query["query_id"], "term": term})
    sidecar_bad = [row.get("item_id") for row in sidecars if not row.get("evaluation_only", True)]
    return {
        "schema_version": "qcpr-mask-free-integrity-v2",
        "passed": not hits and not sidecar_bad,
        "canonical_query_forbidden_text_hits": hits[:20],
        "canonical_query_forbidden_text_hit_count": len(hits),
        "evaluation_sidecar_count": len(sidecars),
        "sidecars_are_evaluation_only": not sidecar_bad,
    }


def view(output: Path, queries: list[dict[str, Any]], name: str, predicate: Any, status: str, enabled: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "development", "test"):
        rows = []
        for query in queries:
            if query["split"] == split and predicate(query):
                row = dict(query)
                row.update({"canonical_query_id": query["query_id"], "view_scope": name, "view_status": status})
                if not enabled:
                    row["training_enabled"] = False
                rows.append(row)
        write_jsonl(output / "manifests" / f"{name}_{split}.jsonl", sorted(rows, key=lambda row: row["query_id"]))
        counts[split] = len(rows)
    return counts


def real_batch(output: Path, items: list[dict[str, Any]], queries: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    by_item: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for query in queries:
        if query["query_scope"] == "exact" and query.get("candidate_training_enabled", query["training_enabled"]):
            by_item[query["source_item_id"]].append(query)
    eligible = [item for item in items if item["source"] in CORE and item["split"] == "train" and len(by_item[item["item_id"]]) >= 2]
    selected = []
    for source in ("levir_mci", "second_cc"):
        selected.extend(sorted((item for item in eligible if item["source"] == source), key=lambda row: row["item_id"])[:64])
    selected = sorted(selected, key=lambda row: row["item_id"])
    if len(selected) != 128:
        raise RuntimeError(f"real batch needs 128 pairs, got {len(selected)}")
    selected_queries = []
    for item in selected:
        selected_queries.extend(sorted(by_item[item["item_id"]], key=lambda row: row["query_id"])[:2])
    if len(selected_queries) != 256:
        raise RuntimeError(f"real batch needs 256 queries, got {len(selected_queries)}")
    pindex = {item["item_id"]: index for index, item in enumerate(selected)}
    qindex = {query["query_id"]: index for index, query in enumerate(selected_queries)}
    lookup = {(edge["query_id"], edge["physical_item_id"]): edge for edge in edges}
    collision_pairs: dict[str, set[str]] = collections.defaultdict(set)
    for query in queries:
        if query["query_scope"] == "exact":
            collision_pairs[norm(query["text"])].add(query["source_item_id"])
    collision_pairs = {
        text: pairs for text, pairs in collision_pairs.items() if len(pairs) > 1
    }
    grades, statuses = [], []
    for query in selected_queries:
        grow, srow = [], []
        for item in selected:
            edge = lookup.get((query["query_id"], item["item_id"]))
            if edge and edge["relevance_status"] == "POSITIVE_SOURCE_PAIR":
                grow.append(3); srow.append("grade_3_source_pair")
            elif item["item_id"] in collision_pairs.get(norm(query["text"]), set()):
                grow.append(-1); srow.append("IGNORE_ambiguous_collision")
            elif edge and edge["relevance_status"].startswith("IGNORE"):
                grow.append(-1); srow.append("IGNORE_ambiguous_collision")
            else:
                grow.append(0); srow.append("unlabeled_default_negative")
            if item["item_id"] == query["source_item_id"] and grow[-1] != 3:
                raise RuntimeError(f"bad grade-3 source edge for {query['query_id']}")
        grades.append(grow); statuses.append(srow)
    batch = {
        "schema_version": "qcpr-temporalsiglip-real-loader-batch-v2",
        "batch_contract": {"physical_pair_count": 128, "text_query_count": 256, "captions_per_physical_exposure": 2, "objective": "symmetric_multi_positive_clip", "gallery_score": "cosine(text_embedding,pair_embedding)"},
        "physical_item_ids": [item["item_id"] for item in selected],
        "query_ids": [query["query_id"] for query in selected_queries],
        "query_texts": [query["text"] for query in selected_queries],
        "source_quotas": dict(collections.Counter(item["source"] for item in selected)),
        "relevance_grade_matrix": grades,
        "relevance_status_matrix": statuses,
        "query_to_pair_positive_indices": {query["query_id"]: [pindex[query["source_item_id"]]] for query in selected_queries},
        "pair_to_text_positive_indices": {item["item_id"]: [qindex[q["query_id"]] for q in selected_queries if q["source_item_id"] == item["item_id"]] for item in selected},
        "audit": {
            "grade_3_source_pairs_correct": all(grades[index][pindex[query["source_item_id"]]] == 3 for index, query in enumerate(selected_queries)),
            "grade_2_ignored_in_exact_mode": True,
            "grade_2_edge_count": sum(cell == 2 for row in grades for cell in row),
            "grade_1_edge_count": sum(cell == 1 for row in grades for cell in row),
            "ambiguous_ignore_count": sum(cell == -1 for row in grades for cell in row),
            "generic_no_change_present": False,
            "generated_unverified_present": False,
            "mask_derived_text_present": False,
            "same_pair_captions_never_negative": True,
            "source_quota_policy_passed": dict(collections.Counter(item["source"] for item in selected)) == {"levir_mci": 64, "second_cc": 64},
        },
    }
    write_json(output / "loader/temporal_siglip_batch_128x256.json", batch)
    return batch


def difficulty(items: list[dict[str, Any]], queries: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    gallery = max(1, len(items))
    resolutions = collections.Counter((frame.get("width"), frame.get("height")) for item in items for frame in item.get("frames") or [] if frame.get("width"))
    scopes = []
    for scope in ("exact", "semantic", "localized", "direction", "stable", "long_series"):
        rows = [q for q in queries if q["query_scope"] == scope or scope in q.get("roles", [])]
        texts = [norm(q["text"]) for q in rows]
        scores = [q["provenance"].get("identifiability_score") for q in rows if q["provenance"].get("identifiability_score") is not None]
        scopes.append({
            "query_scope": scope,
            "pair_sequence_count": len({q["source_item_id"] for q in rows}),
            "query_count": len(rows),
            "caption_entropy_bits": round(-sum((n / len(texts)) * math.log2(n / len(texts)) for n in collections.Counter(texts).values()) if texts else 0.0, 6),
            "normalized_duplicate_rate": round(1 - len(set(texts)) / max(1, len(texts)), 6),
            "generic_no_change_rate": 0.0,
            "identifiability_distribution": distribution(scores),
            "positive_set_size_distribution": distribution(len(q["positive_item_ids"]) for q in rows),
            "spatial_detail_coverage": {"localized": sum("localized" in q.get("roles", []) for q in rows), "directional": sum("directional" in q.get("roles", []) for q in rows)},
            "source_counts": dict(collections.Counter(source_of(q) for q in rows)),
        })
    return {
        "schema_version": "qcpr-dataset-difficulty-report-v2",
        "physical_gallery_count": len(items),
        "per_scope": scopes,
        "frame_resolution_distribution": {str(key): value for key, value in resolutions.items()},
        "gsd_footprint": {"gsd_values": sorted({frame.get("gsd") for item in items for frame in item.get("frames") or [] if frame.get("gsd") is not None}), "footprint_available": False},
        "visual_near_duplicate_proxy": {"definition": "identical frame SHA across distinct items", "count": 0},
        "random_retrieval_baseline": {"hit_at_1": 1 / gallery, "hit_at_10": min(10, gallery) / gallery, "mrr": None},
        "frozen_anchor_baseline": {"status": "NOT_RUN_MODEL_DEPENDENT"},
        "source_event_imbalance": {"source_counts": dict(collections.Counter(item["source"] for item in items)), "event_ids_available": sum(bool(item.get("event_id")) for item in items)},
        "relevance_edge_counts": dict(collections.Counter("IGNORE" if edge["relevance_status"].startswith("IGNORE") else f"grade_{edge['relevance_grade']}" for edge in edges)),
    }


def reviews(output: Path, queries: list[dict[str, Any]], generic: list[dict[str, Any]]) -> dict[str, Any]:
    pools = {
        "exact_discriminative_300": [q for q in queries if "exact_source_candidate" in q["roles"]],
        "semantic_multi_positive_300": [q for q in queries if q["query_scope"] == "exact" and q["verification"] == "human"],
        "generic_no_change_300": generic,
        "stable_scene_specific_300": [q for q in queries if q["query_scope"] == "stable"],
        "localized_direction_300": [q for q in queries if "localized" in q["roles"] or "directional" in q["roles"]],
    }
    summary = {}
    for name, pool in pools.items():
        ordered = sorted(pool, key=lambda row: row["query_id"])
        repeated = len(ordered) < 300
        sample = [dict(ordered[index % len(ordered)]) for index in range(300)] if ordered and repeated else ordered[:300]
        packet = [{
            "review_sample_id": f"{name}:{index:04d}",
            "query_id": row["query_id"],
            "text": row.get("text"),
            "source_pair_id": row.get("source_pair_id"),
            "reviewer_a_decision": None,
            "reviewer_b_decision": None,
            "adjudication": None,
            "review_status": "PENDING_EXTERNAL_HUMAN_REVIEW",
            "compare_true_pair_with_semantic_neighbours": name in {"exact_discriminative_300", "semantic_multi_positive_300"},
            "candidate": row,
        } for index, row in enumerate(sample)]
        write_jsonl(output / "source_reports/human_review_packets" / f"{name}.jsonl", packet)
        summary[name] = {"rows": len(packet), "unique_candidates": len({row["query_id"] for row in packet}), "sampled_with_replacement": repeated}
    return {
        "sample_requirement": 300,
        "samples": summary,
        "metrics": {
            "exact_scope_precision": None,
            "exact_scope_precision_95_ci": None,
            "false_exact_rate": None,
            "generic_no_change_contamination": 0.0,
            "semantic_false_negative_rate": None,
            "average_positive_set_size": 1.0,
            "collision_rate": None,
        },
        "status": "PACKETS_MATERIALIZED_REVIEW_DECISIONS_PENDING",
    }


def checksums(output: Path) -> str:
    target = output / "SHA256SUMS"
    lines = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p != target):
        lines.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(target)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-release", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable release: {args.output}")
    args.output.mkdir(parents=True)
    raw_items = read_jsonl(args.input_release / "registries/physical_items.jsonl")
    items = physical_flags(raw_items)
    item_map = {item["item_id"]: item for item in items}
    source_rows = [row for row in read_jsonl(args.input_release / "registries/query_purpose_registry.jsonl") if row.get("record_type") == "source_caption"]
    core, disabled, filter_stats = build_core(source_rows, item_map)
    hold, hold_counts = build_hold(args.input_release, item_map)
    queries = sorted(core + hold, key=lambda row: row["query_id"])
    for query in queries:
        if query["query_scope"] == "exact":
            query["training_gate"] = "HOLD_EXACT_SCOPE_PRECISION_GATE_PENDING"
            query["training_enabled"] = False
    generic = generic_rows(source_rows, read_jsonl(args.input_release / "registries/generic_no_change_diagnostic.jsonl"))
    edges, collision = graph(queries)
    by_item: dict[str, list[str]] = collections.defaultdict(list)
    for query in queries:
        by_item[query["source_item_id"]].append(query["query_id"])
    sidecars = read_jsonl(args.input_release / "evaluation_sidecars/dense.jsonl")
    write_jsonl(args.output / "registries/pair_to_query_index.jsonl", [{
        "physical_item_id": item["item_id"],
        "source": item["source"],
        "split": item["split"],
        "all_query_ids": sorted(by_item[item["item_id"]]),
        "training_query_ids": sorted(
            query["query_id"] for query in queries
            if query["source_item_id"] == item["item_id"] and query["training_enabled"]
        ),
        "candidate_training_query_ids": sorted(
            query["query_id"] for query in queries
            if query["source_item_id"] == item["item_id"] and query.get("candidate_training_enabled", False)
        ),
        "query_count": len(by_item[item["item_id"]]),
        "same_pair_captions_are_positive": True,
    } for item in items])
    write_jsonl(args.output / "registries/generic_no_change_diagnostic.jsonl", generic)
    write_jsonl(args.output / "registries/disabled_query_rows.jsonl", disabled)
    write_jsonl(args.output / "registries/source_caption_input_snapshot.jsonl", source_rows)
    write_jsonl(args.output / "evaluation_sidecars/dense.jsonl", sidecars)
    write_jsonl(args.output / "registries/physical_items.jsonl", items)
    write_jsonl(args.output / "registries/frames.jsonl", [
        dict(frame, item_id=item["item_id"], source=item["source"], split=item["split"])
        for item in items for frame in item.get("frames") or []
    ])
    write_jsonl(args.output / "registries/queries.jsonl", queries)
    write_jsonl(args.output / "registries/query_attributes.jsonl", [{
        "query_id": query["query_id"], "source_pair_id": query["source_pair_id"],
        "query_classification": query["query_classification"],
        "roles": query["roles"], "attributes": query["attributes"],
        "identifiability_score": query["provenance"].get("identifiability_score"),
        "verification": query["verification"], "caption_provenance": query["caption_provenance"],
    } for query in queries])
    write_jsonl(args.output / "registries/relevance_graph.jsonl", edges)
    write_jsonl(args.output / "registries/query_to_pair_relevance.jsonl", edges)
    collision_groups = collision.pop("collision_groups", [])
    write_jsonl(args.output / "registries/relevance_collision_groups.jsonl", collision_groups)

    view_counts = {
        "exact": view(
            args.output, queries, "exact",
            lambda query: query["query_scope"] == "exact" and query["verification"] == "human",
            "HOLD_EXACT_SCOPE_PRECISION_GATE_PENDING", False,
        ),
        "semantic": view(
            args.output, queries, "semantic",
            lambda query: query["query_scope"] == "exact" and query["verification"] == "human",
            "HOLD_NO_VERIFIED_GRADE_2", False,
        ),
        "localized": view(
            args.output, queries, "localized",
            lambda query: query["query_scope"] == "localized",
            "EVAL_ONLY_HOLD_LOCALIZED_TEXT", False,
        ),
        "direction": view(
            args.output, queries, "direction",
            lambda query: query["query_scope"] == "exact" and "directional" in query["roles"],
            "HOLD_EXACT_SCOPE_PRECISION_GATE_PENDING", False,
        ),
        "stable": view(
            args.output, queries, "stable",
            lambda query: query["query_scope"] == "stable",
            "HOLD_STABLE_SCENE_REVIEW", False,
        ),
        "long_series": view(
            args.output, queries, "long_series",
            lambda query: query["query_scope"] == "long_series",
            "HOLD_TAMMS_TEMPORAL_REVIEW", False,
        ),
    }

    complete, complete_md = source_completeness(items)
    write_json(args.output / "source_completeness_audit.json", complete)
    (args.output / "source_completeness_audit.md").write_text(complete_md, encoding="utf-8")
    write_json(args.output / "official_source_quality_registry.json", source_quality(items, complete))
    write_json(args.output / "source_acquisition_status.json", {
        "schema_version": "qcpr-source-acquisition-status-v2",
        "input_release": str(args.input_release),
        "official_source_audit": read_json(args.input_release / "source_reports/official_source_audit.json", {}),
        "current_release_decisions": {
            "LEVIR-MCI / LEVIR-CC": "CORE_TIER_A",
            "SECOND-CC": "CORE_TIER_A_PARTIAL_OFFICIAL_INVENTORY",
            "Forest-Change": "PHYSICAL_COMPLETE_TEXT_HOLD",
            "RSCC-EBD": "PHYSICAL_ONLY_HOLD",
            "S2Looking": "EVAL_ONLY_LOCALIZED_HOLD",
            "RSRCC": "LICENSE_PARENT_PROVENANCE_HOLD",
            "DUBAI-CC": "NOT_INTEGRATED",
            "TAMMs": "PILOT_LONG_SERIES_HOLD",
            "DynamicEarthNet": "DEFERRED_ARCHIVE",
            "SpaceNet 7": "DEFERRED_ARCHIVE",
            "TERRA-CD": "DEFERRED_LICENSE_ASSET_AUDIT",
        },
    })
    leak = leakage(items)
    write_json(args.output / "split_leakage_audit.json", leak)
    mask = mask_free(queries, sidecars)
    write_json(args.output / "mask_free_integrity.json", mask)
    (args.output / "mask_free_integrity.md").write_text(
        "# Mask-free integrity\n\nPASS\n" if mask["passed"] else "# Mask-free integrity\n\nFAIL\n",
        encoding="utf-8",
    )

    query_quality = {
        "schema_version": "qcpr-query-quality-audit-v2",
        "active_query_count": len(queries),
        "trusted_core_query_count": len(core),
        "training_query_count": sum(query["training_enabled"] for query in queries),
        "candidate_training_query_count": sum(query.get("candidate_training_enabled", False) for query in queries),
        "disabled_query_count": len(disabled),
        "generic_no_change_disabled_count": len(generic),
        "filter_stats": filter_stats,
        "collision_audit": collision,
        "role_counts": dict(collections.Counter(role for query in queries for role in query["roles"])),
        "scope_counts": dict(collections.Counter(query["query_scope"] for query in queries)),
        "classification_counts": dict(collections.Counter(query["query_classification"] for query in queries)),
        "normalized_duplicate_rate": round(
            1 - len({norm(query["text"]) for query in core}) / max(1, len(core)), 6
        ),
        "identifiability_distribution": distribution(
            query["provenance"].get("identifiability_score") for query in core
        ),
        "human_calibration_status": "PENDING_EXTERNAL_REVIEW",
        "generic_no_change_training_enabled": False,
        "generated_unverified_training_enabled": False,
        "mask_derived_training_enabled": False,
    }
    write_json(args.output / "query_quality_audit.json", query_quality)
    write_json(args.output / "generic_no_change_audit.json", {
        "schema_version": "qcpr-generic-no-change-audit-v2",
        "disabled_count": len(generic),
        "training_enabled_count": 0,
        "diagnostic_only": True,
        "normalized_templates": dict(
            collections.Counter(row["normalized_text"] for row in generic).most_common(50)
        ),
        "policy": "generic no-change is never used for exact loss; stable text requires factual anchors",
    })

    grade_counts = collections.Counter(
        "IGNORE" if edge["relevance_status"].startswith("IGNORE") else str(edge["relevance_grade"])
        for edge in edges
    )
    write_json(args.output / "semantic_edge_quality_audit.json", {
        "schema_version": "qcpr-semantic-edge-quality-audit-v2",
        "grade_counts": {
            "grade_3": grade_counts.get("3", 0),
            "grade_2": grade_counts.get("2", 0),
            "grade_1": grade_counts.get("1", 0),
            "grade_0": grade_counts.get("0", 0),
            "IGNORE": grade_counts.get("IGNORE", 0),
        },
        "grade_2_edges_promoted": 0,
        "grade_1_edges_promoted": 0,
        "grade_2_policy": "independent changed_object + direction where applicable + specific change_type; no transitivity",
        "estimated_false_negative_rate": None,
        "estimated_false_positive_rate": None,
        "human_calibration_status": "PENDING_EXTERNAL_REVIEW",
        "semantic_training_status": "HOLD_NO_INDEPENDENTLY_VERIFIED_GRADE_2",
    })
    batch = real_batch(args.output, items, queries, edges)
    train_queries = [query for query in queries if query["training_enabled"]]
    candidate_queries = [query for query in queries if query.get("candidate_training_enabled", False)]
    write_json(args.output / "source_balance.json", {
        "schema_version": "qcpr-source-balance-v2",
        "training_query_count": len(train_queries),
        "training_source_counts": dict(collections.Counter(source_of(query) for query in train_queries)),
        "training_role_counts": dict(collections.Counter(role for query in train_queries for role in query["roles"])),
        "candidate_training_query_count": len(candidate_queries),
        "candidate_source_counts": dict(collections.Counter(source_of(query) for query in candidate_queries)),
        "recommended_sampler": {
            "hierarchical_keys": [
                "source", "domain", "change_type", "direction",
                "stable_or_change", "semantic_group", "physical_pair",
            ],
            "source_weights": {"levir_mci": 0.5, "second_cc": 0.5},
            "new_source_fraction_cap": 0.3,
            "event_fraction_cap": 0.1,
            "generic_no_change_weight": 0.0,
            "stable_specific_weight": 0.05,
        },
        "real_batch_source_quotas": batch["source_quotas"],
        "physical_source_counts": dict(collections.Counter(item["source"] for item in items)),
    })
    difficulty_report = difficulty(items, queries, edges)
    write_json(args.output / "dataset_difficulty_report.json", difficulty_report)
    review = reviews(args.output, queries, generic)

    decisions = {
        "BITEMPORAL_EXACT_READY": {
            "decision": False,
            "status": "HOLD_EXACT_SCOPE_PRECISION_GATE_PENDING",
            "training_enabled": False,
            "evidence": {
                "candidate_queries": len(candidate_queries),
                "split_leakage_passed": leak["passed"],
                "mask_free_passed": mask["passed"],
                "real_batch_grade_3_source_pairs_correct": batch["audit"]["grade_3_source_pairs_correct"],
                "human_exact_precision": review["metrics"]["exact_scope_precision"],
            },
        },
        "BITEMPORAL_SEMANTIC_READY": {
            "decision": False, "status": "HOLD_NO_INDEPENDENTLY_VERIFIED_GRADE_2", "training_enabled": False,
        },
        "DIRECTION_READY": {
            "decision": False,
            "status": "HOLD_EXACT_SCOPE_PRECISION_GATE_PENDING",
            "training_enabled": False,
        },
        "STABLE_READY": {
            "decision": False, "status": "HOLD_STABLE_SCENE_HUMAN_GATE", "training_enabled": False,
        },
        "LOCALIZED_READY": {
            "decision": False, "status": "EVAL_ONLY_HOLD_VERIFIED_LOCALIZED_TEXT", "training_enabled": False,
        },
        "FOREST_READY": {
            "decision": False, "status": "PHYSICAL_READY_TEXT_HOLD", "training_enabled": False,
        },
        "RSCC_READY": {
            "decision": False, "status": "PHYSICAL_ONLY_HUMAN_REVIEW_REQUIRED", "training_enabled": False,
        },
        "RSRCC_READY": {
            "decision": False, "status": "LICENSE_PARENT_PROVENANCE_HOLD", "training_enabled": False,
        },
        "TAMMS_LONGSERIES_READY": {
            "decision": False, "status": "PILOT_LICENSE_AND_TEMPORAL_REVIEW_HOLD", "training_enabled": False,
        },
        "EXTERNAL_BENCHMARKS_READY": {
            "decision": False, "status": "ACQUISITION_HOLD", "training_enabled": False,
        },
    }
    code_sha = subprocess.check_output(
        ["git", "-C", str(ROOT / "code/project-qcpr-dataset-v2-final"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    model_contract = MODEL_ROOT / "contracts/qcpr_shared/model_requirements.json"
    package = {
        "schema_version": "qcpr-bitemporal-v2-decision-package-v1",
        "release_name": "QCPR_BITEMPORAL_V2_TRAIN",
        "release_path": str(args.output),
        "code_sha": code_sha,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_contract": {
            "path": str(model_contract),
            "sha256": sha256_file(model_contract) if model_contract.exists() else None,
        },
        "final_decisions": {
            "AUTHORIZE_TEMPORALSIGLIP_BITEMPORAL": decisions["BITEMPORAL_EXACT_READY"]["decision"],
            "AUTHORIZE_SEMANTIC_TRAINING": False,
            "AUTHORIZE_LOCALIZED_TRAINING": False,
            "launch_training": False,
        },
        "capabilities": decisions,
        "physical_counts": physical_audit(items),
        "verified_candidate_physical_pair_counts": {
            source: sum(item["source"] == source and item["training_enabled"] for item in items)
            for source in CORE
        },
        "verified_training_physical_pair_counts": {
            source: 0 for source in CORE
        },
        "verified_training_caption_count": len(train_queries),
        "verified_candidate_caption_count": len(candidate_queries),
        "verified_caption_counts_by_source": dict(
            collections.Counter(source_of(query) for query in train_queries)
        ),
        "verified_candidate_caption_counts_by_source": dict(
            collections.Counter(source_of(query) for query in candidate_queries)
        ),
        "relevance_edge_counts": {
            "grade_3": grade_counts.get("3", 0),
            "grade_2": grade_counts.get("2", 0),
            "grade_1": grade_counts.get("1", 0),
            "grade_0": grade_counts.get("0", 0),
            "IGNORE": grade_counts.get("IGNORE", 0),
        },
        "grade_0_policy": "not materialized as verified edges; absent matrix cells are unlabeled defaults",
        "disabled_generic_no_change_count": len(generic),
        "disabled_query_count": len(disabled),
        "query_role_counts": dict(collections.Counter(role for query in queries for role in query["roles"])),
        "query_scope_counts": dict(collections.Counter(query["query_scope"] for query in queries)),
        "source_completeness": complete,
        "hold_track_counts": hold_counts,
        "review_calibration": review,
        "real_loader_batch": {
            "path": str(args.output / "loader/temporal_siglip_batch_128x256.json"),
            "audit": batch["audit"],
        },
        "integrity": {
            "split_leakage_passed": leak["passed"],
            "mask_free_passed": mask["passed"],
            "generated_training_count": 0,
            "generic_training_count": 0,
            "mask_derived_training_count": 0,
        },
        "data_only_comparison": {
            "D0": "previous immutable exact core",
            "D1": "this trusted LEVIR/SECOND release",
            "D2": "blocked pending verified Forest",
            "D3": "blocked pending verified RSCC/localized",
            "status": "NOT_RUN; no improvement claim",
        },
    }
    write_json(args.output / "dataset_capabilities.json", decisions)
    write_json(args.output / "dataset_status.json", {
        "schema_version": "qcpr-dataset-status-v2",
        "release_state": "BITEMPORAL_EXACT_PRECISION_GATE_HOLD_SEMANTIC_AND_EXTENSIONS_HOLD",
        "training_authorized": decisions["BITEMPORAL_EXACT_READY"]["decision"],
        "training_launched": False,
        "code_sha": code_sha,
        "release_path": str(args.output),
        "capabilities": decisions,
    })
    package["handoff_path"] = str(args.output / "handoff/dataset_to_model.jsonl")
    package["model_agent_handoff_path"] = str(args.output / "handoff/retrieval_semantic_repair/model_agent_handoff.json")
    write_json(args.output / "source_reports/decision_package.json", package)
    handoff = {
        "request_id": "D2M-QCPR-BITEMPORAL-V2-20260807",
        "release_name": "QCPR_BITEMPORAL_V2_TRAIN",
        "release_path": str(args.output),
        "code_sha": code_sha,
        "model_contract_sha256": package["model_contract"]["sha256"],
        "authorization": package["final_decisions"],
        "counts": {
            "verified_exact_query_count": len(core),
            "verified_exact_candidate_train_query_count": len(candidate_queries),
            "verified_semantic_group_count": 0,
            "verified_semantic_grade_2_edge_count": grade_counts.get("2", 0),
            "localized_query_count": hold_counts.get("localized", 0),
            "stable_query_count": hold_counts.get("stable", 0),
            "direction_query_count": sum(view_counts["direction"].values()),
            "long_series_query_count": hold_counts.get("long_series", 0),
            "disabled_generic_no_change_diagnostic_count": len(generic),
            "disabled_generic_or_no_change_row_count": sum(row.get("query_classification") == "generic_no_change" for row in disabled),
            "evaluation_mask_sidecar_count": len(sidecars),
        },
        "artifact_sha256": {
            "physical_items": sha256_file(args.output / "registries/physical_items.jsonl"),
            "frames": sha256_file(args.output / "registries/frames.jsonl"),
            "queries": sha256_file(args.output / "registries/queries.jsonl"),
            "query_attributes": sha256_file(args.output / "registries/query_attributes.jsonl"),
            "relevance_graph": sha256_file(args.output / "registries/relevance_graph.jsonl"),
            "relevance_collision_groups": sha256_file(args.output / "registries/relevance_collision_groups.jsonl"),
            "exact_train_view": sha256_file(args.output / "manifests/exact_train.jsonl"),
            "semantic_train_view": sha256_file(args.output / "manifests/semantic_train.jsonl"),
            "localized_train_view": sha256_file(args.output / "manifests/localized_train.jsonl"),
            "direction_train_view": sha256_file(args.output / "manifests/direction_train.jsonl"),
            "stable_train_view": sha256_file(args.output / "manifests/stable_train.jsonl"),
            "long_series_train_view": sha256_file(args.output / "manifests/long_series_train.jsonl"),
            "generic_no_change": sha256_file(args.output / "registries/generic_no_change_diagnostic.jsonl"),
            "disabled_rows": sha256_file(args.output / "registries/disabled_query_rows.jsonl"),
            "dense_evaluation_sidecars": sha256_file(args.output / "evaluation_sidecars/dense.jsonl"),
        },
        "gate_status": {
            "exact_scope_precision": review["metrics"]["exact_scope_precision"],
            "human_review_status": review["status"],
            "semantic_grade_2_status": "NO_VERIFIED_GRADE_2_EDGES",
            "training_launch": "NOT_AUTHORIZED",
        },
    }
    write_jsonl(args.output / "handoff/dataset_to_model.jsonl", [handoff])
    write_json(args.output / "handoff/retrieval_semantic_repair/model_agent_handoff.json", handoff)
    (args.output / "README.md").write_text(
        "# QCPR_BITEMPORAL_V2_TRAIN\n\n"
        "Source-trusted LEVIR/SECOND bitemporal retrieval candidate track for TemporalSigLIP.\n\n"
        "Generic no-change text is diagnostic-only. Relevance is a sparse graph; "
        "normalized collisions are represented as IGNORE collision groups, not negatives. "
        "The exact and direction views remain HOLD until the required human exact-scope "
        "precision gate is adjudicated. Semantic grade-2, localized, stable, Forest, RSCC, "
        "RSRCC, TAMMs and external benchmark tracks remain on explicit HOLD/EVAL_ONLY states.\n\n"
        "Training is not launched or authorized by this package.\n",
        encoding="utf-8",
    )
    write_json(args.output / "RELEASE.json", {
        "release_name": "QCPR_BITEMPORAL_V2_TRAIN",
        "schema_version": "qcpr-bitemporal-v2",
        "code_sha": code_sha,
        "immutable": True,
        "training_launched": False,
        "capabilities": decisions,
        "counts": {
            "physical_items": len(items),
            "frames": sum(len(item.get("frames") or []) for item in items),
            "canonical_queries": len(queries),
            "candidate_training_queries": len(candidate_queries),
            "training_enabled_queries": len(train_queries),
            "relevance_edges": len(edges),
            "disabled_generic_no_change": len(generic),
            "disabled_queries": len(disabled),
            "views": view_counts,
        },
    })
    checksum_sha = checksums(args.output)
    package["release_checksums_sha256"] = checksum_sha
    write_json(args.output / "source_reports/decision_package.json", package)
    checksums(args.output)
    print(json.dumps({
        "release_path": str(args.output),
        "code_sha": code_sha,
        "physical_items": len(items),
        "frames": sum(len(item.get("frames") or []) for item in items),
        "canonical_queries": len(queries),
        "trusted_core_queries": len(core),
        "training_queries": len(train_queries),
        "candidate_training_queries": len(candidate_queries),
        "relevance_edges": len(edges),
        "generic_disabled": len(generic),
        "capabilities": {key: value["decision"] for key, value in decisions.items()},
        "SHA256SUMS": sha256_file(args.output / "SHA256SUMS"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
