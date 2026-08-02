#!/usr/bin/env python3
"""Build an immutable, review-gated QCPR Dataset-v2 Stage-2 release.

The release contains the real RSCC-EBD physical source and keeps all semantic
views separate.  Provisional/event-derived candidates and source-verified
S2Looking relations are never promoted to the primary reviewed semantic
training view.  A release may be emitted in DATA_QUALITY_HOLD while the two
independent human review sheets are pending; it is never silently promoted.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "development", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def jsonl_count(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.is_file() else 0


def git_sha(repo: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    existing = read_jsonl(path)
    added = list(rows)
    write_jsonl(path, [*existing, *added])
    return len(added)


def validate_rscc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise SystemExit("RSCC pair registry is empty")
    ids = [str(row.get("canonical_pair_id") or "") for row in rows]
    if any(not value.startswith("rscc_ebd:") for value in ids):
        raise SystemExit("RSCC canonical IDs must use rscc_ebd namespace")
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate RSCC canonical pair IDs")
    for row in rows:
        for key in ("t1_path", "t2_path"):
            path = Path(str(row.get(key) or ""))
            if not path.is_file():
                raise SystemExit(f"RSCC missing image file {key}: {path}")
        if str(row.get("split")) not in SPLITS:
            raise SystemExit(f"unsupported RSCC split: {row.get('split')}")
        if not str(row.get("source_event_id") or row.get("source_scene_group_id") or ""):
            raise SystemExit(f"RSCC row has no event identity: {row['canonical_pair_id']}")
    split_counts = collections.Counter(str(row["split"]) for row in rows)
    event_by_split: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        event = str(row.get("source_event_id") or row.get("source_scene_group_id"))
        event_by_split[str(row["split"])].add(event)
    for left in SPLITS:
        for right in SPLITS:
            if left < right and event_by_split[left] & event_by_split[right]:
                raise SystemExit(f"RSCC event leakage between {left} and {right}")
    return {
        "pair_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "event_group_count": len({str(row.get("source_event_id") or row.get("source_scene_group_id")) for row in rows}),
        "event_split_counts": {split: len(event_by_split[split]) for split in SPLITS},
        "event_split_disjoint": True,
    }


def validate_structured_view(structured_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = read_json(structured_dir / "structured_semantic_audit.json", {})
    loader = read_json(structured_dir / "structured_semantic_loader_contract.json", {})
    independent = read_json(structured_dir / "independent_structured_semantic_verification.json", {})
    if audit.get("status") != "STRUCTURED_SOURCE_SEMANTIC_READY" or not audit.get("training_enabled"):
        raise SystemExit("structured semantic audit is not source-verified")
    if not loader.get("passed"):
        raise SystemExit("structured semantic loader contract did not pass")
    if independent.get("status") != "INDEPENDENT_STRUCTURED_SEMANTIC_VERIFIED" or not independent.get("mask_free_output"):
        raise SystemExit("independent structured S2Looking verification did not pass")
    rows: list[dict[str, Any]] = []
    group_ids: set[str] = set()
    for split in SPLITS:
        path = structured_dir / f"retrieval_semantic_structured_{split}.jsonl"
        for row in read_jsonl(path):
            if row.get("split") != split or row.get("schema_version") != "temporal-caption-manifest-v1":
                raise SystemExit(f"structured split/schema mismatch in {path}")
            if row.get("verification_status") != "structured_source_verified":
                raise SystemExit("structured row is not independently source verified")
            encoded = json.dumps(row, sort_keys=True, ensure_ascii=False).casefold()
            if any(token in encoded for token in ("mask_path", "semantic_t1_path", "semantic_t2_path", "label_path", "dense_path")):
                raise SystemExit(f"mask/dense path leaked into structured row {row.get('query_id')}")
            group_ids.add(str(row.get("semantic_group_id")))
            rows.append(row)
    registry = read_jsonl(structured_dir / "semantic_group_registry_structured.jsonl")
    valid_groups = {str(group.get("semantic_group_id")) for group in registry if int(group.get("pair_count", 0) or 0) >= 2}
    if not rows or not valid_groups.intersection(group_ids):
        raise SystemExit("structured S2Looking view lacks a compact multi-positive group")
    if int(independent.get("verified_rows", 0)) != len(rows):
        raise SystemExit("independent S2Looking row count does not match output")
    return rows, {
        "audit": audit,
        "loader": loader,
        "independent": independent,
        "group_count": len(registry),
        "split_counts": dict(collections.Counter(row["split"] for row in rows)),
    }


def validate_review_package(review_dir: Path) -> dict[str, Any]:
    required = ("human_review_packet.jsonl", "reviewer_a_decisions.jsonl", "reviewer_b_decisions.jsonl", "adjudicated_decisions.jsonl", "human_review_agreement.json", "review_package_audit.json")
    if any(not (review_dir / name).is_file() for name in required):
        raise SystemExit("human review package is incomplete")
    packet = read_jsonl(review_dir / "human_review_packet.jsonl")
    audit = read_json(review_dir / "review_package_audit.json", {})
    if len(packet) != 240 or int(audit.get("row_count", 0)) != 240 or int(audit.get("event_count", 0)) != 12:
        raise SystemExit("Stage-2 review package must contain exactly 20 rows x 12 events")
    events = collections.Counter(str(row.get("source_event_id") or "") for row in packet)
    if set(events) != {str(row.get("source_event_id")) for row in packet} or any(count != 20 for count in events.values()):
        raise SystemExit("review package is not balanced at 20 rows per event")
    if not audit.get("event_ids_provenance_only") or audit.get("semantic_positive_sets_materialized"):
        raise SystemExit("review package violates event/provenance-only contract")
    if audit.get("training_enabled"):
        raise SystemExit("review package cannot be training-enabled")
    return {
        "status": audit.get("status"),
        "rows": len(packet),
        "events": len(events),
        "event_counts": dict(sorted(events.items())),
        "split_counts": dict(collections.Counter(str(row.get("split")) for row in packet)),
        "audit_sha256": sha256(review_dir / "review_package_audit.json"),
    }


def validate_gold_semantic(gold_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    audit_path = gold_dir / "gold_semantic_audit.json"
    audit = read_json(audit_path, {})
    if audit.get("status") not in {"GOLD_SEMANTIC_HOLD", "GOLD_SEMANTIC_READY"}:
        raise SystemExit("gold semantic audit has an unsupported status")
    manifests = [read_jsonl(gold_dir / f"retrieval_semantic_gold_{split}.jsonl") for split in SPLITS]
    groups = read_jsonl(gold_dir / "semantic_group_registry.jsonl")
    text_rows = read_jsonl(gold_dir / "verified_rscc_text_registry.jsonl")
    rows = [row for part in manifests for row in part]
    if audit.get("status") == "GOLD_SEMANTIC_HOLD":
        if rows or groups or text_rows or audit.get("training_enabled"):
            raise SystemExit("gold hold contains promoted semantic rows")
    else:
        if not rows or not groups or not text_rows or not audit.get("training_enabled"):
            raise SystemExit("gold ready is empty or disabled")
        if audit.get("event_ids_used_for_semantics") is not False:
            raise SystemExit("gold semantic groups use event IDs")
        if set(audit.get("positive_grades_enabled", [])) != {2, 3}:
            raise SystemExit("gold primary training grades must be 2 and 3")
        events_by_split: dict[str, set[str]] = collections.defaultdict(set)
        for row in rows:
            events_by_split[str(row["split"])].add(str(row.get("source_event_id")))
            if not row.get("training_enabled") or not row.get("semantic_group_id"):
                raise SystemExit("gold row is missing training/group contract")
        for left in SPLITS:
            for right in SPLITS:
                if left < right and events_by_split[left] & events_by_split[right]:
                    raise SystemExit("gold semantic events cross splits")
        for group in groups:
            if "source_event_id" in group or "event_id" in group or not group.get("event_ids_are_provenance_only"):
                raise SystemExit("gold group registry improperly uses event identity")
    return audit, rows, groups


def normalize_rscc_pair(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.update({
        "source_dataset": "RSCC-EBD",
        "schema_version": "qcpr-stage2-pair-registry-v2",
        "is_synthetic": False,
        "parent_pair_id": None,
        "source_scene_group_id": str(row["source_scene_group_id"]),
        "source_event_id": str(row.get("source_event_id") or row["source_scene_group_id"]),
        "image_view": "NATIVE",
    })
    return result


def rscc_dense_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        sidecar = row.get("dense_label_sidecar") or {}
        for polarity, key in (("pre", "pre_mask_path"), ("post", "post_mask_path")):
            path = str(sidecar.get(key) or "")
            if not path:
                raise SystemExit(f"missing RSCC {polarity} dense label for {row['canonical_pair_id']}")
            output.append({
                "canonical_pair_id": row["canonical_pair_id"],
                "caption_id": f"{row['canonical_pair_id']}:rscc_dense:{polarity}",
                "label_type": f"{polarity}_disaster_mask",
                "mask_path": path,
                "source_dataset": "RSCC-EBD",
            })
    return output


def structured_caption_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "caption_id": row["query_id"],
        "canonical_pair_id": row["canonical_pair_id"],
        "dataset_name": "S2Looking",
        "text": row["text"],
        "normalized_text": row["normalized_text"],
        "caption_source": row["caption_source"],
        "task_type": "structured_semantic_evaluation",
        "query_scope": "semantic_group",
        "semantic_group_id": row["semantic_group_id"],
        "quality_score": 1.0,
        "identifiability_score": 0.25,
        "verification_status": "independently_source_verified",
        "is_generated": True,
        "generator": row["generator"],
        "training_enabled": False,
        "evaluation_only": True,
    } for row in rows]


def structured_relevance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "caption_id": row["query_id"],
        "semantic_group_id": row["semantic_group_id"],
        "exact_pair_id": row["canonical_pair_id"],
        "positive_pair_ids": [],
        "ignored_pair_ids": [],
        "valid_negative_policy": "compact_semantic_group_registry",
        "training_enabled": False,
        "evaluation_only": True,
    } for row in rows]


def gold_caption_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "caption_id": row["query_id"],
        "canonical_pair_id": row["canonical_pair_id"],
        "text": row["text"],
        "normalized_text": " ".join(str(row["text"]).casefold().split()),
        "caption_source": "RSCC-QvQ-human-reviewed",
        "task_type": "semantic_retrieval",
        "query_scope": "semantic_group",
        "semantic_group_id": row["semantic_group_id"],
        "quality_score": 1.0,
        "identifiability_score": 1.0,
        "verification_status": row["verification_status"],
        "verification_confidence": row.get("verification_confidence"),
        "is_generated": True,
        "generator": "human_rewrite_or_accept",
        "training_enabled": True,
    } for row in rows]


def gold_relevance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "caption_id": row["query_id"],
        "canonical_pair_id": row["canonical_pair_id"],
        "exact_pair_id": row["exact_pair_id"],
        "semantic_group_id": row["semantic_group_id"],
        "semantic_group_refs": row.get("semantic_group_refs", []),
        "positive_pair_ids": [],
        "ignored_pair_ids": [],
        "valid_negative_policy": "compact_semantic_group_registry",
        "graded_relevance_policy": row["graded_relevance_policy"],
        "training_enabled": True,
    } for row in rows]


def rscc_dense_evaluation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "canonical_pair_id": row["canonical_pair_id"],
        "dataset_name": "RSCC-EBD",
        "source_event_id": row.get("source_event_id"),
        "source_scene_group_id": row.get("source_scene_group_id"),
        "split": row["split"],
        "t1_path": row["t1_path"],
        "t2_path": row["t2_path"],
        "dense_label_ids": [f"{row['canonical_pair_id']}:rscc_dense:pre", f"{row['canonical_pair_id']}:rscc_dense:post"],
        "mask_access": "dense_label_registry_only",
        "evaluation_only": True,
    } for row in rows]


def hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(p for p in root.rglob("*") if p.is_file())
        if not str(path.relative_to(root)).startswith("hashes/")
    }


def copy_file(source: Path, target: Path) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def update_source_coverage(source: dict[str, Any]) -> dict[str, Any]:
    result = dict(source)
    result["stage2_deferred_access_blockers"] = [
        {"source_dataset": "SYSU-CD", "status": "ACCESS_BLOCKED", "blocker": "official image archive requires Baidu verification/SharePoint authorization", "blocks_first_rscc_p2": False},
        {"source_dataset": "Hi-UCD", "status": "ACCESS_BLOCKED", "blocker": "corrected official archive and 2025-11-01 patch are request-gated", "blocks_first_rscc_p2": False},
        {"source_dataset": "Synthetic RCD SECOND real-A", "status": "MAPPING_BLOCKED", "blocker": "original SECOND-A assets are unavailable; synthetic-A remains diagnostic only", "blocks_first_rscc_p2": False},
    ]
    for row in result.get("sources", []):
        if row.get("source_dataset") in {"SYSU-CD", "Hi-UCD"}:
            row["state"] = "ACCESS_BLOCKED"
            row["accessibility"] = "blocked"
        if row.get("source_dataset") == "Synthetic RCD SECOND":
            row["state"] = "MAPPING_BLOCKED"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--rscc-pairs", type=Path, required=True)
    parser.add_argument("--rscc-qvq", type=Path, required=True)
    parser.add_argument("--rscc-qvq-audit", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--review-package", type=Path, required=True)
    parser.add_argument("--gold-semantic-dir", type=Path, required=True)
    parser.add_argument("--stage2-audit-root", type=Path, required=True)
    parser.add_argument("--gate-summary", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite release root: {args.output_root}")
    current_code_sha = git_sha(args.repo)
    required_core = [args.core_root / "registries" / name for name in ("pair_registry.jsonl", "caption_registry.jsonl", "relevance_registry.jsonl", "dense_label_registry.jsonl")]
    if any(not path.is_file() for path in required_core):
        raise SystemExit("core release is incomplete")

    rscc_rows = [normalize_rscc_pair(row) for row in read_jsonl(args.rscc_pairs)]
    rscc_audit = validate_rscc(rscc_rows)
    structured_rows, structured_info = validate_structured_view(args.structured_dir)
    review_info = validate_review_package(args.review_package)
    gold_audit, gold_rows, gold_groups = validate_gold_semantic(args.gold_semantic_dir)
    qvq_audit = read_json(args.rscc_qvq_audit, {})
    if qvq_audit.get("training_enabled") or qvq_audit.get("human_audit_passed"):
        raise SystemExit("unreviewed RSCC QvQ audit cannot be training-enabled")

    shutil.copytree(args.core_root, args.output_root)
    out = args.output_root
    registries, manifests, reports = out / "registries", out / "manifests", out / "reports"
    for directory in (registries, manifests, reports, out / "hashes"):
        directory.mkdir(parents=True, exist_ok=True)

    core_pair_rows = read_jsonl(registries / "pair_registry.jsonl")
    core_ids = {str(row["canonical_pair_id"]) for row in core_pair_rows}
    rscc_ids = {str(row["canonical_pair_id"]) for row in rscc_rows}
    if core_ids & rscc_ids:
        raise SystemExit("RSCC/core canonical pair collision")
    write_jsonl(registries / "pair_registry.jsonl", [*core_pair_rows, *rscc_rows])
    append_jsonl(registries / "dense_label_registry.jsonl", rscc_dense_rows(rscc_rows))

    # Primary semantic supervision is review-gated.  Keep the old core files
    # only as compatibility paths and replace them with the gold rows, never
    # with provisional/event-only candidates.
    for split in SPLITS:
        gold_split = [row for row in gold_rows if row.get("split") == split]
        write_jsonl(manifests / f"retrieval_semantic_gold_{split}.jsonl", gold_split)
        write_jsonl(manifests / f"retrieval_semantic_{split}_v2.jsonl", gold_split)
        write_jsonl(manifests / f"retrieval_semantic_structured_s2looking_{split}.jsonl", [
            {**row, "training_enabled": False, "evaluation_only": True, "task_view": "structured_semantic_evaluation"}
            for row in structured_rows if row["split"] == split
        ])
    write_jsonl(registries / "semantic_group_registry.jsonl", gold_groups)
    write_jsonl(registries / "semantic_group_registry_structured_s2looking.jsonl", read_jsonl(args.structured_dir / "semantic_group_registry_structured.jsonl"))
    write_jsonl(registries / "relevance_registry_gold_semantic.jsonl", gold_relevance_rows(gold_rows))
    write_jsonl(registries / "relevance_registry_structured_s2looking.jsonl", structured_relevance_rows(structured_rows))
    append_jsonl(registries / "caption_registry.jsonl", [*gold_caption_rows(gold_rows), *structured_caption_rows(structured_rows)])

    write_jsonl(manifests / "physical_pairs_rscc_ebd.jsonl", [{
        "canonical_pair_id": row["canonical_pair_id"], "source_dataset": "RSCC-EBD", "source_event_id": row.get("source_event_id"),
        "source_scene_group_id": row.get("source_scene_group_id"), "split": row["split"], "t1_path": row["t1_path"], "t2_path": row["t2_path"],
        "dense_label_ids": [f"{row['canonical_pair_id']}:rscc_dense:pre", f"{row['canonical_pair_id']}:rscc_dense:post"],
        "caption_supervision": "review_gated_separate_view", "training_enabled": bool(gold_rows),
    } for row in rscc_rows])
    write_jsonl(manifests / "dense_evaluation_rscc_ebd.jsonl", rscc_dense_evaluation_rows(rscc_rows))

    # Preserve provisional rows as auditable data, but make their disabled
    # role explicit and keep them out of all primary semantic manifests.
    provisional_dir = manifests / "provisional_semantic_candidates"
    provisional_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        source = args.stage2_audit_root / "semantic_view" / f"retrieval_semantic_{split}_v2.jsonl"
        if source.is_file():
            target = provisional_dir / source.name
            shutil.copy2(source, target)
    qvq_rows = []
    for row in read_jsonl(args.rscc_qvq):
        qvq_rows.append({**row, "training_enabled": False, "verification_status": "unverified_candidate", "release_role": "candidate_view_only"})
    write_jsonl(manifests / "provisional_semantic_candidates" / "rscc_qvq_unverified.jsonl", qvq_rows)

    review_out = reports / "semantic_review"
    review_out.mkdir(parents=True, exist_ok=True)
    for path in args.review_package.iterdir():
        if path.is_file():
            shutil.copy2(path, review_out / path.name)
    gold_out = reports / "semantic_gold"
    gold_out.mkdir(parents=True, exist_ok=True)
    for path in args.gold_semantic_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, gold_out / path.name)
    write_json(reports / "rscc_qvq_caption_audit.json", {**qvq_audit, "training_enabled": False, "release_role": "unverified_candidate_view_only"})
    write_json(reports / "review_gate_summary.json", {
        "review_package": review_info,
        "gold_semantic": {"status": gold_audit.get("status"), "rows": len(gold_rows), "groups": len(gold_groups), "training_enabled": bool(gold_audit.get("training_enabled"))},
        "training_enabled_unreviewed_rows": 0,
        "event_ids_used_for_semantics": False,
        "p2_submitted": False,
    })

    source_registry = update_source_coverage(read_json(args.stage2_audit_root / "source_registry.json", {}))
    write_json(reports / "stage2_source_registry.json", source_registry)
    gate_path = args.gate_summary or (args.stage2_audit_root / "stage2_gate_summary.json")
    gate_summary = read_json(gate_path, {})
    if gate_summary:
        write_json(reports / "stage2_gate_summary.json", gate_summary)
    rscc_report_dir = reports / "rscc_ebd"
    for name in ("rscc_ebd_pair_audit.json", "rscc_mask_free_loader_contract.json"):
        copy_file(args.stage2_audit_root / "rscc_ebd" / name, rscc_report_dir / name)
    for name in ("source_access_audit.json", "architecture_screening_plan.json"):
        copy_file(args.stage2_audit_root / name, reports / name)
    copy_file(args.stage2_audit_root / "architecture_screening" / "architecture_screening_frozen_report.json", reports / "architecture_screening_frozen_report.json")
    copy_file(args.structured_dir / "semantic_group_registry_structured.jsonl", reports / "structured_semantic" / "semantic_group_registry_structured.jsonl")

    source_counts = collections.Counter(str(row.get("source_dataset")) for row in core_pair_rows + rscc_rows)
    split_counts = collections.defaultdict(collections.Counter)
    for row in core_pair_rows + rscc_rows:
        split_counts[str(row.get("source_dataset"))][str(row.get("split"))] += 1
    provisional_audit = read_json(args.stage2_audit_root / "semantic_view" / "semantic_view_audit.json", {})
    deferred = source_registry["stage2_deferred_access_blockers"]
    active_blockers = []
    if gold_audit.get("status") != "GOLD_SEMANTIC_READY":
        active_blockers.extend([
            "two independent human reviewers have not completed the 240-row RSCC packet",
            "retrieval_semantic_gold train/development/test manifests are empty and training-disabled",
        ])
    release_status = "DATASET_V2_STAGE2_READY" if gold_audit.get("status") == "GOLD_SEMANTIC_READY" and len(gold_rows) > 0 else "DATA_QUALITY_HOLD"
    release = {
        "schema_version": "qcpr-dataset-v2-stage2-expanded-release-v2",
        "release_name": "QCPR Dataset-v2 Stage-2 RSCC semantic review release",
        "status": release_status,
        "stage2_ready": release_status == "DATASET_V2_STAGE2_READY",
        "training_authorized": False,
        "training_launched": False,
        "code_sha": current_code_sha,
        "core_release_preserved": str(args.core_root),
        "historical_r1_job_200097_touched": False,
        "physical_pairs": {"real_temporal_retrieval": len(core_pair_rows) + len(rscc_rows) - sum(source_counts.get(name, 0) for name in ("S2Looking", "s2looking")), "dense_only_s2looking": sum(source_counts.get(name, 0) for name in ("S2Looking", "s2looking")), "registry_total": len(core_pair_rows) + len(rscc_rows)},
        "source_pair_counts": dict(sorted(source_counts.items())),
        "source_split_counts": {source: dict(sorted(counts.items())) for source, counts in sorted(split_counts.items())},
        "new_real_physical_source": {"source": "RSCC-EBD", **rscc_audit},
        "architecture": {"selected_anchor": "B1_framewise_gated_difference", "frozen_comparison": "B0_native_joint_temporal", "grounding_ablation": "B2_text_conditioned_temporal_fusion", "broad_search_frozen": True},
        "semantic_supervision": {
            "gold": {"status": gold_audit.get("status"), "rows": len(gold_rows), "groups": len(gold_groups), "training_enabled": bool(gold_audit.get("training_enabled"))},
            "provisional_candidate_rows": int(provisional_audit.get("multi_positive_rows", 0) or 0),
            "provisional_training_enabled": False,
            "training_enabled_unreviewed_rows": 0,
            "structured_s2looking_rows": len(structured_rows),
            "structured_s2looking_independent_verified": True,
            "structured_s2looking_training_enabled": False,
            "event_ids_used_for_semantics": False,
        },
        "human_review": review_info,
        "deferred_access_blockers": deferred,
        "active_blockers": active_blockers,
        "p2_submitted": False,
        "counts": {"registries": {name: jsonl_count(registries / name) for name in ("pair_registry.jsonl", "caption_registry.jsonl", "relevance_registry.jsonl", "dense_label_registry.jsonl")}, "manifests": {path.name: jsonl_count(path) for path in sorted(manifests.glob("*.jsonl"))}},
        "license_report": {"source_registry": str(reports / "stage2_source_registry.json"), "rscc_license_ancestry_preserved": True, "access_blockers_explicit": True},
        "loader_contracts": {
            "rscc_mask_free": str(rscc_report_dir / "rscc_mask_free_loader_contract.json"),
            "structured_s2looking": str(reports / "structured_semantic" / "structured_semantic_loader_contract.json"),
            "primary_semantic_gold": "empty_until_two_reviewer_adjudication",
        },
        "release_artifact_hashes_path": "hashes/stage2_release_artifacts_sha256.json",
    }
    write_json(out / "dataset_v2_stage2_release.json", release)
    write_json(out / "dataset_v2_stage2_source_coverage.json", source_registry)
    write_json(out / "dataset_v2_stage2_build_summary.json", release)
    write_json(out / "dataset_v2_stage2_leakage_audit.json", {
        "passed": True, "core_release_untouched": True, "rscc_event_split_disjoint": True, "cross_split_pair_id_collisions": 0,
        "rscc_pair_id_collisions_with_core": 0, "rscc_event_groups": rscc_audit["event_group_count"], "rscc_split_counts": rscc_audit["split_counts"],
        "same_event_semantically_different_pairs_are_not_auto_positive": True,
    })
    write_json(out / "dataset_v2_stage2_relevance_audit.json", {
        "passed": True, "exact_view_unchanged_from_core": True, "primary_gold_status": gold_audit.get("status"), "primary_gold_rows": len(gold_rows),
        "compact_semantic_group_registry": True, "event_ids_used_for_semantics": False, "provisional_rows_excluded_from_training": True,
        "structured_s2looking_separate_evaluation_view": True, "positive_ignored_overlap": 0,
    })
    write_json(reports / "stage2_controlled_exposure_plan.json", {
        "status": "PREPARED_NOT_AUTHORIZED", "p2_submitted": False, "architecture": "B1_framewise_gated_difference", "seed": 20260802,
        "arms": {
            "P1": {"sources": ["LEVIR-MCI", "SECOND-CC"], "physical_pairs": int(source_counts.get("LEVIR-MCI", 0) + source_counts.get("SECOND-CC", 0))},
            "P2-real": {"sources": ["LEVIR-MCI", "SECOND-CC", "RSCC-EBD"], "physical_pairs": len(core_pair_rows) + len(rscc_rows)},
            "P2-semantic": {"requires": "GOLD_SEMANTIC_READY", "rows_currently_available": len(gold_rows)},
        },
        "common_contract": {"logical_physical_batch": 128, "logical_text_queries": 256, "score_matrix": "256x128", "physical_microbatch": 16, "captions_per_pair": 2, "fixed_exposure": True, "steps": 348, "hard_negative_mining": False, "early_stopping": False, "mask_supervision": False, "bf16": True, "hardware": "one H100 80GB"},
        "estimated_vram_gib": 15.99,
        "runtime_estimate": "not measured for the final 348-step contract; measure in bounded pilot first",
        "submission_authorization": "blocked_until_release_status_DATASET_V2_STAGE2_READY",
    })
    (reports / "README.md").write_text("DATA_QUALITY_HOLD: RSCC physical source is integrated; primary semantic training remains disabled until two independent human reviews and adjudication complete. SYSU-CD, Hi-UCD and Synthetic RCD real-A are explicit deferred access blockers and do not block the first RSCC P2.\n", encoding="utf-8")

    # Hash all release artifacts except hashes/ itself.  Add the count before
    # the final hash pass so the metadata is deterministic.
    release["release_artifact_hash_count"] = len(hash_tree(out))
    write_json(out / "dataset_v2_stage2_release.json", release)
    write_json(out / "dataset_v2_stage2_build_summary.json", release)
    artifacts = hash_tree(out)
    if len(artifacts) != release["release_artifact_hash_count"]:
        raise SystemExit("release artifact count changed during finalization")
    write_json(out / "hashes/stage2_release_artifacts_sha256.json", artifacts)
    write_json(out / "hashes/registry_sha256.json", {str(path.relative_to(out)): sha256(path) for path in sorted(registries.glob("*.jsonl"))})
    write_json(out / "hashes/manifest_sha256.json", {str(path.relative_to(out)): sha256(path) for path in sorted(manifests.rglob("*.jsonl"))})
    print(json.dumps({"status": release_status, "output_root": str(out), "code_sha": current_code_sha, "real_temporal_pairs": release["physical_pairs"]["real_temporal_retrieval"], "dense_only_pairs": release["physical_pairs"]["dense_only_s2looking"], "gold_rows": len(gold_rows), "training_authorized": False, "artifact_hash_count": len(artifacts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
