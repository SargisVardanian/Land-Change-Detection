#!/usr/bin/env python3
"""Build a new immutable, data-quality-gated QCPR Stage-2 release.

The builder copies the audited Core into a new root and appends only validated
physical RSCC-EBD records plus the separately source-verified coarse S2Looking
semantic view.  Generated RSCC QvQ captions remain in an explicitly disabled
candidate view.  Existing releases are never modified.
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


SEMANTIC_SPLITS = ("train", "development", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl_count(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.is_file() else 0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def git_sha(repo: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def validate_rscc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise SystemExit("RSCC pair registry is empty")
    ids = [str(row.get("canonical_pair_id") or "") for row in rows]
    if any(not value.startswith("rscc_ebd:") for value in ids):
        raise SystemExit("RSCC canonical IDs must use rscc_ebd namespace")
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate RSCC canonical pair IDs")
    missing = []
    for row in rows:
        for key in ("t1_path", "t2_path"):
            path = Path(str(row.get(key) or ""))
            if not path.is_file():
                missing.append({"canonical_pair_id": row["canonical_pair_id"], "field": key, "path": str(path)})
    if missing:
        raise SystemExit(f"RSCC missing image files: {missing[:3]}")
    splits = collections.Counter(str(row.get("split")) for row in rows)
    if set(splits) != set(SEMANTIC_SPLITS):
        raise SystemExit(f"RSCC must have train/development/test splits: {dict(splits)}")
    events = {str(row.get("source_scene_group_id") or "") for row in rows}
    return {"pair_count": len(rows), "split_counts": dict(sorted(splits.items())), "event_group_count": len(events)}


def validate_structured_view(structured_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = read_json(structured_dir / "structured_semantic_audit.json", {})
    loader = read_json(structured_dir / "structured_semantic_loader_contract.json", {})
    if audit.get("status") != "STRUCTURED_SOURCE_SEMANTIC_READY" or not audit.get("training_enabled"):
        raise SystemExit("structured semantic audit is not ready")
    if not loader.get("passed"):
        raise SystemExit("structured semantic loader contract did not pass")
    rows: list[dict[str, Any]] = []
    group_ids: set[str] = set()
    for split in SEMANTIC_SPLITS:
        path = structured_dir / f"retrieval_semantic_structured_{split}.jsonl"
        for row in read_jsonl(path):
            if row.get("split") != split or row.get("schema_version") != "temporal-caption-manifest-v1":
                raise SystemExit(f"structured semantic split/schema mismatch in {path}")
            if row.get("verification_status") != "structured_source_verified" or not row.get("training_enabled"):
                raise SystemExit("structured semantic row is not source-verified/training-enabled")
            encoded = json.dumps(row, sort_keys=True, ensure_ascii=False).casefold()
            if any(token in encoded for token in ("mask_path", "semantic_t1_path", "semantic_t2_path", "label_path", "dense_path")):
                raise SystemExit(f"mask/dense path leaked into structured semantic row {row.get('query_id')}")
            group_ids.add(str(row.get("semantic_group_id")))
            rows.append(row)
    group_registry = read_jsonl(structured_dir / "semantic_group_registry_structured.jsonl")
    valid_groups = {
        str(group.get("semantic_group_id"))
        for group in group_registry
        if int(group.get("pair_count", 0) or 0) >= 2
    }
    if not rows or not group_ids or not valid_groups.intersection(group_ids):
        raise SystemExit("structured semantic view lacks non-empty multi-positive groups")
    return rows, {"audit": audit, "loader": loader, "group_count": len(group_ids)}


def normalize_rscc_pair(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["source_dataset"] = "RSCC-EBD"
    result["schema_version"] = "qcpr-stage2-pair-registry-v2"
    result["is_synthetic"] = False
    result["parent_pair_id"] = None
    result["source_scene_group_id"] = str(row["source_scene_group_id"])
    result["source_event_id"] = str(row.get("source_event_id") or row["source_scene_group_id"])
    result["image_view"] = "NATIVE"
    return result


def rscc_dense_rows(rscc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rscc_rows:
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
    output = []
    for row in rows:
        output.append({
            "caption_id": row["query_id"],
            "canonical_pair_id": row["canonical_pair_id"],
            "dataset_name": "s2looking",
            "text": row["text"],
            "normalized_text": row["normalized_text"],
            "caption_source": row["caption_source"],
            "task_type": "semantic_retrieval",
            "query_scope": "semantic_group",
            "semantic_group_id": row["semantic_group_id"],
            "equivalent_caption_group_id": None,
            "quality_score": 1.0,
            "identifiability_score": 0.25,
            "verification_status": row["verification_status"],
            "is_generated": True,
            "generator": row["generator"],
            "language": "en",
            "change_status": "changed",
        })
    return output


def structured_relevance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "caption_id": row["query_id"],
            "semantic_group_id": row["semantic_group_id"],
            "positive_pair_ids": [],
            "ignored_pair_ids": [],
            "valid_negative_policy": "compact_semantic_group_registry",
        }
        for row in rows
    ]


def rscc_dense_evaluation_rows(rscc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "canonical_pair_id": row["canonical_pair_id"],
            "dataset_name": "rscc_ebd",
            "source_event_id": row.get("source_event_id"),
            "source_scene_group_id": row.get("source_scene_group_id"),
            "split": row["split"],
            "t1_path": row["t1_path"],
            "t2_path": row["t2_path"],
            "dense_label_ids": [f"{row['canonical_pair_id']}:rscc_dense:pre", f"{row['canonical_pair_id']}:rscc_dense:post"],
            "mask_access": "dense_label_registry_only",
            "evaluation_only": True,
        }
        for row in rscc_rows
    ]


def hash_tree(root: Path, *, excluded_prefixes: tuple[str, ...] = ("hashes/",)) -> dict[str, str]:
    values = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = str(path.relative_to(root))
        if any(relative.startswith(prefix) for prefix in excluded_prefixes):
            continue
        values[relative] = sha256(path)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--rscc-pairs", type=Path, required=True)
    parser.add_argument("--rscc-qvq", type=Path, required=True)
    parser.add_argument("--rscc-qvq-audit", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--stage2-audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite release root: {args.output_root}")
    required_core = [args.core_root / "registries" / name for name in ("pair_registry.jsonl", "caption_registry.jsonl", "relevance_registry.jsonl", "dense_label_registry.jsonl")]
    if any(not path.is_file() for path in required_core):
        raise SystemExit("core release is incomplete")
    rscc_rows = [normalize_rscc_pair(row) for row in read_jsonl(args.rscc_pairs)]
    rscc_audit = validate_rscc(rscc_rows)
    structured_rows, structured_info = validate_structured_view(args.structured_dir)
    qvq_audit = read_json(args.rscc_qvq_audit, {})
    if qvq_audit.get("training_enabled") or qvq_audit.get("human_audit_passed"):
        raise SystemExit("unverified RSCC QvQ audit cannot be marked training-enabled")

    shutil.copytree(args.core_root, args.output_root)
    out = args.output_root
    registries = out / "registries"
    manifests = out / "manifests"
    reports = out / "reports"
    for directory in (registries, manifests, reports, out / "hashes"):
        directory.mkdir(parents=True, exist_ok=True)

    core_pair_rows = read_jsonl(registries / "pair_registry.jsonl")
    core_ids = {str(row["canonical_pair_id"]) for row in core_pair_rows}
    rscc_ids = {str(row["canonical_pair_id"]) for row in rscc_rows}
    if core_ids & rscc_ids:
        raise SystemExit("RSCC/core canonical pair collision")
    write_jsonl(registries / "pair_registry.jsonl", core_pair_rows + rscc_rows)
    append_jsonl(registries / "dense_label_registry.jsonl", rscc_dense_rows(rscc_rows))
    append_jsonl(registries / "caption_registry.jsonl", structured_caption_rows(structured_rows))
    append_jsonl(registries / "relevance_registry.jsonl", structured_relevance_rows(structured_rows))
    write_jsonl(registries / "semantic_group_registry_structured.jsonl", read_jsonl(args.structured_dir / "semantic_group_registry_structured.jsonl"))

    for split in SEMANTIC_SPLITS:
        target = manifests / f"retrieval_semantic_{split}_v2.jsonl"
        append_jsonl(target, [row for row in structured_rows if row["split"] == split])
    write_jsonl(manifests / "physical_pairs_rscc_ebd.jsonl", [
        {
            "canonical_pair_id": row["canonical_pair_id"],
            "source_dataset": "RSCC-EBD",
            "source_event_id": row.get("source_event_id"),
            "source_scene_group_id": row.get("source_scene_group_id"),
            "split": row["split"],
            "t1_path": row["t1_path"],
            "t2_path": row["t2_path"],
            "dense_label_ids": [f"{row['canonical_pair_id']}:rscc_dense:pre", f"{row['canonical_pair_id']}:rscc_dense:post"],
            "caption_supervision": "generated_unverified_separate_view",
            "training_enabled": False,
        }
        for row in rscc_rows
    ])
    write_jsonl(manifests / "dense_evaluation_rscc_ebd.jsonl", rscc_dense_evaluation_rows(rscc_rows))
    shutil.copy2(args.rscc_qvq, manifests / "semantic_candidates_unverified_rscc_qvq.jsonl")
    write_json(reports / "stage2_source_registry.json", read_json(args.stage2_audit_root / "source_registry.json", {}))
    for name in ("stage2_gate_summary.json", "stage2_progress_report.json", "p2_bounded_training_plan.json"):
        source = args.stage2_audit_root / name
        if source.is_file():
            shutil.copy2(source, reports / name)
    structured_report_dir = reports / "structured_semantic"
    structured_report_dir.mkdir(parents=True, exist_ok=True)
    for name in ("structured_semantic_audit.json", "structured_semantic_loader_contract.json"):
        shutil.copy2(args.structured_dir / name, structured_report_dir / name)
    qvq_report = dict(qvq_audit)
    qvq_report["training_enabled"] = False
    qvq_report["release_role"] = "unverified_candidate_view_only"
    write_json(reports / "rscc_qvq_caption_audit.json", qvq_report)

    core_counts = {name: jsonl_count(registries / name) for name in ("pair_registry.jsonl", "caption_registry.jsonl", "relevance_registry.jsonl", "dense_label_registry.jsonl")}
    manifest_counts = {path.name: jsonl_count(path) for path in sorted(manifests.glob("*.jsonl"))}
    blockers = [
        "generated RSCC QvQ/detail candidates require independent temporal verification and stratified human audit",
        "Synthetic RCD real-A mapping remains unavailable; synthetic-A is diagnostic only",
        "SYSU-CD official image archive is not integrated",
        "Hi-UCD corrected archive is not integrated",
        "cluster GitHub SSH key is not authorized for remote publication",
    ]
    release = {
        "schema_version": "qcpr-dataset-v2-stage2-candidate-release-v1",
        "release_name": "QCPR Dataset-v2 Stage-2 candidate",
        "status": "DATA_QUALITY_HOLD",
        "stage2_ready": False,
        "training_authorized": False,
        "training_launched": False,
        "code_sha": git_sha(args.repo),
        "core_release_preserved": str(args.core_root),
        "historical_r1_job_200097_touched": False,
        "new_real_physical_source": {"source": "RSCC-EBD", **rscc_audit},
        "structured_semantic": {
            "source": "S2Looking",
            "rows": len(structured_rows),
            "split_counts": dict(collections.Counter(row["split"] for row in structured_rows)),
            "groups": structured_info["group_count"],
            "training_enabled": True,
            "human_verified": False,
            "loader_passed": bool(structured_info["loader"].get("passed")),
        },
        "unverified_views": {"rscc_qvq_rows": qvq_audit.get("caption_count", 0), "training_enabled": False},
        "counts": {"registries": core_counts, "manifests": manifest_counts},
        "blockers": blockers,
    }
    write_json(out / "dataset_v2_stage2_release.json", release)
    write_json(out / "dataset_v2_stage2_source_coverage.json", read_json(args.stage2_audit_root / "source_registry.json", {}))
    write_json(out / "dataset_v2_stage2_build_summary.json", release)
    write_json(out / "dataset_v2_stage2_leakage_audit.json", {
        "passed": True,
        "core_release_untouched": True,
        "cross_split_pair_id_collisions": 0,
        "rscc_pair_id_collisions_with_core": 0,
        "rscc_event_groups": rscc_audit["event_group_count"],
        "rscc_split_counts": rscc_audit["split_counts"],
        "notes": ["Physical RSCC source registry was already identity/decode/pilot validated before assembly."],
    })
    write_json(out / "dataset_v2_stage2_relevance_audit.json", {
        "passed": True,
        "exact_view_unchanged_from_core": True,
        "structured_semantic_view": "compact same-split group registry; no materialized ignored lists",
        "structured_semantic_rows": len(structured_rows),
        "unverified_qvq_excluded_from_training": True,
    })
    (reports / "README.md").write_text("DATA_QUALITY_HOLD: Stage-2 candidate contains new RSCC physical pairs and source-verified coarse S2Looking semantics, but is not training-authorized.\n", encoding="utf-8")

    hashes = hash_tree(out)
    write_json(out / "hashes" / "stage2_release_artifacts_sha256.json", hashes)
    write_json(out / "hashes" / "registry_sha256.json", {str(path.relative_to(out)): sha256(path) for path in sorted(registries.glob("*.jsonl"))})
    write_json(out / "hashes" / "manifest_sha256.json", {str(path.relative_to(out)): sha256(path) for path in sorted(manifests.glob("*.jsonl"))})
    release["release_artifact_hash_count"] = len(hashes)
    release["release_artifact_hashes_path"] = "hashes/stage2_release_artifacts_sha256.json"
    write_json(out / "dataset_v2_stage2_release.json", release)
    write_json(out / "dataset_v2_stage2_build_summary.json", release)
    print(json.dumps({
        "status": release["status"],
        "output_root": str(out),
        "code_sha": release["code_sha"],
        "physical_pairs": len(core_pair_rows) + len(rscc_rows),
        "structured_semantic_rows": len(structured_rows),
        "rscc_pairs": len(rscc_rows),
        "training_authorized": release["training_authorized"],
        "artifact_hash_count": len(hashes),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
