#!/usr/bin/env python3
"""Package an immutable QCPR retrieval-semantic repair release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


REQUIRED_MANIFESTS = [
    f"{scope}_{split}.jsonl"
    for scope in ("exact", "semantic", "localized", "long_series", "direction", "stable")
    for split in ("train", "development", "test")
]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_relevance_graph(path: Path, query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Materialize one deterministic positive edge per query/item relation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    edge_count = 0
    with path.open("wb") as handle:
        ordered_queries = sorted(query_rows, key=lambda row: str(row.get("query_id")))
        for query in ordered_queries:
            query_id = str(query.get("query_id") or "")
            query_scope = str(query.get("query_scope") or "")
            provenance = query.get("provenance") if isinstance(query.get("provenance"), dict) else {}
            grades = query.get("graded_relevance") if isinstance(query.get("graded_relevance"), dict) else {}
            positive_ids = sorted({str(item_id) for item_id in query.get("positive_item_ids") or []})
            for item_id in positive_ids:
                edge = {
                    "schema_version": "qcpr-relevance-graph-edge-v1",
                    "query_id": query_id,
                    "item_id": item_id,
                    "source_item_id": query.get("source_item_id"),
                    "relevance_grade": int(grades[item_id]),
                    "query_scope": query_scope,
                    "purpose": query.get("purpose"),
                    "split": query.get("split"),
                    "verification": query.get("verification"),
                    "training_enabled": bool(query.get("training_enabled")),
                    "diagnostic_only": query_scope == "generic_no_change",
                    "semantic_group_id": provenance.get("semantic_group_id"),
                }
                payload = (json.dumps(edge, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                handle.write(payload)
                digest.update(payload)
                edge_count += 1
    return {
        "schema_version": "qcpr-relevance-graph-edge-v1",
        "path": str(path),
        "edge_count": edge_count,
        "sha256": digest.hexdigest(),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def merge_tree(source: Path, target: Path) -> None:
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            copy_file(path, destination)


def write_checksums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        checksums[str(path.relative_to(root))] = sha256(path)
    text = "".join(f"{digest}  {relative}\n" for relative, digest in sorted(checksums.items()))
    (root / "SHA256SUMS").write_text(text, encoding="utf-8")
    return checksums


def verify_checksums(root: Path) -> dict[str, Any]:
    failures = []
    rows = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for row in rows:
        if "  " not in row:
            failures.append({"row": row, "reason": "invalid_checksum_row"})
            continue
        expected, relative = row.split("  ", 1)
        path = root / relative
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual, "reason": "mismatch"})
    return {"entries": len(rows), "failures": failures, "passed": not failures}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-release", type=Path, required=True)
    parser.add_argument("--repair-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-branch", default="codex/qcpr-dataset-v2-final")
    parser.add_argument("--release-name", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing release: {args.output}")
    if not args.base_release.is_dir() or not args.repair_output.is_dir():
        raise SystemExit("base release and repair output must be directories")
    shutil.copytree(args.base_release, args.output)

    # Candidate manifests and registries replace only the retrieval views;
    # physical base assets remain byte-for-byte copied from the prior release.
    merge_tree(args.repair_output / "manifests", args.output / "manifests")
    copy_file(args.repair_output / "registries/repaired_queries.jsonl", args.output / "registries/queries.jsonl")
    copy_file(args.repair_output / "registries/query_purpose_registry.jsonl", args.output / "registries/query_purpose_registry.jsonl")
    copy_file(args.repair_output / "registries/semantic_groups.jsonl", args.output / "registries/semantic_groups.jsonl")
    copy_file(args.repair_output / "registries/generic_no_change_diagnostic.jsonl", args.output / "registries/generic_no_change_diagnostic.jsonl")
    merge_tree(args.repair_output / "evaluation_sidecars", args.output / "evaluation_sidecars")
    merge_tree(args.repair_output / "audits", args.output / "audits/retrieval_semantic_repair")
    merge_tree(args.repair_output / "source_reports", args.output / "source_reports/retrieval_semantic_repair")
    for packet in sorted((args.repair_output / "source_reports/review_samples").glob("*.jsonl")):
        copy_file(packet, args.output / "source_reports/human_review_packets" / packet.name)
    merge_tree(args.repair_output / "handoff", args.output / "handoff/retrieval_semantic_repair")

    # Forest physical assets are canonicalized by the repair script and are
    # added only to the new release.  The old immutable release is untouched.
    forest_items = read_jsonl(args.repair_output / "registries/forest_physical_items.jsonl")
    existing_items = read_jsonl(args.output / "registries/physical_items.jsonl")
    existing_ids = {str(row.get("item_id")) for row in existing_items}
    existing_items.extend(row for row in forest_items if str(row.get("item_id")) not in existing_ids)
    existing_items.sort(key=lambda row: str(row.get("item_id")))
    with (args.output / "registries/physical_items.jsonl").open("w", encoding="utf-8") as handle:
        for row in existing_items:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    existing_frames = read_jsonl(args.output / "registries/frames.jsonl")
    existing_frame_ids = {str(row.get("frame_id")) for row in existing_frames}
    forest_frames = read_jsonl(args.repair_output / "registries/forest_frames.jsonl")
    existing_frames.extend(row for row in forest_frames if str(row.get("frame_id")) not in existing_frame_ids)
    existing_frames.sort(key=lambda row: str(row.get("frame_id")))
    with (args.output / "registries/frames.jsonl").open("w", encoding="utf-8") as handle:
        for row in existing_frames:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    repair_package = read_json(args.repair_output / "REPAIR_PACKAGE.json", {})
    handoff = read_json(args.repair_output / "handoff/model_agent_handoff.json", {})
    view_readiness = read_json(args.repair_output / "audits/view_readiness.json", {})
    data_only_comparison = read_json(args.repair_output / "audits/data_only_D0_D3_comparison.json", {})
    if data_only_comparison.get("status"):
        repair_package = dict(repair_package)
        repair_package["frozen_model_comparison_status"] = data_only_comparison["status"]
    physical_count = len(existing_items)
    frame_count = sum(len(row.get("frames", [])) for row in existing_items)
    query_rows = read_jsonl(args.output / "registries/queries.jsonl")
    relevance_graph = write_relevance_graph(args.output / "registries/relevance_graph.jsonl", query_rows)
    release = {
        "schema_version": "qcpr-dataset-v2-retrieval-semantic-repair-v1",
        "release_name": args.release_name,
        "release_state": "DATA_QUALITY_HOLD",
        "branch": args.source_branch,
        "code_sha": git_value(args.repo, "rev-parse", "HEAD"),
        "source_release": str(args.base_release),
        "source_release_sha256s": sha256(args.base_release / "SHA256SUMS"),
        "physical_item_count": physical_count,
        "frame_count": frame_count,
        "query_count": len(query_rows),
        "relevance_graph": relevance_graph,
        "required_manifest_names": REQUIRED_MANIFESTS,
        "manifest_counts": {name: len(read_jsonl(args.output / "manifests" / name)) for name in REQUIRED_MANIFESTS},
        "manifest_hashes_before_release_metadata": {name: sha256(args.output / "manifests" / name) for name in REQUIRED_MANIFESTS},
        "view_status": view_readiness.get("views", {}),
        "training_authorized": False,
        "training_submitted": False,
        "main_training_allowed": False,
        "p1_submitted": False,
        "p2_submitted": False,
        "all_query_training_enabled": all(not bool(row.get("training_enabled")) for row in query_rows),
        "repair_package": repair_package,
        "data_only_comparison": data_only_comparison,
        "model_agent_handoff": handoff,
        "source_integration": {
            "forest": "PHYSICAL_SCENE_DISJOINT_CANDIDATE_TEXT_HOLD",
            "tamm": "PHYSICAL_489_SEQUENCE_TEXT_HOLD",
            "rscc": "VERIFIED_TEXT_ZERO_HOLD",
            "dubai_cc": "NOT_INTEGRATED",
            "rsrcc": "NOT_INTEGRATED",
            "dynamic_earth_net": "NOT_INTEGRATED",
            "spacenet_7": "NOT_INTEGRATED",
            "terra_cd": "NOT_INTEGRATED",
        },
    }
    write_json(args.output / "RELEASE.json", release)
    checksums = write_checksums(args.output)
    integrity = verify_checksums(args.output)
    write_json(args.output / "audits/retrieval_semantic_repair_integrity.json", {
        "schema_version": "qcpr-retrieval-semantic-repair-integrity-v1",
        "release": str(args.output),
        "physical_item_count": physical_count,
        "frame_count": frame_count,
        "query_count": len(query_rows),
        "relevance_graph": relevance_graph,
        "manifest_count": len(REQUIRED_MANIFESTS),
        "manifest_nonempty": {name: len(read_jsonl(args.output / "manifests" / name)) > 0 for name in REQUIRED_MANIFESTS},
        "checksum_entry_count_before_integrity_audit": len(checksums),
        "checksum_verification_before_integrity_audit": integrity,
        "forest_physical_added": len(forest_items),
        "all_query_training_enabled_false": all(not bool(row.get("training_enabled")) for row in query_rows),
        "exact_gate_passed": False,
        "stable_gate_passed": False,
        "status": "HOLD_REVIEW_AND_MODEL_COMPARISON_REQUIRED" if integrity["passed"] else "HOLD_CHECKSUM_FAILURE",
    })
    # The integrity audit is part of the immutable release, so regenerate the
    # checksum file once after writing it and verify the final set.
    write_checksums(args.output)
    final_integrity = verify_checksums(args.output)
    if not final_integrity["passed"]:
        raise SystemExit(json.dumps(final_integrity, sort_keys=True))
    print(json.dumps({"release": str(args.output), "physical_items": physical_count, "frames": frame_count, "queries": len(query_rows), "checksum_entries": final_integrity["entries"], "status": "DATA_QUALITY_HOLD"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
