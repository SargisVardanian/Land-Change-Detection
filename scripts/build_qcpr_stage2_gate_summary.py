#!/usr/bin/env python3
"""Build the Stage-2 readiness gate without promoting unreviewed semantics."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SPLITS = ("train", "development", "test")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def review_summary(path: Path) -> dict[str, Any]:
    audit = read_json(path / "review_package_audit.json", {})
    packet = read_jsonl(path / "human_review_packet.jsonl")
    events = collections.Counter(str(row.get("source_event_id") or "") for row in packet)
    return {"status": audit.get("status"), "rows": len(packet), "events": len(events), "rows_per_event": sorted(set(events.values())) if events else [], "training_enabled": bool(audit.get("training_enabled")), "audit_sha256": sha256(path / "review_package_audit.json")}


def gold_summary(path: Path) -> dict[str, Any]:
    audit = read_json(path / "gold_semantic_audit.json", {})
    rows = [row for split in SPLITS for row in read_jsonl(path / f"retrieval_semantic_gold_{split}.jsonl")]
    groups = read_jsonl(path / "semantic_group_registry.jsonl")
    return {"status": audit.get("status"), "rows": len(rows), "groups": len(groups), "training_enabled": bool(audit.get("training_enabled")), "event_ids_used_for_semantics": audit.get("event_ids_used_for_semantics"), "positive_grades_enabled": audit.get("positive_grades_enabled"), "audit_sha256": sha256(path / "gold_semantic_audit.json")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--audit-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--review-package", type=Path, required=True)
    ap.add_argument("--gold-semantic-dir", type=Path, required=True)
    ap.add_argument("--test-result", type=Path, default=None)
    args = ap.parse_args()
    audit = args.audit_root
    registry = read_json(audit / "source_registry.json", {})
    semantic = read_json(audit / "semantic_view/semantic_view_audit.json", {})
    arch = read_json(audit / "architecture_screening_plan.json", {})
    arch_actual = read_json(audit / "architecture_screening/architecture_screening_frozen_report.json", {})
    rscc = read_json(audit / "rscc_ebd/rscc_ebd_pair_audit.json", {})
    rscc_loader = read_json(audit / "rscc_ebd/rscc_mask_free_loader_contract.json", {})
    qvq = read_json(audit / "rscc_ebd/rscc_ebd_qvq_caption_audit.json", {})
    rcd = read_json(audit / "synthetic_rcd/synthetic_rcd_mapping_audit.json", {})
    structured_candidates = sorted((audit / "semantic_view").glob("structured_source_verified_s2looking_*/structured_semantic_audit.json"), key=lambda path: path.stat().st_mtime)
    structured_path = structured_candidates[-1] if structured_candidates else audit / "semantic_view/structured_semantic_audit.json"
    structured = read_json(structured_path, {})
    structured_loader_path = structured_path.parent / "structured_semantic_loader_contract.json"
    structured_loader = read_json(structured_loader_path, {})
    structured_independent_path = structured_path.parent / "independent_structured_semantic_verification.json"
    structured_independent = read_json(structured_independent_path, {})
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.repo, check=True, text=True, capture_output=True).stdout.strip()
    clean = not subprocess.run(["git", "status", "--porcelain"], cwd=args.repo, check=True, text=True, capture_output=True).stdout.splitlines()
    review = review_summary(args.review_package)
    gold = gold_summary(args.gold_semantic_dir)
    rscc_valid = bool(rscc.get("identity_proven") and rscc_loader.get("passed") and int(rscc.get("pair_count", 0) or 0) > 0 and len(rscc.get("split_counts", {})) == 3)
    structured_ready = bool(structured.get("status") == "STRUCTURED_SOURCE_SEMANTIC_READY" and structured_loader.get("passed") and structured_independent.get("status") == "INDEPENDENT_STRUCTURED_SEMANTIC_VERIFIED" and structured_independent.get("mask_free_output") and int(structured_independent.get("verified_rows", 0) or 0) > 0)
    gold_ready = gold.get("status") == "GOLD_SEMANTIC_READY" and gold.get("training_enabled") and int(gold.get("rows", 0)) > 0 and gold.get("event_ids_used_for_semantics") is False and set(gold.get("positive_grades_enabled") or []) == {2, 3}

    test_path = args.test_result or audit / "test_suite/status"
    test_payload = read_json(args.test_result, {}) if args.test_result else {}
    if args.test_result:
        tests_passed = bool(test_payload.get("passed")) and str(test_payload.get("code_sha")) == head
        test_summary = test_payload.get("summary")
    else:
        status_text = test_path.read_text().strip() if test_path.is_file() else "missing"
        log_path = audit / "test_suite/pytest.log"
        log = log_path.read_text(errors="replace") if log_path.is_file() else ""
        match = re.search(r"(?m)^(\d+) passed(?:, (\d+) skipped)?(?:, \d+ warnings)? in ", log)
        tests_passed = status_text == "0" and bool(match)
        test_summary = f"{match.group(1)} passed, {match.group(2) or 0} skipped" if match else None

    deferred_access = [
        {"source_dataset": "SYSU-CD", "status": "ACCESS_BLOCKED", "blocks_first_rscc_p2": False, "blocker": "official archive requires Baidu verification/SharePoint authorization"},
        {"source_dataset": "Hi-UCD", "status": "ACCESS_BLOCKED", "blocks_first_rscc_p2": False, "blocker": "corrected official archive and 2025-11-01 patch are request-gated"},
        {"source_dataset": "Synthetic RCD SECOND real-A", "status": "MAPPING_BLOCKED", "blocks_first_rscc_p2": False, "blocker": "original SECOND-A assets unavailable; synthetic-A diagnostic only"},
    ]
    blockers: list[str] = []
    if not rscc_valid:
        blockers.append("RSCC-EBD physical source is not fully loader-validated")
    if not gold_ready:
        blockers.extend(["two independent human reviews and adjudication are incomplete", "primary gold semantic train/development/test manifests are not training-enabled"])
    if arch.get("status") != "SCREENING_COMPLETE" or not str(arch_actual.get("status", "")).startswith("SCREENING_COMPLETE"):
        blockers.append("frozen B0/B1/B2 architecture screening is incomplete")
    if not structured_ready:
        blockers.append("independent structured S2Looking verification/loader gate is incomplete")
    if not tests_passed:
        blockers.append("full Stage-2 test result is not proven for the current code SHA")
    if not clean:
        blockers.append("Stage-2 worktree is not clean")
    stage2_ready = not blockers
    payload = {
        "schema_version": "qcpr-stage2-gate-summary-v2",
        "status": "DATASET_V2_STAGE2_READY" if stage2_ready else "DATA_QUALITY_HOLD",
        "stage2_ready": stage2_ready,
        "code_sha": head,
        "worktree_clean": clean,
        "new_real_physical_source_present": bool(registry.get("new_real_physical_source_present")),
        "architecture": {"selected_anchor": "B1_framewise_gated_difference", "status": arch.get("status"), "actual_status": arch_actual.get("status"), "broad_search_frozen": True},
        "rscc_ebd": {"pairs": rscc.get("pair_count"), "events": rscc.get("event_count"), "split_counts": rscc.get("split_counts"), "identity_proven": rscc.get("identity_proven"), "loader_passed": rscc_loader.get("passed"), "artifact_sha256": sha256(audit / "rscc_ebd/rscc_ebd_pair_audit.json"), "loader_sha256": sha256(audit / "rscc_ebd/rscc_mask_free_loader_contract.json")},
        "semantic": {"provisional_rows": int(semantic.get("multi_positive_rows", 0) or 0), "provisional_training_enabled_rows": int(semantic.get("training_enabled_rows", 0) or 0), "provisional_groups": int(semantic.get("groups", 0) or 0), "review_required_rows": int(semantic.get("review_required_rows", 0) or 0), "review_package": review, "gold": gold, "event_ids_used_for_semantics": False, "qvq_training_enabled": bool(qvq.get("training_enabled"))},
        "structured_s2looking": {"rows": int(structured.get("output_row_count", 0) or 0), "groups": int(structured.get("group_count", 0) or 0), "independent_verified": structured_ready, "training_enabled_primary": False, "loader_passed": bool(structured_loader.get("passed")), "independent_rows": int(structured_independent.get("verified_rows", 0) or 0), "audit_sha256": sha256(structured_path), "independent_sha256": sha256(structured_independent_path)},
        "deferred_access_blockers": deferred_access,
        "synthetic_rcd": {"mapping_coverage": rcd.get("mapping_coverage"), "status": "MAPPING_BLOCKED", "primary_real_a_mode_enabled": False},
        "tests": {"passed": tests_passed, "summary": test_summary, "code_sha_checked": bool(args.test_result), "artifact_sha256": sha256(args.test_result) if args.test_result else sha256(audit / "test_suite/pytest.log")},
        "training_submitted": False,
        "blockers": blockers,
        "p2_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = ["# QCPR Stage-2 readiness gate", "", f"- Status: {payload['status']}", f"- Code SHA: {head}", f"- Worktree clean: {clean}", f"- Selected anchor: B1_framewise_gated_difference", f"- RSCC-EBD: {rscc.get('pair_count')} pairs; loader={rscc_loader.get('passed')}", f"- Provisional semantic rows: {semantic.get('multi_positive_rows', 0)}; review packet: {review.get('rows', 0)} rows / {review.get('events', 0)} events; gold rows: {gold.get('rows', 0)}", f"- Tests: {test_summary or 'not proven'}", "", "## Active blockers", ""]
    markdown.extend(f"- {item}" for item in blockers)
    markdown.extend(["", "## Deferred access blockers", ""])
    markdown.extend(f"- {item['source_dataset']}: {item['status']} — {item['blocker']}" for item in deferred_access)
    markdown.extend(["", "P2 was not submitted."])
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": payload["status"], "blocker_count": len(blockers), "gold_rows": gold.get("rows", 0)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
