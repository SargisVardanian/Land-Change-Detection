#!/usr/bin/env python3
"""Build a conservative Stage-2 gate summary from immutable audit artifacts."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
from typing import Any

def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--audit-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args=ap.parse_args()
    audit=args.audit_root
    registry=read_json(audit/"source_registry.json", {})
    semantic=read_json(audit/"semantic_view/semantic_view_audit.json", {})
    automated=read_json(audit/"semantic_view/automated_verified_pilot/automated_verified_semantic_pilot_audit.json", {})
    arch=read_json(audit/"architecture_screening_plan.json", {})
    arch_actual=read_json(audit/"architecture_screening/architecture_screening_frozen_report.json", {})
    manual=read_json(audit/"semantic_view/manual_visual_verified_pilot/manual_visual_semantic_pilot_audit.json", {})
    rscc=read_json(audit/"rscc_ebd/rscc_ebd_pair_audit.json", {})
    rscc_loader=read_json(audit/"rscc_ebd/rscc_mask_free_loader_contract.json", {})
    qvq=read_json(audit/"rscc_ebd/rscc_ebd_qvq_caption_audit.json", {})
    rcd=read_json(audit/"synthetic_rcd/synthetic_rcd_mapping_audit.json", {})
    test_status=(audit/"test_suite/status").read_text().strip() if (audit/"test_suite/status").is_file() else "missing"
    test_log=(audit/"test_suite/pytest.log").read_text(errors="replace") if (audit/"test_suite/pytest.log").is_file() else ""
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=args.repo,check=True,text=True,capture_output=True).stdout.strip()
    status=subprocess.run(["git","status","--porcelain"],cwd=args.repo,check=True,text=True,capture_output=True).stdout.splitlines()
    rscc_valid=bool(rscc.get("identity_proven") and rscc_loader.get("passed") and rscc.get("pair_count",0)>0 and len(rscc.get("split_counts",{}))==3)
    semantic_nonempty=all(int(semantic.get("counts",{}).get(s,0))>0 for s in ("train","development","test"))
    semantic_verified=bool(semantic.get("human_audit_rows",0)>0)
    tests_passed=test_status=="0" and "563 passed, 3 skipped" in test_log
    blockers=[]
    if not rscc_valid: blockers.append("RSCC EBD physical pilot is not fully loader-validated")
    if not semantic_nonempty: blockers.append("semantic train/development/test manifests are not all non-empty")
    if not semantic_verified: blockers.append("semantic supervision is provisional: generated QvQ and mask/event-derived groups require independent verifier and stratified human audit")
    if arch.get("status")!="SCREENING_COMPLETE": blockers.append("frozen architecture screening is incomplete; B1/B2 implementation and B3/B4 weight/preprocessing audits remain")
    if rcd.get("mapping_coverage",0) != 1.0: blockers.append("Synthetic RCD real-A mapping remains unavailable; synthetic-A is diagnostic only")
    for name, label in (("SYSU-CD","SYSU-CD official image archive"),("Hi-UCD","Hi-UCD corrected archive")):
        row=next((x for x in registry.get("sources",[]) if x.get("source_dataset")==name), {})
        if row.get("state") not in {"LOADER_VALIDATED","TRAINING_ENABLED"}:
            blockers.append(f"{label} is not integrated")
    payload={
        "schema_version":"qcpr-stage2-gate-summary-v1",
        "status":"DATA_QUALITY_HOLD",
        "stage2_ready":False,
        "code_sha":head,
        "worktree_clean":not status,
        "uncommitted_paths":status,
        "source_registry_status":registry.get("status"),
        "new_real_physical_source_present":bool(registry.get("new_real_physical_source_present")),
        "rscc_ebd":{"pairs":rscc.get("pair_count"),"pilot_pairs":rscc.get("pilot_pair_count"),"events":rscc.get("event_count"),"split_counts":rscc.get("split_counts"),"identity_proven":rscc.get("identity_proven"),"decoded_pilot_passed":rscc.get("decoded_pilot_passed"),"loader_passed":rscc_loader.get("passed"),"loader_artifact_sha256":sha256(audit/"rscc_ebd/rscc_mask_free_loader_contract.json"),"artifact_sha256":sha256(audit/"rscc_ebd/rscc_ebd_pair_audit.json")},
        "rscc_qvq":{"annotation_rows":qvq.get("annotation_rows"),"mapped_pairs":qvq.get("mapped_pairs"),"caption_count":qvq.get("caption_count"),"training_enabled":qvq.get("training_enabled"),"status":qvq.get("status"),"artifact_sha256":sha256(audit/"rscc_ebd/rscc_ebd_qvq_caption_audit.json")},
        "semantic":{"counts":semantic.get("counts"),"groups":semantic.get("groups"),"structured_source_rows":semantic.get("structured_source_rows"),"human_audit_rows":semantic.get("human_audit_rows"),"manual_visual_verified_rows":manual.get("verified_semantic_rows",0),"manual_visual_status":manual.get("status"),"review_required_rows":semantic.get("review_required_rows"),"automated_verified_rows":automated.get("row_count",0),"automated_verified_split_counts":automated.get("split_counts",{}),"automated_verification_status":automated.get("status"),"status":semantic.get("status"),"gate":semantic.get("stage2_gate"),"train_sha256":sha256(audit/"semantic_view/retrieval_semantic_train_v2.jsonl"),"development_sha256":sha256(audit/"semantic_view/retrieval_semantic_development_v2.jsonl"),"test_sha256":sha256(audit/"semantic_view/retrieval_semantic_test_v2.jsonl")},
        "synthetic_rcd":{"mapping_coverage":rcd.get("mapping_coverage"),"mode":rcd.get("mode"),"status":rcd.get("status")},
        "architecture_screening":{"status":arch.get("status"),"plan_sha256":sha256(audit/"architecture_screening_plan.json"),"actual_status":arch_actual.get("status"),"actual_report_sha256":sha256(audit/"architecture_screening/architecture_screening_frozen_report.json")},
        "tests":{"status_file":test_status,"full_suite_passed":tests_passed,"summary":"563 passed, 3 skipped" if tests_passed else None,"log_sha256":sha256(audit/"test_suite/pytest.log")},
        "training_submitted":False,
        "blockers":blockers,
        "artifact_hashes":{str(p.relative_to(audit)):sha256(p) for p in [audit/"source_registry.json",audit/"semantic_view/semantic_view_audit.json",audit/"semantic_view/automated_verified_pilot/automated_verified_semantic_pilot_audit.json",audit/"semantic_view/manual_visual_verified_pilot/manual_visual_semantic_pilot_audit.json",audit/"architecture_screening_plan.json",audit/"architecture_screening/architecture_screening_frozen_report.json"] if p.is_file()},
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    md=["# QCPR Stage-2 gate summary","",f"- Status: {payload['status']}",f"- Code SHA: {head}",f"- Worktree clean: {payload['worktree_clean']}",f"- New real physical source present: {payload['new_real_physical_source_present']}",f"- RSCC EBD: {rscc.get('pair_count')} pairs; project loader {rscc_loader.get('passed')}",f"- Semantic view: {semantic.get('counts')}; structured-source rows {semantic.get('structured_source_rows')}; human-audited rows {semantic.get('human_audit_rows')}",f"- Full suite: {payload['tests']['summary'] or 'not proven'}","", "## Blockers",""]
    md += [f"- {x}" for x in blockers]
    md += ["", "No model training or P2 submission was launched."]
    (args.output.with_suffix(".md")).write_text("\n".join(md)+"\n")
    print(json.dumps({"output":str(args.output),"status":payload["status"],"blocker_count":len(blockers)},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
