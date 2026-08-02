#!/usr/bin/env python3
"""Build a compact automated-verified semantic candidate view from a frozen pilot."""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path
from typing import Any

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()
    accepted=[json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    accepted.sort(key=lambda r:(str(r.get("split")),str(r.get("canonical_pair_id"))))
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index,src in enumerate(accepted):
        event=str(src.get("source_event_id") or "unknown")
        pair=str(src["canonical_pair_id"])
        text=str((src.get("captions") or [""])[0]).strip()
        rows.append({
            "schema_version":"qcpr-stage2-semantic-candidate-v1",
            "query_id":f"{pair}:automated_verified:{index:05d}",
            "canonical_pair_id":pair,
            "text":text,
            "normalized_text":" ".join(text.casefold().split()),
            "split":str(src["split"]),
            "source_dataset":"RSCC-EBD",
            "query_scope":"semantic_group",
            "semantic_group_id":f"rscc_ebd:event:{event}",
            "positive_pair_ids":[pair],
            "graded_relevance_rule":"same_event=1; exact_pair=3 only after human audit",
            "semantic_candidate_count":None,
            "caption_source":"RSCC-QvQ",
            "is_generated":True,
            "verification_status":"automated_frozen_siglip2_verified",
            "verification_method":"SigLIP2 image-text consistency with human LEVIR/SECOND calibration",
            "verification_score":src.get("verification_score"),
            "verification_shuffled_score":src.get("verification_shuffled_score"),
            "training_enabled":False,
            "human_audit_status":"required",
            "t1_path":src.get("t1_path"),
            "t2_path":src.get("t2_path"),
            "provenance":{
                "source_event_id":event,
                "model":"siglip2-base-patch16-256",
                "source_row_status":src.get("verification_status"),
            },
        })
    by_split=collections.Counter(r["split"] for r in rows)
    by_group=collections.Counter(r["semantic_group_id"] for r in rows)
    for split in sorted(by_split):
        path=out/f"retrieval_semantic_automated_verified_{split}.jsonl"
        path.write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in rows if r["split"]==split),encoding="utf-8")
    report={
        "schema_version":"qcpr-stage2-automated-verified-semantic-pilot-v1",
        "status":"AUTOMATED_VERIFIED_HUMAN_REVIEW_REQUIRED",
        "row_count":len(rows),
        "split_counts":dict(sorted(by_split.items())),
        "semantic_group_count":len(by_group),
        "group_size_distribution":dict(collections.Counter(by_group.values())),
        "training_enabled":False,
        "human_audit_passed":False,
        "source_verification":"frozen SigLIP2 consistency screen",
        "notes":["This view is a candidate semantic supervision layer, not a final verified release view.","Exact-pair relevance is not inferred from the verifier; same-event relevance is compact and graded.","Human stratified audit is still required."],
    }
    (out/"automated_verified_semantic_pilot_audit.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"row_count":len(rows),"split_counts":dict(sorted(by_split.items())),"semantic_group_count":len(by_group),"status":report["status"]},sort_keys=True))
    return 0 if rows else 2

if __name__=="__main__":
    raise SystemExit(main())
