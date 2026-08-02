#!/usr/bin/env python3
"""Build provisional graded semantic views from explicit structured provenance."""
from __future__ import annotations
import argparse, collections, json, re
from pathlib import Path
from typing import Any

def read(path: Path) -> list[dict[str,Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--caption-registry",type=Path,required=True); p.add_argument("--pair-registry",type=Path,required=True); p.add_argument("--rscc-pairs",type=Path)
    p.add_argument("--rscc-qvq",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    captions=read(a.caption_registry); pairs={str(r["canonical_pair_id"]):r for r in read(a.pair_registry)}
    if a.rscc_pairs and a.rscc_pairs.exists(): pairs.update({str(r["canonical_pair_id"]):r for r in read(a.rscc_pairs)})
    rows=[]
    s2_groups=collections.defaultdict(list); rscc_groups=collections.defaultdict(list)
    for c in captions:
        if c.get("dataset_name")=="s2looking" and c.get("query_scope")!="generic_no_change":
            text=str(c.get("normalized_text") or ""); direction="appeared" if "appeared" in text else "disappeared" if "disappeared" in text or "demolished" in text else None
            if direction: s2_groups[f"s2looking:building:{direction}"].append(c)
    for c in captions:
        if c.get("dataset_name")=="rscc_ebd" and c.get("query_scope")!="generic_no_change":
            event=str(c.get("source_event_id") or c.get("semantic_group_id") or "")
            if event: rscc_groups[f"rscc_ebd:event:{event}"].append(c)
    # QvQ candidate rows carry the explicit event provenance and are kept generated/unverified.
    if a.rscc_qvq.exists():
        for c in read(a.rscc_qvq):
            for text in c.get("captions",[]):
                rscc_groups[f"rscc_ebd:event:{c['source_event_id']}"].append({"caption_id":f"{c['canonical_pair_id']}:qvq","canonical_pair_id":c["canonical_pair_id"],"text":text,"normalized_text":" ".join(str(text).casefold().split()),"split":c["split"],"dataset_name":"rscc_ebd","caption_source":"RSCC-QvQ","verification_status":"generated_unverified","is_generated":True,"source_event_id":c["source_event_id"]})
    group_rows={}; group_registry=[]; counts=collections.Counter()
    for group, members in sorted({**s2_groups,**rscc_groups}.items()):
        by_split=collections.defaultdict(list)
        for m in members:
                raw_split=str(m.get("split") or pairs.get(str(m["canonical_pair_id"]),{}).get("split") or "unknown"); split_name="development" if raw_split in {"val","validation","dev"} else raw_split; by_split[split_name].append(m)
        for split, split_members in by_split.items():
            pair_ids=sorted({str(m["canonical_pair_id"]) for m in split_members})
            if len(pair_ids)<2: continue
            relation_grade="pair=3;same_relation=2" if group.startswith("s2looking:") else "pair=3;same_event=1"
            group_registry.append({"semantic_group_id":group,"split":split,"pair_ids":pair_ids,"pair_count":len(pair_ids),"graded_relevance_rule":relation_grade,"training_enabled":False})
            for m in split_members:
                pid=str(m["canonical_pair_id"]); relation_grade="pair=3;same_relation=2" if group.startswith("s2looking:") else "pair=3;same_event=1"
                pair=pairs.get(pid,{})
                row={"schema_version":"qcpr-stage2-semantic-view-v1","query_id":str(m.get("caption_id")),"canonical_pair_id":pid,"text":str(m.get("text") or m.get("normalized_text") or ""),"split":split,"source_dataset":str(m.get("dataset_name") or "unknown"),"query_scope":"semantic_group","semantic_group_id":group,"positive_pair_ids":(pair_ids if len(pair_ids)<=256 else []),"semantic_group_pair_count":len(pair_ids),"self_relevance_grade":3,"other_relevance_grade":(2 if group.startswith("s2looking:") else 1),"graded_relevance_rule":relation_grade,"semantic_candidate_count":len(pair_ids),"ignored_pair_ids":[],"training_enabled":False,"caption_source":m.get("caption_source"),"is_generated":bool(m.get("is_generated",False)),"verification_status":("structured_source_verified" if m.get("verification_status")!="generated_unverified" else "generated_unverified"),"verification_mode":"mask_attributes_or_event_identity","human_audit_status":"required_before_stage2_ready","t1_path":pair.get("t1_path"),"t2_path":pair.get("t2_path"),"provenance":{"group_rule":"S2Looking object+direction" if group.startswith("s2looking:") else "RSCC EBD explicit event identity","source_semantic_grade":"relation-level; not exact-pair equivalence"}}
                group_rows.setdefault(split,[]).append(row); counts[split]+=1
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for split in ("train","development","test"):
        out=a.output_dir/f"retrieval_semantic_{split}_v2.jsonl"; values=sorted(group_rows.get(split,[]),key=lambda r:r["query_id"]); out.write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in values),encoding="utf-8")
    (a.output_dir/"semantic_group_registry.jsonl").write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in group_registry),encoding="utf-8")
    summary={"schema_version":"qcpr-stage2-semantic-view-audit-v2","counts":dict(counts),"groups":len(group_registry),"compact_group_registry":True,"group_size_distribution":dict(collections.Counter(x["pair_count"] for x in group_registry)),"multi_positive_rows":sum(counts.values()),"single_positive_rows":0,"materialized_positive_rows":sum(1 for values in group_rows.values() for r in values if r["positive_pair_ids"]),"training_enabled_rows":0,"human_audit_rows":0,"review_required_rows":sum(counts.values()),"structured_source_rows":sum(1 for values in group_rows.values() for r in values if r["verification_status"]=="structured_source_verified"),"status":"PROVISIONAL_MULTI_POSITIVE_CANDIDATES_HUMAN_REVIEW_REQUIRED","stage2_gate":"HOLD","notes":["Each semantic group is stored once in semantic_group_registry.jsonl; large pair lists are not repeated in every row.","Rows with large groups use semantic_group_id and semantic_group_pair_count instead of materialized positive_pair_ids.","Grades are represented by self_relevance_grade/other_relevance_grade and the compact group rule.","No row is training-enabled before independent verifier and stratified human review.","Detailed generated captions remain provenance; they are not promoted as exact equivalence."]}
    (a.output_dir/"semantic_view_audit.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(summary,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
