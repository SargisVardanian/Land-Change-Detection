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
    s2_groups=collections.defaultdict(list)
    excluded_unstructured=[]
    for c in captions:
        if c.get("dataset_name")=="s2looking" and c.get("query_scope")!="generic_no_change":
            text=str(c.get("normalized_text") or ""); direction="appeared" if "appeared" in text else "disappeared" if "disappeared" in text or "demolished" in text else None
            if direction: s2_groups[f"s2looking:building:{direction}"].append(c)
    # RSCC captions/QvQ rows without reviewed structured attributes are audit
    # candidates only.  Event IDs are provenance and split/leakage metadata;
    # they must never define semantic positive sets.
    for c in captions:
        if c.get("dataset_name")=="rscc_ebd" and c.get("query_scope")!="generic_no_change":
            excluded_unstructured.append({
                "canonical_pair_id": c.get("canonical_pair_id"),
                "caption_id": c.get("caption_id"),
                "source_event_id": c.get("source_event_id"),
                "text": c.get("text") or c.get("normalized_text") or "",
                "split": c.get("split"),
                "reason": "unstructured_rscc_caption_event_only_not_semantic",
                "training_enabled": False,
            })
    # QvQ candidate rows carry event provenance but no independently reviewed
    # object/direction/damage attributes, so they remain excluded candidates.
    if a.rscc_qvq.exists():
        for c in read(a.rscc_qvq):
            for text in c.get("captions",[]):
                excluded_unstructured.append({"caption_id":f"{c['canonical_pair_id']}:qvq","canonical_pair_id":c["canonical_pair_id"],"text":text,"split":c["split"],"source_event_id":c.get("source_event_id"),"reason":"unstructured_rscc_qvq_event_only_not_semantic","training_enabled":False})
    group_rows={}; group_registry=[]; counts=collections.Counter()
    for group, members in sorted(s2_groups.items()):
        by_split=collections.defaultdict(list)
        for m in members:
                raw_split=str(m.get("split") or pairs.get(str(m["canonical_pair_id"]),{}).get("split") or "unknown"); split_name="development" if raw_split in {"val","validation","dev"} else raw_split; by_split[split_name].append(m)
        for split, split_members in by_split.items():
            pair_ids=sorted({str(m["canonical_pair_id"]) for m in split_members})
            if len(pair_ids)<2: continue
            relation_grade="pair=3;same_object_and_direction=2;broad_change=1"
            group_registry.append({"semantic_group_id":group,"split":split,"pair_ids":pair_ids,"pair_count":len(pair_ids),"graded_relevance_rule":relation_grade,"training_enabled":False})
            for m in split_members:
                pid=str(m["canonical_pair_id"]); relation_grade="pair=3;same_object_and_direction=2;broad_change=1"
                pair=pairs.get(pid,{})
                row={"schema_version":"qcpr-stage2-semantic-view-v2","query_id":str(m.get("caption_id")),"canonical_pair_id":pid,"text":str(m.get("text") or m.get("normalized_text") or ""),"split":split,"source_dataset":str(m.get("dataset_name") or "unknown"),"query_scope":"semantic_group","semantic_group_id":group,"positive_pair_ids":(pair_ids if len(pair_ids)<=256 else []),"semantic_group_pair_count":len(pair_ids),"self_relevance_grade":3,"other_relevance_grade":2,"graded_relevance_rule":relation_grade,"semantic_candidate_count":len(pair_ids),"ignored_pair_ids":[],"training_enabled":False,"caption_source":m.get("caption_source"),"is_generated":bool(m.get("is_generated",False)),"verification_status":("structured_source_verified" if m.get("verification_status")!="generated_unverified" else "generated_unverified"),"verification_mode":"structured_source_attributes_only","human_audit_status":"required_before_stage2_ready","event_ids_are_provenance_only":True,"t1_path":pair.get("t1_path"),"t2_path":pair.get("t2_path"),"provenance":{"group_rule":"S2Looking object+direction structured candidate","source_semantic_grade":"provisional structured relation; not human-reviewed training supervision"}}
                group_rows.setdefault(split,[]).append(row); counts[split]+=1
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for split in ("train","development","test"):
        out=a.output_dir/f"retrieval_semantic_{split}_v2.jsonl"; values=sorted(group_rows.get(split,[]),key=lambda r:r["query_id"]); out.write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in values),encoding="utf-8")
    (a.output_dir/"semantic_group_registry.jsonl").write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in group_registry),encoding="utf-8")
    (a.output_dir/"unstructured_candidates_excluded.jsonl").write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in excluded_unstructured),encoding="utf-8")
    summary={"schema_version":"qcpr-stage2-semantic-view-audit-v3","counts":dict(counts),"groups":len(group_registry),"compact_group_registry":True,"group_size_distribution":dict(collections.Counter(x["pair_count"] for x in group_registry)),"multi_positive_rows":sum(counts.values()),"single_positive_rows":0,"materialized_positive_rows":sum(1 for values in group_rows.values() for r in values if r["positive_pair_ids"]),"training_enabled_rows":0,"human_audit_rows":0,"review_required_rows":sum(counts.values()),"structured_source_rows":sum(1 for values in group_rows.values() for r in values if r["verification_status"]=="structured_source_verified"),"unstructured_rscc_rows_excluded":len(excluded_unstructured),"event_only_rscc_groups":0,"status":"PROVISIONAL_STRUCTURED_CANDIDATES_HUMAN_REVIEW_REQUIRED","stage2_gate":"HOLD","event_ids_used_for_semantics":False,"stage2_semantic_training_enabled":False,"notes":["Each semantic group is stored once in semantic_group_registry.jsonl; large pair lists are not repeated in every row.","Rows with large groups use semantic_group_id and semantic_group_pair_count instead of materialized positive_pair_ids.","RSCC event identity is provenance only and cannot define a semantic group.","No row is training-enabled before independent verifier and stratified human review.","Detailed generated captions and SigLIP2-only rows remain excluded candidates; they are not semantic positives."]}
    (a.output_dir/"semantic_view_audit.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(summary,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
