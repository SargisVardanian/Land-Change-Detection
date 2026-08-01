#!/usr/bin/env python3
"""Map RSCC QvQ captions only to EBD assets with explicit path proof."""
from __future__ import annotations
import argparse, collections, hashlib, json
from pathlib import Path
from typing import Any

def norm(text: str) -> str:
    return " ".join(str(text).casefold().split())

def split_for(event: str, ordered_events: list[str]) -> str:
    index=ordered_events.index(event)
    return "test" if index < 2 else ("development" if index < 4 else "train")

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--annotations",type=Path,required=True); p.add_argument("--ebd-root",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--pilot",type=int,default=500); a=p.parse_args()
    groups: dict[tuple[str,str],dict[str,Any]]={}; unmatched=[]; total=0; matched=0
    for line in a.annotations.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row=json.loads(line); total+=1; pre=Path(str(row.get("pre_image",""))); post=Path(str(row.get("post_image","")))
        marker=Path("/home/datasets/EBD")
        if marker not in pre.parents or marker not in post.parents: continue
        before=a.ebd_root/pre.relative_to(marker); after=a.ebd_root/post.relative_to(marker)
        if not before.is_file() or not after.is_file(): unmatched.append({"pre":str(pre),"post":str(post),"reason":"EBD asset missing"}); continue
        matched+=1; event=before.parent.parent.name; key=(event,before.stem.removesuffix("_pre_disaster"))
        item=groups.setdefault(key,{"event":event,"before":str(before),"after":str(after),"captions":[],"split":None})
        caption=str(row.get("change_caption","")).strip()
        if caption and caption not in item["captions"]: item["captions"].append(caption)
    ordered_events=sorted({event for event, _ in groups})
    for item in groups.values(): item["split"]=split_for(item["event"],ordered_events)
    rows=[]; provenance=collections.Counter()
    for (event,stem), item in sorted(groups.items()):
        pair=f"rscc_ebd:{event}:{stem}"
        rows.append({"schema_version":"qcpr-stage2-rscc-qvq-caption-v1","canonical_pair_id":pair,"source_dataset":"RSCC-EBD","source_pair_id":f"{event}:{stem}","source_scene_group_id":f"rscc_ebd:event:{event}","source_event_id":event,"split":item["split"],"t1_path":item["before"],"t2_path":item["after"],"captions":item["captions"],"caption_source":"RSCC-QvQ","is_generated":True,"generator":"QvQ-Max","verification_status":"generated_unverified","quality_score":None,"identifiability_score":None,"training_enabled":False,"reason_training_disabled":"independent frozen verifier and stratified human audit pending","query_scope":"semantic_group"})
        provenance["generated_unverified"]+=len(item["captions"])
    rows.sort(key=lambda r:r["canonical_pair_id"]); out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    (out/"rscc_ebd_qvq_caption_pilot.jsonl").write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in rows[:a.pilot]),encoding="utf-8")
    (out/"rscc_ebd_qvq_caption_candidates.jsonl").write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    report={"schema_version":"qcpr-stage2-rscc-qvq-caption-audit-v1","annotation_rows":total,"matched_ebd_annotation_rows":matched,"unmatched_ebd_rows":len(unmatched),"mapped_pairs":len(rows),"pilot_pairs":min(len(rows),a.pilot),"caption_count":sum(len(r["captions"]) for r in rows),"split_counts":dict(collections.Counter(r["split"] for r in rows)),"event_counts":dict(collections.Counter(r["source_event_id"] for r in rows)),"provenance":dict(provenance),"independent_verification_passed":False,"human_audit_passed":False,"training_enabled":False,"status":"CAPTION_VERIFICATION_REQUIRED","unmatched_examples":unmatched[:50],"blockers":["QvQ is generated supervision, not human caption supervision","independent frozen verifier and stratified human review are required before retrieval use"]}
    (out/"rscc_ebd_qvq_caption_audit.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps({k:report[k] for k in ("annotation_rows","matched_ebd_annotation_rows","mapped_pairs","pilot_pairs","caption_count","status")},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
