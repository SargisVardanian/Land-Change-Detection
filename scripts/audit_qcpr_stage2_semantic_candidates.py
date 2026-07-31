#!/usr/bin/env python3
"""Audit candidates; text collisions never become positives automatically."""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--caption-registry",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    rows=[json.loads(x) for x in a.caption_registry.read_text(encoding="utf-8").splitlines() if x.strip()]; by=collections.defaultdict(set); src=collections.Counter(); status=collections.Counter()
    for r in rows:
        if r.get("query_scope")=="generic_no_change": continue
        text=str(r.get("normalized_text") or "").strip(); pair=str(r.get("canonical_pair_id") or "")
        if text and pair: by[text].add(pair); src[str(r.get("dataset_name") or "unknown")]+=1; status[str(r.get("verification_status") or "unknown")]+=1
    candidates=[{"semantic_candidate_id":"candidate:"+hashlib.sha256(text.encode()).hexdigest()[:16],"normalized_text":text,"candidate_pair_ids":sorted(ids),"candidate_kind":"cross_pair_text_collision_candidate","verified":False,"verification_status":"pending_independent_review","positive_pair_ids":[],"ignored_pair_ids":[],"reason":"text collision alone is insufficient evidence"} for text,ids in sorted(by.items()) if len(ids)>1]
    summary={"schema_version":"qcpr-stage2-semantic-audit-v1","caption_rows":len(rows),"non_generic_caption_rows":sum(r.get("query_scope")!="generic_no_change" for r in rows),"candidate_group_count":len(candidates),"verified_cross_pair_group_count":0,"semantic_train_count":0,"semantic_development_count":0,"semantic_test_count":0,"by_source":dict(sorted(src.items())),"by_verification_status":dict(sorted(status.items())),"decision":"SEMANTIC_DATA_HOLD","blocker":"no independently verified cross-pair semantic groups are present"}
    a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/"semantic_candidate_groups.jsonl").write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in candidates),encoding="utf-8"); (a.output_dir/"semantic_audit.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"); (a.output_dir/"semantic_audit.md").write_text("# Stage-2 semantic candidate audit\n\n"+f"- Candidate cross-pair groups: **{len(candidates)}**\n- Independently verified groups: **0**\n- Semantic manifests unlocked: **no**\n\nText collisions are retained for review and are not converted into multi-positive relevance without structured evidence and independent verification.\n",encoding="utf-8"); print(json.dumps(summary,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
