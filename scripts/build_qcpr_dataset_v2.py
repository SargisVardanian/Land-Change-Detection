#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from land_change_detection.data.qcpr_dataset_v2 import build_rows_v2,exclude_cross_split_image_conflicts,jsonl_read,jsonl_write,leakage_audit,stable_manifest_hashes

def main() -> None:
 p=argparse.ArgumentParser();p.add_argument("--manifest-dir",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);a=p.parse_args()
 manifests=[a.manifest_dir/"natural_train_retrieval_manifest.jsonl",a.manifest_dir/"natural_validation_retrieval_manifest.jsonl"]
 source_rows=[row for manifest in manifests for row in jsonl_read(manifest)]
 source_rows,split_resolution=exclude_cross_split_image_conflicts(source_rows)
 summary=build_rows_v2(source_rows,a.output_root/"registries")
 pairs=jsonl_read(a.output_root/"registries/pair_registry.jsonl"); captions=jsonl_read(a.output_root/"registries/caption_registry.jsonl"); relevance=jsonl_read(a.output_root/"registries/relevance_registry.jsonl")
 audit=leakage_audit(pairs); (a.output_root/"reports").mkdir(parents=True,exist_ok=True);(a.output_root/"reports/dataset_v2_leakage_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
 # Immutable task-specific views; dense labels intentionally remain separate.
 cap_by_pair={}; [cap_by_pair.setdefault(c["canonical_pair_id"],[]).append(c) for c in captions]
 rel_by_caption={r["caption_id"]:r for r in relevance}
 for split,source_splits in [("train",{"train"}),("development",{"validation","val","dev","development"}),("test",{"test"})]:
  rows=[r for r in source_rows if str(r.get("split","")).casefold() in source_splits]
  out=[]
  for row in rows:
   for c in cap_by_pair.get(str(row["pair_id"]),[]):
    if c["query_scope"]=="generic_no_change": continue
    out.append({"canonical_pair_id":str(row["pair_id"]),"caption_id":c["caption_id"],"caption":c["text"],"query_scope":c["query_scope"],"positive_pair_ids":rel_by_caption[c["caption_id"]]["positive_pair_ids"],"ignored_pair_ids":rel_by_caption[c["caption_id"]]["ignored_pair_ids"],"t1_path":row.get("t1_path"),"t2_path":row.get("t2_path"),"dataset_name":row.get("dataset_name"),"split":split})
  jsonl_write(a.output_root/f"retrieval_{split}_v2.jsonl",out);jsonl_write(a.output_root/f"grounding_{split}_mask_free_v2.jsonl",out)
 jsonl_write(a.output_root/"dense_evaluation_v2.jsonl",[]);jsonl_write(a.output_root/"scene_language_pretrain_v2.jsonl",[])
 summary["split_conflict_resolution"]=split_resolution
 summary["leakage_audit_passed"]=audit["passed"]
 summary["manifest_hashes"]=stable_manifest_hashes(a.output_root);(a.output_root/"build_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
 if not audit["passed"]: raise RuntimeError("Dataset-v2 leakage audit failed")
if __name__=="__main__":main()
