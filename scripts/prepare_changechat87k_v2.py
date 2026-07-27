#!/usr/bin/env python3
"""Map ChangeChat instructions to existing LEVIR-MCI pair IDs; never copy images."""
from __future__ import annotations
import argparse
from pathlib import Path
from land_change_detection.data.qcpr_dataset_v2 import jsonl_read,jsonl_write,normalize_text
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--instructions",type=Path,required=True);p.add_argument("--levir-pairs",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();known={r["pair_id"] for r in jsonl_read(a.levir_pairs)};out=[]
 for n,row in enumerate(jsonl_read(a.instructions)):
  pair=str(row.get("pair_id") or row.get("image_id") or "")
  if pair not in known: continue
  text=str(row.get("response") or row.get("answer") or "").strip()
  out.append({"caption_id":f"changechat:{pair}:{n}","canonical_pair_id":pair,"text":text,"normalized_text":normalize_text(text),"caption_source":"changechat_gpt_assisted" if row.get("gpt") else "changechat_rule_based","task_type":row.get("task_type","instruction"),"query_scope":"semantic_group","is_generated":bool(row.get("gpt")),"verification_status":"unverified"})
 jsonl_write(a.output,out)
