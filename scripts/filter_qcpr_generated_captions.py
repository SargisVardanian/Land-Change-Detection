#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from land_change_detection.data.qcpr_dataset_v2 import jsonl_read,jsonl_write,normalize_text
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--accepted",type=Path,required=True);p.add_argument("--rejected",type=Path,required=True);p.add_argument("--min-quality",type=float,default=.7);a=p.parse_args();seen=set();good=[];bad=[]
 for row in jsonl_read(a.input):
  text=str(row.get("text") or "").strip(); key=normalize_text(text)
  if not text or key in seen or float(row.get("quality_score",0))<a.min_quality or row.get("verification_status") not in {"verified","human"}: bad.append(row);continue
  seen.add(key);good.append(row)
 jsonl_write(a.accepted,good);jsonl_write(a.rejected,bad)
