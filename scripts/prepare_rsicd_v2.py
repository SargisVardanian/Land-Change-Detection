#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from land_change_detection.data.qcpr_dataset_v2 import jsonl_read,jsonl_write
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();rows=jsonl_read(a.input)
 for r in rows:
  if r.get("t1_path")==r.get("t2_path"): raise ValueError("single-image datasets cannot become fake temporal pairs")
  r["task_type"]="scene_language_pretrain"
 jsonl_write(a.output,rows)
