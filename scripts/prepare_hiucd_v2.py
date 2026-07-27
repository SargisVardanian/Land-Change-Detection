#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from land_change_detection.data.qcpr_dataset_v2 import jsonl_read,jsonl_write
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();rows=jsonl_read(a.input)
 for r in rows:
  frames=sorted(r.get("frames",[]),key=lambda x:str(x.get("timestamp","")))
  if len(frames) not in {2,3}: raise ValueError("Hi-UCD requires T=2 or T=3 ordered frames")
  r["frames"]=frames;r["timestamps"]=[x["timestamp"] for x in frames]
 jsonl_write(a.output,rows)
