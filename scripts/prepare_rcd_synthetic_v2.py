#!/usr/bin/env python3
"""Register RCD synthetic records; masks are written only to dense sidecars."""
from __future__ import annotations
import argparse
from pathlib import Path
from land_change_detection.data.qcpr_dataset_v2 import jsonl_read,jsonl_write
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--records",type=Path,required=True);p.add_argument("--pairs-output",type=Path,required=True);p.add_argument("--dense-output",type=Path,required=True);a=p.parse_args();pairs=[];dense=[]
 for row in jsonl_read(a.records):
  key=str(row["source_pair_id"]);pairs.append({"canonical_pair_id":f"rcd:{row.get('source_dataset','unknown')}:{key}","source_dataset":row.get("source_dataset"),"source_pair_id":key,"source_scene_group_id":key,"frames":row.get("frames",[]),"is_synthetic":True,"synthetic_generator":"RCDGen","split":row.get("split"),"license":row.get("license")})
  if row.get("mask_path"): dense.append({"canonical_pair_id":pairs[-1]["canonical_pair_id"],"binary_change_mask":row["mask_path"]})
 jsonl_write(a.pairs_output,pairs);jsonl_write(a.dense_output,dense)
