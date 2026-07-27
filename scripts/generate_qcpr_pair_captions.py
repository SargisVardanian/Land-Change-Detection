#!/usr/bin/env python3
"""Offline caption proposal contract; does not claim generated text is human."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from land_change_detection.data.qcpr_dataset_v2 import jsonl_read,jsonl_write,normalize_text
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--pairs",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();rows=[]
 for pair in jsonl_read(a.pairs):
  # A proposal needs an external verifier before use; never invent scene facts.
  rows.append({"canonical_pair_id":pair["canonical_pair_id"],"caption_source":"generated_unverified","text":None,"normalized_text":None,"verification_status":"pending_external_generation_and_filter","is_generated":True,"generator":None,"quality_score":0.0,"identifiability_score":0.0})
 jsonl_write(a.output,rows)
