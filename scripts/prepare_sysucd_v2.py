#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from land_change_detection.data.qcpr_dataset_v2 import jsonl_read,jsonl_write
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();jsonl_write(a.output,jsonl_read(a.input))
