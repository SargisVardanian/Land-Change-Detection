#!/usr/bin/env python3
"""Append a deterministic, provenance-complete official download record."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from land_change_detection.data.qcpr_dataset_v2 import sha256_file

def main() -> None:
 p=argparse.ArgumentParser();p.add_argument("--registry",type=Path,required=True);p.add_argument("--dataset",required=True);p.add_argument("--version",required=True);p.add_argument("--source",required=True);p.add_argument("--path",type=Path,required=True);p.add_argument("--license",default="unknown");p.add_argument("--status",choices=("complete","incomplete","manual_access_required"),required=True);a=p.parse_args()
 files=sorted(x for x in a.path.rglob("*") if x.is_file() and not x.name.endswith(".lock")) if a.path.exists() else []
 row={"dataset":a.dataset,"version":a.version,"official_source":a.source,"path":str(a.path),"license":a.license,"status":a.status,"files":len(files),"bytes":sum(x.stat().st_size for x in files),"sha256":sha256_file(files[0]) if len(files)==1 else None}
 a.registry.parent.mkdir(parents=True,exist_ok=True)
 existing=[]
 if a.registry.exists(): existing=[json.loads(x) for x in a.registry.read_text().splitlines() if x.strip()]
 existing=[x for x in existing if not (x.get("dataset")==a.dataset and x.get("version")==a.version)] + [row]
 a.registry.write_text("".join(
     json.dumps(x,sort_keys=True)+"\\n"
     for x in sorted(existing,key=lambda x:(x["dataset"],x["version"]))
 ))
if __name__=="__main__":main()
