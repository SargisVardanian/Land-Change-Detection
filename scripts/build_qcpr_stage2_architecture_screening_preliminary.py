#!/usr/bin/env python3
"""Convert the historical frozen encoder report into an explicit Stage-2 preliminary screen."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--historical-report",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    source=json.loads(a.historical_report.read_text())
    mapping={
        "B0":"current_jina_universat",
        "B3":"remoteclip_vit_b_32",
        "EXTRA_GEORSCLIP":"georsclip",
        "EXTRA_SIGLIP2":"siglip2_base_patch16_256",
    }
    enc={}
    for target, key in mapping.items():
        row=source.get("encoders",{}).get(key)
        if row is None:
            continue
        enc[target]={
            "source_key":key,
            "status":"historical_compatibility_only",
            "metrics":row.get("metrics",{}),
        }
    payload={
        "schema_version":"qcpr-stage2-architecture-screen-preliminary-v1",
        "status":"PRELIMINARY_ONLY",
        "stage2_complete":False,
        "frozen_only":True,
        "source_report":str(a.historical_report),
        "source_report_sha256":sha256(a.historical_report),
        "source_report_manifest":source.get("manifest"),
        "source_report_pair_count":source.get("pair_count"),
        "source_report_query_count":source.get("query_count"),
        "evaluated":enc,
        "not_evaluated":{
            "B1":"implementation required",
            "B2":"implementation required",
            "B4":"LRSCLIP/DGTRS weights and preprocessing not available in this report",
        },
        "not_comparable_to_stage2":"historical report uses the prior audited benchmark and diagnostic exact/semantic protocols; it is not a Stage-2 common real-only screen.",
        "selection":"none",
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(a.output),"status":payload["status"],"evaluated":sorted(enc)},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
