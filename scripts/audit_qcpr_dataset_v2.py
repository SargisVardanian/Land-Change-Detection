#!/usr/bin/env python3
"""Inventory immutable QCPR source datasets without touching their contents."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from land_change_detection.data.qcpr_dataset_v2 import inventory, jsonl_read, sha256_file

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--project-root",type=Path,required=True); p.add_argument("--report-dir",type=Path,required=True); a=p.parse_args()
    root=a.project_root; manifests=root/"datasets/manifests"
    assets={"LEVIR-MCI":root/"datasets/raw/LEVIR-MCI","SECOND-CC":root/"datasets/raw/SECOND-CC-extracted","S2Looking":root/"datasets/raw/S2Looking"}
    report={"datasets":inventory(assets),"manifests":{}}
    for name in ("levir_mci.jsonl","second_cc_canonical.jsonl"):
        path=manifests/name
        rows=jsonl_read(path)
        report["manifests"][name]={"path":str(path),"exists":path.exists(),"sha256":sha256_file(path) if path.exists() else None,"rows":len(rows),"splits":{}}
        for row in rows: report["manifests"][name]["splits"][str(row.get("split","unknown"))]=report["manifests"][name]["splits"].get(str(row.get("split","unknown")),0)+1
    a.report_dir.mkdir(parents=True,exist_ok=True)
    (a.report_dir/"qcpr_dataset_inventory.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\\n")
    lines=["# QCPR Dataset Inventory",""]
    for name,item in report["datasets"].items(): lines.append(f"- **{name}**: exists={item['exists']}; files={item['file_count']}; bytes={item['bytes']}; path=`{item['path']}`")
    for name,item in report["manifests"].items(): lines.append(f"- **{name}**: rows={item['rows']}; sha256=`{item['sha256']}`; splits={item['splits']}")
    (a.report_dir/"qcpr_dataset_inventory.md").write_text("\\n".join(lines)+"\\n")
if __name__=="__main__": main()
