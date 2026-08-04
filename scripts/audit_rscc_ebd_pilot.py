#!/usr/bin/env python3
"""Audit official RSCC EBD multipart archive."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
EXPECTED={"EBD.tar.gz-part-0":(3221225472,"31566711f"),"EBD.tar.gz-part-1":(3221225472,"16e77d45"),"EBD.tar.gz-part-2":(3221225472,"2c475292"),"EBD.tar.gz-part-3":(3221225472,"08594770"),"EBD.tar.gz-part-4":(1603907790,"1f3350dc")}
def sha(path:Path)->str:
    d=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): d.update(b)
    return d.hexdigest()
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--pilot",type=int,default=500); a=p.parse_args(); parts=[]; complete=True
    for name,(expected,prefix) in EXPECTED.items():
        path=a.root/name; item={"file":name,"path":str(path),"expected_bytes":expected,"expected_hf_oid_prefix":prefix,"exists":path.is_file()}
        if path.is_file(): item.update({"actual_bytes":path.stat().st_size,"sha256":sha(path),"size_ok":path.stat().st_size==expected}); item["sha256_oid_prefix_ok"]=item["sha256"].startswith(prefix)
        else: item.update({"actual_bytes":None,"sha256":None,"size_ok":False,"sha256_oid_prefix_ok":False})
        complete &= bool(item["exists"] and item["size_ok"]); parts.append(item)
    extracted=a.root/"extracted"; images=[x for x in extracted.rglob("*") if x.is_file() and x.suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff"}] if extracted.exists() else []
    report={"schema_version":"qcpr-stage2-rscc-ebd-pilot-v1","official_source":"https://huggingface.co/datasets/BiliSakura/RSCC","parts":parts,"archive_complete":complete,"extracted_root":str(extracted),"image_file_count":len(images),"pilot_candidate_count":min(len(images)//2,a.pilot),"pair_identity_proven":False,"loader_validated":False,"state":"DOWNLOADED" if complete else "DOWNLOAD_PARTIAL","blockers":[] if complete else ["all official EBD multipart files are not complete and size-verified"],"notes":["HF Hub OID prefixes are recorded; full SHA256 is retained per part.","RSCC QvQ/xBD metadata is not joined to EBD without explicit frame identity proof."]}
    if complete and not images: report["blockers"].append("archive complete but not extracted or no image inventory found")
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps({"state":report["state"],"archive_complete":complete,"image_file_count":len(images),"output":str(a.output)},sort_keys=True)); return 0 if complete else 2
if __name__=="__main__": raise SystemExit(main())
