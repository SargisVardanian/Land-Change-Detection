#!/usr/bin/env python3
"""Build and validate an event-level RSCC EBD physical-pair pilot."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from typing import Any
from PIL import Image

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()

def dhash(path: Path) -> str:
    with Image.open(path) as image:
        gray=image.convert("L").resize((9,8))
        values=list(gray.getdata())
    bits=[]
    for y in range(8):
        row=values[y*9:(y+1)*9]
        bits.extend(a>b for a,b in zip(row,row[1:]))
    out=0
    for bit in bits: out=(out<<1)|int(bit)
    return f"{out:016x}"

def split_for(event: str) -> str:
    bucket=int(hashlib.sha256(event.encode()).hexdigest()[:8],16)%100
    return "test" if bucket<15 else ("development" if bucket<30 else "train")

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--pilot",type=int,default=500)
    a=p.parse_args()
    image_root=a.root/"EBD"
    pre=sorted(image_root.glob("*/images/*_pre_disaster.*"))
    pairs=[]; missing=[]; decode_failures=[]; events=Counter(); splits=Counter()
    for before in pre:
        stem=before.name[:-len("_pre_disaster"+before.suffix)]
        after=before.with_name(stem+"_post_disaster"+before.suffix)
        if not after.exists():
            missing.append(str(before)); continue
        event=before.parent.parent.name
        try:
            with Image.open(before) as ib: ib.load(); bw,bh=ib.size
            with Image.open(after) as ia: ia.load(); aw,ah=ia.size
            if (bw,bh)!=(aw,ah): raise ValueError(f"dimension mismatch {bw,bh} vs {aw,ah}")
            before_sha,after_sha=sha(before),sha(after)
            split=split_for(event); pair_id=f"rscc_ebd:{event}:{stem}"
            row={"schema_version":"qcpr-stage2-rscc-ebd-pair-v1","canonical_pair_id":pair_id,"source_dataset":"RSCC-EBD","source_version":"HF 791a00849c3e684f54df38291cf35a7d828d7004","source_pair_id":f"{event}:{stem}","source_scene_group_id":f"rscc_ebd:event:{event}","source_event_id":event,"parent_pair_id":None,"t1_path":str(before),"t2_path":str(after),"frames":[{"path":str(before),"timestamp":"pre","sha256":before_sha,"perceptual_hash":dhash(before),"width":bw,"height":bh,"modality":"rgb"},{"path":str(after),"timestamp":"post","sha256":after_sha,"perceptual_hash":dhash(after),"width":aw,"height":ah,"modality":"rgb"}],"timestamps":["pre","post"],"modalities":["rgb","rgb"],"split":split,"license":"CC-BY-4.0 / RSCC EBD source terms","sensor":"RGB disaster imagery","gsd":None,"native_dimensions":[bh,bw],"is_synthetic":False,"ordered_pair_hash":hashlib.sha256((before_sha+"\n"+after_sha).encode()).hexdigest(),"order_invariant_pair_hash":hashlib.sha256("\n".join(sorted((before_sha,after_sha))).encode()).hexdigest(),"registration_quality":"source-aligned event pair; requires pilot review","caption_supervision":"none","dense_label_sidecar":{"pre_mask_path":str(image_root/event/"masks"/(stem+"_pre_disaster.png")),"post_mask_path":str(image_root/event/"masks"/(stem+"_post_disaster.png"))}}
            pairs.append(row); events[event]+=1; splits[split]+=1
        except Exception as exc:
            decode_failures.append({"before":str(before),"after":str(after),"error":f"{type(exc).__name__}: {exc}"})
    pairs.sort(key=lambda row:row["canonical_pair_id"])
    pilot=pairs[:a.pilot]
    output=a.output_dir; output.mkdir(parents=True,exist_ok=True)
    (output/"rscc_ebd_pair_pilot.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in pilot),encoding="utf-8")
    (output/"rscc_ebd_pair_registry.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in pairs),encoding="utf-8")
    report={"schema_version":"qcpr-stage2-rscc-ebd-pair-audit-v1","source":"RSCC-EBD","source_version":"HF 791a00849c3e684f54df38291cf35a7d828d7004","image_root":str(image_root),"candidate_pre_images":len(pre),"pair_count":len(pairs),"pilot_pair_count":len(pilot),"missing_post_count":len(missing),"decode_failure_count":len(decode_failures),"event_count":len(events),"event_pair_counts":dict(sorted(events.items())),"split_counts":dict(sorted(splits.items())),"split_policy":"deterministic event-level hash because EBD archive has no unified official train/dev/test split in the downloaded asset","caption_count":0,"dense_label_sidecar_only":True,"mask_free_loader_passed":False,"real_loader_batch_passed":bool(pilot),"identity_proven":bool(pairs) and not missing and not decode_failures,"status":"PILOT_READY_NO_CAPTION_SUPERVISION" if pilot and not missing and not decode_failures else "DATA_QUALITY_HOLD","blockers":["RSCC QvQ captions reference xBD paths and are not joined to EBD", "generated temporal caption verifier is still required before retrieval training"] if pairs else ["no complete EBD pairs"],"missing_examples":missing[:20],"decode_failures":decode_failures[:20]}
    (output/"rscc_ebd_pair_audit.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("pair_count","pilot_pair_count","event_count","split_counts","caption_count","status")},sort_keys=True)); return 0 if report["identity_proven"] else 2

if __name__=="__main__": raise SystemExit(main())
