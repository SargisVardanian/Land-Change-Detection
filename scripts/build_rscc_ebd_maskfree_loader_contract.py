#!/usr/bin/env python3
"""Build a mask-free RSCC QvQ manifest and execute the real project loader contract."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from land_change_detection.training.temporal_caption_dataset import TemporalCaptionManifestDataset
from land_change_detection.temporal_caption_manifest import SCHEMA_VERSION

FORBIDDEN=("mask","semantic","label","dense")

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--qvq",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--image-size",type=int,default=64)
    args=ap.parse_args()
    rows=[]
    for line in args.qvq.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        src=json.loads(line)
        rows.append({
            "schema_version":SCHEMA_VERSION,
            "dataset_name":"rscc_ebd",
            "pair_id":str(src["canonical_pair_id"]),
            "original_id":str(src["source_pair_id"]),
            "split":str(src["split"]),
            "t1_path":str(src["t1_path"]),
            "t2_path":str(src["t2_path"]),
            "captions":[str(x) for x in src.get("captions",[]) if str(x).strip()],
            "normalized_caption_groups":[" ".join(str(x).casefold().split()) for x in src.get("captions",[]) if str(x).strip()],
            "caption_source":"model_generated",
            "time_order":["before","after"],
            "mask_path":None,
            "semantic_t1_path":None,
            "semantic_t2_path":None,
            "source_metadata":{
                "source_dataset":"RSCC-EBD",
                "source_event_id":src.get("source_event_id"),
                "verification_status":src.get("verification_status"),
                "training_enabled":False,
                "reason_training_disabled":"independent verifier and stratified human audit pending",
                "retrieval_supervision":False,
            },
        })
    rows.sort(key=lambda r:r["pair_id"])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=args.output_dir/"rscc_ebd_qvq_mask_free_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    forbidden=[]
    for r in rows:
        for key,value in r.items():
            if any(token in key.casefold() for token in ("mask","semantic","dense","label")) and value not in (None,False,[],{}):
                forbidden.append({"key":key})
        for key in r.get("source_metadata",{}):
            if any(token in key.casefold() for token in FORBIDDEN):
                forbidden.append({"key":"source_metadata."+key})
    # The metadata contains explicit null target fields only; this is allowed.
    forbidden=[x for x in forbidden if x["key"] not in {"mask_path","semantic_t1_path","semantic_t2_path"}]
    per_split={}
    errors=[]
    for split in sorted({r["split"] for r in rows}):
        try:
            ds=TemporalCaptionManifestDataset(manifest,split=split,image_size=args.image_size,max_pairs=4,load_segmentation_targets=False,exclude_rscc_model_generated=False)
            if len(ds)==0:
                raise RuntimeError("empty loader split")
            loader=DataLoader(ds,batch_size=min(2,len(ds)),shuffle=False,num_workers=0,collate_fn=lambda batch: {
                "t1":torch.stack([x.t1 for x in batch]),
                "t2":torch.stack([x.t2 for x in batch]),
                "captions":[x.captions for x in batch],
            })
            batch=next(iter(loader))
            if batch["t1"].ndim!=4 or batch["t2"].ndim!=4 or batch["t1"].shape!=batch["t2"].shape:
                raise RuntimeError(f"bad batch shapes {tuple(batch['t1'].shape)} {tuple(batch['t2'].shape)}")
            if any(ds[i].metadata.get("mask_path") is not None for i in range(min(2,len(ds)))):
                raise RuntimeError("mask path unexpectedly present")
            per_split[split]={"dataset_rows":len(ds),"batch_t1_shape":list(batch["t1"].shape),"batch_t2_shape":list(batch["t2"].shape),"load_segmentation_targets":False,"passed":True}
        except Exception as exc:
            errors.append({"split":split,"error":f"{type(exc).__name__}: {exc}"})
            per_split[split]={"passed":False}
    report={
        "schema_version":"qcpr-stage2-rscc-mask-free-loader-contract-v1",
        "manifest":str(manifest),
        "manifest_rows":len(rows),
        "split_counts":dict(Counter(r["split"] for r in rows)),
        "mask_free_forbidden_key_count":len(forbidden),
        "mask_free_forbidden_keys":forbidden[:20],
        "project_loader":"TemporalCaptionManifestDataset",
        "load_segmentation_targets":False,
        "per_split":per_split,
        "errors":errors,
        "passed":not forbidden and not errors and len(rows)>0,
    }
    (args.output_dir/"rscc_mask_free_loader_contract.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"manifest_rows":len(rows),"split_counts":report["split_counts"],"passed":report["passed"],"errors":len(errors)},sort_keys=True))
    return 0 if report["passed"] else 2

if __name__=="__main__":
    raise SystemExit(main())
