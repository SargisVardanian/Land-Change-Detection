#!/usr/bin/env python3
"""Frozen SigLIP2 consistency verification for a bounded RSCC QvQ pilot."""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import hashlib

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def image_tensor(path: str, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image=image.convert("RGB").resize((size,size), Image.Resampling.BICUBIC)
        array=np.asarray(image,dtype=np.float32)/255.0
    return torch.from_numpy(array).permute(2,0,1)

def file_sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values,dtype=np.float64),q)) if values else float("nan")

class FrozenVerifier:
    def __init__(self, model_path: Path, device: torch.device, image_size: int):
        self.device=device
        self.image_size=image_size
        self.model=AutoModel.from_pretrained(model_path,local_files_only=True).eval().to(device)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
        image_processor=getattr(self.model.config,"vision_config",None)
        self.hidden_dim=int(image_processor.hidden_size)
        processor_mean=list(processor.image_processor.image_mean)
        processor_std=list(processor.image_processor.image_std)
        self.mean=torch.tensor(processor_mean,dtype=torch.float32,device=device).view(1,3,1,1)
        self.std=torch.tensor(processor_std,dtype=torch.float32,device=device).view(1,3,1,1)
        self.max_length=int(self.model.config.text_config.max_position_embeddings)
        self.special_ids=set(int(x) for x in self.tokenizer.all_special_ids)
    @torch.inference_mode()
    def images(self, paths: list[str], batch_size: int) -> torch.Tensor:
        outputs=[]
        for start in range(0,len(paths),batch_size):
            x=torch.stack([image_tensor(p,self.image_size) for p in paths[start:start+batch_size]]).to(self.device)
            x=(x-self.mean)/self.std
            hidden=self.model.vision_model(pixel_values=x).last_hidden_state
            pooled=hidden.mean(dim=1)
            outputs.append(torch.nn.functional.normalize(pooled,dim=-1).cpu())
        return torch.cat(outputs)
    @torch.inference_mode()
    def texts(self, texts: list[str], batch_size: int) -> torch.Tensor:
        outputs=[]
        for start in range(0,len(texts),batch_size):
            enc=self.tokenizer(texts[start:start+batch_size],padding=True,truncation=True,max_length=self.max_length,return_tensors="pt")
            ids=enc["input_ids"].to(self.device)
            mask=enc["attention_mask"].to(self.device).bool()
            hidden=self.model.text_model(input_ids=ids,attention_mask=mask).last_hidden_state
            valid=mask.clone()
            if self.special_ids:
                special=torch.tensor(sorted(self.special_ids),device=self.device,dtype=ids.dtype)
                valid &= ~torch.isin(ids,special)
            empty=~valid.any(dim=1)
            valid[empty]=mask[empty]
            pooled=(hidden*valid.unsqueeze(-1)).sum(dim=1)/valid.sum(dim=1,keepdim=True).clamp_min(1)
            outputs.append(torch.nn.functional.normalize(pooled,dim=-1).cpu())
        return torch.cat(outputs)

def row_caption(row: dict[str,Any]) -> str:
    for key in ("text","caption","change_caption"):
        if row.get(key):
            return str(row[key])
    captions=row.get("captions") or []
    return str(captions[0]) if captions else ""

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path",type=Path,required=True)
    ap.add_argument("--qvq-pilot",type=Path,required=True)
    ap.add_argument("--human-caption-registry",type=Path,required=True)
    ap.add_argument("--pair-registry",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--pilot",type=int,default=500)
    ap.add_argument("--calibration",type=int,default=100)
    ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--seed",type=int,default=20260802)
    args=ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("frozen verifier requires CUDA allocation")
    device=torch.device("cuda")
    verifier=FrozenVerifier(args.model_path,device,image_size=256)
    qvq=[r for r in read_jsonl(args.qvq_pilot) if r.get("captions")][:args.pilot]
    pair_lookup={str(row["canonical_pair_id"]):row for row in read_jsonl(args.pair_registry)}
    human=[]
    for source_row in read_jsonl(args.human_caption_registry):
        row=dict(source_row)
        pair=pair_lookup.get(str(row.get("canonical_pair_id")))
        if pair:
            row["t1_path"]=pair.get("t1_path")
            row["t2_path"]=pair.get("t2_path")
        source=str(row.get("caption_source") or row.get("source"))
        text=row_caption(row)
        if source=="human" and text and row.get("t1_path") and row.get("t2_path") and Path(str(row["t1_path"])).is_file() and Path(str(row["t2_path"])).is_file():
            human.append(row)
        if len(human)>=args.calibration:
            break
    if len(human)<10:
        raise RuntimeError(f"insufficient human calibration rows: {len(human)}")
    calibration_rows=human
    cal_text=verifier.texts([row_caption(r) for r in calibration_rows],args.batch_size)
    cal_paths=[str(r["t1_path"]) for r in calibration_rows]+[str(r["t2_path"]) for r in calibration_rows]
    cal_img=verifier.images(cal_paths,args.batch_size)
    n=len(calibration_rows)
    cal_scores=torch.maximum((cal_text*cal_img[:n]).sum(-1),(cal_text*cal_img[n:]).sum(-1)).tolist()
    rotated=torch.roll(cal_img[n:],shifts=1,dims=0)
    cal_negative=(cal_text*rotated).sum(-1).tolist()
    positive_p05=percentile(cal_scores,5)
    negative_median=float(np.median(np.asarray(cal_negative)))
    candidate_text=verifier.texts([row_caption(r) for r in qvq],args.batch_size)
    candidate_paths=[str(r["t1_path"]) for r in qvq]+[str(r["t2_path"]) for r in qvq]
    candidate_img=verifier.images(candidate_paths,args.batch_size)
    m=len(qvq)
    candidate_scores=torch.maximum((candidate_text*candidate_img[:m]).sum(-1),(candidate_text*candidate_img[m:]).sum(-1)).tolist()
    shuffled=torch.roll(candidate_img[m:],shifts=1,dims=0)
    shuffled_scores=(candidate_text*shuffled).sum(-1).tolist()
    accepted=[]
    rows=[]
    for row,score,negative in zip(qvq,candidate_scores,shuffled_scores,strict=True):
        passed=bool(score>=positive_p05 and score>=negative)
        copy=dict(row)
        copy["verification_status"]="automated_frozen_siglip2_verified" if passed else "automated_frozen_siglip2_rejected"
        copy["verification_model"]="siglip2-base-patch16-256"
        copy["verification_score"]=float(score)
        copy["verification_shuffled_score"]=float(negative)
        copy["verification_thresholds"]={"human_calibration_positive_p05":positive_p05,"candidate_shuffled_minimum":0.0}
        copy["training_enabled"]=False
        copy["human_audit_status"]="required"
        rows.append(copy)
        if passed: accepted.append(copy)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"rscc_qvq_siglip2_verified_pilot.jsonl").write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in accepted),encoding="utf-8")
    report={
        "schema_version":"qcpr-stage2-rscc-qvq-frozen-verification-v1",
        "status":"AUTOMATED_VERIFICATION_ONLY",
        "model_path":str(args.model_path),
        "model_sha256":file_sha256(args.model_path / "model.safetensors") if (args.model_path / "model.safetensors").is_file() else None,
        "pilot_rows":len(qvq),
        "calibration_rows":len(calibration_rows),
        "accepted_rows":len(accepted),
        "accepted_rate":len(accepted)/len(qvq) if qvq else 0.0,
        "rejected_rows":len(qvq)-len(accepted),
        "calibration_positive_p05":positive_p05,
        "calibration_negative_median":negative_median,
        "candidate_score_summary":{"min":min(candidate_scores) if candidate_scores else None,"median":float(np.median(candidate_scores)) if candidate_scores else None,"max":max(candidate_scores) if candidate_scores else None},
        "human_audit_passed":False,
        "training_enabled":False,
        "notes":["This is an image-text consistency screen, not proof of every caption detail.","Accepted rows remain disabled until stratified human review.","No model parameters were updated."],
    }
    (args.output_dir/"rscc_qvq_siglip2_verification_pilot.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:report[k] for k in ("status","pilot_rows","calibration_rows","accepted_rows","accepted_rate","human_audit_passed","training_enabled")},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
