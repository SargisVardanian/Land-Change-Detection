from __future__ import annotations
import argparse,json,math,sys,time
from dataclasses import asdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qcpr_single_pass_runtime import evaluate_retrieval,forbidden_mask_keys,git_sha,save_checkpoint,sha256
from land_change_detection.models.qcpr_single_pass import build_pair_masks,multi_positive_sigmoid_loss
from land_change_detection.models.qcpr_single_pass_factory import build_single_pass_retriever
from land_change_detection.training.qcpr_single_pass_data import CaptionCollisionIndex,MaskFreePairDataset,PairCaptionCollator

def args():
 p=argparse.ArgumentParser()
 p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--manifest-dir",type=Path,required=True)
 p.add_argument("--universat-source",default="/mnt/weka/svardanyan/rs_change_project/external/UniverSat")
 p.add_argument("--universat-checkpoint",default="/mnt/weka/svardanyan/rs_change_project/models/universat-base")
 p.add_argument("--jina-model",default="/mnt/weka/svardanyan/rs_change_project/models/jina-v5-text-small-retrieval")
 p.add_argument("--output-grid",type=int,default=32); p.add_argument("--epochs",type=int,default=30)
 p.add_argument("--patience",type=int,default=5); p.add_argument("--workers",type=int,default=8)
 p.add_argument("--batch-size",type=int,default=24); p.add_argument("--accumulation",type=int,default=2)
 p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--text-lr",type=float,default=2e-5)
 p.add_argument("--weight-decay",type=float,default=.05); p.add_argument("--seed",type=int,default=20260725)
 return p.parse_args()

def loader(dataset,batch,workers,epoch,seed):
 return DataLoader(dataset,batch_size=batch,shuffle=True,num_workers=workers,pin_memory=True,
  collate_fn=PairCaptionCollator(2,seed,epoch),drop_last=True)

def train_step(model,batch,collisions,device,scale):
 if forbidden_mask_keys(batch): raise RuntimeError("segmentation target in retrieval batch")
 with torch.autocast("cuda",dtype=torch.bfloat16):
  out=model(batch["images"].to(device,non_blocking=True),batch["captions"])
 collision_sets=[collisions.collisions(pid,text) for pid,text in zip(batch["query_pair_ids"],batch["normalized_captions"],strict=True)]
 pos,exc=build_pair_masks(batch["query_pair_ids"],batch["pair_ids"],collision_sets,device)
 loss,stats=multi_positive_sigmoid_loss(out.score_matrix,pos,exc)
 (loss/scale).backward()
 return loss,stats,out

def main():
 a=args(); torch.manual_seed(a.seed); device=torch.device("cuda"); a.output_dir.mkdir(parents=True,exist_ok=True)
 train_manifest=a.manifest_dir/"natural_train_retrieval_manifest.jsonl"
 val_manifest=a.manifest_dir/"natural_validation_retrieval_manifest.jsonl"
 collision_path=a.manifest_dir/"caption_quality_audit.jsonl"
 train=MaskFreePairDataset(train_manifest,"train"); val=MaskFreePairDataset(val_manifest,"val")
 if set(train.pair_ids)&set(val.pair_ids): raise RuntimeError("train/dev leakage")
 collisions=CaptionCollisionIndex(collision_path)
 model=build_single_pass_retriever(universat_source=a.universat_source,
  universat_checkpoint=a.universat_checkpoint,jina_model=a.jina_model,device=device,output_grid=a.output_grid)
 text_parameters=list(model.text_projection.parameters())
 local_projection=getattr(model.text_encoder,"local_projection",None)
 if local_projection is not None: text_parameters.extend(local_projection.parameters())
 groups=[{"params":model.pair_encoder.parameters(),"lr":a.lr},
         {"params":text_parameters,"lr":a.text_lr},
         {"params":[model.logit_scale],"lr":a.text_lr}]
 optimizer=torch.optim.AdamW(groups,weight_decay=a.weight_decay)
 total=math.ceil(len(train)/a.batch_size)*a.epochs/max(a.accumulation,1)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(int(total),1),eta_min=1e-6)
 config=vars(a)|{"git_sha":git_sha(),"train_manifest_sha256":sha256(train_manifest),
  "val_manifest_sha256":sha256(val_manifest),"collision_audit_sha256":sha256(collision_path),
  "load_segmentation_targets":False,"captions_per_pair":2,"bf16":True}
 (a.output_dir/"run_config.json").write_text(json.dumps(config,indent=2,sort_keys=True,default=str)+"\n")
 # Embedded first real batch check and in-process OOM fallback.
 actual_batch,actual_accum=a.batch_size,a.accumulation
 while True:
  first=next(iter(loader(train,actual_batch,a.workers,0,a.seed))); optimizer.zero_grad(set_to_none=True)
  try:
   torch.cuda.reset_peak_memory_stats()
   loss,stats,out=train_step(model,first,collisions,device,actual_accum)
   grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
   if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
    raise RuntimeError("non-finite loss/gradients or no intended gradients")
   if out.pair.contextual_patch_tokens.shape[1]!=a.output_grid**2: raise RuntimeError("patch shape contract")
   if out.pair.pair_search_vector.shape[-1]!=512 or out.text.text_search_vector.shape[-1]!=512:
    raise RuntimeError("search vector contract")
   peak=torch.cuda.max_memory_allocated()/2**30; optimizer.zero_grad(set_to_none=True); break
  except torch.cuda.OutOfMemoryError:
   optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
   if (actual_batch,actual_accum)!=(a.batch_size,a.accumulation): raise
   actual_batch,actual_accum=16,3
 config.update({"actual_batch_size":actual_batch,"actual_gradient_accumulation":actual_accum,
  "effective_physical_batch":actual_batch*actual_accum,"first_batch_peak_gib":peak})
 (a.output_dir/"run_config.json").write_text(json.dumps(config,indent=2,sort_keys=True,default=str)+"\n")
 best=None; stale=0; history=a.output_dir/"metrics.jsonl"
 for epoch in range(a.epochs):
  model.train(); optimizer.zero_grad(set_to_none=True); sums={"loss":0.,"positive":0.,"negative":0.}; count=0
  for step,batch in enumerate(loader(train,actual_batch,a.workers,epoch,a.seed),1):
   loss,stats,_=train_step(model,batch,collisions,device,actual_accum)
   sums["loss"]+=float(loss); sums["positive"]+=float(stats["positive_similarity"]); sums["negative"]+=float(stats["negative_similarity"]); count+=1
   if step%actual_accum==0:
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
  if count%actual_accum: optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
  model.eval(); metrics,rows=evaluate_retrieval(model,val,min(8,actual_batch),a.workers,device)
  current=(metrics["all"]["recall_at_5"],metrics["all"]["recall_at_1"],metrics["all"]["mrr"])
  record={"epoch":epoch+1,**{k:v/max(count,1) for k,v in sums.items()},
   "logit_scale":float(model.scale),"development":metrics,"selector":current}
  with history.open("a") as f: f.write(json.dumps(record,sort_keys=True)+"\n")
  save_checkpoint(a.output_dir/"last.pt",model,optimizer,epoch+1,metrics,config,"retrieval")
  if best is None or current>best:
   best=current; stale=0; save_checkpoint(a.output_dir/"best_retrieval.pt",model,optimizer,epoch+1,metrics,config,"retrieval")
   with (a.output_dir/"best_top10.jsonl").open("w") as f:
    for row in rows: f.write(json.dumps(row)+"\n")
  else: stale+=1
  if stale>=a.patience: break
 (a.output_dir/"training_complete.json").write_text(json.dumps({"best_selector":best,"epochs":epoch+1,
  "peak_memory_gib":torch.cuda.max_memory_allocated()/2**30},indent=2)+"\n")

if __name__=="__main__": main()
