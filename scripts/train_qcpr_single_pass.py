from __future__ import annotations
import argparse,hashlib,json,math,random,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qcpr_single_pass_runtime import evaluate_retrieval,forbidden_mask_keys,git_sha,save_checkpoint,sha256
from land_change_detection.models.qcpr_single_pass import build_pair_masks,multi_positive_sigmoid_loss
from land_change_detection.models.qcpr_single_pass_factory import build_single_pass_retriever
from land_change_detection.training.qcpr_single_pass_data import CaptionCollisionIndex,MaskFreePairDataset,PairCaptionCollator

def arguments():
 p=argparse.ArgumentParser()
 p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--manifest-dir",type=Path,required=True)
 p.add_argument("--universat-source",default="/mnt/weka/svardanyan/rs_change_project/external/UniverSat")
 p.add_argument("--universat-checkpoint",default="/mnt/weka/svardanyan/rs_change_project/models/universat-base")
 p.add_argument("--jina-model",default="/mnt/weka/svardanyan/rs_change_project/models/jina-v5-text-small-retrieval")
 p.add_argument("--output-grid",type=int,default=32); p.add_argument("--epochs",type=int,default=25)
 p.add_argument("--min-epochs",type=int,default=8); p.add_argument("--patience",type=int,default=5)
 p.add_argument("--workers",type=int,default=8); p.add_argument("--batch-size",type=int,default=16)
 p.add_argument("--accumulation",type=int,default=4); p.add_argument("--weight-decay",type=float,default=.05)
 p.add_argument("--mrr-improvement-tolerance",type=float,default=1e-8)
 p.add_argument("--recall-regression-tolerance",type=float,default=.002)
 p.add_argument("--seed",type=int,default=20260725); p.add_argument("--resume",type=Path)
 return p.parse_args()

def loader(dataset,batch,workers,epoch,seed):
 return DataLoader(dataset,batch_size=batch,shuffle=True,num_workers=workers,pin_memory=True,
  collate_fn=PairCaptionCollator(2,seed,epoch),drop_last=True)

def no_decay(name,parameter):
 lowered=name.casefold()
 return parameter.ndim<=1 or lowered.endswith("bias") or "norm" in lowered or "layer_scale" in lowered or "logit_" in lowered

def optimizer_groups(model,weight_decay):
 specifications=[
  ("dense_token_adapters",list(model.pair_encoder.token_adapters.named_parameters()),5e-5),
  ("pair_cross_attention",list(model.pair_encoder.pair_blocks.named_parameters()),1e-4),
  ("pair_output",[(name,getattr(model.pair_encoder,name)) for name in ()],2e-4)]
 output=[]; covered=set()
 def add(label,named,lr):
  decay=[]; nodecay=[]
  for name,parameter in named:
   if not parameter.requires_grad or id(parameter) in covered: continue
   covered.add(id(parameter)); (nodecay if no_decay(name,parameter) else decay).append(parameter)
  if decay: output.append({"name":label+"/decay","params":decay,"lr":lr,"weight_decay":weight_decay})
  if nodecay: output.append({"name":label+"/no_decay","params":nodecay,"lr":lr,"weight_decay":0.})
 add("dense_token_adapters",model.pair_encoder.token_adapters.named_parameters(),5e-5)
 add("pair_cross_attention",model.pair_encoder.pair_blocks.named_parameters(),1e-4)
 pair_output=[]
 for name in ("pair_token","baseline_norm","baseline_projection","delta_norm","delta_projection"):
  value=getattr(model.pair_encoder,name)
  pair_output.extend([(name,value)] if isinstance(value,torch.nn.Parameter) else
                     [(name+"."+sub,p) for sub,p in value.named_parameters()])
 add("pair_output",pair_output,2e-4)
 text=list(model.text_projection.named_parameters())
 local=getattr(model.text_encoder,"local_projection",None)
 if local is not None: text.extend(("jina_local."+name,p) for name,p in local.named_parameters())
 add("text_adapter",text,5e-5)
 add("logits",[("logit_scale",model.logit_scale),("logit_bias",model.logit_bias)],1e-4)
 intended={id(p) for p in model.parameters() if p.requires_grad}
 if covered!=intended: raise RuntimeError(f"optimizer coverage mismatch missing={len(intended-covered)} extra={len(covered-intended)}")
 return output

def frozen_fingerprint(module):
 h=hashlib.sha256()
 for name,p in module.named_parameters():
  if p.requires_grad: continue
  flat=p.detach().reshape(-1)
  sample=torch.cat((flat[:min(128,flat.numel())],flat[-min(128,flat.numel()):])).float().cpu()
  h.update(name.encode()); h.update(sample.numpy().tobytes())
 return h.hexdigest()

def grad_norm(module):
 values=[p.grad.detach().float().norm()**2 for p in module.parameters() if p.grad is not None]
 return float(torch.stack(values).sum().sqrt()) if values else 0.

def train_step(model,batch,collisions,device,accumulation):
 if forbidden_mask_keys(batch): raise RuntimeError("segmentation target in retrieval batch")
 with torch.autocast("cuda",dtype=torch.bfloat16):
  out=model(batch["images"].to(device,non_blocking=True),batch["captions"])
  collision_sets=[collisions.collisions(pid,text) for pid,text in zip(
      batch["query_pair_ids"],batch["normalized_captions"],strict=True)]
  positive,excluded=build_pair_masks(batch["query_pair_ids"],batch["pair_ids"],collision_sets,device)
  loss,stats=multi_positive_sigmoid_loss(out.score_matrix,positive,excluded)
 (loss/accumulation).backward()
 cosine=out.text.text_search_vector@out.pair.pair_search_vector.T
 valid_negative=(~positive)&(~excluded)
 stats=stats|{"mean_positive_cosine":cosine.masked_select(positive).mean(),
             "mean_negative_cosine":cosine.masked_select(valid_negative).mean(),
             "text_embedding_norm":out.text.text_search_vector.norm(dim=-1).mean(),
             "pair_embedding_norm":out.pair.pair_search_vector.norm(dim=-1).mean()}
 return loss,stats,out

def main():
 a=arguments(); torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
 torch.set_float32_matmul_precision("high"); device=torch.device("cuda"); a.output_dir.mkdir(parents=True,exist_ok=True)
 train_manifest=a.manifest_dir/"natural_train_retrieval_manifest.jsonl"
 val_manifest=a.manifest_dir/"natural_validation_retrieval_manifest.jsonl"; collision_path=a.manifest_dir/"caption_quality_audit.jsonl"
 train=MaskFreePairDataset(train_manifest,"train"); val=MaskFreePairDataset(val_manifest,"val")
 if set(train.pair_ids)&set(val.pair_ids): raise RuntimeError("train/dev leakage")
 collisions=CaptionCollisionIndex(collision_path)
 model=build_single_pass_retriever(universat_source=a.universat_source,universat_checkpoint=a.universat_checkpoint,
  jina_model=a.jina_model,device=device,output_grid=a.output_grid)
 optimizer=torch.optim.AdamW(optimizer_groups(model,a.weight_decay))
 optimizer_steps_per_epoch=math.ceil(math.ceil(len(train)/a.batch_size)/a.accumulation)
 total_steps=optimizer_steps_per_epoch*a.epochs; warmup=max(int(total_steps*.05),1)
 def schedule(step):
  if step<warmup: return max(step,1)/warmup
  progress=min(max((step-warmup)/max(total_steps-warmup,1),0.),1.)
  return .5*(1+math.cos(math.pi*progress))
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
 start_epoch=0; global_step=0; actual_batch=a.batch_size; actual_accum=a.accumulation
 baseline_metrics=None
 if a.resume:
  payload=torch.load(a.resume,map_location=device,weights_only=False)
  if payload.get("role")!="retrieval": raise RuntimeError("retrieval can resume only retrieval checkpoint")
  model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
  start_epoch=int(payload["epoch"]); global_step=int(payload.get("global_step",0))
  rng=payload.get("rng",{}); torch.set_rng_state(rng.get("torch",torch.get_rng_state()).cpu())
  if torch.cuda.is_available() and rng.get("cuda"): torch.cuda.set_rng_state_all(rng["cuda"])
  if rng.get("python"): random.setstate(rng["python"])
  if rng.get("numpy"): np.random.set_state(rng["numpy"])
  contract=payload.get("batch_contract",{}); actual_batch=int(contract.get("physical_batch",actual_batch))
  actual_accum=int(contract.get("accumulation_steps",actual_accum))
  baseline_metrics=payload.get("retrieval_baseline_metrics")
 if baseline_metrics is None:
  model.eval(); baseline_metrics,_=evaluate_retrieval(model,val,min(16,actual_batch),a.workers,device)
 config=vars(a)|{"git_sha":git_sha(),"train_manifest_sha256":sha256(train_manifest),
  "val_manifest_sha256":sha256(val_manifest),"collision_audit_sha256":sha256(collision_path),
  "load_segmentation_targets":False,"captions_per_item":2,"scientific_loss":"balanced_multi_positive_siglip",
  "model_config":asdict(model.cfg)}
 (a.output_dir/"run_config.json").write_text(json.dumps(config,indent=2,sort_keys=True,default=str)+"\n")
 visual_before=frozen_fingerprint(model.visual_encoder); text_before=frozen_fingerprint(model.text_encoder)
 while True:
  first=next(iter(loader(train,actual_batch,a.workers,start_epoch,a.seed))); optimizer.zero_grad(set_to_none=True)
  try:
   torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
   loss,stats,out=train_step(model,first,collisions,device,actual_accum)
   expected_shape=(actual_batch*2,actual_batch)
   if out.score_matrix.shape!=expected_shape or len(first["captions"])!=actual_batch*2:
    raise RuntimeError(f"fixed caption/query contract violated: expected {expected_shape}")
   grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
   if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads): raise RuntimeError("first batch finite-gradient check")
   if out.pair.adapted_dense_tokens.shape[1:]!=(a.output_grid**2,model.cfg.visual_dim): raise RuntimeError("native dense shape")
   if frozen_fingerprint(model.visual_encoder)!=visual_before or frozen_fingerprint(model.text_encoder)!=text_before:
    raise RuntimeError("frozen backbone fingerprint changed")
   allocated=torch.cuda.max_memory_allocated()/2**30; reserved=torch.cuda.max_memory_reserved()/2**30
   optimizer.zero_grad(set_to_none=True); break
  except torch.cuda.OutOfMemoryError:
   optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
   fallback=(max(a.batch_size//2,1),a.accumulation*2)
   if (actual_batch,actual_accum)==fallback: raise
   actual_batch,actual_accum=fallback
 props=torch.cuda.get_device_properties(device)
 batch_contract={"gpu_name":props.name,"total_vram_gib":props.total_memory/2**30,
  "temporal_series_length":int(first["images"].shape[1]),"native_grid":[a.output_grid,a.output_grid],
  "dense_token_count":a.output_grid**2,"physical_batch":actual_batch,"captions_per_item":2,
  "text_queries_per_microbatch":actual_batch*2,"accumulation_steps":actual_accum,
  "contrastive_matrix_shape":[actual_batch*2,actual_batch],
  "effective_physical_batch":actual_batch*actual_accum,"peak_allocated_gib":allocated,
  "peak_reserved_gib":reserved,"oom_fallback_used":actual_batch!=a.batch_size}
 (a.output_dir/"batch_contract.json").write_text(json.dumps(batch_contract,indent=2,sort_keys=True)+"\n")
 best=None; best_metrics=None; stale=0; history=a.output_dir/"metrics.jsonl"
 for epoch in range(start_epoch,a.epochs):
  model.train(); optimizer.zero_grad(set_to_none=True)
  sums={key:0. for key in ("loss","positive_loss","negative_loss","positive_cosine","negative_cosine",
                            "text_norm","pair_norm","dense_adapter_grad","pair_attention_grad","pair_output_grad","text_grad")}
  count=0
  for micro,batch in enumerate(loader(train,actual_batch,a.workers,epoch,a.seed),1):
   loss,stats,_=train_step(model,batch,collisions,device,actual_accum)
   sums["loss"]+=float(loss); sums["positive_loss"]+=float(stats["positive_loss"]); sums["negative_loss"]+=float(stats["negative_loss"])
   sums["positive_cosine"]+=float(stats["mean_positive_cosine"]); sums["negative_cosine"]+=float(stats["mean_negative_cosine"])
   sums["text_norm"]+=float(stats["text_embedding_norm"]); sums["pair_norm"]+=float(stats["pair_embedding_norm"])
   sums["dense_adapter_grad"]+=grad_norm(model.pair_encoder.token_adapters)
   sums["pair_attention_grad"]+=grad_norm(model.pair_encoder.pair_blocks)
   sums["pair_output_grad"]+=grad_norm(model.pair_encoder.delta_projection)+grad_norm(model.pair_encoder.baseline_projection)
   sums["text_grad"]+=grad_norm(model.text_projection); count+=1
   if micro%actual_accum==0:
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step+=1
  if count%actual_accum:
   torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
   optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step+=1
  model.eval(); metrics,rows=evaluate_retrieval(model,val,min(16,actual_batch),a.workers,device)
  current=(metrics["all"]["mrr"],-metrics["all"]["median_rank"],metrics["all"]["recall_at_1"],
           metrics["all"]["recall_at_5"],metrics["all"]["recall_at_10"])
  record={"epoch":epoch+1,**{key:value/max(count,1) for key,value in sums.items()},
   "logit_scale":float(model.scale),"logit_bias":float(model.logit_bias),"development":metrics,"selector":current}
  with history.open("a") as handle: handle.write(json.dumps(record,sort_keys=True)+"\n")
  extra={"batch_contract":batch_contract,"native_grid_metadata":out.pair.metadata,
         "retrieval_baseline_metrics":baseline_metrics}
  save_checkpoint(a.output_dir/"last.pt",model,optimizer,epoch+1,metrics,config,"retrieval",
                  scheduler=scheduler,global_step=global_step,extra=extra)
  if best is None or current>best:
   best=current; best_metrics=metrics; stale=0
   save_checkpoint(a.output_dir/"best_retrieval.pt",model,optimizer,epoch+1,metrics,config,"retrieval",
                   scheduler=scheduler,global_step=global_step,extra=extra)
   with (a.output_dir/"best_top10.jsonl").open("w") as handle:
    for row in rows: handle.write(json.dumps(row)+"\n")
  else: stale+=1
  if epoch+1>=a.min_epochs and stale>=a.patience: break
 checkpoint=a.output_dir/"best_retrieval.pt"
 baseline_all=baseline_metrics["all"]; accepted_all=best_metrics["all"]
 mrr_improved=accepted_all["mrr"]>baseline_all["mrr"]+a.mrr_improvement_tolerance
 median_improved=accepted_all["median_rank"]<baseline_all["median_rank"]
 recall_constraints=(accepted_all["recall_at_5"]+a.recall_regression_tolerance>=baseline_all["recall_at_5"] and
                     accepted_all["recall_at_10"]+a.recall_regression_tolerance>=baseline_all["recall_at_10"])
 finite_metrics=all(math.isfinite(float(value)) for metrics in (baseline_metrics,best_metrics)
                    for subset in metrics.values() for value in subset.values())
 checkpoint_valid=checkpoint.is_file()
 acceptance={"passed":bool(mrr_improved and recall_constraints and finite_metrics and checkpoint_valid),
  "mrr_improved":bool(mrr_improved),"median_rank_improved":bool(median_improved),"recall_constraints_passed":bool(recall_constraints),
  "finite_metrics":bool(finite_metrics),"checkpoint_exists":bool(checkpoint_valid),
  "baseline_metrics":baseline_metrics,"accepted_metrics":best_metrics,"checkpoint_path":str(checkpoint),
  "checkpoint_sha256":sha256(checkpoint),
  "mrr_improvement_tolerance":a.mrr_improvement_tolerance,"recall_regression_tolerance":a.recall_regression_tolerance,
  "recall_constraint":"Recall@5 and Recall@10 may fall at most by the explicit tolerance; median rank is diagnostic only"}
 (a.output_dir/"retrieval_acceptance.json").write_text(json.dumps(acceptance,indent=2,sort_keys=True)+"\n")
 (a.output_dir/"training_complete.json").write_text(json.dumps({"best_selector":best,"epochs":epoch+1,
  "global_step":global_step,"batch_contract":batch_contract},indent=2)+"\n")
 if not acceptance["passed"]:
  raise RuntimeError("retrieval scientific acceptance gate failed; grounding chain is blocked")
if __name__=="__main__": main()
