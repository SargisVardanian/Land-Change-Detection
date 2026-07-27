from __future__ import annotations
import argparse,json,math,random,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qcpr_single_pass_runtime import collect_gallery,git_sha,save_checkpoint,sha256
from land_change_detection.models.qcpr_single_pass import QueryConditionedLocalizer,balanced_siglip_loss
from land_change_detection.models.qcpr_single_pass_factory import build_single_pass_retriever
from land_change_detection.training.qcpr_single_pass_data import CaptionCollisionIndex,MaskFreePairDataset

def arguments():
 p=argparse.ArgumentParser(); p.add_argument("--retrieval-checkpoint",type=Path,required=True)
 p.add_argument("--retrieval-acceptance",type=Path,required=True)
 p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--manifest-dir",type=Path,required=True)
 p.add_argument("--universat-source",default="/mnt/weka/svardanyan/rs_change_project/external/UniverSat")
 p.add_argument("--universat-checkpoint",default="/mnt/weka/svardanyan/rs_change_project/models/universat-base")
 p.add_argument("--jina-model",default="/mnt/weka/svardanyan/rs_change_project/models/jina-v5-text-small-retrieval")
 p.add_argument("--output-grid",type=int,default=32); p.add_argument("--epochs",type=int,default=20)
 p.add_argument("--min-epochs",type=int,default=6); p.add_argument("--patience",type=int,default=5)
 p.add_argument("--query-batch",type=int,default=8); p.add_argument("--candidates",type=int,default=4)
 p.add_argument("--hard-cache-size",type=int,default=32)
 p.add_argument("--validation-candidates",type=int,default=16)
 p.add_argument("--accumulation",type=int,default=2); p.add_argument("--workers",type=int,default=8)
 p.add_argument("--weight-decay",type=float,default=.05); p.add_argument("--seed",type=int,default=20260725)
 p.add_argument("--resume",type=Path); return p.parse_args()

def no_decay(name,p):
 lowered=name.casefold()
 return p.ndim<=1 or lowered.endswith("bias") or "norm" in lowered or "logit_" in lowered

def make_optimizer(localizer,weight_decay):
 groups=[]; specifications=[
  ("grounding_attention",[(n,p) for n,p in localizer.named_parameters()
     if any(key in n for key in ("global_query_projection","text_token_projection","blocks",
                                  "relevance_query_projection","relevance_key_projection","relevance_bias"))],1e-4),
  ("grounded_projection",[(n,p) for n,p in localizer.named_parameters() if "dense_value_projection" in n],2e-4),
  ("grounding_logits",[("logit_scale",localizer.logit_scale),("logit_bias",localizer.logit_bias)],1e-4)]
 covered=set()
 for label,named,lr in specifications:
  for decay in (True,False):
   parameters=[p for n,p in named if (not no_decay(n,p))==decay and id(p) not in covered]
   covered.update(id(p) for p in parameters)
   if parameters: groups.append({"name":label+("/decay" if decay else "/no_decay"),"params":parameters,
                                 "lr":lr,"weight_decay":weight_decay if decay else 0.})
 intended={id(p) for p in localizer.parameters() if p.requires_grad}
 if covered!=intended: raise RuntimeError("grounding optimizer coverage mismatch")
 return torch.optim.AdamW(groups)

@torch.no_grad()
def mine(model,dataset,collisions,device,workers,candidate_count):
 gallery=collect_gallery(model,dataset,16,workers,device); queries=[]; qvec=[]
 for start in range(0,len(dataset),16):
  items=[dataset[i] for i in range(start,min(start+16,len(dataset)))]
  captions=[]; identities=[]
  for item in items:
   for caption,normalized in list(zip(item.captions,item.normalized_captions,strict=True))[:2]:
    captions.append(caption); identities.append((item.pair_id,caption,normalized))
  encoded=model.encode_texts(captions); qvec.append(encoded.text_search_vector.cpu()); queries.extend(identities)
 vectors=torch.cat(qvec); scores=vectors@gallery["vectors"].T; order=scores.argsort(1,descending=True)
 index={pair:i for i,pair in enumerate(gallery["pair_ids"])}; rows=[]
 for row,(pair_id,caption,normalized) in enumerate(queries):
  excluded=collisions.collisions(pair_id,normalized)|{pair_id}; hard=[]
  for candidate in order[row].tolist():
   if gallery["pair_ids"][candidate] not in excluded: hard.append(int(candidate))
   if len(hard)==candidate_count-1: break
  if len(hard)!=candidate_count-1: raise RuntimeError("insufficient valid retrieval hard negatives")
  indices=[index[pair_id]]+hard
  rows.append({"query_pair_id":pair_id,"caption":caption,"normalized_caption_group":normalized,
   "candidate_indices":indices,"candidate_pair_ids":[gallery["pair_ids"][i] for i in indices],
   "candidate_scores":[float(scores[row,i]) for i in indices]})
 return rows

def sample_candidate_subsets(rows,candidate_count,seed):
 sampled=[]
 for row in rows:
  if len(row["candidate_indices"])<candidate_count: raise RuntimeError("hard-negative cache too small")
  order=list(range(1,len(row["candidate_indices"])))
  random.Random(f"{seed}:{row['query_pair_id']}:{row['normalized_caption_group']}").shuffle(order)
  chosen=[0]+order[:candidate_count-1]
  copied=dict(row)
  for key in ("candidate_indices","candidate_pair_ids","candidate_scores"):
   copied[key]=[row[key][index] for index in chosen]
  sampled.append(copied)
 return sampled

def grouped_epoch_rows(rows,seed):
 groups=defaultdict(list)
 for row in rows: groups[row["query_pair_id"]].append(row)
 keys=list(groups); random.Random(seed).shuffle(keys)
 return [row for key in keys for row in groups[key]]

def aligned_batch(dataset,rows,start,count,device):
 selected=rows[start:start+count]; items=[]
 for row in selected: items.extend(dataset[i] for i in row["candidate_indices"])
 return selected,[row["caption"] for row in selected],torch.stack([item.images for item in items]).to(device,non_blocking=True)

def combined_grounding_loss(scores,text_vectors,grounded,rows,k,localizer):
 query_count=len(rows); base_positive=torch.zeros_like(scores,dtype=torch.bool); base_positive[:,0]=True
 base_negative=~base_positive; flat_scores=[scores.reshape(-1)]; flat_positive=[base_positive.reshape(-1)]
 flat_negative=[base_negative.reshape(-1)]
 true_grounded=grounded[torch.arange(query_count,device=grounded.device)*k]
 same=localizer.scale*(text_vectors@true_grounded.T)+localizer.logit_bias
 same_positive=torch.zeros((query_count,query_count),dtype=torch.bool,device=same.device)
 same_negative=torch.zeros_like(same_positive)
 for i,left in enumerate(rows):
  for j,right in enumerate(rows):
   if i!=j and left["query_pair_id"]==right["query_pair_id"] and left["normalized_caption_group"]!=right["normalized_caption_group"]:
    same_negative[i,j]=True
 if same_negative.any():
  involved=same_negative.any(dim=0)|same_negative.any(dim=1)
  same_positive[torch.arange(query_count,device=same.device)[involved],torch.arange(query_count,device=same.device)[involved]]=True
  flat_scores.append(same.reshape(-1)); flat_positive.append(same_positive.reshape(-1)); flat_negative.append(same_negative.reshape(-1))
 all_scores=torch.cat(flat_scores); positive=torch.cat(flat_positive); negative=torch.cat(flat_negative)
 loss,stats=balanced_siglip_loss(all_scores[None],positive[None],negative[None])
 stats["same_pair_query_negatives"]=same_negative.sum()
 return loss,stats

def evidence_diagnostics(localizer,out,repeated_text,patches,k):
 probabilities=out.relevance_probabilities; count=probabilities.shape[1]; keep=max(count//10,1)
 top=probabilities.topk(keep,dim=-1).indices; mask=torch.zeros_like(probabilities,dtype=torch.bool)
 mask.scatter_(1,top,True); values=localizer.dense_value_projection(patches)
 def score(weights):
  weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
  embedding=torch.nn.functional.normalize(torch.einsum("bn,bnd->bd",weights,values),dim=-1)
  return localizer.scale*(torch.nn.functional.normalize(repeated_text,dim=-1)*embedding).sum(-1)+localizer.logit_bias
 insertion=score(probabilities*mask); deletion=score(probabilities*(~mask)); uniform=score(torch.ones_like(probabilities))
 true_rows=torch.arange(0,probabilities.shape[0],k,device=probabilities.device)
 argmax=probabilities.argmax(-1); fixed=float(torch.bincount(argmax,minlength=count).max()/argmax.numel())
 return {"normalized_entropy":float(out.diagnostics["entropy"].mean()/math.log(count)),
  "effective_patch_count":float(out.diagnostics["effective_patch_count"].mean()),
  "concentration":float(out.diagnostics["concentration"].mean()),
  "spatial_variance":float(out.diagnostics["spatial_variance"].mean()),
  "insertion_score":float(insertion.mean()),"deletion_score":float(deletion.mean()),
  "deletion_drop":float((out.grounded_score[true_rows]-deletion[true_rows]).mean()),
  "uniform_score":float(uniform[true_rows].mean()),
  "learned_uniform_margin":float((out.grounded_score[true_rows]-uniform[true_rows]).mean()),"fixed_location_ratio":fixed}

def grounded(model,localizer,rows,texts,images,k,device):
 with torch.no_grad():
  text=model.encode_texts(texts); pair=model.encode_pairs(images)
  query=text.text_search_vector.to(device); tokens=text.contextual_text_tokens.to(device)
  attention=text.attention_mask.to(device); content=text.content_mask.to(device)
 repeated_query=query.repeat_interleave(k,0); repeated_tokens=tokens.repeat_interleave(k,0)
 repeated_attention=attention.repeat_interleave(k,0); repeated_content=content.repeat_interleave(k,0)
 output=localizer(repeated_query,repeated_tokens,repeated_attention,repeated_content,pair.adapted_dense_tokens)
 scores=output.grounded_score.reshape(len(texts),k)
 loss,stats=combined_grounding_loss(scores,query,output.grounded_visual_embedding,rows,k,localizer)
 diagnostics=evidence_diagnostics(localizer,output,repeated_query,pair.adapted_dense_tokens,k)
 if len(rows)>1:
  query_changes=[]; pair_changes=[]
  maps=output.native_pooling_map.reshape(len(rows),k,localizer.grid_size,localizer.grid_size)
  for i in range(len(rows)):
   pair_changes.append((maps[i,0]-maps[i,1]).abs().mean())
   for j in range(i+1,len(rows)):
    if (rows[i]["query_pair_id"]==rows[j]["query_pair_id"] and
        rows[i]["normalized_caption_group"]!=rows[j]["normalized_caption_group"]):
     query_changes.append((maps[i,0]-maps[j,0]).abs().mean())
  diagnostics["same_query_different_pair_map_change"]=float(torch.stack(pair_changes).mean())
  diagnostics["same_pair_different_query_map_change"]=float(torch.stack(query_changes).mean()) if query_changes else 0.
 return loss,scores,output,stats,diagnostics

@torch.no_grad()
def development(model,localizer,dataset,rows,batch,k,device):
 reciprocal=0.; recall1=0; diagnostics=defaultdict(float); total=0
 for start in range(0,len(rows),batch):
  selected,texts,images=aligned_batch(dataset,rows,start,batch,device)
  loss,scores,out,stats,diag=grounded(model,localizer,selected,texts,images,k,device)
  ranks=(scores.argsort(1,descending=True)==0).nonzero()[:,1]+1
  reciprocal+=float((1/ranks.float()).sum()); recall1+=int((ranks==1).sum()); total+=len(selected)
  for key,value in diag.items(): diagnostics[key]+=value*len(selected)
 return {"grounding_mrr":reciprocal/total,"grounding_recall_at_1":recall1/total,
         **{key:value/total for key,value in diagnostics.items()}}

def main():
 a=arguments(); torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
 torch.set_float32_matmul_precision("high"); device=torch.device("cuda"); a.output_dir.mkdir(parents=True,exist_ok=True)
 train_path=a.manifest_dir/"natural_train_retrieval_manifest.jsonl"; val_path=a.manifest_dir/"natural_validation_retrieval_manifest.jsonl"
 collision_path=a.manifest_dir/"caption_quality_audit.jsonl"
 train=MaskFreePairDataset(train_path,"train"); val=MaskFreePairDataset(val_path,"val"); collisions=CaptionCollisionIndex(collision_path)
 model=build_single_pass_retriever(universat_source=a.universat_source,universat_checkpoint=a.universat_checkpoint,
  jina_model=a.jina_model,device=device,output_grid=a.output_grid)
 acceptance=json.loads(a.retrieval_acceptance.read_text())
 checkpoint_hash=sha256(a.retrieval_checkpoint)
 if not acceptance.get("passed"): raise RuntimeError("retrieval scientific acceptance gate failed")
 if acceptance.get("checkpoint_sha256")!=checkpoint_hash: raise RuntimeError("retrieval acceptance SHA mismatch")
 payload=torch.load(a.retrieval_checkpoint,map_location=device,weights_only=False)
 if payload.get("role")!="retrieval": raise RuntimeError("grounding requires accepted retrieval checkpoint")
 model.load_state_dict(payload["model"]); model.eval()
 for parameter in model.parameters(): parameter.requires_grad_(False)
 localizer=QueryConditionedLocalizer(grid_size=a.output_grid).to(device); optimizer=make_optimizer(localizer,a.weight_decay)
 train_rows=mine(model,train,collisions,device,a.workers,a.hard_cache_size)
 val_rows=mine(model,val,collisions,device,a.workers,a.validation_candidates)
 with (a.output_dir/"hard_candidates.jsonl").open("w") as handle:
  for row in train_rows: handle.write(json.dumps(row)+"\n")
 steps_per_epoch=math.ceil(math.ceil(len(train_rows)/a.query_batch)/a.accumulation); total_steps=steps_per_epoch*a.epochs
 warmup=max(int(total_steps*.05),1)
 def schedule(step):
  if step<warmup: return max(step,1)/warmup
  progress=min(max((step-warmup)/max(total_steps-warmup,1),0.),1.); return .5*(1+math.cos(math.pi*progress))
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
 start_epoch=0; global_step=0; actual_batch=a.query_batch; actual_accum=a.accumulation
 if a.resume:
  resumed=torch.load(a.resume,map_location=device,weights_only=False)
  if resumed.get("role")!="grounding": raise RuntimeError("grounding resumes only a grounding checkpoint")
  if resumed.get("accepted_retrieval_sha256")!=sha256(a.retrieval_checkpoint): raise RuntimeError("retrieval lineage mismatch")
  localizer.load_state_dict(resumed["localizer"]); optimizer.load_state_dict(resumed["optimizer"]); scheduler.load_state_dict(resumed["scheduler"])
  start_epoch=int(resumed["epoch"]); global_step=int(resumed.get("global_step",0))
  rng=resumed.get("rng",{}); torch.set_rng_state(rng.get("torch",torch.get_rng_state()).cpu())
  if torch.cuda.is_available() and rng.get("cuda"): torch.cuda.set_rng_state_all(rng["cuda"])
  if rng.get("python"): random.setstate(rng["python"])
  if rng.get("numpy"): np.random.set_state(rng["numpy"])
  contract=resumed.get("batch_contract",{}); actual_batch=int(contract.get("grounding_query_batch",actual_batch))
  actual_accum=int(contract.get("grounding_accumulation",actual_accum))
 config=vars(a)|{"git_sha":git_sha(),"retrieval_checkpoint_sha256":sha256(a.retrieval_checkpoint),
  "load_segmentation_targets":False,"optimizer_lineage":"fresh_grounding_optimizer",
  "scientific_loss":"balanced_multi_positive_siglip_grounded_region_only"}
 while True:
  sampled=sample_candidate_subsets(train_rows,a.candidates,a.seed)
  rows=grouped_epoch_rows(sampled,a.seed)[:actual_batch]; selected,texts,images=aligned_batch(train,rows,0,actual_batch,device)
  optimizer.zero_grad(set_to_none=True)
  try:
   torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
   with torch.autocast("cuda",dtype=torch.bfloat16):
    loss,scores,out,stats,diag=grounded(model,localizer,selected,texts,images,a.candidates,device)
   (loss/actual_accum).backward(); grads=[p.grad for p in localizer.parameters() if p.grad is not None]
   if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads): raise RuntimeError("grounding first-batch check")
   allocated=torch.cuda.max_memory_allocated()/2**30; reserved=torch.cuda.max_memory_reserved()/2**30
   optimizer.zero_grad(set_to_none=True); break
  except torch.cuda.OutOfMemoryError:
   optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
   if (actual_batch,actual_accum)!=(a.query_batch,a.accumulation): raise
   actual_batch,actual_accum=4,4
 props=torch.cuda.get_device_properties(device)
 validation_query_batch=max(1,min(actual_batch,32//a.validation_candidates))
 retrieval_contract=payload.get("batch_contract",{})
 batch_contract={"gpu_name":props.name,"total_vram_gib":props.total_memory/2**30,
  "temporal_series_length":retrieval_contract.get("temporal_series_length",2),
  "retrieval_physical_batch":retrieval_contract.get("physical_batch"),
  "retrieval_captions_per_item":retrieval_contract.get("captions_per_item"),
  "retrieval_accumulation":retrieval_contract.get("accumulation_steps"),
  "native_grid":[a.output_grid,a.output_grid],"dense_token_count":a.output_grid**2,
  "grounding_query_batch":actual_batch,"training_candidates_per_query":a.candidates,
  "hard_negative_cache_size":a.hard_cache_size,"validation_candidates_per_query":a.validation_candidates,
  "validation_query_batch":validation_query_batch,"validation_combinations_per_forward":validation_query_batch*a.validation_candidates,
  "query_candidate_combinations":actual_batch*a.candidates,"grounding_accumulation":actual_accum,
  "peak_allocated_gib":allocated,"peak_reserved_gib":reserved,"oom_fallback_used":actual_batch!=a.query_batch}
 (a.output_dir/"batch_contract.json").write_text(json.dumps(batch_contract,indent=2,sort_keys=True)+"\n")
 (a.output_dir/"run_config.json").write_text(json.dumps(config,indent=2,sort_keys=True,default=str)+"\n")
 best=None; best_metrics=None; stale=0
 for epoch in range(start_epoch,a.epochs):
  localizer.train(); sampled=sample_candidate_subsets(train_rows,a.candidates,a.seed+epoch)
  ordered=grouped_epoch_rows(sampled,a.seed+epoch); optimizer.zero_grad(set_to_none=True)
  sums=defaultdict(float); steps=0
  for start in range(0,len(ordered),actual_batch):
   selected,texts,images=aligned_batch(train,ordered,start,actual_batch,device)
   with torch.autocast("cuda",dtype=torch.bfloat16):
    loss,scores,out,stats,diag=grounded(model,localizer,selected,texts,images,a.candidates,device)
   (loss/actual_accum).backward(); steps+=1; sums["loss"]+=float(loss)
   sums["positive_loss"]+=float(stats["positive_loss"]); sums["negative_loss"]+=float(stats["negative_loss"])
   sums["same_pair_query_negatives"]+=float(stats["same_pair_query_negatives"])
   for key,value in diag.items(): sums[key]+=value
   if steps%actual_accum==0:
    torch.nn.utils.clip_grad_norm_(localizer.parameters(),1.); optimizer.step(); scheduler.step()
    optimizer.zero_grad(set_to_none=True); global_step+=1
  if steps%actual_accum:
   torch.nn.utils.clip_grad_norm_(localizer.parameters(),1.); optimizer.step(); scheduler.step()
   optimizer.zero_grad(set_to_none=True); global_step+=1
  localizer.eval(); metrics=development(model,localizer,val,val_rows,validation_query_batch,a.validation_candidates,device)
  record={"epoch":epoch+1,**{key:value/max(steps,1) for key,value in sums.items()},
          "grounding_logit_scale":float(localizer.scale),"grounding_logit_bias":float(localizer.logit_bias),
          "development":metrics}
  with (a.output_dir/"metrics.jsonl").open("a") as handle: handle.write(json.dumps(record,sort_keys=True)+"\n")
  selector=(metrics["grounding_mrr"],metrics["grounding_recall_at_1"])
  extra={"accepted_retrieval_checkpoint":str(a.retrieval_checkpoint),
         "accepted_retrieval_sha256":sha256(a.retrieval_checkpoint),"batch_contract":batch_contract,
         "grounding_candidate_configuration":{"training_candidates":a.candidates,"cache_size":a.hard_cache_size,
          "validation_candidates":a.validation_candidates,"source":"frozen_retriever_scores"}}
  save_checkpoint(a.output_dir/"last.pt",model,optimizer,epoch+1,metrics,config,"grounding",localizer,
                  scheduler=scheduler,global_step=global_step,extra=extra)
  if best is None or selector>best:
   best=selector; best_metrics=metrics; stale=0
   save_checkpoint(a.output_dir/"best_grounding.pt",model,optimizer,epoch+1,metrics,config,"grounding",localizer,
                   scheduler=scheduler,global_step=global_step,extra=extra)
  else: stale+=1
  if epoch+1>=a.min_epochs and stale>=a.patience: break
 (a.output_dir/"training_complete.json").write_text(json.dumps({"best_selector":best,"epochs":epoch+1,
  "global_step":global_step,"batch_contract":batch_contract},indent=2)+"\n")
 checkpoint=a.output_dir/"best_grounding.pt"
 gates={"learned_map_beats_uniform":best_metrics["learned_uniform_margin"]>0,
        "top_evidence_deletion_lowers_score":best_metrics["deletion_drop"]>0,
        "non_equivalent_queries_change_map":best_metrics["same_pair_different_query_map_change"]>1e-6}
 grounding_acceptance={"passed":bool(all(gates.values())),"gates":gates,"metrics":best_metrics,
  "checkpoint_path":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),
  "map_semantics":{"pooling_weights":"softmax distribution used by grounded retrieval loss",
                   "soft_relevance":"sigmoid logits; uncalibrated query-conditioned relevance, not segmentation probability"},
  "selection_candidate_count":a.validation_candidates,
  "note":"Failure is preserved as a non-collapse result; no handcrafted loss is added automatically."}
 (a.output_dir/"grounding_acceptance.json").write_text(json.dumps(grounding_acceptance,indent=2,sort_keys=True)+"\n")

if __name__=="__main__": main()
