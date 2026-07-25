from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qcpr_single_pass_runtime import collect_gallery,git_sha,save_checkpoint,sha256
from land_change_detection.models.qcpr_single_pass import QueryConditionedLocalizer,multi_positive_sigmoid_loss
from land_change_detection.models.qcpr_single_pass_factory import build_single_pass_retriever
from land_change_detection.training.qcpr_single_pass_data import MaskFreePairDataset

def parse():
 p=argparse.ArgumentParser(); p.add_argument("--retrieval-checkpoint",type=Path,required=True)
 p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--manifest-dir",type=Path,required=True)
 p.add_argument("--universat-source",default="/mnt/weka/svardanyan/rs_change_project/external/UniverSat")
 p.add_argument("--universat-checkpoint",default="/mnt/weka/svardanyan/rs_change_project/models/universat-base")
 p.add_argument("--jina-model",default="/mnt/weka/svardanyan/rs_change_project/models/jina-v5-text-small-retrieval")
 p.add_argument("--output-grid",type=int,default=32); p.add_argument("--epochs",type=int,default=20)
 p.add_argument("--patience",type=int,default=4); p.add_argument("--query-batch",type=int,default=8)
 p.add_argument("--candidates",type=int,default=4); p.add_argument("--workers",type=int,default=8)
 p.add_argument("--lr",type=float,default=1e-4); return p.parse_args()

@torch.no_grad()
def mine(model,dataset,device,workers):
 gallery=collect_gallery(model,dataset,8,workers,device); queries=[]; qvec=[]
 for start in range(0,len(dataset),16):
  items=[dataset[i] for i in range(start,min(start+16,len(dataset)))]
  captions=[]; identities=[]
  for item in items:
   for caption in item.captions[:2]: captions.append(caption); identities.append((item.pair_id,caption))
  enc=model.encode_texts(captions); qvec.append(enc.text_search_vector.cpu()); queries.extend(identities)
 vectors=torch.cat(qvec); scores=vectors@gallery["vectors"].T; order=scores.argsort(1,descending=True)
 index={p:i for i,p in enumerate(gallery["pair_ids"])}; candidates=[]
 for row,(pair_id,caption) in enumerate(queries):
  hard=[int(x) for x in order[row] if int(x)!=index[pair_id]][:3]
  candidates.append({"query_pair_id":pair_id,"caption":caption,"candidate_indices":[index[pair_id]]+hard,
   "candidate_pair_ids":[gallery["pair_ids"][index[pair_id]]]+[gallery["pair_ids"][x] for x in hard]})
 return candidates

def aligned_batch(dataset,rows,start,count,device):
 selected=rows[start:start+count]; texts=[x["caption"] for x in selected]
 items=[] 
 for row in selected:
  items.extend(dataset[i] for i in row["candidate_indices"])
 images=torch.stack([x.images for x in items]).to(device,non_blocking=True)
 return selected,texts,images

def grounded(model,localizer,texts,images,k,device):
 with torch.no_grad():
  text=model.encode_texts(texts).text_search_vector.to(device); patches=model.encode_pairs(images).contextual_patch_tokens
 repeated=text.repeat_interleave(k,0); output=localizer(repeated,patches)
 scores=model.scale.detach()*output.grounded_score.reshape(len(texts),k)
 positive=torch.zeros_like(scores,dtype=torch.bool); positive[:,0]=True
 return multi_positive_sigmoid_loss(scores,positive)[0],scores,output

def dev_score(model,localizer,dataset,rows,batch,k,device):
 correct=0; reciprocal=0.
 with torch.no_grad():
  for start in range(0,len(rows),batch):
   selected,texts,images=aligned_batch(dataset,rows,start,batch,device)
   _,scores,_=grounded(model,localizer,texts,images,k,device)
   ranks=scores.argsort(1,descending=True)
   true=(ranks==0).nonzero()[:,1]+1; correct+=int((true==1).sum()); reciprocal+=float((1/true.float()).sum())
 return {"grounding_recall_at_1":correct/len(rows),"grounding_mrr":reciprocal/len(rows)}

def main():
 a=parse(); device=torch.device("cuda"); a.output_dir.mkdir(parents=True,exist_ok=True)
 train_path=a.manifest_dir/"natural_train_retrieval_manifest.jsonl"; val_path=a.manifest_dir/"natural_validation_retrieval_manifest.jsonl"
 train=MaskFreePairDataset(train_path,"train"); val=MaskFreePairDataset(val_path,"val")
 model=build_single_pass_retriever(universat_source=a.universat_source,universat_checkpoint=a.universat_checkpoint,
  jina_model=a.jina_model,device=device,output_grid=a.output_grid)
 payload=torch.load(a.retrieval_checkpoint,map_location=device,weights_only=False)
 if payload.get("role")!="retrieval": raise RuntimeError("localization requires retrieval model checkpoint, not optimizer resume")
 model.load_state_dict(payload["model"]); model.eval()
 for p in model.parameters(): p.requires_grad_(False)
 localizer=QueryConditionedLocalizer(grid_size=a.output_grid).to(device)
 optimizer=torch.optim.AdamW(localizer.parameters(),lr=a.lr,weight_decay=.05)
 train_rows=mine(model,train,device,a.workers); val_rows=mine(model,val,device,a.workers)
 with (a.output_dir/"hard_candidates.jsonl").open("w") as f:
  for row in train_rows: f.write(json.dumps(row)+"\n")
 config=vars(a)|{"git_sha":git_sha(),"retrieval_checkpoint_sha256":sha256(a.retrieval_checkpoint),
  "load_segmentation_targets":False,"optimizer_lineage":"fresh_localization_optimizer","candidate_pairs":4}
 # Embedded first real batch and in-process OOM fallback.
 actual=a.query_batch
 while True:
  optimizer.zero_grad(set_to_none=True)
  try:
   torch.cuda.reset_peak_memory_stats(); _,texts,images=aligned_batch(train,train_rows,0,actual,device)
   with torch.autocast("cuda",dtype=torch.bfloat16): loss,scores,out=grounded(model,localizer,texts,images,a.candidates,device)
   loss.backward(); grads=[p.grad for p in localizer.parameters() if p.grad is not None]
   if not torch.isfinite(loss) or not grads or not all(torch.isfinite(x).all() for x in grads): raise RuntimeError("first batch gradient check")
   if out.soft_map.shape[-2:]!=(a.output_grid,a.output_grid): raise RuntimeError("localization spatial contract")
   peak=torch.cuda.max_memory_allocated()/2**30; optimizer.zero_grad(set_to_none=True); break
  except torch.cuda.OutOfMemoryError:
   optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
   if actual!=a.query_batch: raise
   actual=4
 config.update({"actual_query_batch":actual,"candidate_pairs_per_query":a.candidates,"first_batch_peak_gib":peak})
 (a.output_dir/"run_config.json").write_text(json.dumps(config,indent=2,sort_keys=True,default=str)+"\n")
 best=None; stale=0
 for epoch in range(a.epochs):
  localizer.train(); permutation=torch.randperm(len(train_rows)).tolist(); ordered=[train_rows[i] for i in permutation]
  total=0.; steps=0
  for start in range(0,len(ordered),actual):
   _,texts,images=aligned_batch(train,ordered,start,actual,device); optimizer.zero_grad(set_to_none=True)
   with torch.autocast("cuda",dtype=torch.bfloat16): loss,_,_=grounded(model,localizer,texts,images,a.candidates,device)
   loss.backward(); torch.nn.utils.clip_grad_norm_(localizer.parameters(),1.); optimizer.step(); total+=float(loss); steps+=1
  localizer.eval(); metrics=dev_score(model,localizer,val,val_rows,actual,a.candidates,device)
  with (a.output_dir/"metrics.jsonl").open("a") as f: f.write(json.dumps({"epoch":epoch+1,"loss":total/steps,**metrics})+"\n")
  selector=(metrics["grounding_recall_at_1"],metrics["grounding_mrr"])
  save_checkpoint(a.output_dir/"last.pt",model,optimizer,epoch+1,metrics,config,"localization",localizer)
  if best is None or selector>best:
   best=selector; stale=0; save_checkpoint(a.output_dir/"best_localization.pt",model,optimizer,epoch+1,metrics,config,"localization",localizer)
  else: stale+=1
  if stale>=a.patience: break
 (a.output_dir/"training_complete.json").write_text(json.dumps({"best_selector":best,"epochs":epoch+1},indent=2)+"\n")
if __name__=="__main__": main()
