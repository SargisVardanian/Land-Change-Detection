from __future__ import annotations
import hashlib,json,math,os,random,subprocess
import numpy as np
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader
from land_change_detection.models.qcpr_single_pass import build_pair_masks
from land_change_detection.training.qcpr_single_pass_data import AllCaptionCollator,PairCaptionCollator

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def forbidden_mask_keys(batch):
    return sorted(set(batch)&{"mask","masks","query_masks","semantic_targets","segmentation_targets"})

@torch.no_grad()
def collect_gallery(model,dataset,batch_size,workers,device,with_patches=False):
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,
        collate_fn=PairCaptionCollator(1),pin_memory=True)
    vectors=[]; patches=[]; ids=[]; statuses=[]
    for batch in loader:
        if forbidden_mask_keys(batch): raise RuntimeError("mask data in retrieval batch")
        encoded=model.encode_pairs(batch["images"].to(device,non_blocking=True))
        vectors.append(encoded.pair_search_vector.cpu())
        if with_patches: patches.append(encoded.contextual_patch_tokens.cpu())
        ids.extend(batch["pair_ids"]); statuses.extend(batch["change_status"])
    return {"vectors":torch.cat(vectors),"patches":torch.cat(patches) if patches else None,
            "pair_ids":ids,"change_status":statuses}

@torch.no_grad()
def collect_queries(model,dataset,batch_size,workers,device):
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,
        collate_fn=AllCaptionCollator(),pin_memory=True)
    vectors=[]; captions=[]; pair_ids=[]; normalized=[]
    for batch in loader:
        text=model.encode_texts(batch["captions"])
        vectors.append(text.text_search_vector.cpu()); captions.extend(batch["captions"])
        pair_ids.extend(batch["query_pair_ids"]); normalized.extend(batch["normalized_captions"])
    return {"vectors":torch.cat(vectors),"captions":captions,"pair_ids":pair_ids,"normalized":normalized}

def exact_metrics(scores,query_pair_ids,gallery_pair_ids,statuses=None):
    order=scores.argsort(dim=1,descending=True); index={p:i for i,p in enumerate(gallery_pair_ids)}
    ranks=torch.tensor([(order[i]==index[p]).nonzero()[0,0].item()+1 for i,p in enumerate(query_pair_ids)])
    def pack(mask):
        r=ranks[mask]; return {"queries":int(r.numel()),"recall_at_1":float((r<=1).float().mean()),
          "recall_at_5":float((r<=5).float().mean()),"recall_at_10":float((r<=10).float().mean()),
          "mrr":float((1/r.float()).mean()),"mean_rank":float(r.float().mean()),
          "median_rank":float(r.float().median()),"ndcg_at_10":float(torch.where(r<=10,1/torch.log2(r.float()+1),0).mean())}
    result={"all":pack(torch.ones_like(ranks,dtype=torch.bool))}
    if statuses:
        qstatus=[statuses[index[p]] for p in query_pair_ids]
        for name in ("changed","no_change"):
            mask=torch.tensor([x==name for x in qstatus])
            if mask.any(): result[name]=pack(mask)
    return result,ranks,order[:,:10]

def evaluate_retrieval(model,dataset,batch_size,workers,device):
    gallery=collect_gallery(model,dataset,batch_size,workers,device)
    queries=collect_queries(model,dataset,batch_size,workers,device)
    scores=queries["vectors"]@gallery["vectors"].T
    metrics,ranks,top10=exact_metrics(scores,queries["pair_ids"],gallery["pair_ids"],gallery["change_status"])
    rows=[]
    for i in range(len(queries["captions"])):
        rows.append({"query_id":f"query:{i}","query":queries["captions"][i],
          "true_pair_id":queries["pair_ids"][i],"top10_pair_ids":[gallery["pair_ids"][j] for j in top10[i]],
          "top10_scores":[float(scores[i,j]) for j in top10[i]],"true_pair_rank":int(ranks[i])})
    return metrics,rows

def save_checkpoint(path,model,optimizer,epoch,metrics,config,role,localizer=None,
                    scheduler=None,global_step=0,extra=None):
    payload={"role":role,"model":model.state_dict(),
      "optimizer":optimizer.state_dict() if optimizer else None,
      "scheduler":scheduler.state_dict() if scheduler else None,
      "scaler":None,"epoch":epoch,"global_step":global_step,
      "metrics":metrics,"config":config,
      "rng":{"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               "python":random.getstate(),"numpy":np.random.get_state()}}
    if localizer is not None: payload["localizer"]=localizer.state_dict()
    if extra: payload.update(extra)
    torch.save(payload,path)
    loaded=torch.load(path,map_location="cpu",weights_only=False)
    if loaded["role"]!=role or loaded["epoch"]!=epoch: raise RuntimeError("checkpoint round-trip mismatch")

def git_sha():
    return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
