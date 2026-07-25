from __future__ import annotations
import json, random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset
from land_change_detection.data.unichange_mci import _load_rgb

@dataclass(frozen=True)
class RetrievalItem:
    pair_id:str; dataset_name:str; images:torch.Tensor; captions:list[str]
    normalized_captions:list[str]; change_status:str; metadata:dict[str,Any]

class MaskFreePairDataset(Dataset):
    """Physical-pair dataset that never opens or returns segmentation targets."""
    def __init__(self,manifest:str|Path,split:str,image_size:int=256):
        self.manifest=Path(manifest); self.image_size=image_size; self.samples=[]
        with self.manifest.open(encoding="utf-8") as handle:
            for line in handle:
                row=json.loads(line)
                if row.get("split")!=split: continue
                captions=[str(x) for x in row.get("captions",[]) if str(x).strip()]
                if captions: self.samples.append(row)
        self.samples.sort(key=lambda x:str(x["pair_id"]))
        self.pair_ids=[str(x["pair_id"]) for x in self.samples]
        if len(self.pair_ids)!=len(set(self.pair_ids)): raise ValueError("duplicate physical pair ids")
    def __len__(self): return len(self.samples)
    def __getitem__(self,index:int)->RetrievalItem:
        row=self.samples[index]
        # Deliberately no access to mask_path/semantic paths.
        images=torch.stack((_load_rgb(row["t1_path"],self.image_size),_load_rgb(row["t2_path"],self.image_size)))
        captions=[str(x) for x in row["captions"] if str(x).strip()]
        normalized=[" ".join(str(x).casefold().strip(" .").split()) for x in captions]
        status="no_change" if row.get("source_metadata",{}).get("changeflag")==0 else "changed"
        return RetrievalItem(str(row["pair_id"]),str(row["dataset_name"]),images,captions,normalized,status,
            {"t1_path":row["t1_path"],"t2_path":row["t2_path"],"split":row["split"]})

class CaptionCollisionIndex:
    def __init__(self,audit_path:str|Path):
        clusters=defaultdict(set); caption_cluster={}
        with Path(audit_path).open(encoding="utf-8") as handle:
            for line in handle:
                row=json.loads(line); pair=str(row.get("pair_id","")); caption=str(row.get("normalized_caption",""))
                cluster=str(row.get("duplicate_caption_cluster",""))
                if pair and caption and cluster: clusters[cluster].add(pair); caption_cluster[(pair,caption)]=cluster
        self.caption_cluster=caption_cluster; self.clusters=dict(clusters)
    def collisions(self,pair_id:str,normalized_caption:str)->set[str]:
        cluster=self.caption_cluster.get((pair_id,normalized_caption))
        return set() if cluster is None else set(self.clusters.get(cluster,set()))-{pair_id}

class PairCaptionCollator:
    def __init__(self,captions_per_pair:int=2,seed:int=20260725,epoch:int=0):
        self.count=captions_per_pair; self.seed=seed; self.epoch=epoch
    def __call__(self,items:list[RetrievalItem])->dict[str,Any]:
        images=torch.stack([x.images for x in items]); captions=[]; query_pair_ids=[]; normalized=[]
        for item in items:
            rng=random.Random(f"{self.seed}:{self.epoch}:{item.pair_id}")
            indices=list(range(len(item.captions))); rng.shuffle(indices)
            chosen=(indices*((self.count+len(indices)-1)//len(indices)))[:self.count]
            for index in chosen:
                captions.append(item.captions[index]); normalized.append(item.normalized_captions[index])
                query_pair_ids.append(item.pair_id)
        batch={"images":images,"pair_ids":[x.pair_id for x in items],"captions":captions,
               "normalized_captions":normalized,"query_pair_ids":query_pair_ids,
               "change_status":[x.change_status for x in items],"metadata":[x.metadata for x in items]}
        forbidden={"mask","masks","query_masks","semantic_targets","segmentation_targets"}
        if forbidden&batch.keys(): raise RuntimeError("mask tensor entered a mask-free batch")
        return batch

class AllCaptionCollator:
    def __call__(self,items:list[RetrievalItem])->dict[str,Any]:
        captions=[]; query_pair_ids=[]; normalized=[]
        for item in items:
            captions.extend(item.captions); normalized.extend(item.normalized_captions)
            query_pair_ids.extend([item.pair_id]*len(item.captions))
        return {"images":torch.stack([x.images for x in items]),"pair_ids":[x.pair_id for x in items],
                "captions":captions,"normalized_captions":normalized,"query_pair_ids":query_pair_ids,
                "change_status":[x.change_status for x in items],"metadata":[x.metadata for x in items]}
