from __future__ import annotations
from typing import Iterable
import torch
from torch import Tensor

def _ranks(scores:Tensor,relevance:Tensor)->Tensor:
    if scores.ndim != 2 or relevance.shape != scores.shape: raise ValueError("scores and relevance must have equal [Q,P] shapes")
    ranked=relevance.gather(1,scores.argsort(dim=1,descending=True)); first=ranked.float().argmax(dim=1)+1; missing=ranked.sum(dim=1)==0; return torch.where(missing,torch.full_like(first,scores.shape[1]+1),first)

def candidate_hit_at_k(scores:Tensor,relevance:Tensor,k:int)->float:
    if k<=0: raise ValueError("k must be positive")
    order=scores.argsort(dim=1,descending=True)[:,:k]; return float(relevance.gather(1,order).any(dim=1).float().mean())

def multi_positive_recall_at_k(scores:Tensor,relevance:Tensor,k:int)->float:
    if k<=0: raise ValueError("k must be positive")
    order=scores.argsort(dim=1,descending=True)[:,:k]; hits=relevance.gather(1,order).sum(dim=1).float(); total=relevance.sum(dim=1).float().clamp_min(1.); return float((hits/total).mean())

def precision_at_k(scores:Tensor,relevance:Tensor,k:int)->float:
    if k<=0: raise ValueError("k must be positive")
    order=scores.argsort(dim=1,descending=True)[:,:k]; return float(relevance.gather(1,order).sum(dim=1).float().div(float(k)).mean())

def mrr_full(scores:Tensor,relevance:Tensor)->float: return float((1./_ranks(scores,relevance).float()).mean())
def mrr_at_k(scores:Tensor,relevance:Tensor,k:int)->float:
    ranks=_ranks(scores,relevance); return float(torch.where(ranks<=k,1./ranks.float(),torch.zeros_like(ranks,dtype=torch.float32)).mean())
def map_at_k(scores:Tensor,relevance:Tensor,k:int)->float:
    if k<=0: raise ValueError("k must be positive")
    order=scores.argsort(dim=1,descending=True)[:,:k]; ranked=relevance.gather(1,order).float(); cumulative=ranked.cumsum(1); pos=torch.arange(1,k+1,device=scores.device,dtype=torch.float32).view(1,-1); denom=relevance.sum(1).float().clamp_min(1.); return float(((cumulative/pos*ranked).sum(1)/denom).mean())
def full_gallery_metrics(scores:Tensor,relevance:Tensor,ks:Iterable[int]=(1,5,10,50,100,500))->dict[str,float]:
    result={"mrr_full":mrr_full(scores,relevance)}
    for k in ks:
        kk=min(int(k),scores.shape[1]); result[f"candidate_hit_at_{k}"]=candidate_hit_at_k(scores,relevance,kk); result[f"multi_positive_recall_at_{k}"]=multi_positive_recall_at_k(scores,relevance,kk); result[f"precision_at_{k}"]=precision_at_k(scores,relevance,kk); result[f"mrr_at_{k}"]=mrr_at_k(scores,relevance,kk); result[f"map_at_{k}"]=map_at_k(scores,relevance,kk)
    return result
