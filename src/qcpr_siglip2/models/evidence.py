from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ..config.schema import Siglip2TemporalConfig

@dataclass
class EvidenceOutput:
    evidence_logits: Tensor
    evidence_weights: Tensor
    evidence_map: Tensor
    evidence_vector: Tensor
    evidence_gate: Tensor

class EvidenceBottleneck(nn.Module):
    """One text-token-to-temporal-patch evidence path used by the final score."""
    def __init__(self,config:Siglip2TemporalConfig)->None:
        super().__init__(); self.hidden_size=config.hidden_size; self.evidence_temperature=float(config.evidence_temperature); self.raw_gate=nn.Parameter(torch.tensor(config.evidence_gate_raw_init))
    @property
    def evidence_gate(self)->Tensor: return .5*torch.sigmoid(self.raw_gate)
    def forward(self,text_tokens:Tensor,text_mask:Tensor,visual_tokens:Tensor,*,frame_count:int,patch_count:int)->EvidenceOutput:
        if text_tokens.ndim != 3 or text_mask.ndim != 2 or visual_tokens.ndim != 3: raise ValueError("evidence inputs must be [Q,L,D], [Q,L] and [P,M,D]")
        q,l,d=text_tokens.shape; p,m,vd=visual_tokens.shape
        if d != self.hidden_size or vd != self.hidden_size or text_mask.shape != (q,l): raise ValueError("evidence dimensions do not match hidden size")
        if m != frame_count*patch_count: raise ValueError("visual token count does not match temporal dimensions")
        valid=text_mask.bool()
        if torch.any(valid.sum(dim=1)==0): raise ValueError("every query needs at least one valid text token")
        similarity=torch.einsum("qld,pmd->qplm",F.normalize(text_tokens,dim=-1),F.normalize(visual_tokens,dim=-1)); similarity=similarity.masked_fill(~valid[:,None,:,None],torch.finfo(similarity.dtype).min); logits=torch.logsumexp(similarity,dim=2); weights=torch.softmax(logits/self.evidence_temperature,dim=-1); vector=torch.einsum("qpm,pmd->qpd",weights,visual_tokens); mapped=weights.reshape(q,p,frame_count,patch_count); side=int(math.isqrt(patch_count))
        if side*side != patch_count: raise ValueError("evidence map requires square native patch grid")
        return EvidenceOutput(logits,weights,mapped.reshape(q,p,frame_count,side,side),vector,self.evidence_gate)
