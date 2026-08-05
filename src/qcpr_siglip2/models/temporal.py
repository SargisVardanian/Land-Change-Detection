from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ..config.schema import Siglip2TemporalConfig

class LayerScaleTransformerBlock(nn.Module):
    def __init__(self, config: Siglip2TemporalConfig)->None:
        super().__init__(); self.norm_attn=nn.LayerNorm(config.hidden_size); self.attention=nn.MultiheadAttention(config.hidden_size,config.attention_heads,dropout=config.dropout,batch_first=True); self.attn_scale=nn.Parameter(torch.full((config.hidden_size,),config.layer_scale_init)); self.norm_ffn=nn.LayerNorm(config.hidden_size); self.ffn=nn.Sequential(nn.Linear(config.hidden_size,config.mlp_size),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(config.mlp_size,config.hidden_size)); self.ffn_scale=nn.Parameter(torch.full((config.hidden_size,),config.layer_scale_init))
    def forward(self,hidden: Tensor)->Tensor:
        normalized=self.norm_attn(hidden); attention,_=self.attention(normalized,normalized,normalized,need_weights=False); hidden=hidden+attention*self.attn_scale; return hidden+self.ffn(self.norm_ffn(hidden))*self.ffn_scale

@dataclass
class TemporalAdapterOutput:
    pair_cls: Tensor
    frame_cls: Tensor
    temporal_patch_tokens: Tensor
    pair_initial: Tensor
    frame_count: int
    patch_count: int

class TemporalTransformerAdapter(nn.Module):
    """Exactly two pre-norm temporal blocks above frozen SigLIP-2 tokens."""
    def __init__(self,config: Siglip2TemporalConfig)->None:
        super().__init__(); config.validate(); self.config=config; self.frame_position=nn.Parameter(torch.zeros(config.max_frames,config.hidden_size)); self.frame_type=nn.Parameter(torch.zeros(1,1,config.hidden_size)); self.patch_type=nn.Parameter(torch.zeros(1,1,config.hidden_size)); self.spatial_position=nn.Parameter(torch.zeros(1,config.hidden_size,config.base_grid,config.base_grid)); self.time_projection=nn.Linear(2,config.hidden_size,bias=False); nn.init.zeros_(self.time_projection.weight); nn.init.trunc_normal_(self.frame_position,std=.02); nn.init.trunc_normal_(self.spatial_position,std=.02); self.blocks=nn.ModuleList([LayerScaleTransformerBlock(config) for _ in range(config.temporal_layers)])
    def _spatial_position(self,n:int,*,device:torch.device,dtype:torch.dtype)->Tensor:
        side=int(math.isqrt(n))
        if side*side != n: raise ValueError("native patch count must form a square grid")
        value=F.interpolate(self.spatial_position.float(),size=(side,side),mode="bicubic",align_corners=False); return value.flatten(2).transpose(1,2).to(device=device,dtype=dtype)
    def forward(self,frame_tokens:Tensor,frame_embeddings:Tensor,timestamps:Tensor|None=None)->TemporalAdapterOutput:
        if frame_tokens.ndim != 4 or frame_embeddings.ndim != 3: raise ValueError("frame tokens must be [B,T,N,D] and frame embeddings [B,T,D]")
        b,t,n,d=frame_tokens.shape
        if d != self.config.hidden_size or frame_embeddings.shape != (b,t,d): raise ValueError("temporal feature shapes do not match config")
        if t<2 or t>self.config.max_frames: raise ValueError("frame count outside supported range")
        pair_initial=F.normalize(frame_embeddings.sum(dim=1),dim=-1); spatial=self._spatial_position(n,device=frame_tokens.device,dtype=frame_tokens.dtype)
        if timestamps is None: tf=torch.zeros((b,t,2),device=frame_tokens.device,dtype=frame_tokens.dtype)
        else:
            if timestamps.shape != (b,t): raise ValueError("timestamps must be [B,T]")
            first=timestamps[:,:1]; delta=timestamps-first; scale=delta.abs().amax(dim=1,keepdim=True).clamp_min(1.0); tf=torch.stack((timestamps-first,delta/scale),dim=-1).to(frame_tokens.dtype)
        # Metadata features may follow BF16 backbone activations while this
        # small projection is retained in its module dtype.  Normalize the
        # input to the projection dtype, then return the learned bias in the
        # token dtype for mixed-precision-safe residual addition.
        time_bias=self.time_projection(tf.to(dtype=self.time_projection.weight.dtype)).to(dtype=frame_tokens.dtype); seq=[pair_initial.unsqueeze(1)]
        for i in range(t):
            frame_bias=self.frame_position[i].view(1,1,-1).to(frame_tokens.dtype); time=time_bias[:,i:i+1]; seq.extend((frame_embeddings[:,i:i+1]+frame_bias+self.frame_type.to(frame_tokens.dtype)+time,frame_tokens[:,i]+spatial+self.patch_type.to(frame_tokens.dtype)+frame_bias+time))
        hidden=torch.cat(seq,dim=1)
        for block in self.blocks: hidden=block(hidden)
        pair=F.normalize(hidden[:,0],dim=-1); frames=[]; patches=[]; offset=1
        for _ in range(t): frames.append(hidden[:,offset]); patches.append(hidden[:,offset+1:offset+1+n]); offset+=1+n
        return TemporalAdapterOutput(pair,torch.stack(frames,dim=1),torch.cat(patches,dim=1),pair_initial,t,n)
