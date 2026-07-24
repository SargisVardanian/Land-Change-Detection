from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F

class ReZeroTextBlock(nn.Module):
    def __init__(self, dim:int=512, heads:int=8, ffn_dim:int=2048):
        super().__init__()
        self.attn_norm=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.ffn_norm=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,ffn_dim),nn.GELU(),nn.Linear(ffn_dim,dim))
        self.alpha_attn=nn.Parameter(torch.zeros(()))
        self.alpha_ffn=nn.Parameter(torch.zeros(()))
    def forward(self,x:Tensor,key_padding_mask:Tensor|None=None)->Tensor:
        y,_=self.attn(self.attn_norm(x),self.attn_norm(x),self.attn_norm(x),key_padding_mask=key_padding_mask,need_weights=False)
        x=x+self.alpha_attn*y
        return x+self.alpha_ffn*self.ffn(self.ffn_norm(x))

class SharedTextAttentionAdapter(nn.Module):
    def __init__(self,dim:int=512,heads:int=8,ffn_dim:int=2048,depth:int=2):
        super().__init__()
        self.blocks=nn.ModuleList([ReZeroTextBlock(dim,heads,ffn_dim) for _ in range(depth)])
        self.base_projection=nn.Identity()
        self.adapted_projection=nn.Linear(dim,dim)
        self.attention_query=nn.Parameter(torch.randn(dim)*0.02)
        self.beta_text=nn.Parameter(torch.zeros(()))
        nn.init.eye_(self.adapted_projection.weight)
        nn.init.zeros_(self.adapted_projection.bias)
    @staticmethod
    def safe_attention_mask(attention_mask:Tensor)->Tensor:
        attention=attention_mask.bool().clone()
        attention[:, 0] |= ~attention.any(dim=1)
        return attention
    @classmethod
    def safe_content_mask(cls,content_mask:Tensor,attention_mask:Tensor)->Tensor:
        attention=cls.safe_attention_mask(attention_mask)
        content=content_mask.bool() & attention
        return torch.where(~content.any(dim=1,keepdim=True), attention, content)
    def forward(self,base_global:Tensor,tokens:Tensor,attention_mask:Tensor,content_mask:Tensor)->tuple[Tensor,Tensor,Tensor]:
        attention=self.safe_attention_mask(attention_mask)
        key_padding_mask=~attention
        adapted=tokens
        for block in self.blocks:
            adapted=block(adapted,key_padding_mask)
        valid=self.safe_content_mask(content_mask,attention_mask)
        weights=valid.to(adapted.dtype).unsqueeze(-1)
        mean=(adapted*weights).sum(dim=1)/weights.sum(dim=1).clamp_min(1.0)
        scores=torch.einsum("bld,d->bl",F.normalize(adapted,dim=-1),F.normalize(self.attention_query,dim=0))
        scores=scores.masked_fill(~valid,float("-inf"))
        attention_weights=torch.softmax(scores,dim=1)
        attended=torch.einsum("bl,bld->bd",attention_weights,adapted)
        pooled=0.5*(mean+attended)
        global_embedding=F.normalize(self.base_projection(base_global)+self.beta_text*self.adapted_projection(pooled),dim=-1)
        return global_embedding,adapted,self.safe_attention_mask(attention_mask)
