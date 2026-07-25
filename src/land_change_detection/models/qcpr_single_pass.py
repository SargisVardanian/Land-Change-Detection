from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
from torch.nn import functional as F

@dataclass(frozen=True)
class SinglePassConfig:
    visual_dim:int=768; text_dim:int=512; d_model:int=384; retrieval_dim:int=512
    grid_size:int=32; layers:int=6; heads:int=6; mlp_ratio:int=4; dropout:float=.1

@dataclass(frozen=True)
class PairEncoding:
    pair_cls:Tensor; contextual_patch_tokens:Tensor; pair_search_vector:Tensor
    per_time_tokens:Tensor; metadata:dict[str,Any]

@dataclass(frozen=True)
class TextEncoding:
    text_search_vector:Tensor; contextual_text_tokens:Tensor
    attention_mask:Tensor; content_mask:Tensor; base_text_embedding:Tensor
    metadata:dict[str,Any]

@dataclass(frozen=True)
class RetrievalOutput:
    pair:PairEncoding; text:TextEncoding; score_matrix:Tensor

@dataclass(frozen=True)
class LocalizationOutput:
    patch_relevance_logits:Tensor; patch_relevance_probability:Tensor
    soft_map:Tensor; upsampled_soft_map:Tensor|None
    regional_search_vector:Tensor; grounded_score:Tensor
    cross_attention_weights:Tensor

class BiTemporalPairTransformer(nn.Module):
    def __init__(self,cfg:SinglePassConfig):
        super().__init__(); self.cfg=cfg
        if cfg.grid_size < 1: raise ValueError("grid_size must be positive")
        self.temporal_projection=nn.Linear(cfg.visual_dim*5,cfg.d_model)
        self.cls_pair=nn.Parameter(torch.zeros(1,1,cfg.d_model))
        self.position_2d=nn.Parameter(torch.zeros(1,cfg.grid_size**2,cfg.d_model))
        self.cls_position=nn.Parameter(torch.zeros(1,1,cfg.d_model))
        layer=nn.TransformerEncoderLayer(cfg.d_model,cfg.heads,cfg.d_model*cfg.mlp_ratio,
            cfg.dropout,"gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,cfg.layers,enable_nested_tensor=False)
        self.final_norm=nn.LayerNorm(cfg.d_model); self.pair_projection=nn.Linear(cfg.d_model,cfg.retrieval_dim)
        nn.init.trunc_normal_(self.cls_pair,std=.02); nn.init.trunc_normal_(self.position_2d,std=.02)
        nn.init.trunc_normal_(self.cls_position,std=.02)

    def forward(self,x:Tensor)->PairEncoding:
        if x.ndim!=4 or x.shape[1:]!=(2,self.cfg.grid_size**2,self.cfg.visual_dim):
            raise ValueError(f"expected [B,2,{self.cfg.grid_size**2},{self.cfg.visual_dim}], got {tuple(x.shape)}")
        a,b=x[:,0],x[:,1]; d=b-a
        z=self.temporal_projection(torch.cat((a,b,d,d.abs(),a*b),-1))+self.position_2d
        cls=self.cls_pair.expand(x.shape[0],-1,-1)+self.cls_position
        y=self.final_norm(self.encoder(torch.cat((cls,z),1)))
        if y.shape[1]!=1+self.cfg.grid_size**2: raise RuntimeError("pair sequence length mismatch")
        return PairEncoding(y[:,0],y[:,1:],F.normalize(self.pair_projection(y[:,0]),dim=-1),x,
            {"grid_size":self.cfg.grid_size,"patch_count":self.cfg.grid_size**2,"sequence_length":1+self.cfg.grid_size**2})

class TextSearchProjection(nn.Module):
    """Two-layer adapter over frozen Jina tokens, seeded by Jina pooled CLS/EOS."""
    def __init__(self,cfg:SinglePassConfig):
        super().__init__(); self.cfg=cfg
        layer=nn.TransformerEncoderLayer(cfg.text_dim,8,2048,cfg.dropout,"gelu",batch_first=True,norm_first=True)
        self.adapter=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.final_norm=nn.LayerNorm(cfg.text_dim)
        self.projection=nn.Linear(cfg.text_dim,cfg.retrieval_dim)
        if cfg.text_dim==cfg.retrieval_dim: nn.init.eye_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    def forward(self,base:Tensor,tokens:Tensor,attention:Tensor,content:Tensor,metadata=None)->TextEncoding:
        if base.ndim!=2 or base.shape[-1]!=self.cfg.text_dim: raise ValueError("text pooled shape")
        if tokens.shape[:2]!=attention.shape or content.shape!=attention.shape: raise ValueError("text mask shape")
        sequence=torch.cat((base.unsqueeze(1),tokens),dim=1)
        valid=torch.cat((torch.ones(base.shape[0],1,dtype=torch.bool,device=attention.device),attention.bool()),dim=1)
        adapted=self.final_norm(self.adapter(sequence,src_key_padding_mask=~valid))
        return TextEncoding(F.normalize(self.projection(adapted[:,0]),dim=-1),adapted[:,1:],attention.bool(),
            content.bool(),base,metadata or {})

class QCPRSinglePassRetriever(nn.Module):
    def __init__(self,visual_encoder:nn.Module,text_encoder:nn.Module,cfg:SinglePassConfig=SinglePassConfig()):
        super().__init__(); self.cfg=cfg; self.visual_encoder=visual_encoder; self.text_encoder=text_encoder
        self.pair_encoder=BiTemporalPairTransformer(cfg); self.text_projection=TextSearchProjection(cfg)
        self.logit_scale=nn.Parameter(torch.tensor(1/.07).log()); self.freeze_backbones()
    def freeze_backbones(self):
        for m in (self.visual_encoder,self.text_encoder):
            for p in m.parameters(): p.requires_grad_(False)
            m.eval()
        # The base Jina transformer stays frozen; its external 1024->512 token
        # projection belongs to the permitted trainable text adapter path.
        local_projection=getattr(self.text_encoder,"local_projection",None)
        if local_projection is not None:
            for p in local_projection.parameters(): p.requires_grad_(True)
    def train(self,mode=True):
        super().train(mode); self.visual_encoder.eval(); self.text_encoder.eval(); return self
    @property
    def scale(self): return self.logit_scale.exp().clamp(max=100.)
    def encode_pairs(self,images:Tensor)->PairEncoding:
        with torch.no_grad(): f=self.visual_encoder(images)
        out=self.pair_encoder(f.features if hasattr(f,"features") else f)
        meta=dict(out.metadata); meta.update(getattr(f,"metadata",{}))
        return PairEncoding(out.pair_cls,out.contextual_patch_tokens,out.pair_search_vector,out.per_time_tokens,meta)
    def encode_texts(self,captions:list[str])->TextEncoding:
        with torch.no_grad(): f=self.text_encoder(captions,role="query")
        return self.text_projection(f.global_embedding,f.token_embeddings,f.attention_mask,
            f.content_token_mask,getattr(f,"metadata",{}))
    def forward(self,images:Tensor,captions:list[str])->RetrievalOutput:
        p=self.encode_pairs(images); t=self.encode_texts(captions); dev=p.pair_search_vector.device
        t=TextEncoding(t.text_search_vector.to(dev),t.contextual_text_tokens.to(dev),
            t.attention_mask.to(dev),t.content_mask.to(dev),t.base_text_embedding.to(dev),t.metadata)
        return RetrievalOutput(p,t,self.scale*(t.text_search_vector@p.pair_search_vector.T))

class QueryPatchAttentionBlock(nn.Module):
    def __init__(self,d:int=384,heads:int=6,dropout:float=.1):
        super().__init__(); self.qn=nn.LayerNorm(d); self.pn=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True)
        self.fn=nn.LayerNorm(d); self.ffn=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*d,d))
    def forward(self,q:Tensor,p:Tensor):
        pn=self.pn(p); update,w=self.attn(self.qn(q),pn,pn,need_weights=True,average_attn_weights=False)
        q=q+update; return q+self.ffn(self.fn(q)),w

class QueryConditionedLocalizer(nn.Module):
    def __init__(self,text_dim=512,patch_dim=384,retrieval_dim=512,heads=6,layers=2,dropout=.1,grid_size=32):
        super().__init__(); self.grid_size=grid_size; self.patch_dim=patch_dim; self.query_projection=nn.Linear(text_dim,patch_dim)
        self.blocks=nn.ModuleList(QueryPatchAttentionBlock(patch_dim,heads,dropout) for _ in range(layers))
        self.patch_bias=nn.Sequential(nn.LayerNorm(patch_dim),nn.Linear(patch_dim,patch_dim),nn.GELU(),nn.Linear(patch_dim,1))
        self.regional_projection=nn.Linear(patch_dim,retrieval_dim)
    def forward(self,text:Tensor,patches:Tensor,output_size:tuple[int,int]|None=None)->LocalizationOutput:
        if patches.ndim!=3 or patches.shape[1:]!=(self.grid_size**2,self.patch_dim): raise ValueError("patches must be [B,G*G,384]")
        if text.shape[0]!=patches.shape[0]: raise ValueError("aligned query-pair batches only")
        q=self.query_projection(text).unsqueeze(1); history=[]
        for block in self.blocks: q,w=block(q,patches); history.append(w.squeeze(2))
        agree=torch.einsum("bd,bpd->bp",F.normalize(q.squeeze(1),dim=-1),F.normalize(patches,dim=-1))
        prior=torch.stack(history).mean((0,2)).clamp_min(1e-8).log()
        logits=agree+self.patch_bias(patches).squeeze(-1)+prior; prob=logits.softmax(-1)
        region=torch.einsum("bp,bpd->bd",prob,patches)
        regional=F.normalize(self.regional_projection(region),dim=-1)
        score=(regional*F.normalize(text,dim=-1)).sum(-1); soft=prob.reshape(-1,self.grid_size,self.grid_size)
        up=None if output_size is None else F.interpolate(soft[:,None],output_size,mode="bilinear",align_corners=False)[:,0]
        return LocalizationOutput(logits,prob,soft,up,regional,score,torch.stack(history,1))

def multi_positive_sigmoid_loss(scores:Tensor,positive_mask:Tensor,excluded_mask:Tensor|None=None):
    if scores.shape!=positive_mask.shape: raise ValueError("mask shape")
    pos=positive_mask.bool(); exc=torch.zeros_like(pos) if excluded_mask is None else excluded_mask.bool()
    valid=(~exc)|pos; targets=torch.where(pos,torch.ones_like(scores),-torch.ones_like(scores))
    loss=(-F.logsigmoid(targets*scores)).masked_select(valid).mean(); neg=valid&~pos
    stats={"positive_similarity":scores.masked_select(pos).mean(),
           "negative_similarity":scores.masked_select(neg).mean(),
           "positive_entries":pos.sum(),"valid_entries":valid.sum()}
    return loss,stats

def build_pair_masks(query_pair_ids:list[str],gallery_pair_ids:list[str],
                     query_collision_pair_ids:list[set[str]]|None=None,device=None):
    pos=torch.tensor([[q==p for p in gallery_pair_ids] for q in query_pair_ids],dtype=torch.bool,device=device)
    exc=torch.zeros_like(pos)
    if query_collision_pair_ids:
        for i,collisions in enumerate(query_collision_pair_ids):
            for j,pair_id in enumerate(gallery_pair_ids):
                if pair_id!=query_pair_ids[i] and pair_id in collisions: exc[i,j]=True
    return pos,exc
