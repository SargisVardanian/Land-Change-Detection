from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

@dataclass(frozen=True)
class SinglePassConfig:
    visual_dim:int=768
    text_dim:int=512
    retrieval_dim:int=512
    grid_size:int=32
    token_adapter_layers:int=2
    token_bottleneck_ratio:int=4
    pair_layers:int=3
    pair_heads:int=12
    ffn_ratio:int=4
    dropout:float=.1
    layer_scale_init:float=1e-3
    text_adapter_layers:int=2
    text_heads:int=8
    max_logit_scale:float=100.0

@dataclass(frozen=True)
class PairEncoding:
    pair_token:Tensor
    adapted_dense_tokens:Tensor
    pair_search_vector:Tensor
    native_dense_tokens:Tensor
    metadata:dict[str,Any]
    @property
    def contextual_patch_tokens(self)->Tensor:
        return self.adapted_dense_tokens

@dataclass(frozen=True)
class TextEncoding:
    text_search_vector:Tensor
    contextual_text_tokens:Tensor
    attention_mask:Tensor
    content_mask:Tensor
    base_text_embedding:Tensor
    metadata:dict[str,Any]

@dataclass(frozen=True)
class RetrievalOutput:
    pair:PairEncoding
    text:TextEncoding
    score_matrix:Tensor

@dataclass(frozen=True)
class LocalizationOutput:
    relevance_logits:Tensor
    relevance_probabilities:Tensor
    soft_relevance_probabilities:Tensor
    grounded_visual_embedding:Tensor
    native_soft_map:Tensor
    native_pooling_map:Tensor
    upsampled_soft_map:Tensor|None
    grounded_score:Tensor
    cross_attention_weights:Tensor
    diagnostics:dict[str,Tensor]
    @property
    def patch_relevance_logits(self)->Tensor: return self.relevance_logits
    @property
    def patch_relevance_probability(self)->Tensor: return self.relevance_probabilities
    @property
    def soft_map(self)->Tensor: return self.native_soft_map
    @property
    def regional_search_vector(self)->Tensor: return self.grounded_visual_embedding

class ResidualBottleneckTokenAdapter(nn.Module):
    """Identity-initialized residual adapter applied independently to dense tokens."""
    def __init__(self,dim:int=768,bottleneck_ratio:int=4,dropout:float=.1):
        super().__init__()
        hidden=max(dim//bottleneck_ratio,1)
        self.norm=nn.LayerNorm(dim)
        self.down=nn.Linear(dim,hidden)
        self.activation=nn.GELU()
        self.dropout=nn.Dropout(dropout)
        self.up=nn.Linear(hidden,dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    def forward(self,x:Tensor)->Tensor:
        return x+self.up(self.dropout(self.activation(self.down(self.norm(x)))))

class PairCrossAttentionBlock(nn.Module):
    """One PAIR query attends linearly over all native dense tokens."""
    def __init__(self,dim:int=768,heads:int=12,ffn_ratio:int=4,dropout:float=.1,layer_scale_init:float=1e-3):
        super().__init__()
        self.query_norm=nn.LayerNorm(dim)
        self.token_norm=nn.LayerNorm(dim)
        self.cross_attention=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True)
        self.attention_layer_scale=nn.Parameter(torch.full((dim,),layer_scale_init))
        self.ffn_norm=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,dim*ffn_ratio),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(dim*ffn_ratio,dim),nn.Dropout(dropout))
        self.ffn_layer_scale=nn.Parameter(torch.full((dim,),layer_scale_init))
    def forward(self,pair:Tensor,tokens:Tensor)->tuple[Tensor,Tensor]:
        normalized=self.token_norm(tokens)
        update,weights=self.cross_attention(self.query_norm(pair),normalized,normalized,
            need_weights=True,average_attn_weights=False)
        pair=pair+self.attention_layer_scale*update
        pair=pair+self.ffn_layer_scale*self.ffn(self.ffn_norm(pair))
        return pair,weights

class DeepResidualPairAdapter(nn.Module):
    """Dense identity adapters plus deep PAIR cross-attention pooling."""
    def __init__(self,cfg:SinglePassConfig):
        super().__init__(); self.cfg=cfg
        if cfg.visual_dim%cfg.pair_heads: raise ValueError("native visual dimension must divide attention heads")
        self.token_adapters=nn.ModuleList(ResidualBottleneckTokenAdapter(
            cfg.visual_dim,cfg.token_bottleneck_ratio,cfg.dropout) for _ in range(cfg.token_adapter_layers))
        self.pair_token=nn.Parameter(torch.zeros(1,1,cfg.visual_dim))
        nn.init.trunc_normal_(self.pair_token,std=.02)
        self.pair_blocks=nn.ModuleList(PairCrossAttentionBlock(cfg.visual_dim,cfg.pair_heads,
            cfg.ffn_ratio,cfg.dropout,cfg.layer_scale_init) for _ in range(cfg.pair_layers))
        self.baseline_norm=nn.LayerNorm(cfg.visual_dim)
        self.baseline_projection=nn.Linear(cfg.visual_dim,cfg.retrieval_dim)
        self.delta_norm=nn.LayerNorm(cfg.visual_dim)
        self.delta_projection=nn.Linear(cfg.visual_dim,cfg.retrieval_dim)
        nn.init.zeros_(self.delta_projection.weight)
        nn.init.zeros_(self.delta_projection.bias)

    def adapt_tokens(self,native:Tensor)->Tensor:
        adapted=native
        for layer in self.token_adapters: adapted=layer(adapted)
        return adapted

    def baseline_vector(self,adapted:Tensor)->Tensor:
        return self.baseline_projection(self.baseline_norm(adapted.mean(dim=1)))

    def forward(self,native:Tensor,metadata:dict[str,Any]|None=None)->PairEncoding:
        if native.ndim!=3 or native.shape[-1]!=self.cfg.visual_dim:
            raise ValueError(f"native dense tokens must be [B,N,{self.cfg.visual_dim}]")
        native=native.detach()
        adapted=self.adapt_tokens(native)
        pair=self.pair_token.expand(native.shape[0],-1,-1)
        for block in self.pair_blocks: pair,_=block(pair,adapted)
        pair=pair[:,0]
        baseline=self.baseline_vector(adapted)
        delta=self.delta_projection(self.delta_norm(pair))
        search=F.normalize(baseline+delta,dim=-1)
        meta=dict(metadata or {})
        meta.update({"dense_token_count":native.shape[1],"native_hidden_dim":native.shape[-1],
                     "token_adapter_layers":len(self.token_adapters),"pair_cross_attention_layers":len(self.pair_blocks),
                     "pair_attention_complexity":"linear_in_dense_tokens"})
        return PairEncoding(pair,adapted,search,native,meta)

class TextResidualAdapterBlock(nn.Module):
    """Pre-normalized ReZero block that is an exact identity at initialization."""
    def __init__(self,dim:int,heads:int,ffn_dim:int=2048,dropout:float=.1):
        super().__init__()
        self.attention_norm=nn.LayerNorm(dim)
        self.attention=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True)
        self.attention_gate=nn.Parameter(torch.zeros(()))
        self.ffn_norm=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,ffn_dim),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(ffn_dim,dim),nn.Dropout(dropout))
        self.ffn_gate=nn.Parameter(torch.zeros(()))
    def forward(self,x:Tensor,padding_mask:Tensor)->Tensor:
        normalized=self.attention_norm(x)
        update,_=self.attention(normalized,normalized,normalized,key_padding_mask=padding_mask,need_weights=False)
        x=x+self.attention_gate*update
        return x+self.ffn_gate*self.ffn(self.ffn_norm(x))

class TextSearchProjection(nn.Module):
    """Identity-anchored adapter over frozen Jina representations."""
    def __init__(self,cfg:SinglePassConfig):
        super().__init__(); self.cfg=cfg
        self.adapter=nn.ModuleList(TextResidualAdapterBlock(cfg.text_dim,cfg.text_heads,2048,cfg.dropout)
                                   for _ in range(cfg.text_adapter_layers))
        self.base_projection=nn.Linear(cfg.text_dim,cfg.retrieval_dim)
        self.delta_norm=nn.LayerNorm(cfg.text_dim)
        self.delta_projection=nn.Linear(cfg.text_dim,cfg.retrieval_dim)
        if cfg.text_dim==cfg.retrieval_dim: nn.init.eye_(self.base_projection.weight)
        nn.init.zeros_(self.base_projection.bias)
        nn.init.zeros_(self.delta_projection.weight)
        nn.init.zeros_(self.delta_projection.bias)
    def forward(self,base:Tensor,tokens:Tensor,attention:Tensor,content:Tensor,metadata=None)->TextEncoding:
        if base.ndim!=2 or base.shape[-1]!=self.cfg.text_dim: raise ValueError("text pooled shape")
        if tokens.shape[:2]!=attention.shape or content.shape!=attention.shape: raise ValueError("text mask shape")
        adapted=tokens
        for block in self.adapter: adapted=block(adapted,~attention.bool())
        pool_mask=content.bool()
        pool_mask=torch.where(pool_mask.any(dim=1,keepdim=True),pool_mask,attention.bool())
        weights=pool_mask.to(adapted.dtype)
        pooled=(adapted*weights.unsqueeze(-1)).sum(1)/weights.sum(1,keepdim=True).clamp_min(1)
        search=F.normalize(self.base_projection(base)+self.delta_projection(self.delta_norm(pooled)),dim=-1)
        return TextEncoding(search,adapted,
            attention.bool(),content.bool(),base,metadata or {})

class QCPRSinglePassRetriever(nn.Module):
    def __init__(self,visual_encoder:nn.Module,text_encoder:nn.Module,cfg:SinglePassConfig=SinglePassConfig()):
        super().__init__(); self.cfg=cfg; self.visual_encoder=visual_encoder; self.text_encoder=text_encoder
        self.pair_encoder=DeepResidualPairAdapter(cfg); self.text_projection=TextSearchProjection(cfg)
        self.logit_scale=nn.Parameter(torch.tensor(10.0).log())
        self.logit_bias=nn.Parameter(torch.tensor(-10.0))
        self.freeze_backbones()
    def freeze_backbones(self):
        for module in (self.visual_encoder,self.text_encoder):
            for parameter in module.parameters(): parameter.requires_grad_(False)
            module.eval()
        local_projection=getattr(self.text_encoder,"local_projection",None)
        if local_projection is not None:
            for parameter in local_projection.parameters(): parameter.requires_grad_(True)
    def train(self,mode=True):
        super().train(mode); self.visual_encoder.eval(); self.text_encoder.eval(); return self
    @property
    def scale(self)->Tensor:
        return self.logit_scale.exp().clamp(min=1e-3,max=self.cfg.max_logit_scale)
    def encode_pairs(self,images:Tensor,dates:Tensor|None=None)->PairEncoding:
        native=self.visual_encoder(images,dates=dates)
        tokens=native.features if hasattr(native,"features") else native
        metadata=dict(getattr(native,"metadata",{}))
        return self.pair_encoder(tokens.detach(),metadata)
    def encode_texts(self,captions:list[str])->TextEncoding:
        features=self.text_encoder(captions,role="query")
        return self.text_projection(features.global_embedding,features.token_embeddings,
            features.attention_mask,features.content_token_mask,getattr(features,"metadata",{}))
    def forward(self,images:Tensor,captions:list[str],dates:Tensor|None=None)->RetrievalOutput:
        pair=self.encode_pairs(images,dates); text=self.encode_texts(captions); device=pair.pair_search_vector.device
        text=TextEncoding(text.text_search_vector.to(device),text.contextual_text_tokens.to(device),
            text.attention_mask.to(device),text.content_mask.to(device),text.base_text_embedding.to(device),text.metadata)
        logits=self.scale*(text.text_search_vector@pair.pair_search_vector.T)+self.logit_bias
        return RetrievalOutput(pair,text,logits)

class GroundingCrossAttentionBlock(nn.Module):
    def __init__(self,dim:int=768,heads:int=12,ffn_ratio:int=4,dropout:float=.1):
        super().__init__(); self.dim=dim; self.heads=heads; self.head_dim=dim//heads
        self.query_norm=nn.LayerNorm(dim); self.token_norm=nn.LayerNorm(dim)
        self.query_projection=nn.Linear(dim,dim); self.key_projection=nn.Linear(dim,dim)
        self.value_projection=nn.Linear(dim,dim); self.output_projection=nn.Linear(dim,dim)
        self.ffn_norm=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,dim*ffn_ratio),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*ffn_ratio,dim))
        self.dropout=nn.Dropout(dropout)
    def forward(self,query:Tensor,tokens:Tensor)->tuple[Tensor,Tensor]:
        batch,count,_=tokens.shape
        q=self.query_projection(self.query_norm(query)).reshape(batch,1,self.heads,self.head_dim).transpose(1,2)
        normalized=self.token_norm(tokens)
        k=self.key_projection(normalized).reshape(batch,count,self.heads,self.head_dim).transpose(1,2)
        v=self.value_projection(normalized).reshape(batch,count,self.heads,self.head_dim).transpose(1,2)
        logits=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
        weights=logits.softmax(dim=-1)
        context=(weights@v).transpose(1,2).reshape(batch,1,self.dim)
        query=query+self.dropout(self.output_projection(context))
        query=query+self.dropout(self.ffn(self.ffn_norm(query)))
        return query,weights.squeeze(2)

class QueryConditionedLocalizer(nn.Module):
    """Grounding score has no global PAIR embedding bypass."""
    def __init__(self,text_dim=512,patch_dim=768,retrieval_dim=512,heads=12,layers=2,
                 ffn_ratio=4,dropout=.1,grid_size=32):
        super().__init__(); self.grid_size=grid_size; self.patch_dim=patch_dim
        self.global_query_projection=nn.Linear(text_dim,patch_dim)
        self.text_token_projection=nn.Linear(text_dim,patch_dim)
        self.blocks=nn.ModuleList(GroundingCrossAttentionBlock(patch_dim,heads,ffn_ratio,dropout)
                                  for _ in range(layers))
        self.relevance_query_projection=nn.Linear(patch_dim,patch_dim)
        self.relevance_key_projection=nn.Linear(patch_dim,patch_dim)
        self.relevance_bias=nn.Sequential(nn.LayerNorm(patch_dim),nn.Linear(patch_dim,1))
        self.dense_value_projection=nn.Linear(patch_dim,retrieval_dim)
        self.logit_scale=nn.Parameter(torch.tensor(10.0).log())
        self.logit_bias=nn.Parameter(torch.tensor(-10.0))

    def _text_query(self,global_text:Tensor,text_tokens:Tensor,content_mask:Tensor,attention_mask:Tensor)->Tensor:
        mask=content_mask.bool()
        fallback=attention_mask.bool()
        mask=torch.where(mask.any(dim=1,keepdim=True),mask,fallback)
        weights=mask.to(text_tokens.dtype); pooled=(text_tokens*weights.unsqueeze(-1)).sum(1)/weights.sum(1,keepdim=True).clamp_min(1)
        return (self.global_query_projection(global_text)+self.text_token_projection(pooled)).unsqueeze(1)

    @property
    def scale(self)->Tensor: return self.logit_scale.exp().clamp(1e-3,100.0)

    def forward(self,text:Tensor,text_tokens:Tensor,attention_mask:Tensor,content_mask:Tensor,
                patches:Tensor,output_size:tuple[int,int]|None=None)->LocalizationOutput:
        if patches.ndim!=3 or patches.shape[-1]!=self.patch_dim: raise ValueError("adapted tokens must be [B,N,Dnative]")
        if text.shape[0]!=patches.shape[0] or text_tokens.shape[0]!=patches.shape[0]:
            raise ValueError("grounding requires aligned query-candidate combinations")
        query=self._text_query(text,text_tokens,content_mask,attention_mask); history=[]
        for block in self.blocks: query,weights=block(query,patches); history.append(weights)
        q=F.normalize(self.relevance_query_projection(query[:,0]),dim=-1)
        k=F.normalize(self.relevance_key_projection(patches),dim=-1)
        relevance=torch.einsum("bd,bnd->bn",q,k)+self.relevance_bias(patches).squeeze(-1)
        probabilities=relevance.softmax(dim=-1)
        soft_relevance=relevance.sigmoid()
        projected=self.dense_value_projection(patches)
        grounded=F.normalize(torch.einsum("bn,bnd->bd",probabilities,projected),dim=-1)
        grounded_score=self.scale*(F.normalize(text,dim=-1)*grounded).sum(-1)+self.logit_bias
        count=patches.shape[1]
        if self.grid_size*self.grid_size!=count: raise ValueError("native grid metadata/token count mismatch")
        pooling_map=probabilities.reshape(-1,self.grid_size,self.grid_size)
        soft_map=soft_relevance.reshape(-1,self.grid_size,self.grid_size)
        upsampled=None if output_size is None else F.interpolate(soft_map[:,None],output_size,mode="bilinear",align_corners=False)[:,0]
        entropy=-(probabilities*probabilities.clamp_min(1e-8).log()).sum(-1)
        diagnostics={"entropy":entropy,"concentration":probabilities.amax(-1),
                     "spatial_variance":pooling_map.var(dim=(1,2),unbiased=False),"effective_patch_count":entropy.exp()}
        return LocalizationOutput(relevance,probabilities,soft_relevance,grounded,soft_map,pooling_map,upsampled,grounded_score,
                                  torch.stack(history,dim=1),diagnostics)

def balanced_siglip_loss(scores:Tensor,positive_mask:Tensor,valid_negative_mask:Tensor):
    if scores.shape!=positive_mask.shape or scores.shape!=valid_negative_mask.shape:
        raise ValueError("score and supervision masks must match")
    positives=scores.masked_select(positive_mask.bool())
    negatives=scores.masked_select(valid_negative_mask.bool())
    if positives.numel()==0 or negatives.numel()==0: raise ValueError("balanced SigLIP requires positive and negative entries")
    positive_loss=F.softplus(-positives).mean()
    negative_loss=F.softplus(negatives).mean()
    loss=.5*positive_loss+.5*negative_loss
    stats={"positive_loss":positive_loss,"negative_loss":negative_loss,
           "positive_similarity":positives.mean(),"negative_similarity":negatives.mean(),
           "positive_entries":torch.tensor(positives.numel(),device=scores.device),
           "negative_entries":torch.tensor(negatives.numel(),device=scores.device)}
    return loss,stats

def multi_positive_sigmoid_loss(scores:Tensor,positive_mask:Tensor,excluded_mask:Tensor|None=None):
    positive=positive_mask.bool()
    excluded=torch.zeros_like(positive) if excluded_mask is None else excluded_mask.bool()
    valid_negative=(~positive)&(~excluded)
    return balanced_siglip_loss(scores,positive,valid_negative)

def build_pair_masks(query_pair_ids:list[str],gallery_pair_ids:list[str],
                     query_collision_pair_ids:list[set[str]]|None=None,device=None):
    positive=torch.tensor([[query==pair for pair in gallery_pair_ids] for query in query_pair_ids],
                          dtype=torch.bool,device=device)
    excluded=torch.zeros_like(positive)
    if query_collision_pair_ids:
        for row,collisions in enumerate(query_collision_pair_ids):
            for column,pair_id in enumerate(gallery_pair_ids):
                if pair_id!=query_pair_ids[row] and pair_id in collisions: excluded[row,column]=True
    return positive,excluded
