from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ..backbones.siglip2 import ImageEncoding, Siglip2Backbone, TextEncoding
from ..config.schema import Siglip2TemporalConfig
from ..contracts import validate_feature_contract
from .evidence import EvidenceBottleneck, EvidenceOutput
from .temporal import TemporalAdapterOutput, TemporalTransformerAdapter

@dataclass
class RetrievalForwardOutput:
    score_matrix: Tensor
    global_score_matrix: Tensor
    text_embedding: Tensor
    pair_cls: Tensor
    pair_for_query: Tensor
    evidence: EvidenceOutput
    temporal: TemporalAdapterOutput

class Siglip2TemporalRetrievalModel(nn.Module):
    """Minimal SigLIP-2 temporal retriever with a causal evidence score."""
    def __init__(self,backbone:Siglip2Backbone|None,config:Siglip2TemporalConfig|None=None)->None:
        super().__init__(); self.config=(config or Siglip2TemporalConfig()).validate(); self.backbone=backbone; self.temporal_adapter=TemporalTransformerAdapter(self.config); self.evidence_bottleneck=EvidenceBottleneck(self.config); self.log_temperature=nn.Parameter(torch.tensor(self.config.retrieval_temperature).log())
    @property
    def retrieval_temperature(self)->Tensor: return self.log_temperature.clamp(min=-5.,max=2.).exp()
    def forward_from_features(self,frame_tokens:Tensor,frame_embeddings:Tensor,text_tokens:Tensor,text_embeddings:Tensor,text_mask:Tensor,*,timestamps:Tensor|None=None)->RetrievalForwardOutput:
        validate_feature_contract(frame_tokens,frame_embeddings,text_tokens,text_embeddings,text_mask,hidden_size=self.config.hidden_size,expected_patch_tokens=self.config.expected_patch_tokens); temporal=self.temporal_adapter(frame_tokens,frame_embeddings,timestamps=timestamps); text_embedding=F.normalize(text_embeddings,dim=-1); pair=temporal.pair_cls; global_score=text_embedding@pair.transpose(0,1); evidence=self.evidence_bottleneck(text_tokens,text_mask,temporal.temporal_patch_tokens,frame_count=temporal.frame_count,patch_count=temporal.patch_count); pair_for_query=F.normalize(pair.unsqueeze(0)+evidence.evidence_gate*evidence.evidence_vector,dim=-1); score=torch.einsum("qd,qpd->qp",text_embedding,pair_for_query)/self.retrieval_temperature; return RetrievalForwardOutput(score,global_score/self.retrieval_temperature,text_embedding,pair,pair_for_query,evidence,temporal)
    def forward(self,pixel_values:Tensor,input_ids:Tensor,attention_mask:Tensor,*,pixel_attention_mask:Tensor|None=None,spatial_shapes:Tensor|None=None,timestamps:Tensor|None=None)->RetrievalForwardOutput:
        if self.backbone is None: raise RuntimeError("raw-input forward requires a Siglip2Backbone")
        image:ImageEncoding=self.backbone.encode_images(pixel_values,pixel_attention_mask=pixel_attention_mask,spatial_shapes=spatial_shapes); text:TextEncoding=self.backbone.encode_text(input_ids,attention_mask); return self.forward_from_features(image.patch_tokens,image.pooled_embedding,text.token_embeddings,text.pooled_embedding,text.attention_mask,timestamps=timestamps)
    def trainable_parameter_report(self)->dict[str,dict[str,int]]:
        report={}
        for name,module in (("temporal_adapter",self.temporal_adapter),("evidence_bottleneck",self.evidence_bottleneck),("retrieval_temperature",nn.ParameterList([self.log_temperature]))):
            ps=list(module.parameters()); report[name]={"parameter_count":sum(p.numel() for p in ps),"trainable_count":sum(p.numel() for p in ps if p.requires_grad)}
        if self.backbone is not None:
            for name,module in (("siglip2_vision_backbone",self.backbone.vision_model),("siglip2_text_backbone",self.backbone.text_model)):
                ps=list(module.parameters()); report[name]={"parameter_count":sum(p.numel() for p in ps),"trainable_count":sum(p.numel() for p in ps if p.requires_grad)}
        return report
