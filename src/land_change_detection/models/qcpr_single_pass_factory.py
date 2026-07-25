from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from torch import Tensor,nn
from land_change_detection.backbones.jina_v5_text import JinaV5TextConfig,JinaV5TextEncoder
from land_change_detection.backbones.universat_backend import UniverSatBackendConfig,UniverSatJointBackend
from land_change_detection.models.qcpr_single_pass import QCPRSinglePassRetriever,SinglePassConfig

@dataclass(frozen=True)
class NativeDenseSeriesFeatures:
    features:Tensor
    grid_height:int
    grid_width:int
    metadata:dict[str,Any]

class JointUniverSatSeriesEncoder(nn.Module):
    """Frozen joint temporal-spatial UniverSat; accepts arbitrary series length T."""
    def __init__(self,backend:UniverSatJointBackend,output_grid:int=32,visual_dim:int=768):
        super().__init__(); self.model=backend.model; self.adapter_spec=backend.config.adapter_spec
        self.output_grid=output_grid; self.visual_dim=visual_dim
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        self.model.eval()
    def train(self,mode=True): super().train(False); self.model.eval(); return self
    def forward(self,series:Tensor,dates:Tensor|None=None)->NativeDenseSeriesFeatures:
        if series.ndim!=5: raise ValueError("temporal series must be [B,T,C,H,W]")
        batch,time_steps=series.shape[:2]
        if time_steps<1: raise ValueError("temporal series must contain at least one timestamp")
        if dates is None:
            dates=torch.arange(time_steps,device=series.device,dtype=torch.long)[None].expand(batch,-1)
        if dates.shape!=(batch,time_steps): raise ValueError("dates must be [B,T]")
        payload={self.adapter_spec.modality_name:series,self.adapter_spec.date_key:dates}
        with torch.inference_mode():
            output=self.model.encode(payload,**self.adapter_spec.encode_kwargs(self.output_grid))
        if isinstance(output,(tuple,list)): output=output[0]
        if isinstance(output,dict):
            output=next(value for key in ("tokens","last_hidden_state","features","x")
                        if isinstance((value:=output.get(key)),Tensor))
        if output.ndim==4: output=output.flatten(2).transpose(1,2)
        expected=self.output_grid**2
        if output.shape!=(batch,expected,self.visual_dim):
            raise ValueError(f"joint UniverSat expected {(batch,expected,self.visual_dim)}, got {tuple(output.shape)}")
        output=output.detach()
        return NativeDenseSeriesFeatures(output,self.output_grid,self.output_grid,{
            "backend":"universat_joint_temporal_spatial","temporal_series_length":time_steps,
            "grid_height":self.output_grid,"grid_width":self.output_grid,
            "dense_token_count":expected,"native_hidden_dim":self.visual_dim,
            "frozen":True,"dates_shape":list(dates.shape)})

def build_single_pass_retriever(*,universat_source:str|Path,universat_checkpoint:str|Path,
                                jina_model:str|Path,device:torch.device,output_grid:int=32)->QCPRSinglePassRetriever:
    backend=UniverSatJointBackend(UniverSatBackendConfig(source_dir=str(universat_source),
        checkpoint_dir=str(universat_checkpoint),output_grid=output_grid,freeze=True))
    visual=JointUniverSatSeriesEncoder(backend,output_grid=output_grid,visual_dim=768)
    text=JinaV5TextEncoder(JinaV5TextConfig(model_path=jina_model,max_length=256,
        global_projection_mode="matryoshka_truncate",freeze=True))
    return QCPRSinglePassRetriever(visual,text,SinglePassConfig(grid_size=output_grid)).to(device)
