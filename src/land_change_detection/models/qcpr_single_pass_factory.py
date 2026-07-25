from __future__ import annotations
from pathlib import Path
import torch
from land_change_detection.backbones.jina_v5_text import JinaV5TextConfig,JinaV5TextEncoder
from land_change_detection.backbones.sequence_universat import SequenceUniverSatEncoder
from land_change_detection.backbones.universat_backend import UniverSatBackendConfig,UniverSatJointBackend
from land_change_detection.backbones.universat_frame_backend import UniverSatFrameBackend
from land_change_detection.models.qcpr_single_pass import QCPRSinglePassRetriever,SinglePassConfig

def build_single_pass_retriever(*,universat_source:str|Path,universat_checkpoint:str|Path,
                                jina_model:str|Path,device:torch.device,output_grid:int=32)->QCPRSinglePassRetriever:
    loader=UniverSatJointBackend(UniverSatBackendConfig(source_dir=str(universat_source),
        checkpoint_dir=str(universat_checkpoint),output_grid=output_grid,freeze=True))
    visual=SequenceUniverSatEncoder(UniverSatFrameBackend(loader.model,output_grid=output_grid),
        output_grid=output_grid,visual_dim=768,freeze=True)
    text=JinaV5TextEncoder(JinaV5TextConfig(model_path=jina_model,max_length=256,
        global_projection_mode="matryoshka_truncate",freeze=True))
    return QCPRSinglePassRetriever(visual,text,SinglePassConfig(grid_size=output_grid)).to(device)
