from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone

import torch

import train_unichange_v2_retrieval as base
from land_change_detection.backbones.jina_v5_text import JinaV5TextConfig, JinaV5TextEncoder
from land_change_detection.backbones.sequence_universat import SequenceUniverSatEncoder
from land_change_detection.backbones.universat_backend import UniverSatBackendConfig, UniverSatJointBackend
from land_change_detection.backbones.universat_frame_backend import UniverSatFrameBackend
from land_change_detection.models.retrieval_heads import RetrievalProjectionHead
from land_change_detection.models.temporal_change_encoder import TemporalChangeEncoder, TemporalChangeEncoderConfig
from land_change_detection.models.unichange_v2_retrieval import UniChangeV2RetrievalModel
from land_change_detection.training.rng_state import restore_rng_state


# The shared training script restores checkpoints through its module-level
# helper. Cluster entry points import this module before training, so replace
# that helper with the implementation that converts map_location-moved RNG
# tensors back to CPU ByteTensors before calling PyTorch RNG APIs.
base._restore_rng_state = restore_rng_state


def strict_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def build_model(config, device):
    loader = UniverSatJointBackend(UniverSatBackendConfig(source_dir=config.universat_source, checkpoint_dir=config.universat_checkpoint, output_grid=config.output_grid, freeze=True))
    frame = UniverSatFrameBackend(loader.model, output_grid=config.output_grid)
    visual = SequenceUniverSatEncoder(frame, output_grid=config.output_grid, visual_dim=768, freeze=True)
    text = JinaV5TextEncoder(JinaV5TextConfig(model_path=config.jina_model, max_length=96, global_projection_mode="matryoshka_truncate", freeze=True))
    temporal = TemporalChangeEncoder(TemporalChangeEncoderConfig(input_dim=768, hidden_dim=512, depth=4, heads=8, ffn_dim=2048, grid_size=config.output_grid, window_size=8, global_tokens=4))
    return UniChangeV2RetrievalModel(visual, temporal, text, RetrievalProjectionHead(512)).to(device)


def run_metadata() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {"git_commit": commit, "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""), "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
