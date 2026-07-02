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
from land_change_detection.models.retrieval_heads import RetrievalProjectionHead, TextEmbeddingAdapter
from land_change_detection.models.temporal_change_encoder import TemporalChangeEncoder, TemporalChangeEncoderConfig
from land_change_detection.models.unichange_v2_retrieval import UniChangeV2RetrievalModel
from land_change_detection.training.rng_state import restore_rng_state


# Cluster entry points import this module before training. Replace the legacy
# helper with map-location-safe RNG restoration.
base._restore_rng_state = restore_rng_state


def strict_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def build_model(config, device):
    loader = UniverSatJointBackend(
        UniverSatBackendConfig(
            source_dir=config.universat_source,
            checkpoint_dir=config.universat_checkpoint,
            output_grid=config.output_grid,
            freeze=True,
        )
    )
    frame = UniverSatFrameBackend(loader.model, output_grid=config.output_grid)
    visual = SequenceUniverSatEncoder(
        frame,
        output_grid=config.output_grid,
        visual_dim=768,
        freeze=True,
    )
    text = JinaV5TextEncoder(
        JinaV5TextConfig(
            model_path=config.jina_model,
            max_length=96,
            global_projection_mode="matryoshka_truncate",
            freeze=True,
        )
    )
    temporal = TemporalChangeEncoder(
        TemporalChangeEncoderConfig(
            input_dim=768,
            hidden_dim=512,
            depth=int(getattr(config, "temporal_depth", 4)),
            heads=8,
            ffn_dim=2048,
            grid_size=config.output_grid,
            window_size=8,
            global_tokens=4,
            use_direction_embeddings=bool(getattr(config, "use_direction_embeddings", False)),
            use_explicit_change_fusion=bool(getattr(config, "use_explicit_change_fusion", False)),
            max_time_steps=2,
        )
    )
    retrieval_head = RetrievalProjectionHead(
        512,
        trainable_temperature=bool(getattr(config, "trainable_temperature", False)),
        initial_temperature=float(getattr(config, "initial_temperature", 0.07)),
        max_logit_scale=float(getattr(config, "max_logit_scale", 100.0)),
    )
    text_adapter = (
        TextEmbeddingAdapter(512, hidden_dim=int(getattr(config, "text_adapter_hidden_dim", 512)))
        if bool(getattr(config, "use_text_adapter", False))
        else None
    )
    return UniChangeV2RetrievalModel(visual, temporal, text, retrieval_head, text_adapter=text_adapter).to(device)


def run_metadata() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {
        "git_commit": commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
