from __future__ import annotations

from dataclasses import dataclass

import torch

from land_change_detection.backbones.jina_v5_text import JinaV5TextConfig, JinaV5TextEncoder
from land_change_detection.backbones.sequence_universat import SequenceUniverSatEncoder
from land_change_detection.backbones.universat_backend import UniverSatBackendConfig, UniverSatJointBackend
from land_change_detection.backbones.universat_frame_backend import UniverSatFrameBackend
from land_change_detection.models.qcpr_v3 import QCPRV3Config, QCPRV3GenericGrounding
from land_change_detection.models.retrieval_heads import RetrievalProjectionHead, TextEmbeddingAdapter
from land_change_detection.models.temporal_change_encoder import TemporalChangeEncoder, TemporalChangeEncoderConfig
from land_change_detection.models.unichange_v3_retrieval import UniChangeV3RetrievalModel


@dataclass(frozen=True)
class QCPRV3BackboneConfig:
    universat_source: str
    universat_checkpoint: str
    jina_model: str
    output_grid: int = 32
    temporal_depth: int = 6
    text_max_length: int = 256
    use_direction_embeddings: bool = True
    use_explicit_change_fusion: bool = True
    use_text_adapter: bool = True
    text_adapter_hidden_dim: int = 512
    trainable_temperature: bool = True
    initial_temperature: float = 0.07
    max_logit_scale: float = 100.0


def build_clean_v3_model(
    config: QCPRV3BackboneConfig,
    *,
    device: torch.device,
    grounding_config: QCPRV3Config | None = None,
) -> UniChangeV3RetrievalModel:
    """Build v3 directly from pretrained backbones, without a v2 model instance."""
    loader = UniverSatJointBackend(
        UniverSatBackendConfig(
            source_dir=config.universat_source,
            checkpoint_dir=config.universat_checkpoint,
            output_grid=config.output_grid,
            freeze=True,
        )
    )
    visual = SequenceUniverSatEncoder(
        UniverSatFrameBackend(loader.model, output_grid=config.output_grid),
        output_grid=config.output_grid,
        visual_dim=768,
        freeze=True,
    )
    text = JinaV5TextEncoder(
        JinaV5TextConfig(
            model_path=config.jina_model,
            max_length=config.text_max_length,
            global_projection_mode="matryoshka_truncate",
            freeze=True,
        )
    )
    temporal = TemporalChangeEncoder(
        TemporalChangeEncoderConfig(
            input_dim=768,
            hidden_dim=512,
            depth=config.temporal_depth,
            heads=8,
            ffn_dim=2048,
            grid_size=config.output_grid,
            window_size=8,
            global_tokens=4,
            use_direction_embeddings=config.use_direction_embeddings,
            use_explicit_change_fusion=config.use_explicit_change_fusion,
            max_time_steps=2,
        )
    )
    retrieval = RetrievalProjectionHead(
        512,
        trainable_temperature=config.trainable_temperature,
        initial_temperature=config.initial_temperature,
        max_logit_scale=config.max_logit_scale,
    )
    adapter = (
        TextEmbeddingAdapter(512, hidden_dim=config.text_adapter_hidden_dim)
        if config.use_text_adapter
        else None
    )
    return UniChangeV3RetrievalModel(
        visual,
        temporal,
        text,
        retrieval,
        QCPRV3GenericGrounding(grounding_config or QCPRV3Config(visual_source_dim=768)),
        text_adapter=adapter,
    ).to(device)
