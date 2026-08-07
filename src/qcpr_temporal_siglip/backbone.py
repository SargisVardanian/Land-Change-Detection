from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class TemporalSigLIPBackbone(nn.Module):
    """The single official SigLIP2 tower pair used by TemporalSigLIP.

    This wrapper deliberately exposes only frame and text encoding.  Temporal
    fusion, retrieval scoring and localization live in the active package and
    are not delegated to the historical evidence/reranking implementation.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        local_files_only: bool = True,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # Keep feature-only CPU/unit tests independent of the optional
        # Transformers installation.  The real backbone is imported only when
        # a checkpoint-backed model is constructed.
        from qcpr_siglip2.backbones.siglip2 import Siglip2Backbone

        self.encoder: Any = Siglip2Backbone(
            checkpoint,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )

    @property
    def hidden_size(self) -> int:
        return self.encoder.hidden_size

    @property
    def runtime_class(self) -> str:
        return self.encoder.runtime_class

    @property
    def phase_b_top_blocks(self) -> int:
        return int(self.encoder.phase_b_top_blocks)

    def freeze_towers(self) -> None:
        self.encoder.freeze_all()

    def enable_stage_b(self, *, top_blocks: int = 2, gradient_checkpointing: bool = True) -> None:
        if top_blocks != 2:
            raise ValueError("Stage B requires exactly the last two backbone blocks")
        self.encoder.enable_phase_b_top_blocks(
            top_blocks=top_blocks,
            gradient_checkpointing=gradient_checkpointing,
        )

    def encode_images(
        self,
        pixel_values: Tensor,
        *,
        pixel_attention_mask: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
    ) -> Any:
        return self.encoder.encode_images(
            pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        content_mask: Tensor | None = None,
    ) -> Any:
        return self.encoder.encode_text(
            input_ids,
            attention_mask,
            content_mask=content_mask,
        )

    def trainable_scope(self) -> dict[str, int]:
        vision = list(self.encoder.vision_model.parameters())
        text = list(self.encoder.text_model.parameters())
        return {
            "vision_parameter_count": sum(p.numel() for p in vision),
            "vision_trainable_count": sum(p.numel() for p in vision if p.requires_grad),
            "text_parameter_count": sum(p.numel() for p in text),
            "text_trainable_count": sum(p.numel() for p in text if p.requires_grad),
        }
