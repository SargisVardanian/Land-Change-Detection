from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel
from transformers.models.siglip.modeling_siglip import (
    create_bidirectional_mask as create_siglip_mask,
)

try:
    from transformers.models.siglip2.modeling_siglip2 import (
        create_bidirectional_mask as create_siglip2_mask,
    )
except ImportError:
    create_siglip2_mask = create_siglip_mask


@dataclass
class ImageEncoding:
    patch_tokens: Tensor
    pooled_embedding: Tensor


@dataclass
class TextEncoding:
    token_embeddings: Tensor
    pooled_embedding: Tensor
    attention_mask: Tensor


class Siglip2Backbone(nn.Module):
    """Pinned SigLIP-2 repository loaded through the checkpoint's native class."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        local_files_only: bool = True,
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            str(model_path), local_files_only=local_files_only, dtype=torch_dtype
        )
        self.runtime_class = type(self.model).__name__
        self.is_fixed_siglip = self.runtime_class == "SiglipModel"
        self.vision_model, self.text_model = (
            self.model.vision_model,
            self.model.text_model,
        )
        self.hidden_size = int(self.vision_model.config.hidden_size)
        if int(self.text_model.config.hidden_size) != self.hidden_size:
            raise ValueError("SigLIP vision/text hidden sizes differ")
        self.phase_b_top_blocks = 0
        self.gradient_checkpointing = False
        self.freeze_all()

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.phase_b_top_blocks = 0
        self.model.eval()

    def _set_block_scope(
        self, tower: nn.Module, top_blocks: int, final_names: tuple[str, ...]
    ) -> None:
        for p in tower.parameters():
            p.requires_grad = False
        layers = list(tower.encoder.layers)
        if not 0 <= top_blocks <= len(layers):
            raise ValueError("invalid top block count")
        for layer in layers[-top_blocks:] if top_blocks else []:
            for p in layer.parameters():
                p.requires_grad = True
        for name in final_names:
            module = getattr(tower, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True

    def enable_phase_b_top_blocks(
        self, top_blocks: int = 2, *, gradient_checkpointing: bool = False
    ) -> None:
        self.freeze_all()
        self._set_block_scope(self.vision_model, top_blocks, ("post_layernorm", "head"))
        self._set_block_scope(self.text_model, top_blocks, ("final_layer_norm", "head"))
        self.phase_b_top_blocks = top_blocks
        self.gradient_checkpointing = gradient_checkpointing
        self.model.eval()
        for tower, names in (
            (self.vision_model, ("post_layernorm", "head")),
            (self.text_model, ("final_layer_norm", "head")),
        ):
            for layer in list(tower.encoder.layers)[-top_blocks:]:
                layer.train()
            for name in names:
                module = getattr(tower, name, None)
                if module is not None:
                    module.train()

    def train(self, mode: bool = True) -> Siglip2Backbone:
        super().train(mode)
        self.model.eval()
        if self.phase_b_top_blocks:
            for tower in (self.vision_model, self.text_model):
                for layer in list(tower.encoder.layers)[-self.phase_b_top_blocks :]:
                    layer.train(mode)
        return self

    @staticmethod
    def _default_spatial_shapes(pixel_values: Tensor) -> Tensor:
        h, w = pixel_values.shape[-2:]
        return torch.tensor(
            [[h, w]], dtype=torch.long, device=pixel_values.device
        ).repeat(pixel_values.shape[0], 1)

    def _vision_forward(
        self,
        pixel_values: Tensor,
        pixel_attention_mask: Tensor | None,
        spatial_shapes: Tensor,
    ) -> Any:
        tower = self.vision_model
        if self.phase_b_top_blocks == 0:
            with torch.no_grad():
                if self.is_fixed_siglip:
                    return tower(pixel_values=pixel_values)
                return tower(
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask,
                    spatial_shapes=spatial_shapes,
                )
        if self.is_fixed_siglip:
            hidden = tower.embeddings(pixel_values, interpolate_pos_encoding=False)
            attention = None
        else:
            hidden = tower.embeddings(pixel_values, spatial_shapes)
            attention = create_siglip2_mask(
                config=tower.config,
                inputs_embeds=hidden,
                attention_mask=pixel_attention_mask,
            )
        layers = list(tower.encoder.layers)
        with torch.no_grad():
            for layer in layers[: -self.phase_b_top_blocks]:
                hidden = layer(hidden, attention)
        if self.gradient_checkpointing:
            hidden = hidden.detach().requires_grad_(True)
        for layer in layers[-self.phase_b_top_blocks :]:
            if self.gradient_checkpointing:
                hidden = checkpoint(
                    lambda value, module=layer: module(value, attention),
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = layer(hidden, attention)
        hidden = tower.post_layernorm(hidden)
        if getattr(tower, "use_head", True):
            try:
                pooled = (
                    tower.head(hidden, pixel_attention_mask)
                    if not self.is_fixed_siglip
                    else tower.head(hidden)
                )
            except TypeError:
                pooled = tower.head(hidden)
        else:
            pooled = hidden.mean(dim=1)
        return type(
            "VisionOutput", (), {"last_hidden_state": hidden, "pooler_output": pooled}
        )()

    def _text_forward(self, input_ids: Tensor, attention_mask: Tensor | None) -> Any:
        tower = self.text_model
        if self.phase_b_top_blocks == 0:
            with torch.no_grad():
                return tower(input_ids=input_ids, attention_mask=attention_mask)
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        hidden = tower.embeddings(input_ids=input_ids, position_ids=None)
        attention = create_siglip_mask(
            config=tower.config, inputs_embeds=hidden, attention_mask=attention_mask
        )
        layers = list(tower.encoder.layers)
        with torch.no_grad():
            for layer in layers[: -self.phase_b_top_blocks]:
                hidden = layer(hidden, attention)
        if self.gradient_checkpointing:
            hidden = hidden.detach().requires_grad_(True)
        for layer in layers[-self.phase_b_top_blocks :]:
            if self.gradient_checkpointing:
                hidden = checkpoint(
                    lambda value, module=layer: module(value, attention),
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = layer(hidden, attention)
        hidden = tower.final_layer_norm(hidden)
        pooled = tower.head(hidden[:, -1, :])
        return type(
            "TextOutput", (), {"last_hidden_state": hidden, "pooler_output": pooled}
        )()

    def encode_images(
        self,
        pixel_values: Tensor,
        *,
        pixel_attention_mask: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
    ) -> ImageEncoding:
        if pixel_values.ndim != 5:
            raise ValueError("pixel_values must be [B,T,C,H,W]")
        b, t, c, h, w = pixel_values.shape
        flat = pixel_values.reshape(b * t, c, h, w)
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                (b * t, h, w), dtype=torch.bool, device=pixel_values.device
            )
        else:
            pixel_attention_mask = pixel_attention_mask.reshape(
                b * t, *pixel_attention_mask.shape[-2:]
            )
        spatial_shapes = (
            self._default_spatial_shapes(flat)
            if spatial_shapes is None
            else spatial_shapes.reshape(b * t, 2).to(
                device=pixel_values.device, dtype=torch.long
            )
        )
        output = self._vision_forward(flat, pixel_attention_mask, spatial_shapes)
        tokens = output.last_hidden_state.reshape(
            b, t, output.last_hidden_state.shape[1], -1
        )
        pooled = output.pooler_output.reshape(b, t, -1)
        if tokens.shape[-1] != self.hidden_size or pooled.shape[-1] != self.hidden_size:
            raise ValueError("native SigLIP-2 output dimension mismatch")
        return ImageEncoding(tokens, pooled)

    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        content_mask: Tensor | None = None,
    ) -> TextEncoding:
        output = self._text_forward(input_ids, attention_mask)
        tokens = output.last_hidden_state
        pooled = output.pooler_output
        if tokens.shape[-1] != self.hidden_size or pooled.shape[-1] != self.hidden_size:
            raise ValueError("native SigLIP-2 text dimension mismatch")
        evidence_mask = attention_mask if content_mask is None else content_mask
        if evidence_mask.shape != input_ids.shape:
            raise ValueError("content_mask must match input_ids")
        return TextEncoding(tokens, pooled, evidence_mask.bool())

    def parameter_scope_report(self) -> dict[str, Any]:
        def report(module: nn.Module) -> dict[str, int | bool]:
            ps = list(module.parameters())
            return {
                "parameter_count": sum(p.numel() for p in ps),
                "trainable_count": sum(p.numel() for p in ps if p.requires_grad),
                "requires_grad": any(p.requires_grad for p in ps),
            }

        return {
            "runtime_class": self.runtime_class,
            "vision_backbone": report(self.vision_model),
            "text_backbone": report(self.text_model),
            "phase_b_top_blocks": self.phase_b_top_blocks,
            "gradient_checkpointing": self.gradient_checkpointing,
            "frozen_lower_blocks": self.phase_b_top_blocks > 0,
        }
