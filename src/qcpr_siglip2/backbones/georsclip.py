"""Safe, frozen GeoRSCLIP ViT-B/32 baseline wrapper.

The checkpoint is a PyTorch state-dict archive.  This module deliberately uses
``weights_only=True`` and requires an exact state-dict load into an explicitly
constructed OpenCLIP model.  It is a separate baseline and is never mixed
with the SigLIP-2 backbone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .siglip2 import ImageEncoding, TextEncoding


@dataclass(frozen=True)
class GeoRSCLIPLoadAudit:
    checkpoint: str
    model_name: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    weights_only: bool


class GeoRSCLIPBackbone(nn.Module):
    """Frozen OpenCLIP ViT-B/32 loader exposing patch and text tokens."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        model_name: str = "ViT-B-32",
        local_only: bool = True,
    ) -> None:
        super().__init__()
        del local_only  # OpenCLIP receives no remote pretrained identifier here.
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - depends on baseline env
            raise RuntimeError(
                "GeoRSCLIP requires the OpenCLIP baseline environment"
            ) from exc

        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise TypeError("GeoRSCLIP checkpoint must contain a state dict")
        self.model = open_clip.create_model(model_name, pretrained=None)
        missing, unexpected = self.model.load_state_dict(dict(state), strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "GeoRSCLIP state-dict mismatch: "
                f"missing={list(missing)}, unexpected={list(unexpected)}"
            )
        self.load_audit = GeoRSCLIPLoadAudit(
            checkpoint=str(checkpoint),
            model_name=model_name,
            missing_keys=tuple(missing),
            unexpected_keys=tuple(unexpected),
            weights_only=True,
        )
        self.hidden_size = (
            int(self.model.visual.proj.shape[-1])
            if self.model.visual.proj is not None
            else int(self.model.visual.width)
        )
        self.patch_grid = (
            int(self.model.visual.grid_size[0])
            if hasattr(self.model.visual, "grid_size")
            else 7
        )
        self.freeze_all()

    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def _vision_tokens(self, images: Tensor) -> tuple[Tensor, Tensor]:
        visual = self.model.visual
        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_embedding = visual.class_embedding.to(dtype=x.dtype, device=x.device)
        class_tokens = class_embedding.expand(x.shape[0], 1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        x = x + visual.positional_embedding.to(dtype=x.dtype, device=x.device)
        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj
        return x[:, 1:], x[:, 0]

    def encode_images(self, pixel_values: Tensor) -> ImageEncoding:
        if pixel_values.ndim != 5:
            raise ValueError("pixel_values must be [B,T,C,H,W]")
        batch, frames, channels, height, width = pixel_values.shape
        del channels, height, width
        with torch.no_grad():
            patches, pooled = self._vision_tokens(
                pixel_values.reshape(batch * frames, *pixel_values.shape[2:])
            )
        return ImageEncoding(
            patch_tokens=patches.reshape(
                batch, frames, patches.shape[1], patches.shape[2]
            ),
            pooled_embedding=pooled.reshape(batch, frames, pooled.shape[-1]),
        )

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor) -> TextEncoding:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must be [B,L]")
        model = self.model
        with torch.no_grad():
            # Current OpenCLIP builds use batch-first transformer blocks.  Do
            # not transpose here: doing so makes a [B, L, D] input look like
            # a one-token batch and causes the causal mask to have the wrong
            # shape under recent PyTorch versions.
            x = model.token_embedding(input_ids).to(model.token_embedding.weight.dtype)
            x = x + model.positional_embedding[: input_ids.shape[1]].to(
                dtype=x.dtype, device=x.device
            )
            x = model.transformer(x, attn_mask=model.attn_mask)
            x = model.ln_final(x)
            if model.text_projection is not None:
                x = x @ model.text_projection
            eot_index = input_ids.argmax(dim=-1)
            pooled = x[torch.arange(x.shape[0], device=x.device), eot_index]
        return TextEncoding(x, pooled, attention_mask.bool())

    def parameter_scope_report(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self.model).__name__,
            "hidden_size": self.hidden_size,
            "patch_grid": self.patch_grid,
            "vision_trainable_count": sum(
                p.numel() for p in self.model.visual.parameters() if p.requires_grad
            ),
            "text_trainable_count": sum(
                p.numel()
                for p in self.model.transformer.parameters()
                if p.requires_grad
            ),
            "load_audit": {
                "checkpoint": self.load_audit.checkpoint,
                "model_name": self.load_audit.model_name,
                "missing_keys": list(self.load_audit.missing_keys),
                "unexpected_keys": list(self.load_audit.unexpected_keys),
                "weights_only": self.load_audit.weights_only,
            },
        }
