"""Frozen SigLIP2 dense image/text features for QCPR v3.1 grounding only."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SigLIP2GroundingText:
    token_embeddings: Tensor
    attention_mask: Tensor
    content_mask: Tensor


class FrozenSigLIP2GroundingBackbone(nn.Module):
    """Expose aligned 16x16 image patches and language tokens.

    The large frozen model is intentionally held outside PyTorch's registered
    module tree. It is reconstructed from the immutable local model path and is
    therefore absent from student checkpoints and optimizers.
    """

    def __init__(self, model_path: str | Path):
        super().__init__()
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        path = str(Path(model_path))
        model = AutoModel.from_pretrained(path, local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        object.__setattr__(self, "_frozen_model", model)
        object.__setattr__(self, "_tokenizer", tokenizer)
        self.model_path = path
        self.hidden_dim = int(model.config.vision_config.hidden_size)
        self.image_size = int(model.config.vision_config.image_size)
        self.patch_size = int(model.config.vision_config.patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.text_max_length = int(model.config.text_config.max_position_embeddings)
        image_processor = processor.image_processor
        mean = torch.tensor(image_processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(image_processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    @property
    def frozen_model(self) -> nn.Module:
        return self.__dict__["_frozen_model"]

    @property
    def tokenizer(self) -> Any:
        return self.__dict__["_tokenizer"]

    def _apply(self, fn):
        super()._apply(fn)
        self.frozen_model._apply(fn)
        return self

    def train(self, mode: bool = True):
        super().train(False)
        self.frozen_model.eval()
        return self

    def encode_images(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[1:3] != (2, 3):
            raise ValueError("SigLIP2 grounding images must be [B,2,3,H,W]")
        batch, time, channels, height, width = images.shape
        if (height, width) != (self.image_size, self.image_size):
            raise ValueError(
                f"SigLIP2 grounding expects {self.image_size}x{self.image_size}, "
                f"got {height}x{width}"
            )
        flat = images.reshape(batch * time, channels, height, width)
        pixel_values = (flat - self.image_mean.to(flat)) / self.image_std.to(flat)
        with torch.no_grad():
            output = self.frozen_model.vision_model(pixel_values=pixel_values)
        tokens = output.last_hidden_state
        expected = self.grid_size * self.grid_size
        if tokens.shape != (batch * time, expected, self.hidden_dim):
            raise RuntimeError(
                f"SigLIP2 dense image contract expected "
                f"{(batch * time, expected, self.hidden_dim)}, got {tuple(tokens.shape)}"
            )
        if not torch.isfinite(tokens).all():
            raise RuntimeError("SigLIP2 image tokens contain NaN or Inf")
        return tokens.reshape(batch, time, expected, self.hidden_dim)

    def encode_texts(self, captions: list[str], *, device: torch.device) -> SigLIP2GroundingText:
        encoded = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention = encoded.get("attention_mask")
        if attention is None:
            attention = input_ids.ne(int(self.tokenizer.pad_token_id))
        else:
            attention = attention.to(device).bool()
        with torch.no_grad():
            output = self.frozen_model.text_model(
                input_ids=input_ids,
                attention_mask=attention,
            )
        tokens = output.last_hidden_state
        if tokens.shape[:2] != input_ids.shape or tokens.shape[-1] != self.hidden_dim:
            raise RuntimeError("SigLIP2 dense text token contract failed")
        content = attention.clone()
        special_ids = set(int(value) for value in self.tokenizer.all_special_ids)
        for row in range(input_ids.shape[0]):
            token_strings = self.tokenizer.convert_ids_to_tokens(input_ids[row].tolist())
            for column, (token_id, token) in enumerate(
                zip(input_ids[row].tolist(), token_strings, strict=True)
            ):
                normalized = str(token).casefold().lstrip("▁Ġ").strip(".,;:!?")
                if token_id in special_ids or not normalized:
                    content[row, column] = False
        fallback = attention & ~torch.isin(
            input_ids,
            torch.tensor(sorted(special_ids), device=device, dtype=input_ids.dtype),
        )
        empty = ~content.any(dim=1)
        if bool(empty.any()):
            content[empty] = fallback[empty]
        if bool((~content.any(dim=1)).any()):
            raise RuntimeError("SigLIP2 content mask removed every token")
        if not torch.isfinite(tokens).all():
            raise RuntimeError("SigLIP2 text tokens contain NaN or Inf")
        return SigLIP2GroundingText(tokens, attention, content)

    def provenance(self) -> dict[str, Any]:
        return {
            "kind": "siglip2",
            "model_path": self.model_path,
            "hidden_dim": self.hidden_dim,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "dense_grid": [self.grid_size, self.grid_size],
            "text_max_length": self.text_max_length,
            "frozen": True,
            "checkpoint_serialization": "external_reference_not_embedded",
        }
