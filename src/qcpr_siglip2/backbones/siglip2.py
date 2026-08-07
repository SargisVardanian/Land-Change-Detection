from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer
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


def _scalar_token_id(value: Any, *, name: str) -> int:
    """Validate a tokenizer special-token id at the model boundary."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def validate_tokenizer_contract(tokenizer: Any, embedding_vocab_size: int) -> dict[str, Any]:
    """Return a checked tokenizer/embedding vocabulary contract.

    The pinned checkpoint has a Gemma tokenizer with a 256k vocabulary while
    its inherited SigLIP config contains the old 32k special-token defaults.
    This boundary check makes the discrepancy explicit and rejects an unsafe
    tokenizer rather than silently clipping or remapping token ids.
    """

    if embedding_vocab_size <= 0:
        raise ValueError("embedding_vocab_size must be positive")
    tokenizer_vocab_size = int(getattr(tokenizer, "vocab_size"))
    if tokenizer_vocab_size != embedding_vocab_size:
        raise ValueError(
            "tokenizer and model embedding vocabularies differ: "
            f"{tokenizer_vocab_size} != {embedding_vocab_size}"
        )
    ids = {
        "bos_token_id": _scalar_token_id(
            getattr(tokenizer, "bos_token_id", None), name="bos_token_id"
        ),
        "eos_token_id": _scalar_token_id(
            getattr(tokenizer, "eos_token_id", None), name="eos_token_id"
        ),
        "pad_token_id": _scalar_token_id(
            getattr(tokenizer, "pad_token_id", None), name="pad_token_id"
        ),
    }
    invalid = {name: value for name, value in ids.items() if value >= embedding_vocab_size}
    if invalid:
        raise ValueError(
            "tokenizer special-token ids exceed the model vocabulary: "
            f"{invalid} >= {embedding_vocab_size}"
        )
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "model_embedding_vocab_size": embedding_vocab_size,
        "special_token_ids": ids,
        "special_token_ids_in_range": True,
    }


def synchronize_tokenizer_config(model: Any, contract: dict[str, Any]) -> dict[str, Any]:
    """Synchronize runtime text-config metadata with the validated tokenizer.

    This changes configuration metadata only.  It does not alter checkpoint
    weights or token embeddings.  The before/after values are returned so the
    compatibility report can distinguish the upstream config warning from the
    verified runtime contract.
    """

    ids = contract.get("special_token_ids")
    if not isinstance(ids, dict):
        raise ValueError("tokenizer contract has no special_token_ids")
    text_model = getattr(model, "text_model", None)
    text_config = getattr(text_model, "config", None)
    model_config = getattr(getattr(model, "config", None), "text_config", None)
    if text_config is None or model_config is None:
        raise ValueError("SigLIP model has no text configuration")
    before = {
        name: getattr(text_config, name, None)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id")
    }
    for name, value in ids.items():
        setattr(text_config, name, int(value))
        setattr(model_config, name, int(value))
    after = {
        name: getattr(text_config, name, None)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id")
    }
    if after != {name: int(value) for name, value in ids.items()}:
        raise RuntimeError("runtime text configuration did not synchronize")
    return {"config_ids_before_sync": before, "config_ids_after_sync": after}


def _load_patched_local_config(model_path: str | Path, tokenizer_contract: dict[str, Any]) -> Any | None:
    """Build a corrected local SigLIP config before model construction.

    Transformers 5.x validates the inherited SigLIP defaults while parsing
    the checkpoint config.  Constructing the nested text config with the
    already-validated tokenizer ids avoids creating an invalid runtime config
    for the local pinned checkpoint.  If a non-local identifier is supplied,
    the normal AutoModel path remains available and the post-load sync still
    enforces the contract.
    """

    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return None
    from transformers.models.siglip.configuration_siglip import (
        SiglipConfig,
        SiglipTextConfig,
        SiglipVisionConfig,
    )

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    text_raw = dict(raw.get("text_config", {}))
    vision_raw = dict(raw.get("vision_config", {}))
    text_raw.update(
        {
            name: int(value)
            for name, value in tokenizer_contract["special_token_ids"].items()
        }
    )
    text_config = SiglipTextConfig(**text_raw)
    vision_config = SiglipVisionConfig(**vision_raw)
    return SiglipConfig(text_config=text_config, vision_config=vision_config)


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
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=local_files_only
        )
        # The tokenizer is loaded before the model so the model constructor can
        # be supplied a corrected nested text config for the pinned local
        # checkpoint.  The actual embedding size is checked again after load.
        tokenizer_vocab_size = int(getattr(tokenizer, "vocab_size"))
        tokenizer_contract = {
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "special_token_ids": {
                "bos_token_id": _scalar_token_id(
                    getattr(tokenizer, "bos_token_id", None), name="bos_token_id"
                ),
                "eos_token_id": _scalar_token_id(
                    getattr(tokenizer, "eos_token_id", None), name="eos_token_id"
                ),
                "pad_token_id": _scalar_token_id(
                    getattr(tokenizer, "pad_token_id", None), name="pad_token_id"
                ),
            },
        }
        config = _load_patched_local_config(model_path, tokenizer_contract)
        self.model = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=local_files_only,
            dtype=torch_dtype,
            **({"config": config} if config is not None else {}),
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
        embedding_vocab_size = int(self.text_model.get_input_embeddings().num_embeddings)
        checked_contract = validate_tokenizer_contract(tokenizer, embedding_vocab_size)
        checked_contract.update(synchronize_tokenizer_config(self.model, checked_contract))
        checked_contract["runtime_config_ids_valid"] = True
        self.tokenizer_contract = checked_contract
        self.tokenizer = tokenizer
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
