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
    patch_valid_mask: Tensor | None = None
    spatial_shapes: Tensor | None = None
    native_image_size: Tensor | None = None
    processed_patch_grid: Tensor | None = None
    transform_hash: str | None = None
    token_coordinates: Tensor | None = None
    chunk_plan_hashes: tuple[str, ...] | None = None
    processing_mode: str = "DIRECT_NAFLEX"
    force_region_reduction: bool = False
    chunk_counts: tuple[int, ...] | None = None
    chunk_valid_patch_counts: tuple[tuple[int, ...], ...] | None = None
    chunk_size: tuple[int, int] | None = None
    chunk_overlap: tuple[int, int] | None = None
    tile_batch_size: int | None = None
    overview_processed_patch_grid: Tensor | None = None


@dataclass
class TextEncoding:
    token_embeddings: Tensor
    pooled_embedding: Tensor
    attention_mask: Tensor


def _scalar_token_id(value: Any, *, name: str) -> int:
    """Validate a tokenizer special-token id at the model boundary."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
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
    tokenizer_vocab_size = int(tokenizer.vocab_size)
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
        raise TypeError("tokenizer contract has no special_token_ids")
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
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    text_raw = dict(raw.get("text_config", {}))
    vision_raw = dict(raw.get("vision_config", {}))
    text_raw.update(
        {
            name: int(value)
            for name, value in tokenizer_contract["special_token_ids"].items()
        }
    )
    if raw.get("model_type") == "siglip2":
        from transformers.models.siglip2.configuration_siglip2 import (
            Siglip2Config,
            Siglip2TextConfig,
            Siglip2VisionConfig,
        )

        text_config = Siglip2TextConfig(**text_raw)
        vision_config = Siglip2VisionConfig(**vision_raw)
        return Siglip2Config(text_config=text_config, vision_config=vision_config)

    from transformers.models.siglip.configuration_siglip import (
        SiglipConfig,
        SiglipTextConfig,
        SiglipVisionConfig,
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
        tokenizer_vocab_size = int(tokenizer.vocab_size)
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
        self.is_naflex = "Siglip2" in self.runtime_class and not self.is_fixed_siglip
        self.vision_model, self.text_model = (
            self.model.vision_model,
            self.model.text_model,
        )
        self.hidden_size = int(self.vision_model.config.hidden_size)
        patch_size = getattr(self.vision_model.config, "patch_size", 16)
        if isinstance(patch_size, (list, tuple)):
            if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
                raise ValueError("only square SigLIP patch sizes are supported")
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
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
    def _default_spatial_shapes(
        pixel_values: Tensor, *, patch_size: int, fixed_image: bool = False
    ) -> Tensor:
        if pixel_values.ndim == 4 and not fixed_image:
            n = int(pixel_values.shape[1])
            side = int(n**0.5)
            grid = (side, side) if side * side == n else (1, n)
            return torch.tensor(
                [grid], dtype=torch.long, device=pixel_values.device
            ).repeat(pixel_values.shape[0], 1)
        h, w = pixel_values.shape[-2:]
        return torch.tensor(
            [[max(1, h // patch_size), max(1, w // patch_size)]],
            dtype=torch.long,
            device=pixel_values.device,
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
        native_image_size: Tensor | None = None,
        transform_hash: str | None = None,
    ) -> ImageEncoding:
        if pixel_values.ndim not in (4, 5):
            raise ValueError(
                "pixel_values must be [B,T,C,H,W] FixRes or [B,T,N,patch_dim] NaFlex"
            )
        b, t = pixel_values.shape[:2]
        flat = pixel_values.reshape(b * t, *pixel_values.shape[2:])
        input_patch_count = int(pixel_values.shape[2]) if pixel_values.ndim == 4 else None
        if pixel_values.ndim == 5 and self.is_naflex:
            raise ValueError(
                "NaFlex requires processor patch tensors [B,T,N,patch_dim]; "
                "raw image tensors would hide the explicit patch budget"
            )

        if spatial_shapes is None:
            spatial_shapes_flat = self._default_spatial_shapes(
                flat, patch_size=self.patch_size, fixed_image=self.is_fixed_siglip
            )
        elif spatial_shapes.ndim == 3 and spatial_shapes.shape[:2] == (b, t):
            spatial_shapes_flat = spatial_shapes.reshape(b * t, 2)
        elif spatial_shapes.ndim == 2 and spatial_shapes.shape == (b * t, 2):
            spatial_shapes_flat = spatial_shapes
        else:
            raise ValueError("spatial_shapes must be [B,T,2] or [B*T,2]")
        spatial_shapes_flat = spatial_shapes_flat.to(
            device=pixel_values.device, dtype=torch.long
        )

        if pixel_attention_mask is not None:
            if (
                pixel_attention_mask.ndim == 3
                and pixel_attention_mask.shape[:2] == (b, t)
            ) or (
                pixel_attention_mask.ndim == 2
                and pixel_attention_mask.shape[0] == b * t
            ):
                pixel_attention_mask = pixel_attention_mask.reshape(b * t, -1)
            else:
                raise ValueError(
                    "pixel_attention_mask must be [B,T,N] or [B*T,N] patch mask"
                )
            pixel_attention_mask = pixel_attention_mask.to(
                device=pixel_values.device, dtype=torch.bool
            )
        elif self.is_naflex:
            if input_patch_count is None:
                raise ValueError("NaFlex patch mask is required for patch tensors")
            pixel_attention_mask = torch.arange(
                input_patch_count, device=pixel_values.device
            ).view(1, -1) < (
                spatial_shapes_flat[:, 0] * spatial_shapes_flat[:, 1]
            ).unsqueeze(1)
        output = self._vision_forward(
            flat, pixel_attention_mask, spatial_shapes_flat
        )
        tokens = output.last_hidden_state.reshape(
            b, t, output.last_hidden_state.shape[1], -1
        )
        pooled = output.pooler_output.reshape(b, t, -1)
        if tokens.shape[-1] != self.hidden_size or pooled.shape[-1] != self.hidden_size:
            raise ValueError("native SigLIP-2 output dimension mismatch")
        output_patch_count = int(tokens.shape[2])
        if pixel_attention_mask is None:
            patch_valid_mask = torch.ones(
                b * t,
                output_patch_count,
                dtype=torch.bool,
                device=tokens.device,
            )
        elif pixel_attention_mask.shape[1] == output_patch_count:
            patch_valid_mask = pixel_attention_mask
        else:
            raise ValueError(
                "processor patch mask length does not match native visual token count"
            )
        if torch.any(
            spatial_shapes_flat[:, 0] * spatial_shapes_flat[:, 1] > output_patch_count
        ):
            raise ValueError("processor spatial_shapes exceed native visual token count")
        expected_valid = spatial_shapes_flat[:, 0] * spatial_shapes_flat[:, 1]
        actual_valid = patch_valid_mask.sum(dim=-1)
        if not torch.equal(expected_valid, actual_valid):
            raise ValueError(
                "native patch-valid mask counts disagree with processed_patch_grid"
            )
        reshaped_shapes = spatial_shapes_flat.reshape(b, t, 2)
        if not torch.equal(
            reshaped_shapes,
            reshaped_shapes[:, :1].expand_as(reshaped_shapes),
        ):
            raise ValueError("T1/T2 native patch grids must be compatible")
        shapes = reshaped_shapes
        if native_image_size is not None:
            if native_image_size.ndim == 3 and native_image_size.shape[:2] == (b, t):
                native_image_size = native_image_size.to(
                    device=tokens.device, dtype=torch.long
                )
            elif native_image_size.ndim == 2 and native_image_size.shape == (b * t, 2):
                native_image_size = native_image_size.reshape(b, t, 2).to(
                    device=tokens.device, dtype=torch.long
                )
            else:
                raise ValueError("native_image_size must be [B,T,2] or [B*T,2]")
        return ImageEncoding(
            patch_tokens=tokens,
            pooled_embedding=pooled,
            patch_valid_mask=patch_valid_mask.reshape(b, t, output_patch_count),
            spatial_shapes=shapes,
            native_image_size=native_image_size,
            processed_patch_grid=shapes,
            transform_hash=transform_hash,
        )

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
            "is_naflex": self.is_naflex,
            "patch_size": self.patch_size,
            "vision_backbone": report(self.vision_model),
            "text_backbone": report(self.text_model),
            "phase_b_top_blocks": self.phase_b_top_blocks,
            "gradient_checkpointing": self.gradient_checkpointing,
            "frozen_lower_blocks": self.phase_b_top_blocks > 0,
        }
