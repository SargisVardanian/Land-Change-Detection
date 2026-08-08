"""Mask-free real-image and text batching for the SigLIP-2 track."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor

from ..backbones.siglip2 import ImageEncoding, Siglip2Backbone, TextEncoding


@dataclass(frozen=True)
class RawFeatureBatch:
    """Backbone outputs for one physical pair/query slice."""

    frame_tokens: Tensor
    frame_embeddings: Tensor
    text_tokens: Tensor
    text_embeddings: Tensor
    text_mask: Tensor
    patch_valid_mask: Tensor | None = None
    spatial_shapes: Tensor | None = None


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def processor_inputs(
    processor: Any,
    pair_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    query_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    device: torch.device,
    *,
    max_num_patches: int | None = None,
    synchronized_transform: Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]]
    | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Decode each T1/T2 exactly once and tokenize the paired queries."""

    image_inputs = processor_image_inputs(
        processor,
        pair_rows,
        device,
        max_num_patches=max_num_patches,
        synchronized_transform=synchronized_transform,
    )
    text_inputs = processor_text_inputs(processor, query_rows, device)
    return image_inputs, text_inputs


def processor_image_inputs(
    processor: Any,
    pair_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    device: torch.device,
    *,
    max_num_patches: int | None = None,
    synchronized_transform: Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]]
    | None = None,
) -> dict[str, Any]:
    """Decode T1/T2 images and return the processor's tensor inputs."""

    images: list[Image.Image] = []
    native_sizes: list[tuple[int, int]] = []
    for row in pair_rows:
        loaded: list[Image.Image] = []
        for field in ("t1_path", "t2_path"):
            path = Path(str(row[field]))
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                loaded.append(image.convert("RGB"))
        native_sizes.extend((image.height, image.width) for image in loaded)
        if synchronized_transform is not None:
            loaded[0], loaded[1] = synchronized_transform(loaded[0], loaded[1])
        if loaded[0].size != loaded[1].size:
            raise ValueError(
                "T1/T2 image geometry differs; apply one synchronized transform before processing"
            )
        images.extend(loaded)

    image_processor = getattr(processor, "image_processor", processor)
    processor_name = type(image_processor).__name__.lower()
    is_naflex = "siglip2" in processor_name or hasattr(
        image_processor, "max_num_patches"
    )
    if is_naflex and max_num_patches is None:
        raise ValueError(
            "explicit max_num_patches is required for NaFlex; refusing the processor default"
        )
    if max_num_patches is not None:
        if not isinstance(max_num_patches, int) or max_num_patches <= 0:
            raise ValueError("max_num_patches must be a positive integer")
        image_inputs = processor(
            images=images, max_num_patches=max_num_patches, return_tensors="pt"
        )
    else:
        image_inputs = processor(images=images, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"]
    pair_count = len(pair_rows)
    image_inputs["pixel_values"] = pixel_values.reshape(
        pair_count, 2, *pixel_values.shape[1:]
    ).to(device)
    image_inputs["native_image_size"] = torch.tensor(
        native_sizes, dtype=torch.long, device=device
    ).reshape(pair_count, 2, 2)
    if max_num_patches is not None and image_inputs.get("spatial_shapes") is not None:
        shape_values = image_inputs["spatial_shapes"].detach().cpu().tolist()
        transform_payload = "\n".join(
            f"{size[0]}x{size[1]}:{shape[0]}x{shape[1]}:{max_num_patches}"
            for size, shape in zip(native_sizes, shape_values)
        )
        image_inputs["transform_hash"] = sha256(
            transform_payload.encode("utf-8")
        ).hexdigest()
    for key in ("pixel_attention_mask", "spatial_shapes"):
        value = image_inputs.get(key)
        if value is not None:
            if key == "pixel_attention_mask":
                value = value.reshape(pair_count, 2, -1)
            else:
                value = value.reshape(pair_count, 2, 2)
            image_inputs[key] = value.to(device)

    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in image_inputs.items()
    }


def processor_text_inputs(
    processor: Any,
    query_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    device: torch.device,
) -> dict[str, Tensor]:
    """Tokenize only text rows; no image or label fields are consulted."""

    texts = [str(row["caption"]) for row in query_rows]
    text_inputs = processor(text=texts, return_tensors="pt", padding="max_length")
    input_ids = text_inputs["input_ids"]
    attention_mask = text_inputs.get("attention_mask")
    if attention_mask is None:
        pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", 0)
        attention_mask = input_ids.ne(0 if pad_id is None else int(pad_id))
    content_mask = attention_mask.bool()
    special_ids = getattr(getattr(processor, "tokenizer", None), "all_special_ids", ())
    for token_id in special_ids or ():
        content_mask = content_mask & input_ids.ne(int(token_id))
    text_inputs["input_ids"] = input_ids.to(device)
    text_inputs["attention_mask"] = attention_mask.to(device)
    text_inputs["content_mask"] = content_mask.to(device)
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in text_inputs.items()
    }


def encode_real_images(
    backbone: Siglip2Backbone,
    processor: Any,
    pair_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    no_grad: bool = True,
    max_num_patches: int | None = None,
    synchronized_transform: Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]]
    | None = None,
) -> ImageEncoding:
    image_inputs = processor_image_inputs(
        processor,
        pair_rows,
        device,
        max_num_patches=max_num_patches,
        synchronized_transform=synchronized_transform,
    )
    context = torch.no_grad() if no_grad else nullcontext()
    with context, _device_autocast(device, dtype):
        return backbone.encode_images(**image_inputs)


def encode_real_text(
    backbone: Siglip2Backbone,
    processor: Any,
    query_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    no_grad: bool = True,
) -> TextEncoding:
    text_inputs = processor_text_inputs(processor, query_rows, device)
    context = torch.no_grad() if no_grad else nullcontext()
    with context, _device_autocast(device, dtype):
        return backbone.encode_text(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
            content_mask=text_inputs["content_mask"],
        )


def encode_real_features(
    backbone: Any,
    processor: Any,
    pair_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    query_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    no_grad: bool = True,
    max_num_patches: int | None = None,
    synchronized_transform: Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]]
    | None = None,
) -> RawFeatureBatch:
    image = encode_real_images(
        backbone,
        processor,
        pair_rows,
        device,
        dtype=dtype,
        no_grad=no_grad,
        max_num_patches=max_num_patches,
        synchronized_transform=synchronized_transform,
    )
    text = encode_real_text(
        backbone,
        processor,
        query_rows,
        device,
        dtype=dtype,
        no_grad=no_grad,
    )
    return RawFeatureBatch(
        frame_tokens=image.patch_tokens,
        frame_embeddings=image.pooled_embedding,
        text_tokens=text.token_embeddings,
        text_embeddings=text.pooled_embedding,
        text_mask=text.attention_mask,
        patch_valid_mask=image.patch_valid_mask,
        spatial_shapes=image.spatial_shapes,
    )


def build_relevance_masks(
    query_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    pair_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[str, int | bool]]:
    """Build explicit positives/ignored masks without text-collision inference."""

    pair_index = {
        str(row["canonical_pair_id"]): index for index, row in enumerate(pair_rows)
    }
    positive = torch.zeros(
        (len(query_rows), len(pair_rows)), dtype=torch.bool, device=device
    )
    ignored = torch.zeros_like(positive)
    for query_index, row in enumerate(query_rows):
        positive_ids = row.get("positive_pair_ids")
        if not isinstance(positive_ids, list) or not positive_ids:
            raise ValueError(
                f"query {row.get('caption_id', query_index)} lacks explicit positives"
            )
        for item_id in positive_ids:
            index = pair_index.get(str(item_id))
            if index is not None:
                positive[query_index, index] = True
        ignored_ids = row.get("ignored_pair_ids", [])
        if not isinstance(ignored_ids, list):
            raise TypeError("ignored_pair_ids must be a list when present")
        for item_id in ignored_ids:
            index = pair_index.get(str(item_id))
            if index is not None:
                ignored[query_index, index] = True
    if torch.any(positive.sum(dim=1) == 0):
        raise ValueError("every query needs a positive in the logical batch")
    if torch.any(positive & ignored):
        raise ValueError("positive and ignored relevance overlap")
    return positive, ignored, {
        "query_count": len(query_rows),
        "pair_count": len(pair_rows),
        "multi_positive_queries": int((positive.sum(dim=1) > 1).sum()),
        "text_collision_not_used_as_positive": True,
    }
