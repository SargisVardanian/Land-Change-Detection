"""Mask-free real-image and text batching for the SigLIP-2 track."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
import inspect
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor

from ..backbones.siglip2 import ImageEncoding, Siglip2Backbone, TextEncoding
from .chunking import (
    SynchronizedChunkPlan,
    apply_synchronized_chunk_plan,
    build_synchronized_chunk_plan,
    chunk_patch_coordinates,
)


# The pinned SigLIP2 tokenizer has an effectively unbounded tokenizer-side
# ``model_max_length``.  The audited processor gate, however, uses a fixed
# 64-token scientific contract.  Keep this policy in one model-side helper so
# training, milestone evaluation and route audits cannot silently diverge.
PINNED_SIGLIP2_MAX_TEXT_LENGTH = 64


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
    native_image_size: Tensor | None = None
    processed_patch_grid: Tensor | None = None
    transform_hash: str | None = None
    token_coordinates: Tensor | None = None
    processing_mode: str = "DIRECT_NAFLEX"
    force_region_reduction: bool = False


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
    synchronized_sequence_transform: Callable[
        [list[Image.Image]], list[Image.Image]
    ]
    | None = None,
    is_naflex: bool | None = None,
    max_text_length: int = PINNED_SIGLIP2_MAX_TEXT_LENGTH,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Decode each T1/T2 exactly once and tokenize the paired queries."""

    image_inputs = processor_image_inputs(
        processor,
        pair_rows,
        device,
        max_num_patches=max_num_patches,
        synchronized_transform=synchronized_transform,
        synchronized_sequence_transform=synchronized_sequence_transform,
        is_naflex=is_naflex,
    )
    text_inputs = processor_text_inputs(
        processor, query_rows, device, max_text_length=max_text_length
    )
    return image_inputs, text_inputs


def processor_image_inputs(
    processor: Any,
    pair_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    device: torch.device,
    *,
    max_num_patches: int | None = None,
    synchronized_transform: Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]]
    | None = None,
    synchronized_sequence_transform: Callable[
        [list[Image.Image]], list[Image.Image]
    ]
    | None = None,
    is_naflex: bool | None = None,
) -> dict[str, Any]:
    """Decode T1/T2 images and return the processor's tensor inputs."""

    if synchronized_transform is not None and synchronized_sequence_transform is not None:
        raise ValueError(
            "provide either synchronized_transform or synchronized_sequence_transform"
        )
    images: list[Image.Image] = []
    frame_count: int | None = None
    native_sizes: list[tuple[int, int]] = []
    for row in pair_rows:
        frame_paths = row.get("frames")
        if frame_paths is None:
            frame_paths = [row["t1_path"], row["t2_path"]]
        if not isinstance(frame_paths, Sequence) or isinstance(frame_paths, (str, bytes)):
            raise TypeError("frames must be a sequence of image paths")
        if len(frame_paths) < 2:
            raise ValueError("temporal items require at least two frames")
        if frame_count is None:
            frame_count = len(frame_paths)
        elif len(frame_paths) != frame_count:
            raise ValueError("all temporal items in a batch must have the same frame count")
        loaded: list[Image.Image] = []
        for frame_path in frame_paths:
            path = Path(str(frame_path))
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                loaded.append(image.convert("RGB"))
        native_sizes.extend((image.height, image.width) for image in loaded)
        if synchronized_sequence_transform is not None:
            loaded = synchronized_sequence_transform(loaded)
            if len(loaded) != frame_count:
                raise ValueError("synchronized sequence transform changed frame count")
        if synchronized_transform is not None:
            if len(loaded) != 2:
                raise ValueError(
                    "synchronized_transform is only valid for two-frame items; "
                    "use synchronized_sequence_transform for T>2"
                )
            loaded[0], loaded[1] = synchronized_transform(loaded[0], loaded[1])
        if any(image.size != loaded[0].size for image in loaded[1:]):
            raise ValueError(
                "temporal image geometry differs; apply one synchronized transform before processing"
            )
        images.extend(loaded)

    image_processor = getattr(processor, "image_processor", processor)
    processor_name = type(image_processor).__name__.lower()
    if is_naflex is None:
        is_naflex = "siglip2" in processor_name or hasattr(
            image_processor, "max_num_patches"
        )
    elif not isinstance(is_naflex, bool):
        raise TypeError("is_naflex must be a boolean when supplied")
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
    if not is_naflex:
        # Fixed-resolution SigLIP processors may expose fields with names used
        # by NaFlex, but their masks are pixel/image metadata rather than
        # native patch-valid masks.  Passing them into the variable-token
        # contract produces a false grid-count mismatch.  Fixed models use
        # their native square grid derived by the backbone instead.
        image_inputs.pop("pixel_attention_mask", None)
        image_inputs.pop("spatial_shapes", None)
    pixel_values = image_inputs["pixel_values"]
    if frame_count is None:
        raise ValueError("at least one temporal item is required")
    pair_count = len(pair_rows)
    image_inputs["pixel_values"] = pixel_values.reshape(
        pair_count, frame_count, *pixel_values.shape[1:]
    ).to(device)
    image_inputs["native_image_size"] = torch.tensor(
        native_sizes, dtype=torch.long, device=device
    ).reshape(pair_count, frame_count, 2)
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
                value = value.reshape(pair_count, frame_count, -1)
            else:
                value = value.reshape(pair_count, frame_count, 2)
            image_inputs[key] = value.to(device)

    processed_shapes = image_inputs.get("spatial_shapes")
    if processed_shapes is not None:
        if processed_shapes.shape != (pair_count, frame_count, 2):
            raise ValueError("processor spatial_shapes must reshape to [B,T,2]")
        if not torch.equal(processed_shapes, processed_shapes[:, :1].expand_as(processed_shapes)):
            raise ValueError(
                "all frames of a temporal item must use a compatible processed NaFlex patch grid"
            )
        attention_mask = image_inputs.get("pixel_attention_mask")
        if attention_mask is not None:
            counts = attention_mask.sum(dim=-1)
            expected = processed_shapes[..., 0] * processed_shapes[..., 1]
            if not torch.equal(counts, expected):
                raise ValueError(
                    "pixel_attention_mask counts disagree with spatial_shapes"
                )

    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in image_inputs.items()
    }


def processor_text_inputs(
    processor: Any,
    query_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    device: torch.device,
    *,
    max_text_length: int = PINNED_SIGLIP2_MAX_TEXT_LENGTH,
) -> dict[str, Tensor]:
    """Tokenize only text rows; no image or label fields are consulted."""

    if not isinstance(max_text_length, int) or max_text_length <= 0:
        raise ValueError("max_text_length must be a positive integer")
    texts = [str(row["caption"]) for row in query_rows]
    call_signature = inspect.signature(processor.__call__)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in call_signature.parameters.values()
    )
    if accepts_kwargs or {"truncation", "max_length"}.issubset(call_signature.parameters):
        text_inputs = processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_length,
        )
    else:
        # Small test doubles from the legacy contract predate the explicit
        # tokenizer policy.  Real SigLIP2 processors accept **kwargs and never
        # use this compatibility branch.
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


def _query_text(row: dict[str, Any]) -> str:
    value = row.get("caption", row.get("text"))
    if value is None:
        raise ValueError("query row has neither caption nor text")
    return str(value)


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def text_preprocessing_audit(
    processor: Any,
    query_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    max_text_length: int = PINNED_SIGLIP2_MAX_TEXT_LENGTH,
) -> dict[str, Any]:
    """Audit the exact tokenizer policy without mutating source text.

    ``raw_token_lengths`` are measured before truncation and are therefore the
    only lengths used to calculate the truncation fraction.  Processed lengths
    are also reported to make padding and the fixed maximum explicit.
    """

    if not isinstance(max_text_length, int) or max_text_length <= 0:
        raise ValueError("max_text_length must be a positive integer")
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("processor must expose a tokenizer for the audit")
    texts = [_query_text(row) for row in query_rows]
    raw = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    processed = tokenizer(
        texts,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_text_length,
    )
    raw_ids = raw.get("input_ids")
    processed_ids = processed.get("input_ids")
    if not isinstance(raw_ids, list) or not isinstance(processed_ids, list):
        raise TypeError("tokenizer audit requires list input_ids outputs")
    raw_lengths = [len(row) for row in raw_ids]
    processed_lengths = [
        min(len(row), max_text_length)
        for row in processed_ids
    ]
    lower_case = getattr(tokenizer, "do_lower_case", None)
    if lower_case is True:
        lowercase_policy = "tokenizer_lowercases"
    elif lower_case is False:
        lowercase_policy = "tokenizer_preserves_case"
    else:
        lowercase_policy = "dataset_text_preserved_tokenizer_default"
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    return {
        "query_count": len(texts),
        "lowercase_policy": lowercase_policy,
        "tokenizer_do_lower_case": lower_case,
        "tokenizer_model_max_length": (
            int(tokenizer_max)
            if isinstance(tokenizer_max, int) and tokenizer_max < 10**9
            else "unbounded_or_unspecified"
        ),
        "max_text_length": max_text_length,
        "fraction_of_queries_truncated": float(
            sum(length > max_text_length for length in raw_lengths) / max(len(raw_lengths), 1)
        ),
        "max_observed_token_length": max(raw_lengths, default=0),
        "raw_token_length_percentiles": {
            "p50": _percentile(raw_lengths, 50),
            "p90": _percentile(raw_lengths, 90),
            "p95": _percentile(raw_lengths, 95),
            "p99": _percentile(raw_lengths, 99),
        },
        "processed_token_length_percentiles": {
            "p50": _percentile(processed_lengths, 50),
            "p90": _percentile(processed_lengths, 90),
            "p95": _percentile(processed_lengths, 95),
            "p99": _percentile(processed_lengths, 99),
        },
        "padding": "max_length",
        "truncation": "right_side_tokenizer_truncation",
        "text_source_mutated": False,
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
    synchronized_sequence_transform: Callable[
        [list[Image.Image]], list[Image.Image]
    ]
    | None = None,
    is_naflex: bool | None = None,
) -> ImageEncoding:
    if is_naflex is None:
        is_naflex = getattr(backbone, "is_naflex", None)
    image_inputs = processor_image_inputs(
        processor,
        pair_rows,
        device,
        max_num_patches=max_num_patches,
        synchronized_transform=synchronized_transform,
        synchronized_sequence_transform=synchronized_sequence_transform,
        is_naflex=is_naflex,
    )
    context = torch.no_grad() if no_grad else nullcontext()
    with context, _device_autocast(device, dtype):
        return backbone.encode_images(**image_inputs)


def encode_large_scene_images(
    backbone: Siglip2Backbone,
    processor: Any,
    pair_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    device: torch.device,
    *,
    chunk_size: tuple[int, int],
    overlap: tuple[int, int] = (0, 0),
    max_num_patches: int,
    tile_batch_size: int = 4,
    overview_max_num_patches: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    no_grad: bool = True,
) -> ImageEncoding:
    """Encode synchronized native-resolution chunks through one shared backbone.

    Chunk embeddings are never exposed as ANN keys. Their native-coordinate
    patch tokens are packed into one temporal item and consumed by the bounded,
    query-independent reducer before the canonical single-vector output.
    """

    if not pair_rows:
        raise ValueError("at least one large-scene temporal item is required")
    if not backbone.is_naflex:
        raise ValueError("large-scene native chunking requires a NaFlex backbone")
    if max_num_patches <= 0:
        raise ValueError("max_num_patches must be explicit and positive")
    if tile_batch_size <= 0 or overview_max_num_patches <= 0:
        raise ValueError("tile batch size and overview budget must be positive")
    all_chunk_frames: list[Image.Image] = []
    overview_frames: list[Image.Image] = []
    chunk_records: list[tuple[int, SynchronizedChunkPlan, int]] = []
    frame_count: int | None = None
    plans: list[SynchronizedChunkPlan] = []
    for pair_index, row in enumerate(pair_rows):
        frame_paths = row.get("frames")
        if frame_paths is None:
            frame_paths = [row["t1_path"], row["t2_path"]]
        if not isinstance(frame_paths, Sequence) or isinstance(
            frame_paths, (str, bytes)
        ):
            raise TypeError("frames must be a sequence of image paths")
        if len(frame_paths) < 2:
            raise ValueError("temporal items require at least two frames")
        if frame_count is None:
            frame_count = len(frame_paths)
        elif len(frame_paths) != frame_count:
            raise ValueError("all temporal items must have the same frame count")
        frames: list[Image.Image] = []
        for value in frame_paths:
            path = Path(str(value))
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))
        if any(frame.size != frames[0].size for frame in frames[1:]):
            raise ValueError("large-scene temporal frames must share native geometry")
        overview_frames.extend(frames)
        plan = build_synchronized_chunk_plan(
            (frames[0].height, frames[0].width),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        plans.append(plan)
        for chunk_index, temporal_chunk in enumerate(
            apply_synchronized_chunk_plan(frames, plan)
        ):
            chunk_records.append((pair_index, plan, chunk_index))
            all_chunk_frames.extend(temporal_chunk)
    if frame_count is None:
        raise AssertionError("frame count was not initialized")
    total_chunks = len(chunk_records)
    combined_hash = sha256(
        "\n".join(plan.transform_hash for plan in plans).encode("utf-8")
    ).hexdigest()
    encoded_chunks: list[
        tuple[Tensor, Tensor, Tensor, Tensor, tuple[int, int]]
    ] = []
    for start in range(0, total_chunks, tile_batch_size):
        stop = min(start + tile_batch_size, total_chunks)
        frame_start = start * frame_count
        frame_stop = stop * frame_count
        processed = processor(
            images=all_chunk_frames[frame_start:frame_stop],
            max_num_patches=max_num_patches,
            return_tensors="pt",
        )
        pixel_values = processed["pixel_values"]
        patch_mask = processed.get("pixel_attention_mask")
        spatial_shapes = processed.get("spatial_shapes")
        if patch_mask is None or spatial_shapes is None:
            raise ValueError("NaFlex processor must return patch masks and spatial shapes")
        chunk_batch = stop - start
        pixel_values = pixel_values.reshape(
            chunk_batch, frame_count, *pixel_values.shape[1:]
        ).to(device)
        patch_mask = patch_mask.reshape(chunk_batch, frame_count, -1).to(device)
        spatial_shapes = spatial_shapes.reshape(chunk_batch, frame_count, 2).to(device)
        if not torch.equal(
            spatial_shapes, spatial_shapes[:, :1].expand_as(spatial_shapes)
        ):
            raise ValueError("every temporal chunk must have a synchronized patch grid")
        native_sizes = torch.tensor(
            [
                [
                    plan.chunks[chunk_index].valid_height,
                    plan.chunks[chunk_index].valid_width,
                ]
                for _, plan, chunk_index in chunk_records[start:stop]
            ],
            dtype=torch.long,
            device=device,
        ).view(chunk_batch, 1, 2).expand(chunk_batch, frame_count, 2).clone()
        with (torch.no_grad() if no_grad else nullcontext()), _device_autocast(
            device, dtype
        ):
            encoded_batch = backbone.encode_images(
                pixel_values,
                pixel_attention_mask=patch_mask,
                spatial_shapes=spatial_shapes,
                native_image_size=native_sizes,
                transform_hash=combined_hash,
            )
        for local_index in range(chunk_batch):
            grid_values = tuple(
                int(value) for value in spatial_shapes[local_index, 0].tolist()
            )
            encoded_chunks.append(
                (
                    encoded_batch.patch_tokens[local_index],
                    encoded_batch.patch_valid_mask[local_index],
                    encoded_batch.spatial_shapes[local_index],
                    encoded_batch.pooled_embedding[local_index],
                    grid_values,
                )
            )

    overview_processed = processor(
        images=overview_frames,
        max_num_patches=overview_max_num_patches,
        return_tensors="pt",
    )
    overview_pixels = overview_processed["pixel_values"].reshape(
        len(pair_rows), frame_count, *overview_processed["pixel_values"].shape[1:]
    ).to(device)
    overview_mask = overview_processed.get("pixel_attention_mask")
    overview_shapes = overview_processed.get("spatial_shapes")
    if overview_mask is None or overview_shapes is None:
        raise ValueError("NaFlex overview must return patch masks and spatial shapes")
    overview_mask = overview_mask.reshape(len(pair_rows), frame_count, -1).to(device)
    overview_shapes = overview_shapes.reshape(len(pair_rows), frame_count, 2).to(device)
    overview_native_size = torch.tensor(
        [[plan.native_height, plan.native_width] for plan in plans],
        dtype=torch.long,
        device=device,
    ).view(len(pair_rows), 1, 2).expand(len(pair_rows), frame_count, 2).clone()
    with (torch.no_grad() if no_grad else nullcontext()), _device_autocast(
        device, dtype
    ):
        overview = backbone.encode_images(
            overview_pixels,
            pixel_attention_mask=overview_mask,
            spatial_shapes=overview_shapes,
            native_image_size=overview_native_size,
            transform_hash=combined_hash,
        )

    hidden = encoded_chunks[0][0].shape[-1]
    chunks_per_pair = [len(plan.chunks) for plan in plans]
    packed_counts = [0] * len(pair_rows)
    for encoded_index, (pair_index, _plan, _chunk_index) in enumerate(chunk_records):
        packed_counts[pair_index] += encoded_chunks[encoded_index][0].shape[1]
    packed_count = max(packed_counts)
    batch = len(pair_rows)
    tokens = encoded_chunks[0][0].new_zeros(batch, frame_count, packed_count, hidden)
    valid = torch.zeros(
        batch, frame_count, packed_count, dtype=torch.bool, device=device
    )
    coordinates = tokens.new_zeros(batch, frame_count, packed_count, 2)
    pair_chunk_offsets = [0] * batch
    chunk_valid_counts: list[list[int]] = [[] for _ in range(batch)]
    for encoded_index, (pair_index, plan, chunk_index) in enumerate(chunk_records):
        chunk_tokens, processor_valid, _chunk_shapes, _chunk_pooled, grid = (
            encoded_chunks[encoded_index]
        )
        patch_count = chunk_tokens.shape[1]
        offset = pair_chunk_offsets[pair_index]
        pair_chunk_offsets[pair_index] += patch_count
        chunk_valid_counts[pair_index].append(int(processor_valid[0].sum()))
        tokens[pair_index, :, offset : offset + patch_count] = chunk_tokens
        chunk_coordinates, geometry_valid = chunk_patch_coordinates(
            plan.chunks[chunk_index],
            native_size=(plan.native_height, plan.native_width),
            patch_grid=grid,
            device=device,
            dtype=chunk_tokens.dtype,
        )
        coordinate_count = chunk_coordinates.shape[0]
        coordinates[
            pair_index, :, offset : offset + coordinate_count
        ] = chunk_coordinates.view(1, coordinate_count, 2)
        geometry_mask = torch.zeros(
            patch_count, dtype=torch.bool, device=device
        )
        geometry_mask[:coordinate_count] = geometry_valid
        valid[pair_index, :, offset : offset + patch_count] = (
            processor_valid & geometry_mask.view(1, patch_count)
        )
    pooled = overview.pooled_embedding
    valid_counts = valid.sum(dim=-1)
    if not torch.equal(valid_counts, valid_counts[:, :1].expand_as(valid_counts)):
        raise ValueError("synchronized chunks produced unequal temporal valid counts")
    packed_shapes = torch.stack(
        (torch.ones_like(valid_counts), valid_counts), dim=-1
    )
    original_sizes = torch.tensor(
        [[plan.native_height, plan.native_width] for plan in plans],
        dtype=torch.long,
        device=device,
    ).view(batch, 1, 2).expand(batch, frame_count, 2).clone()
    return ImageEncoding(
        patch_tokens=tokens,
        pooled_embedding=pooled,
        patch_valid_mask=valid,
        spatial_shapes=packed_shapes,
        native_image_size=original_sizes,
        processed_patch_grid=packed_shapes,
        transform_hash=combined_hash,
        token_coordinates=coordinates,
        chunk_plan_hashes=tuple(plan.transform_hash for plan in plans),
        processing_mode="HIERARCHICAL_NATIVE",
        force_region_reduction=True,
        chunk_counts=tuple(chunks_per_pair),
        chunk_valid_patch_counts=tuple(
            tuple(values) for values in chunk_valid_counts
        ),
        chunk_size=tuple(map(int, chunk_size)),
        chunk_overlap=tuple(map(int, overlap)),
        tile_batch_size=int(tile_batch_size),
        overview_processed_patch_grid=overview.processed_patch_grid,
    )


def encode_real_text(
    backbone: Siglip2Backbone,
    processor: Any,
    query_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    no_grad: bool = True,
    max_text_length: int = PINNED_SIGLIP2_MAX_TEXT_LENGTH,
) -> TextEncoding:
    text_inputs = processor_text_inputs(
        processor, query_rows, device, max_text_length=max_text_length
    )
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
    synchronized_sequence_transform: Callable[
        [list[Image.Image]], list[Image.Image]
    ]
    | None = None,
    is_naflex: bool | None = None,
    representation_mode: str = "DIRECT_NAFLEX",
    hierarchical_chunk_size: tuple[int, int] = (128, 128),
    hierarchical_tile_batch_size: int = 4,
    hierarchical_overview_max_num_patches: int | None = None,
    max_text_length: int = PINNED_SIGLIP2_MAX_TEXT_LENGTH,
) -> RawFeatureBatch:
    if representation_mode not in {"DIRECT_NAFLEX", "HIERARCHICAL_NATIVE"}:
        raise ValueError(f"unsupported representation_mode: {representation_mode}")
    if representation_mode == "DIRECT_NAFLEX":
        image = encode_real_images(
            backbone,
            processor,
            pair_rows,
            device,
            dtype=dtype,
            no_grad=no_grad,
            max_num_patches=max_num_patches,
            synchronized_transform=synchronized_transform,
            synchronized_sequence_transform=synchronized_sequence_transform,
            is_naflex=is_naflex,
        )
    else:
        if max_num_patches is None:
            raise ValueError("hierarchical mode requires max_num_patches")
        overview_budget = (
            max_num_patches
            if hierarchical_overview_max_num_patches is None
            else hierarchical_overview_max_num_patches
        )
        image = encode_large_scene_images(
            backbone,
            processor,
            pair_rows,
            device,
            chunk_size=hierarchical_chunk_size,
            max_num_patches=max_num_patches,
            tile_batch_size=hierarchical_tile_batch_size,
            overview_max_num_patches=overview_budget,
            dtype=dtype,
            no_grad=no_grad,
        )
    text = encode_real_text(
        backbone,
        processor,
        query_rows,
        device,
        dtype=dtype,
        no_grad=no_grad,
        max_text_length=max_text_length,
    )
    return RawFeatureBatch(
        frame_tokens=image.patch_tokens,
        frame_embeddings=image.pooled_embedding,
        text_tokens=text.token_embeddings,
        text_embeddings=text.pooled_embedding,
        text_mask=text.attention_mask,
        patch_valid_mask=image.patch_valid_mask,
        spatial_shapes=image.spatial_shapes,
        native_image_size=image.native_image_size,
        processed_patch_grid=image.processed_patch_grid,
        transform_hash=image.transform_hash,
        token_coordinates=image.token_coordinates,
        processing_mode=image.processing_mode,
        force_region_reduction=image.force_region_reduction,
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
