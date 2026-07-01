from __future__ import annotations

import torch
from torch.nn import functional as F

from land_change_detection.data.temporal_sample import TemporalChangeBatch, TemporalChangeSample


def _resize_image(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)


def _resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 2:
        resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=size, mode="nearest").squeeze(0).squeeze(0)
        return resized.to(mask.dtype)
    if mask.ndim == 3:
        resized = F.interpolate(mask.unsqueeze(0).float(), size=size, mode="nearest").squeeze(0)
        return resized.to(mask.dtype)
    raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")


def collate_temporal_change_samples(samples: list[TemporalChangeSample], image_size: int | tuple[int, int] = 256) -> TemporalChangeBatch:
    if not samples:
        raise ValueError("Cannot collate an empty temporal batch.")
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    max_t = max(int(sample.images.shape[0]) for sample in samples)
    first = samples[0]
    channels = int(first.images.shape[1])
    images = first.images.new_zeros((len(samples), max_t, channels, size[0], size[1]))
    timestamps = first.timestamps.new_zeros((len(samples), max_t))
    valid = torch.zeros((len(samples), max_t), dtype=torch.bool)
    binary_masks: list[torch.Tensor] = []
    semantic_masks: list[torch.Tensor] = []
    transition_labels: list[torch.Tensor] = []
    event_components = []
    event_texts: list[list[str]] = []
    metadata = []

    for batch_index, sample in enumerate(samples):
        sample.validate()
        t = int(sample.images.shape[0])
        if int(sample.images.shape[1]) != channels:
            raise ValueError("All samples in a temporal batch must have the same channel count.")
        resized_frames = [_resize_image(sample.images[idx], size) for idx in range(t)]
        images[batch_index, :t] = torch.stack(resized_frames)
        timestamps[batch_index, :t] = sample.timestamps
        if sample.temporal_valid_mask is None:
            valid[batch_index, :t] = True
        else:
            valid[batch_index, :t] = sample.temporal_valid_mask.to(torch.bool)
        if sample.binary_mask is not None:
            binary_masks.append(_resize_mask(sample.binary_mask, size))
        if sample.semantic_masks is not None:
            semantic_masks.append(_resize_mask(sample.semantic_masks, size))
        if sample.transition_labels is not None:
            transition_labels.append(_resize_mask(sample.transition_labels, size))
        event_components.append(sample.event_components)
        event_texts.append(sample.event_texts or [])
        metadata.append(sample.metadata)

    return TemporalChangeBatch(
        images=images,
        timestamps=timestamps,
        temporal_valid_mask=valid,
        pair_ids=[sample.pair_id for sample in samples],
        captions=[sample.captions for sample in samples],
        binary_mask=torch.stack(binary_masks) if len(binary_masks) == len(samples) else None,
        semantic_masks=torch.stack(semantic_masks) if len(semantic_masks) == len(samples) else None,
        transition_labels=torch.stack(transition_labels) if len(transition_labels) == len(samples) else None,
        event_components=event_components,
        event_texts=event_texts,
        metadata=metadata,
    )
