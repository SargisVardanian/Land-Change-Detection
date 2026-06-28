from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class MaskRLE:
    height: int
    width: int
    counts: tuple[int, ...]
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventIndexRecord:
    event_id: str
    pair_id: str
    slot_id: int
    event_embedding: list[float]
    presence_score: float
    mask_rle: MaskRLE
    bbox_xyxy: tuple[int, int, int, int] | None
    transition_label: str | None = None
    short_caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairIndexRecord:
    pair_id: str
    split: str
    dataset: str
    t1: str
    t2: str
    global_pair_embedding: list[float]
    event_ids: tuple[str, ...]
    captions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def encode_binary_mask_rle(mask: Tensor, threshold: float = 0.5) -> MaskRLE:
    if mask.ndim != 2:
        raise ValueError("mask must have shape [H, W].")
    binary = (mask.detach().cpu().to(torch.float32) >= threshold).flatten().tolist()
    counts: list[int] = []
    current = 0
    run = 0
    for value in binary:
        bit = 1 if value else 0
        if bit == current:
            run += 1
        else:
            counts.append(run)
            current = bit
            run = 1
    counts.append(run)
    return MaskRLE(height=int(mask.shape[0]), width=int(mask.shape[1]), counts=tuple(counts), threshold=float(threshold))


def decode_binary_mask_rle(rle: MaskRLE) -> Tensor:
    values: list[int] = []
    bit = 0
    for count in rle.counts:
        values.extend([bit] * int(count))
        bit = 1 - bit
    expected = rle.height * rle.width
    if len(values) != expected:
        raise ValueError(f"RLE length {len(values)} does not match mask size {expected}.")
    return torch.tensor(values, dtype=torch.bool).view(rle.height, rle.width)


def mask_bbox_xyxy(mask: Tensor, threshold: float = 0.5) -> tuple[int, int, int, int] | None:
    if mask.ndim != 2:
        raise ValueError("mask must have shape [H, W].")
    coords = (mask.detach().cpu().to(torch.float32) >= threshold).nonzero(as_tuple=False)
    if coords.numel() == 0:
        return None
    y_min, x_min = coords.min(dim=0).values.tolist()
    y_max, x_max = coords.max(dim=0).values.tolist()
    return int(x_min), int(y_min), int(x_max), int(y_max)


def build_event_records(
    pair_id: str,
    event_embeddings: Tensor,
    event_presence: Tensor,
    event_masks: Tensor,
    grid_height: int,
    grid_width: int,
    presence_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    metadata: dict[str, Any] | None = None,
) -> tuple[EventIndexRecord, ...]:
    if event_embeddings.ndim != 2 or event_presence.ndim != 1 or event_masks.ndim != 2:
        raise ValueError("Expected event_embeddings [K,D], event_presence [K], event_masks [K,N].")
    if event_embeddings.shape[0] != event_presence.shape[0] or event_embeddings.shape[0] != event_masks.shape[0]:
        raise ValueError("Event tensors must agree on K.")
    if event_masks.shape[1] != grid_height * grid_width:
        raise ValueError("event_masks flattened size does not match grid dimensions.")
    records: list[EventIndexRecord] = []
    probs = torch.sigmoid(event_presence.detach().cpu().to(torch.float32))
    for slot_id, presence in enumerate(probs.tolist()):
        if presence < presence_threshold:
            continue
        mask = event_masks[slot_id].detach().cpu().to(torch.float32).view(grid_height, grid_width)
        event_id = f"{pair_id}::event_{slot_id:02d}"
        records.append(
            EventIndexRecord(
                event_id=event_id,
                pair_id=pair_id,
                slot_id=slot_id,
                event_embedding=event_embeddings[slot_id].detach().cpu().to(torch.float32).tolist(),
                presence_score=float(presence),
                mask_rle=encode_binary_mask_rle(mask, threshold=mask_threshold),
                bbox_xyxy=mask_bbox_xyxy(mask, threshold=mask_threshold),
                metadata=dict(metadata or {}),
            )
        )
    return tuple(records)
