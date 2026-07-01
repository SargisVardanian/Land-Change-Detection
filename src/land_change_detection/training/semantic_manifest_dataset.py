from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from land_change_detection.semantic_surface import select_prithvi_bands


@dataclass(frozen=True)
class SemanticManifestRecord:
    sample_id: str
    dataset_name: str
    task: str
    before_path: str
    after_path: str
    change_mask_path: str | None = None
    semantic_before_path: str | None = None
    semantic_after_path: str | None = None
    before_ms_path: str | None = None
    after_ms_path: str | None = None
    caption: str | None = None
    split: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SemanticManifestRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            dataset_name=str(payload["dataset_name"]),
            task=str(payload["task"]),
            before_path=str(payload["before_path"]),
            after_path=str(payload["after_path"]),
            change_mask_path=str(payload["change_mask_path"]) if payload.get("change_mask_path") else None,
            semantic_before_path=str(payload["semantic_before_path"]) if payload.get("semantic_before_path") else None,
            semantic_after_path=str(payload["semantic_after_path"]) if payload.get("semantic_after_path") else None,
            before_ms_path=str(payload["before_ms_path"]) if payload.get("before_ms_path") else None,
            after_ms_path=str(payload["after_ms_path"]) if payload.get("after_ms_path") else None,
            caption=str(payload["caption"]) if payload.get("caption") is not None else None,
            split=str(payload["split"]) if payload.get("split") is not None else None,
            metadata=dict(payload.get("metadata", {})),
        )


def load_semantic_manifest(path: str | Path) -> list[SemanticManifestRecord]:
    rows: list[SemanticManifestRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(SemanticManifestRecord.from_dict(json.loads(line)))
    return rows


def filter_semantic_supervised_records(records: list[SemanticManifestRecord]) -> list[SemanticManifestRecord]:
    return [
        record
        for record in records
        if record.semantic_before_path
        and record.semantic_after_path
        and record.task == "semantic_transition_segmentation"
    ]


class SemanticManifestDataset:
    def __init__(self, records: list[SemanticManifestRecord], *, input_channels: int = 6, ignore_index: int = -1):
        self.records = records
        self.input_channels = input_channels
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        record = self.records[index]
        before = _load_input_tensor(record.before_path, record.before_ms_path, self.input_channels)
        after = _load_input_tensor(record.after_path, record.after_ms_path, self.input_channels)

        if record.change_mask_path:
            change_mask = np.array(Image.open(record.change_mask_path), dtype=np.float32)
            if change_mask.ndim == 3:
                change_mask = change_mask[..., 0]
            change_mask = (change_mask > 0).astype(np.float32)
        else:
            change_mask = np.zeros(before.shape[1:], dtype=np.float32)

        if record.semantic_before_path and record.semantic_after_path:
            before_target = np.array(Image.open(record.semantic_before_path), dtype=np.int64)
            after_target = np.array(Image.open(record.semantic_after_path), dtype=np.int64)
            if before_target.ndim == 3:
                before_target = before_target[..., 0]
            if after_target.ndim == 3:
                after_target = after_target[..., 0]
        else:
            before_target = np.full(before.shape[1:], self.ignore_index, dtype=np.int64)
            after_target = np.full(after.shape[1:], self.ignore_index, dtype=np.int64)

        return {
            "sample_id": record.sample_id,
            "task": record.task,
            "dominant_transition": str((record.metadata or {}).get("dominant_transition") or (record.metadata or {}).get("transition_label") or ""),
            "before": torch.tensor(before, dtype=torch.float32),
            "after": torch.tensor(after, dtype=torch.float32),
            "before_target": torch.tensor(before_target, dtype=torch.long),
            "after_target": torch.tensor(after_target, dtype=torch.long),
            "change_target": torch.tensor(change_mask, dtype=torch.float32),
        }


def _adapt_channels(arr, channels: int):
    import numpy as np

    if arr.shape[0] == channels:
        return arr
    if arr.shape[0] > channels:
        return arr[:channels]
    repeats = int(np.ceil(channels / arr.shape[0]))
    tiled = np.tile(arr, (repeats, 1, 1))
    return tiled[:channels]


def _load_input_tensor(rgb_path: str, ms_path: str | None, channels: int):
    import numpy as np
    from PIL import Image

    if ms_path:
        ms_arr = _load_multispectral(ms_path)
        if channels == 6 and ms_arr.shape[0] >= 13:
            return select_prithvi_bands(ms_arr)
        return _adapt_channels(ms_arr, channels)

    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    rgb = np.moveaxis(rgb, -1, 0)
    return _adapt_channels(rgb, channels)


def _load_multispectral(path: str):
    import numpy as np

    source = Path(path)
    if source.suffix.lower() == ".npy":
        arr = np.load(source)
    else:
        import tifffile

        arr = tifffile.imread(source)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[0] not in (3, 6, 9, 13):
        arr = np.moveaxis(arr, -1, 0)
    return arr.astype(np.float32)
