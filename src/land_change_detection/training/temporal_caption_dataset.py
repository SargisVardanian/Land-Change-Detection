from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as TF

from land_change_detection.data.event_targets import ComponentTargets, TemporalContext, build_component_targets
from land_change_detection.data.unichange_mci import _load_mask, _load_rgb
from land_change_detection.temporal_caption_manifest import SCHEMA_VERSION, load_json_rows


@dataclass(frozen=True)
class TemporalCaptionItem:
    pair_id: str
    dataset_name: str
    t1: Tensor
    t2: Tensor
    captions: list[str]
    mask: Tensor
    components: ComponentTargets
    temporal_context: TemporalContext
    metadata: dict[str, Any]


def _blank_mask_like(path: str | Path, image_size: int | None) -> Tensor:
    with Image.open(path) as image:
        width, height = (image_size, image_size) if image_size is not None else (image.width, image.height)
    return torch.zeros(int(height), int(width), dtype=torch.float32)


class TemporalCaptionManifestDataset(Dataset[TemporalCaptionItem]):
    def __init__(
        self,
        manifests: str | Path | Sequence[str | Path],
        *,
        split: str = "train",
        image_size: int | None = 224,
        output_grid: int = 36,
        max_pairs: int | None = None,
        allowed_caption_sources: set[str] | None = None,
        exclude_rscc_model_generated: bool = True,
    ):
        manifest_paths = [manifests] if isinstance(manifests, (str, Path)) else list(manifests)
        rows: list[dict[str, Any]] = []
        for path in manifest_paths:
            for row in load_json_rows(path):
                if row.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError(f"Unsupported manifest schema in {path}: {row.get('schema_version')!r}")
                if split != "all" and row.get("split") != split:
                    continue
                caption_source = str(row.get("caption_source", ""))
                if allowed_caption_sources is not None and caption_source not in allowed_caption_sources:
                    continue
                if exclude_rscc_model_generated and row.get("dataset_name") == "rscc" and caption_source == "model_generated":
                    continue
                if not [caption for caption in row.get("captions", []) if str(caption).strip()]:
                    continue
                rows.append(row)
        rows = sorted(rows, key=lambda row: str(row["pair_id"]))
        if max_pairs is not None:
            rows = rows[: max(0, max_pairs)]
        self.samples = rows
        self.image_size = image_size
        self.output_grid = output_grid
        self.indices_by_dataset: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            self.indices_by_dataset[str(row["dataset_name"])].append(index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TemporalCaptionItem:
        row = self.samples[index]
        mask_path = row.get("mask_path")
        if mask_path:
            mask = _load_mask(mask_path, self.image_size)
        else:
            mask = _blank_mask_like(row["t1_path"], self.image_size)
        return TemporalCaptionItem(
            pair_id=str(row["pair_id"]),
            dataset_name=str(row["dataset_name"]),
            t1=_load_rgb(row["t1_path"], self.image_size),
            t2=_load_rgb(row["t2_path"], self.image_size),
            captions=[str(caption) for caption in row.get("captions", []) if str(caption).strip()],
            mask=mask,
            components=build_component_targets(mask, grid_size=(self.output_grid, self.output_grid), min_area=4),
            temporal_context=TemporalContext(),
            metadata={
                "manifest_schema_version": row.get("schema_version"),
                "dataset_name": row.get("dataset_name"),
                "split": row.get("split"),
                "caption_source": row.get("caption_source"),
                "normalized_caption_groups": row.get("normalized_caption_groups", []),
                "t1_path": row.get("t1_path"),
                "t2_path": row.get("t2_path"),
                "mask_path": row.get("mask_path"),
                "semantic_t1_path": row.get("semantic_t1_path"),
                "semantic_t2_path": row.get("semantic_t2_path"),
                "source_metadata": row.get("source_metadata", {}),
            },
        )


def _stable_seed(seed: int, epoch: int, label: str) -> int:
    payload = f"{seed}:{epoch}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & 0x7FFF_FFFF_FFFF_FFFF


class DeterministicWeightedDatasetSampler(Sampler[int]):
    def __init__(
        self,
        dataset: TemporalCaptionManifestDataset,
        *,
        weights: dict[str, float],
        seed: int,
        epoch: int,
        num_samples: int | None = None,
    ):
        self.dataset = dataset
        self.weights = {str(name): float(value) for name, value in weights.items() if float(value) > 0.0}
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.num_samples = int(num_samples) if num_samples is not None else len(dataset)
        missing = [name for name in self.weights if name not in dataset.indices_by_dataset]
        if missing:
            raise ValueError(f"Sampling weights reference datasets not present in manifest: {missing}")
        if not self.weights:
            raise ValueError("At least one positive dataset sampling weight is required")

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        names = sorted(self.weights)
        probs = torch.tensor([self.weights[name] for name in names], dtype=torch.float64)
        probs = probs / probs.sum()
        generator = torch.Generator().manual_seed(_stable_seed(self.seed, self.epoch, "dataset_mix"))
        name_choices = torch.multinomial(probs, num_samples=self.num_samples, replacement=True, generator=generator)
        positions: dict[str, int] = {name: 0 for name in names}
        shuffled: dict[str, list[int]] = {}
        for name in names:
            indices = list(self.dataset.indices_by_dataset[name])
            g = torch.Generator().manual_seed(_stable_seed(self.seed, self.epoch, name))
            order = torch.randperm(len(indices), generator=g).tolist() if indices else []
            shuffled[name] = [indices[index] for index in order]
        for choice in name_choices.tolist():
            name = names[int(choice)]
            items = shuffled[name]
            if not items:
                continue
            position = positions[name]
            if position and position % len(items) == 0:
                cycle_id = math.floor(position / len(items))
                g = torch.Generator().manual_seed(_stable_seed(self.seed, self.epoch + cycle_id, name))
                order = torch.randperm(len(items), generator=g).tolist()
                items = [items[index] for index in order]
                shuffled[name] = items
            yield items[position % len(items)]
            positions[name] = position + 1


def parse_dataset_weights(values: Sequence[str] | dict[str, float] | None) -> dict[str, float]:
    if values is None:
        return {}
    if isinstance(values, dict):
        return {str(key): float(value) for key, value in values.items()}
    weights: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Dataset weight must use dataset=weight syntax: {value!r}")
        name, raw_weight = value.split("=", 1)
        weights[name.strip()] = float(raw_weight)
    return weights


def load_dataset_config(path: str | Path | None) -> tuple[list[str], list[str], dict[str, float]]:
    if path is None:
        return [], [], {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        [str(item) for item in payload.get("train_manifests", [])],
        [str(item) for item in payload.get("val_manifests", [])],
        {str(key): float(value) for key, value in payload.get("dataset_sampling_weights", {}).items()},
    )
