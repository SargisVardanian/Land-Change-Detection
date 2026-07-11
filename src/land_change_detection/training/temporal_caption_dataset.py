from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
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


_INTEGER_SEMANTIC_MODES = {"P", "L", "I", "I;16", "I;16B", "I;16L"}
_TUPLE_SEMANTIC_MODES = {"RGB", "RGBA"}


def _load_semantic_labels(path: str | Path, image_size: int | None) -> Tensor:
    """Load semantic class identities without converting them to luminance."""

    with Image.open(path) as image:
        mode = image.mode
        if mode not in _INTEGER_SEMANTIC_MODES | _TUPLE_SEMANTIC_MODES:
            raise ValueError(f"Unsupported semantic-map mode {mode!r} for {path}; expected indexed/integer or RGB/RGBA labels")
        if image_size is not None:
            image = image.resize((int(image_size), int(image_size)), resample=Image.Resampling.NEAREST)
        array = np.asarray(image)
    if mode in _INTEGER_SEMANTIC_MODES:
        if array.ndim != 2:
            raise ValueError(f"Integer semantic map {path} must be rank-2, got {array.shape}")
        return torch.from_numpy(array.copy()).to(torch.int64)
    expected_channels = 3 if mode == "RGB" else 4
    if array.ndim != 3 or array.shape[-1] != expected_channels:
        raise ValueError(f"{mode} semantic map {path} must have {expected_channels} channels, got {array.shape}")
    return torch.from_numpy(array.copy()).to(torch.uint8)


def _semantic_change_mask(t1_path: str | Path, t2_path: str | Path, image_size: int | None) -> Tensor:
    first = _load_semantic_labels(t1_path, image_size)
    second = _load_semantic_labels(t2_path, image_size)
    if first.shape != second.shape:
        raise ValueError(f"Semantic maps must align after normalization, got {tuple(first.shape)} and {tuple(second.shape)}")
    if first.ndim == 2:
        return (first != second).float()
    return torch.any(first != second, dim=-1).float()


def _segmentation_target_contract(row: dict[str, Any], *, has_mask: bool, has_semantics: bool) -> tuple[str, float]:
    source_metadata = row.get("source_metadata", {})
    explicit_kind = (
        row.get("segmentation_target_kind")
        or row.get("seg_supervision_mode")
        or source_metadata.get("segmentation_target_kind")
        or source_metadata.get("seg_supervision_mode")
    )
    if has_mask:
        if explicit_kind in {"query_specific", "query-specific"} or row.get("query_mask_path"):
            return "query_specific", 1.0
        if str(row.get("dataset_name", "")).casefold() in {"levir_mci", "levir_cc"}:
            return "binary_generic", 0.5
        return "binary_generic", 0.5
    if has_semantics:
        return "semantic_transition_union", 0.25
    return "none", 0.0


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
        self.selection_metadata: dict[str, Any] = {
            "requested_max_pairs": max_pairs,
            "omitted_datasets": [],
            "selected_counts_by_dataset": {},
            "available_counts_by_dataset": {},
            "selection_mode": "full" if max_pairs is None else "truncated",
        }
        if max_pairs is not None:
            if max_pairs < 0:
                raise ValueError("max_pairs must be non-negative")
            rows, selection_metadata = _select_manifest_rows(rows, max_pairs)
            self.selection_metadata.update(selection_metadata)
        self.samples = rows
        self.image_size = image_size
        self.output_grid = output_grid
        self.indices_by_dataset: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            self.indices_by_dataset[str(row["dataset_name"])].append(index)
        self.selection_metadata["selected_counts_by_dataset"] = {
            name: len(indices) for name, indices in sorted(self.indices_by_dataset.items())
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TemporalCaptionItem:
        row = self.samples[index]
        mask_path = row.get("mask_path")
        has_semantics = bool(row.get("semantic_t1_path") and row.get("semantic_t2_path"))
        target_kind, supervision_weight = _segmentation_target_contract(
            row,
            has_mask=bool(mask_path),
            has_semantics=has_semantics,
        )
        if mask_path:
            mask = _load_mask(mask_path, self.image_size)
            segmentation_target_source = target_kind
        elif has_semantics:
            mask = _semantic_change_mask(row["semantic_t1_path"], row["semantic_t2_path"], self.image_size)
            segmentation_target_source = target_kind
        else:
            mask = _blank_mask_like(row["t1_path"], self.image_size)
            segmentation_target_source = "none"
        changed_mask = mask
        changed_paths = row.get("source_metadata", {}).get("changed_mask_paths", [])
        if changed_paths:
            changed_masks = [_load_mask(path, self.image_size) for path in changed_paths]
            changed_mask = torch.stack(changed_masks).amax(dim=0)
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
                "retrieval_supervision": bool(row.get("retrieval_supervision", row.get("source_metadata", {}).get("retrieval_supervision", True))),
                "seg_supervision_mode": row.get("seg_supervision_mode", row.get("source_metadata", {}).get("seg_supervision_mode")),
                "query_mask_path": row.get("query_mask_path", row.get("source_metadata", {}).get("query_mask_path")),
                "segmentation_supervision": supervision_weight > 0.0,
                "segmentation_target_source": segmentation_target_source,
                "segmentation_target_kind": target_kind,
                "segmentation_supervision_weight": supervision_weight,
                "changed_mask": changed_mask,
                "change_type": row.get("change_type", row.get("source_metadata", {}).get("change_type")),
            },
        )


def _stable_seed(seed: int, epoch: int, label: str) -> int:
    payload = f"{seed}:{epoch}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & 0x7FFF_FFFF_FFFF_FFFF


def _group_rows_by_dataset(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset_name"])].append(row)
    return {name: sorted(items, key=lambda item: str(item["pair_id"])) for name, items in grouped.items()}


def _deterministic_dataset_priority(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    return sorted(grouped, key=lambda name: (-len(grouped[name]), name))


def _allocate_multidataset_quotas(counts: dict[str, int], target_total: int) -> dict[str, int]:
    if target_total <= 0 or not counts:
        return {name: 0 for name in sorted(counts)}
    dataset_names = sorted(counts)
    if len(dataset_names) == 1:
        name = dataset_names[0]
        return {name: min(target_total, counts[name])}
    if target_total < len(dataset_names):
        quotas = {name: 0 for name in dataset_names}
        for name in _deterministic_dataset_priority({name: [None] * counts[name] for name in dataset_names})[:target_total]:
            quotas[name] = 1
        return quotas

    quotas = {name: 1 for name in dataset_names}
    remaining = target_total - len(dataset_names)
    residual_capacity = {name: counts[name] - 1 for name in dataset_names}
    available_remaining = sum(max(capacity, 0) for capacity in residual_capacity.values())
    if remaining <= 0 or available_remaining <= 0:
        return quotas

    floor_additions: dict[str, int] = {name: 0 for name in dataset_names}
    remainders: list[tuple[Fraction, int, str]] = []
    for name in dataset_names:
        capacity = max(residual_capacity[name], 0)
        if capacity == 0:
            remainders.append((Fraction(0, 1), 0, name))
            continue
        share = Fraction(remaining * capacity, available_remaining)
        floor_value = min(int(share), capacity)
        floor_additions[name] = floor_value
        remainders.append((share - floor_value, capacity, name))

    quotas = {name: quotas[name] + floor_additions[name] for name in dataset_names}
    slots_left = remaining - sum(floor_additions.values())
    for _, _, name in sorted(remainders, key=lambda item: (-item[0], -item[1], item[2])):
        if slots_left <= 0:
            break
        if quotas[name] >= counts[name]:
            continue
        quotas[name] += 1
        slots_left -= 1

    while slots_left > 0:
        progress = False
        for name in _deterministic_dataset_priority({name: [None] * counts[name] for name in dataset_names}):
            if quotas[name] >= counts[name]:
                continue
            quotas[name] += 1
            slots_left -= 1
            progress = True
            if slots_left == 0:
                break
        if not progress:
            break
    return quotas


def _select_manifest_rows(rows: Sequence[dict[str, Any]], max_pairs: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_total = min(max_pairs, len(rows))
    grouped = _group_rows_by_dataset(rows)
    counts = {name: len(items) for name, items in sorted(grouped.items())}
    metadata: dict[str, Any] = {
        "requested_max_pairs": max_pairs,
        "available_counts_by_dataset": counts,
        "omitted_datasets": [],
        "selection_mode": "single_dataset" if len(grouped) <= 1 else "multidataset_stratified",
    }
    if target_total == 0 or not rows:
        metadata["omitted_datasets"] = sorted(grouped)
        return [], metadata
    if len(grouped) <= 1:
        return list(rows[:target_total]), metadata

    quotas = _allocate_multidataset_quotas(counts, target_total)
    selected: list[dict[str, Any]] = []
    for name in sorted(grouped):
        selected.extend(grouped[name][: quotas.get(name, 0)])
    selected = sorted(selected, key=lambda row: (str(row["dataset_name"]), str(row["pair_id"])))
    if len(selected) != target_total:
        raise RuntimeError(
            f"Deterministic max_pairs selection produced {len(selected)} rows, expected {target_total}"
        )
    pair_ids = [str(row["pair_id"]) for row in selected]
    if len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("Deterministic max_pairs selection produced duplicate pair IDs")
    metadata["omitted_datasets"] = sorted(name for name in grouped if quotas.get(name, 0) <= 0)
    return selected, metadata


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


def load_dataset_config(path: str | Path | None) -> tuple[list[str], list[str], dict[str, float], dict[str, Any]]:
    if path is None:
        return [], [], {}, {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    options: dict[str, Any] = {}
    for key in (
        "allowed_caption_sources",
        "semantic_soft_target_weight",
        "semantic_teacher_top_k",
        "semantic_teacher_temperature",
        "structured_fna_weight",
    ):
        if key in payload:
            options[key] = payload[key]
    return (
        [str(item) for item in payload.get("train_manifests", [])],
        [str(item) for item in payload.get("val_manifests", [])],
        {str(key): float(value) for key, value in payload.get("dataset_sampling_weights", {}).items()},
        options,
    )
