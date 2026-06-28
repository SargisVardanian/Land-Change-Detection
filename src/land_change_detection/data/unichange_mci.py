from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from land_change_detection.data.event_targets import ComponentTargets, TemporalContext, build_component_targets
from land_change_detection.data.unichange_subset import load_subset_pair_ids, resolve_levir_mci_root
from land_change_detection.levir_mci import LevirMciSample, discover_levir_mci_samples


@dataclass(frozen=True)
class UniChangeMciItem:
    pair_id: str
    t1: Tensor
    t2: Tensor
    captions: list[str]
    mask: Tensor
    components: ComponentTargets
    temporal_context: TemporalContext
    metadata: dict[str, Any]


def _load_rgb(path: str | Path, image_size: int | None) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size is not None:
            image = image.resize((image_size, image_size), Image.BILINEAR)
        return TF.to_tensor(image)


def _load_mask(path: str | Path, image_size: int | None) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        if image_size is not None:
            image = image.resize((image_size, image_size), Image.NEAREST)
        return (TF.to_tensor(image).squeeze(0) > 0.5).to(torch.float32)


def _captions_for_sample(sample: LevirMciSample) -> list[str]:
    captions = [caption for caption in sample.captions if caption.strip()]
    if not captions and sample.caption.strip():
        captions = [sample.caption]
    return captions


class UniChangeMciDataset(Dataset[UniChangeMciItem]):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        image_size: int | None = 224,
        output_grid: int = 36,
        min_component_area: int = 4,
        max_pairs: int | None = None,
        subset_file: str | Path | None = None,
    ):
        self.data_root = resolve_levir_mci_root(data_root).root
        samples = [sample for sample in discover_levir_mci_samples(self.data_root) if split == "all" or sample.split == split]
        samples = [sample for sample in samples if _captions_for_sample(sample)]
        if subset_file is not None:
            selected = set(load_subset_pair_ids(subset_file))
            samples = [sample for sample in samples if sample.sample_id in selected]
        if max_pairs is not None:
            samples = samples[: max(0, max_pairs)]
        self.samples = samples
        self.image_size = image_size
        self.output_grid = output_grid
        self.min_component_area = min_component_area

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> UniChangeMciItem:
        sample = self.samples[index]
        mask = _load_mask(sample.binary_change_mask, self.image_size)
        return UniChangeMciItem(
            pair_id=sample.sample_id,
            t1=_load_rgb(sample.image_before, self.image_size),
            t2=_load_rgb(sample.image_after, self.image_size),
            captions=_captions_for_sample(sample),
            mask=mask,
            components=build_component_targets(
                mask,
                grid_size=(self.output_grid, self.output_grid),
                min_area=self.min_component_area,
            ),
            temporal_context=TemporalContext(),
            metadata={
                "split": sample.split,
                "before_path": sample.image_before,
                "after_path": sample.image_after,
                "mask_path": sample.binary_change_mask,
                **sample.metadata,
            },
        )


def collate_unichange_mci(items: list[UniChangeMciItem]) -> dict[str, Any]:
    captions: list[str] = []
    caption_to_pair: list[int] = []
    for pair_index, item in enumerate(items):
        captions.extend(item.captions)
        caption_to_pair.extend([pair_index] * len(item.captions))
    return {
        "pair_ids": [item.pair_id for item in items],
        "t1": torch.stack([item.t1 for item in items], dim=0),
        "t2": torch.stack([item.t2 for item in items], dim=0),
        "captions": captions,
        "caption_to_pair": torch.tensor(caption_to_pair, dtype=torch.long),
        "masks": torch.stack([item.mask for item in items], dim=0),
        "components": [item.components for item in items],
        "temporal_context": [item.temporal_context.to_dict() for item in items],
        "metadata": [item.metadata for item in items],
    }
