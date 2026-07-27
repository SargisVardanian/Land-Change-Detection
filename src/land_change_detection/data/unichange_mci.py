from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
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
        # S2Looking stores directional labels in color channels (red for one
        # direction, blue for the other). Converting to luminance before a
        # 0.5 threshold silently erased both. Decode foreground from any
        # non-zero channel before spatial normalization.
        mask = TF.pil_to_tensor(image).ne(0).any(dim=0).to(torch.float32)
    if image_size is None or tuple(mask.shape) == (image_size, image_size):
        return mask
    target_size = (int(image_size), int(image_size))
    if target_size[0] <= mask.shape[-2] and target_size[1] <= mask.shape[-1]:
        return F.adaptive_max_pool2d(mask[None, None], target_size)[0, 0]
    return F.interpolate(mask[None, None], size=target_size, mode="nearest")[0, 0]


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
            selected_ids = load_subset_pair_ids(subset_file)
            order = {pair_id: index for index, pair_id in enumerate(selected_ids)}
            samples = sorted([sample for sample in samples if sample.sample_id in order], key=lambda sample: order[sample.sample_id])
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


def collate_unichange_mci(items: list[UniChangeMciItem], reverse_probability: float = 0.0, rng: Any | None = None) -> dict[str, Any]:
    if rng is None:
        import random

        rng = random
    captions: list[str] = []
    caption_to_pair: list[int] = []
    t1_items: list[Tensor] = []
    t2_items: list[Tensor] = []
    temporal_contexts: list[dict[str, Any]] = []
    direction_targets: list[int] = []
    for pair_index, item in enumerate(items):
        captions.extend(item.captions)
        caption_to_pair.extend([pair_index] * len(item.captions))
        reverse = bool(reverse_probability > 0.0 and rng.random() < reverse_probability)
        if reverse:
            t1_items.append(item.t2)
            t2_items.append(item.t1)
            temporal_contexts.append(TemporalContext(before_index=1, after_index=0).to_dict())
            direction_targets.append(0)
        else:
            t1_items.append(item.t1)
            t2_items.append(item.t2)
            temporal_contexts.append(item.temporal_context.to_dict())
            direction_targets.append(1)
    return {
        "pair_ids": [item.pair_id for item in items],
        "t1": torch.stack(t1_items, dim=0),
        "t2": torch.stack(t2_items, dim=0),
        "captions": captions,
        "caption_to_pair": torch.tensor(caption_to_pair, dtype=torch.long),
        "masks": torch.stack([item.mask for item in items], dim=0),
        "components": [item.components for item in items],
        "temporal_context": temporal_contexts,
        "direction_target": torch.tensor(direction_targets, dtype=torch.long),
        "metadata": [item.metadata for item in items],
    }
