from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import train_unichange_v2_retrieval as base
from land_change_detection.models.retrieval_heads import (
    caption_detail_score,
    classify_caption_semantics,
    normalize_caption_text,
    semantic_text_to_pair_set_loss,
    semantic_teacher_relevance_matrix,
    stable_caption_group_ids,
    structured_fna_relevance_matrix,
)
from land_change_detection.models.qcpr import (
    conditional_instance_discrimination_loss,
    local_positive_negative_margin_loss,
    segmentation_loss_components,
    structured_auxiliary_evidence_loss,
    temporal_channel_loss_components,
)
from land_change_detection.run_metadata import file_sha256, path_fingerprint
from land_change_detection.temporal_caption_manifest import manifest_file_fingerprint
from land_change_detection.training.temporal_caption_dataset import (
    DeterministicWeightedDatasetSampler,
    TemporalCaptionManifestDataset,
    load_dataset_config,
    parse_dataset_weights,
)
from land_change_detection.training.runtime_device import assert_runtime_tensor_devices, resolve_runtime_device
from ucv2_cluster_common import build_model, run_metadata, strict_device
from ucv2_retrieval_metrics import relevance_aware_retrieval_metrics


@dataclass(frozen=True)
class Stage1NextConfig:
    data_root: str
    output_dir: str
    universat_source: str
    universat_checkpoint: str
    jina_model: str
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = 256
    output_grid: int = 32
    batch_size: int = 32
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_steps: int | None = None
    epochs: int = 25
    num_workers: int = 8
    learning_rate: float = 2e-4
    retrieval_head_lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    min_lr: float = 1e-6
    grad_clip_norm: float = 1.0
    use_bf16: bool = True
    device: str = "cuda"
    seed: int = 20260701
    smoke: bool = False
    fake_backbones: bool = False
    synthetic_data: bool = False
    train_manifests: tuple[str, ...] = ()
    val_manifests: tuple[str, ...] = ()
    localization_val_manifests: tuple[str, ...] = ()
    dataset_config: str | None = None
    dataset_sampling_weights: tuple[str, ...] = ("levir_mci=0.55", "second_cc=0.45")
    allowed_caption_sources: tuple[str, ...] = ()
    target_aware_mask_crop: bool = False
    target_crop_context: float = 2.0

    temporal_depth: int = 4
    use_direction_embeddings: bool = True
    use_explicit_change_fusion: bool = True
    trainable_temperature: bool = True
    initial_temperature: float = 0.07
    max_logit_scale: float = 100.0

    max_captions_per_pair: int = 2
    caption_frequency_power: float = 0.5
    text_to_pair_weight: float = 0.75
    pair_to_text_weight: float = 0.25
    semantic_soft_target_weight: float = 0.25
    semantic_teacher_top_k: int = 8
    semantic_teacher_temperature: float = 0.05
    use_text_adapter: bool = True
    text_adapter_hidden_dim: int = 512
    text_adapter_lr: float = 2e-5
    text_max_length: int = 256
    caption_sampling_seed: int | None = None
    enable_conflict_filtering: bool = False
    conflict_mask_threshold: float = 0.05
    mask_fraction_boundaries: tuple[float, float, float] = (0.0, 0.01, 0.05)
    negative_queue_size: int = 0
    similarity_query_chunk_size: int = 0
    similarity_candidate_chunk_size: int = 0

    train_eval_pairs: int = 1024
    train_eval_interval: int = 2
    checkpoint_interval_steps: int = 500
    early_stopping_patience: int = 4
    early_stopping_min_improvement: float = 0.002
    enable_patch_reranker: bool = False
    qcpr_architecture_version: str = "v1"
    enable_temporal_explanation_channels: bool = False
    qcpr_alpha: float = 1.0
    qcpr_beta: float = 0.25
    qcpr_local_loss_weight: float = 1.0
    fused_retrieval_loss_weight: float = 0.25
    global_embedding_preservation_weight: float = 0.0
    freeze_global_retrieval: bool = False
    query_segmentation_loss_weight: float = 0.2
    changed_channel_loss_weight: float = 0.2
    appeared_channel_loss_weight: float = 0.2
    disappeared_channel_loss_weight: float = 0.2
    temporal_reversal_consistency_loss_weight: float = 0.05
    structured_fna_weight: float = 0.0
    local_margin_loss_weight: float = 0.1
    local_margin: float = 0.1
    conditional_instance_loss_weight: float = 0.01
    structured_auxiliary_loss_weight: float = 0.05


class FrequencyBalancedCaptionCollator:
    def __init__(
        self,
        caption_frequencies: dict[str, int],
        *,
        max_captions_per_pair: int | None,
        frequency_power: float,
        seed: int,
        epoch: int,
        training: bool,
    ):
        self.caption_frequencies = caption_frequencies
        self.max_captions_per_pair = max_captions_per_pair
        self.frequency_power = float(frequency_power)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.training = bool(training)

    def _item_seed(self, pair_id: str) -> int:
        payload = f"{self.seed}:{self.epoch}:{pair_id}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & 0x7FFF_FFFF_FFFF_FFFF

    def _select_captions(self, pair_id: str, captions: list[str]) -> list[str]:
        limit = self.max_captions_per_pair
        if not self.training or limit is None or limit <= 0 or len(captions) <= limit:
            return captions
        if limit >= len(captions):
            return captions
        detail_scores = [caption_detail_score(caption) for caption in captions]
        detail_index = max(range(len(captions)), key=lambda index: (detail_scores[index], -index))
        weights = torch.tensor(
            [
                max(self.caption_frequencies.get(normalize_caption_text(caption), 1), 1)
                ** (-self.frequency_power)
                for caption in captions
            ],
            dtype=torch.float64,
        )
        generator = torch.Generator().manual_seed(self._item_seed(pair_id))
        # Rotate one guaranteed caption slot by epoch so five-caption LEVIR rows
        # are covered within a bounded number of epochs, then fill remaining
        # slots with deterministic rare-caption-weighted sampling.
        rotation_index = self.epoch % len(captions)
        if rotation_index == detail_index and len(captions) > 1:
            rotation_index = (rotation_index + 1) % len(captions)
        weights[rotation_index] = 0.0
        weights[detail_index] = 0.0
        remaining = max(limit - 2, 0)
        sampled: list[int] = []
        if remaining:
            if float(weights.sum().item()) <= 0.0:
                candidates = [index for index in range(len(captions)) if index != rotation_index]
                sampled = candidates[:remaining]
            else:
                sampled = torch.multinomial(weights, num_samples=remaining, replacement=False, generator=generator).tolist()
        indices = [detail_index, rotation_index, *sampled]
        return [captions[index] for index in sorted(indices)]

    def __call__(self, items: list[Any]) -> dict[str, Any]:
        images = torch.stack(
            [torch.stack([item.t1, item.t2], dim=0) for item in items],
            dim=0,
        )
        captions: list[str] = []
        caption_to_pair: list[int] = []
        query_masks: list[torch.Tensor] = []
        query_change_types: list[str | None] = []
        query_segmentation_supervision: list[bool] = []
        for pair_index, item in enumerate(items):
            selected = self._select_captions(str(item.pair_id), list(item.captions))
            if not selected:
                raise RuntimeError(f"Pair {item.pair_id} has no usable captions")
            captions.extend(selected)
            caption_to_pair.extend([pair_index] * len(selected))
            available = list(item.captions)
            used: set[int] = set()
            for caption in selected:
                source_index = next(
                    index for index, value in enumerate(available)
                    if index not in used and value == caption
                )
                used.add(source_index)
                item_query_masks = getattr(item, "query_masks", [item.mask for _ in available])
                item_query_changes = getattr(
                    item, "query_change_types",
                    [getattr(item, "metadata", {}).get("change_type") for _ in available],
                )
                query_masks.append(item_query_masks[source_index].float())
                query_change_types.append(item_query_changes[source_index])
                query_segmentation_supervision.append(
                    bool(getattr(item, "metadata", {}).get("segmentation_supervision", False))
                )
        return {
            "pair_ids": [str(item.pair_id) for item in items],
            "dataset_names": [str(getattr(item, "dataset_name", getattr(item, "metadata", {}).get("dataset_name", "unknown"))) for item in items],
            "images": images,
            "timestamps": torch.tensor([[0.0, 1.0] for _ in items], dtype=torch.float32),
            "temporal_valid_mask": torch.ones(len(items), 2, dtype=torch.bool),
            "changed_masks": torch.stack(
                [getattr(item, "metadata", {}).get("changed_mask", item.mask).float() for item in items], dim=0
            ),
            "captions": captions,
            "caption_to_pair": torch.tensor(caption_to_pair, dtype=torch.long),
            "query_masks": torch.stack(query_masks, dim=0),
            "query_change_types": query_change_types,
            "query_segmentation_supervision": torch.tensor(
                query_segmentation_supervision,
                dtype=torch.bool,
            ),
            "mask_fractions": torch.tensor(
                [float(item.mask.float().mean().item()) for item in items],
                dtype=torch.float32,
            ),
            "masks": torch.stack([item.mask.float() for item in items], dim=0),
            "retrieval_supervision": torch.tensor(
                [bool(getattr(item, "metadata", {}).get("retrieval_supervision", True)) for item in items],
                dtype=torch.bool,
            ),
            "segmentation_supervision": torch.tensor(
                [bool(getattr(item, "metadata", {}).get("segmentation_supervision", False)) for item in items],
                dtype=torch.bool,
            ),
            "segmentation_weights": torch.tensor(
                [float(getattr(item, "metadata", {}).get("segmentation_supervision_weight", 0.0)) for item in items],
                dtype=torch.float32,
            ),
            "segmentation_target_kinds": [
                str(getattr(item, "metadata", {}).get("segmentation_target_kind", "none")) for item in items
            ],
            "change_types": [
                getattr(item, "metadata", {}).get("change_type") for item in items
            ],
            "segmentation_target_sources": [
                str(getattr(item, "metadata", {}).get("segmentation_target_source", "none")) for item in items
            ],
        }


def retrieval_supervision_selection(batch: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
    """Select retrieval-supervised pairs with one explicit tensor-device contract.

    ``retrieval_supervision`` and ``caption_to_pair`` may arrive from a CPU
    collator even when model outputs are on CUDA.  Every tensor used for
    indexing is moved to ``device`` before any indexing operation; Python
    metadata such as captions and pair IDs deliberately remains on CPU.
    """
    requested_device = resolve_runtime_device(device)
    pair_mask = torch.as_tensor(batch["retrieval_supervision"], device=requested_device, dtype=torch.bool)
    caption_to_pair = torch.as_tensor(batch["caption_to_pair"], device=requested_device, dtype=torch.long)
    if pair_mask.ndim != 1:
        raise ValueError(f"retrieval_supervision must be rank-1, got {tuple(pair_mask.shape)}")
    if caption_to_pair.ndim != 1:
        raise ValueError(f"caption_to_pair must be rank-1, got {tuple(caption_to_pair.shape)}")
    pair_count = pair_mask.numel()
    if caption_to_pair.numel() and (
        int(caption_to_pair.min().item()) < 0 or int(caption_to_pair.max().item()) >= pair_count
    ):
        raise ValueError(f"caption_to_pair values must be in [0, {pair_count}), got out-of-range values")

    caption_mask = pair_mask.index_select(0, caption_to_pair)
    selected_pairs = torch.nonzero(pair_mask, as_tuple=False).flatten()
    selected_queries = torch.nonzero(caption_mask, as_tuple=False).flatten()
    inverse = torch.full((pair_mask.numel(),), -1, dtype=torch.long, device=requested_device)
    inverse.index_copy_(0, selected_pairs, torch.arange(selected_pairs.numel(), device=requested_device))
    selected_pair_indices = caption_to_pair.index_select(0, selected_queries)
    selected_mapping = inverse.index_select(0, selected_pair_indices)
    if bool((selected_mapping < 0).any()):
        raise RuntimeError("selected_mapping contains an unsupervised or negative compact pair index")
    result = {
        "pair_mask": pair_mask,
        "caption_mask": caption_mask,
        "selected_pairs": selected_pairs,
        "selected_queries": selected_queries,
        "selected_mapping": selected_mapping,
    }
    assert_runtime_tensor_devices(result, requested_device)
    return result


def _captions_from_sample(sample: Any) -> list[str]:
    captions = [str(caption) for caption in getattr(sample, "captions", []) if str(caption).strip()]
    fallback = str(getattr(sample, "caption", ""))
    if not captions and fallback.strip():
        captions = [fallback]
    return captions


def caption_frequencies(dataset: Dataset) -> dict[str, int]:
    frequencies: Counter[str] = Counter()
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        for sample in samples:
            frequencies.update(normalize_caption_text(caption) for caption in _captions_from_sample(sample))
    else:
        for index in range(len(dataset)):
            item = dataset[index]
            frequencies.update(normalize_caption_text(caption) for caption in item.captions)
    return dict(frequencies)


def _make_collator(
    dataset: Dataset,
    config: Stage1NextConfig,
    *,
    epoch: int,
    training: bool,
    frequencies: dict[str, int] | None = None,
) -> FrequencyBalancedCaptionCollator:
    return FrequencyBalancedCaptionCollator(
        frequencies if frequencies is not None else (caption_frequencies(dataset) if training else {}),
        max_captions_per_pair=config.max_captions_per_pair if training else None,
        frequency_power=config.caption_frequency_power,
        seed=config.caption_sampling_seed if config.caption_sampling_seed is not None else config.seed,
        epoch=epoch,
        training=training,
    )


def make_train_loader(
    dataset: Dataset,
    config: Stage1NextConfig,
    frequencies: dict[str, int],
    epoch: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed + epoch)
    weights = parse_dataset_weights(config.dataset_sampling_weights)
    sampler = None
    shuffle = True
    if weights and isinstance(dataset, TemporalCaptionManifestDataset):
        sampler = DeterministicWeightedDatasetSampler(dataset, weights=weights, seed=config.seed, epoch=epoch)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator,
        num_workers=config.num_workers,
        collate_fn=_make_collator(dataset, config, epoch=epoch, training=True, frequencies=frequencies),
        pin_memory=True,
        persistent_workers=False,
    )


def make_eval_loader(dataset: Dataset, config: Stage1NextConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_make_collator(dataset, config, epoch=0, training=False),
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )


def _no_decay(name: str, parameter: nn.Parameter) -> bool:
    lowered = name.casefold()
    return (
        parameter.ndim <= 1
        or lowered.endswith("bias")
        or "norm" in lowered
        or "global_queries" in lowered
        or "direction_embeddings" in lowered
        or "logit_scale" in lowered
    )


def _module_parameter_groups(
    module: nn.Module,
    *,
    prefix: str,
    learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        full_name = f"{prefix}.{name}" if prefix else name
        if _no_decay(name, parameter):
            no_decay.append(parameter)
            no_decay_names.append(full_name)
        else:
            decay.append(parameter)
            decay_names.append(full_name)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"name": f"{prefix}_decay", "params": decay, "param_names": decay_names, "lr": learning_rate, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"name": f"{prefix}_no_decay", "params": no_decay, "param_names": no_decay_names, "lr": learning_rate, "weight_decay": 0.0})
    return groups


def make_optimizer(model: nn.Module, config: Stage1NextConfig) -> torch.optim.Optimizer:
    groups = [
        *_module_parameter_groups(
            model.temporal_encoder,
            prefix="temporal_encoder",
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
        *_module_parameter_groups(
            model.retrieval_head,
            prefix="retrieval_head",
            learning_rate=config.retrieval_head_lr,
            weight_decay=config.weight_decay,
        ),
    ]
    if getattr(model, "text_adapter", None) is not None:
        groups.extend(
            _module_parameter_groups(
                model.text_adapter,
                prefix="text_adapter",
                learning_rate=config.text_adapter_lr,
                weight_decay=config.weight_decay,
            )
        )
    if getattr(model, "patch_reranker", None) is not None:
        groups.extend(
            _module_parameter_groups(
                model.patch_reranker,
                prefix="patch_reranker",
                learning_rate=config.retrieval_head_lr,
                weight_decay=config.weight_decay,
            )
        )
    if not groups:
        raise RuntimeError("No trainable Stage-1 parameters were found")
    seen: set[int] = set()
    for group in groups:
        for parameter in group["params"]:
            identity = id(parameter)
            if identity in seen:
                raise RuntimeError("A trainable parameter appears in multiple optimizer groups")
            seen.add(identity)
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != trainable:
        raise RuntimeError("Optimizer groups do not cover every trainable parameter exactly once")
    return torch.optim.AdamW(groups)


def freeze_global_retrieval_modules(model: nn.Module) -> list[str]:
    """Freeze the v1-compatible candidate generator before local-reranker training."""
    frozen: list[str] = []
    for name in ("temporal_encoder", "retrieval_head", "text_adapter"):
        module = getattr(model, name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        frozen.append(name)
    return frozen


def trainable_stage1_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Stage1NextConfig,
    *,
    epoch_index: int,
    next_batch_index: int,
    step: int,
    best_scores: dict[str, float],
    metrics: dict[str, Any],
    selection_metric: str | None = None,
    selection_value: float | None = None,
    temporal_supervised_pairs: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporal_head = getattr(getattr(model, "patch_reranker", None), "temporal_channel_head", None)
    temporal_gradients = [] if temporal_head is None else [parameter.grad for parameter in temporal_head.parameters() if parameter.requires_grad]
    temporal_gradient_finite = bool(temporal_gradients) and all(
        gradient is not None and torch.isfinite(gradient).all().item() for gradient in temporal_gradients
    )
    temporal_gradient_nonzero = bool(temporal_gradients) and any(
        gradient is not None and torch.any(gradient != 0).item() for gradient in temporal_gradients
    )
    temporal_channels_available = config.qcpr_architecture_version == "v2"
    temporal_channels_trained = bool(
        temporal_channels_available
        and config.enable_temporal_explanation_channels
        and temporal_supervised_pairs > 0
        and temporal_gradient_finite
        and temporal_gradient_nonzero
    )
    temporal_channel_provenance = {
        "supervision": {
            "enabled": bool(config.enable_temporal_explanation_channels),
            "supervised_pairs": int(temporal_supervised_pairs),
            "sources": ["query-specific appeared/disappeared labels", "generic changed masks"],
        },
        "gradients": {
            "temporal_channel_head_finite": temporal_gradient_finite,
            "temporal_channel_head_nonzero": temporal_gradient_nonzero,
        },
    }
    torch.save(
        {
            "stage1_next": True,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": base._rng_state(),
            "config": asdict(config),
            **run_metadata(),
            "epoch_index": epoch_index,
            "next_batch_index": next_batch_index,
            "step": step,
            "best_scores": best_scores,
            "metrics": metrics,
            "selection_metric": selection_metric,
            "selection_value": selection_value,
            "temporal_channels_available": temporal_channels_available,
            "temporal_channels_trained": temporal_channels_trained,
            "temporal_channel_provenance": temporal_channel_provenance,
        },
        tmp,
    )
    os.replace(tmp, path)


def _append_history(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(base._json_ready(payload)) + "\n")


def _prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def validate_config(config: Stage1NextConfig) -> None:
    if config.max_captions_per_pair < 0:
        raise ValueError("max_captions_per_pair must be non-negative")
    if config.caption_frequency_power < 0:
        raise ValueError("caption_frequency_power must be non-negative")
    if config.text_adapter_lr <= 0:
        raise ValueError("text_adapter_lr must be positive")
    if config.text_to_pair_weight < 0 or config.pair_to_text_weight < 0:
        raise ValueError("loss direction weights must be non-negative")
    if not 0.0 <= config.semantic_soft_target_weight <= 1.0:
        raise ValueError("semantic_soft_target_weight must be in [0, 1]")
    if config.semantic_teacher_top_k <= 0:
        raise ValueError("semantic_teacher_top_k must be positive")
    if config.semantic_teacher_temperature <= 0:
        raise ValueError("semantic_teacher_temperature must be positive")
    if config.text_max_length <= 0:
        raise ValueError("text_max_length must be positive")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if config.early_stopping_min_improvement < 0:
        raise ValueError("early_stopping_min_improvement must be non-negative")
    if config.max_steps is not None and config.max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")
    if config.qcpr_alpha < 0 or config.qcpr_beta < 0 or config.qcpr_alpha + config.qcpr_beta <= 0:
        raise ValueError("QCPR fusion weights must be non-negative and not both zero")
    if config.qcpr_architecture_version not in {"v1", "v2"}:
        raise ValueError("qcpr_architecture_version must be v1 or v2")
    if config.enable_temporal_explanation_channels and config.qcpr_architecture_version != "v2":
        raise ValueError("temporal explanation channels require QCPR v2")
    if config.qcpr_local_loss_weight < 0 or config.fused_retrieval_loss_weight < 0 or config.global_embedding_preservation_weight < 0 or config.query_segmentation_loss_weight < 0:
        raise ValueError("QCPR loss weights must be non-negative")
    if not 0.0 <= config.structured_fna_weight <= 1.0:
        raise ValueError("structured_fna_weight must be in [0, 1]")
    if config.local_margin_loss_weight < 0 or config.local_margin < 0 or config.conditional_instance_loss_weight < 0 or config.structured_auxiliary_loss_weight < 0:
        raise ValueError("QCPR ranking auxiliary loss weights and margins must be non-negative")
    if config.text_to_pair_weight + config.pair_to_text_weight <= 0:
        raise ValueError("at least one loss direction weight must be positive")
    if sorted(config.mask_fraction_boundaries) != list(config.mask_fraction_boundaries):
        raise ValueError("mask_fraction_boundaries must be sorted")
    if config.conflict_mask_threshold < 0:
        raise ValueError("conflict_mask_threshold must be non-negative")
    if config.train_manifests and not config.val_manifests:
        raise ValueError("val_manifests must be provided when train_manifests are used")
    parse_dataset_weights(config.dataset_sampling_weights)


def _build_stage1_datasets(config: Stage1NextConfig) -> tuple[Dataset, Dataset]:
    config_train, config_val, config_weights, config_options = load_dataset_config(config.dataset_config)
    train_manifests = tuple(config_train or config.train_manifests)
    val_manifests = tuple(config_val or config.val_manifests)
    if config_weights:
        object.__setattr__(config, "dataset_sampling_weights", tuple(f"{name}={value}" for name, value in sorted(config_weights.items())))
    for key in (
        "allowed_caption_sources", "localization_val_manifests", "semantic_soft_target_weight",
        "semantic_teacher_top_k", "semantic_teacher_temperature", "structured_fna_weight",
        "fused_retrieval_loss_weight", "global_embedding_preservation_weight",
        "freeze_global_retrieval", "local_margin_loss_weight", "local_margin",
        "conditional_instance_loss_weight", "structured_auxiliary_loss_weight",
    ):
        if key in config_options:
            value = config_options[key]
            object.__setattr__(config, key, tuple(value) if key in {"allowed_caption_sources", "localization_val_manifests"} else value)
    if config_train:
        object.__setattr__(config, "train_manifests", tuple(config_train))
    if config_val:
        object.__setattr__(config, "val_manifests", tuple(config_val))
    if train_manifests or val_manifests:
        if not train_manifests or not val_manifests:
            raise ValueError("Both train and validation manifests are required for manifest-based Stage-1-next training")
        train = TemporalCaptionManifestDataset(
            train_manifests,
            split=config.train_split,
            image_size=config.image_size,
            output_grid=config.output_grid,
            max_pairs=config.max_train_samples,
            allowed_caption_sources=set(config.allowed_caption_sources) if config.allowed_caption_sources else None,
            target_aware_mask_crop=config.target_aware_mask_crop,
            target_crop_context=config.target_crop_context,
        )
        val = TemporalCaptionManifestDataset(
            val_manifests,
            split=config.val_split,
            image_size=config.image_size,
            output_grid=config.output_grid,
            max_pairs=config.max_val_samples,
            allowed_caption_sources=set(config.allowed_caption_sources) if config.allowed_caption_sources else None,
            target_aware_mask_crop=config.target_aware_mask_crop,
            target_crop_context=config.target_crop_context,
        )
        return train, val
    return base._build_datasets(config)


def _build_stage1_full_count_datasets(config: Stage1NextConfig) -> tuple[Dataset | None, Dataset | None]:
    train_manifests = tuple(config.train_manifests)
    val_manifests = tuple(config.val_manifests)
    if not train_manifests and not val_manifests:
        return None, None
    if not train_manifests or not val_manifests:
        raise ValueError("Both train and validation manifests are required for manifest-based Stage-1-next counting")
    allowed_sources = set(config.allowed_caption_sources) if config.allowed_caption_sources else None
    train = TemporalCaptionManifestDataset(
        train_manifests,
        split=config.train_split,
        image_size=config.image_size,
        output_grid=config.output_grid,
        max_pairs=None,
        allowed_caption_sources=allowed_sources,
    )
    val = TemporalCaptionManifestDataset(
        val_manifests,
        split=config.val_split,
        image_size=config.image_size,
        output_grid=config.output_grid,
        max_pairs=None,
        allowed_caption_sources=allowed_sources,
    )
    return train, val


def _assert_stage1_disjoint(train: Dataset, val: Dataset) -> None:
    def ids(dataset: Dataset) -> set[str]:
        samples = getattr(dataset, "samples", None)
        if samples is not None:
            values: set[str] = set()
            for sample in samples:
                if isinstance(sample, dict):
                    values.add(str(sample.get("pair_id")))
                else:
                    values.add(str(getattr(sample, "sample_id", getattr(sample, "pair_id", ""))))
            return values
        return {str(getattr(dataset[index], "pair_id")) for index in range(len(dataset))}

    overlap = ids(train) & ids(val)
    if overlap:
        raise RuntimeError(f"Train/validation pair ID leakage detected: {sorted(overlap)[:10]}")


def _stage1_data_metadata(
    config: Stage1NextConfig,
    train: Dataset,
    val: Dataset,
    *,
    full_train: Dataset | None = None,
    full_val: Dataset | None = None,
) -> dict[str, Any]:
    train_manifests = [str(path) for path in config.train_manifests]
    val_manifests = [str(path) for path in config.val_manifests]
    data_mode = "mixed" if train_manifests or val_manifests else "levir_only"
    weights = parse_dataset_weights(config.dataset_sampling_weights) if data_mode == "mixed" else {}
    count_train = full_train if full_train is not None else train
    count_val = full_val if full_val is not None else val
    dataset_names: set[str] = set()
    for dataset in (count_train, count_val):
        mapping = getattr(dataset, "indices_by_dataset", None)
        if isinstance(mapping, dict):
            dataset_names.update(str(name) for name in mapping)
        else:
            for index in range(min(len(dataset), 64)):
                item = dataset[index]
                dataset_names.add(str(getattr(item, "dataset_name", getattr(item, "metadata", {}).get("dataset_name", "levir_mci"))))
    return {
        "data_mode": data_mode,
        "train_manifests": train_manifests,
        "val_manifests": val_manifests,
        "manifest_fingerprints": {
            "train": {path: manifest_file_fingerprint(path) for path in train_manifests},
            "validation": {path: manifest_file_fingerprint(path) for path in val_manifests},
        },
        "dataset_names": sorted(dataset_names),
        "dataset_weights": weights,
        "train_row_count": len(count_train),
        "validation_row_count": len(count_val),
    }


def _dataset_index_counts(dataset: Dataset) -> dict[str, int]:
    mapping = getattr(dataset, "indices_by_dataset", None)
    if isinstance(mapping, dict):
        return {str(name): len(indices) for name, indices in sorted(mapping.items())}
    counts: Counter[str] = Counter()
    for index in range(len(dataset)):
        item = dataset[index]
        counts[str(getattr(item, "dataset_name", getattr(item, "metadata", {}).get("dataset_name", "unknown")))] += 1
    return dict(sorted(counts.items()))


def _mixed_subset_coverage_details(
    dataset: Dataset,
    *,
    expected_datasets: set[str],
    max_pairs: int | None,
    subset_name: str,
) -> dict[str, Any]:
    expected = sorted(str(name) for name in expected_datasets)
    selected_counts = _dataset_index_counts(dataset)
    available_counts = dict(sorted(getattr(dataset, "selection_metadata", {}).get("available_counts_by_dataset", {}).items()))
    if not available_counts:
        available_counts = dict(selected_counts)
    omitted_datasets = list(getattr(dataset, "selection_metadata", {}).get("omitted_datasets", []))
    missing_expected = [name for name in expected if selected_counts.get(name, 0) <= 0]
    passed = not missing_expected
    details = {
        "subset_name": subset_name,
        "requested_max_pairs": max_pairs,
        "expected_datasets": expected,
        "available_datasets": sorted(name for name, count in available_counts.items() if int(count) > 0),
        "selected_datasets": sorted(name for name, count in selected_counts.items() if int(count) > 0),
        "sample_counts_by_dataset": selected_counts,
        "available_counts_by_dataset": available_counts,
        "omitted_datasets": omitted_datasets,
        "missing_expected_datasets": missing_expected,
        "mixed_subset_coverage_passed": passed,
    }
    if not passed:
        raise ValueError(
            f"Mixed-smoke coverage error for {subset_name}: max_pairs={max_pairs}; "
            f"expected_datasets={expected}; available_datasets={details['available_datasets']}; "
            f"selected_datasets={details['selected_datasets']}; missing_expected_datasets={missing_expected}; "
            f"available_counts_by_dataset={available_counts}; selected_counts_by_dataset={selected_counts}; "
            f"omitted_datasets={omitted_datasets}"
        )
    return details


def _expected_train_datasets(full_train: Dataset, configured_weights: dict[str, float]) -> set[str]:
    available = {name for name, count in _dataset_index_counts(full_train).items() if count > 0}
    positive = {str(name) for name, weight in configured_weights.items() if float(weight) > 0.0}
    return positive & available


def _expected_manifest_datasets(full_dataset: Dataset) -> set[str]:
    return {name for name, count in _dataset_index_counts(full_dataset).items() if count > 0}


def build_localization_validation_datasets(config: Stage1NextConfig) -> tuple[Dataset | None, Dataset | None]:
    manifests = tuple(config.localization_val_manifests)
    if not manifests:
        return None, None
    allowed_sources = set(config.allowed_caption_sources) if config.allowed_caption_sources else None
    common = {
        "split": config.val_split,
        "image_size": config.image_size,
        "output_grid": config.output_grid,
        "allowed_caption_sources": allowed_sources,
    }
    selected = TemporalCaptionManifestDataset(manifests, max_pairs=config.max_val_samples, **common)
    full = TemporalCaptionManifestDataset(manifests, max_pairs=None, **common)
    return selected, full


def _sample_mask_fraction(sample: Any) -> float:
    mask = getattr(sample, "mask", None)
    if mask is not None:
        return float(torch.as_tensor(mask).float().mean().item())
    path = getattr(sample, "binary_change_mask", None)
    if path is not None:
        try:
            from land_change_detection.data.unichange_mci import _load_mask

            return float(_load_mask(path, None).float().mean().item())
        except Exception:
            return float("nan")
    return float("nan")


def audit_dataset_conflicts(dataset: Dataset, config: Stage1NextConfig) -> tuple[dict[str, Any], list[int]]:
    counts: Counter[str] = Counter()
    affected: list[int] = []
    samples = getattr(dataset, "samples", None)
    iterable = samples if samples is not None else [dataset[index] for index in range(len(dataset))]
    for index, sample in enumerate(iterable):
        captions = _captions_from_sample(sample)
        semantics = [classify_caption_semantics(caption) for caption in captions]
        no_change_all = bool(semantics) and all(item["no_change"] for item in semantics)
        changed_all = bool(semantics) and all(item["changed"] for item in semantics)
        appeared = any(item["appeared"] for item in semantics)
        disappeared = any(item["disappeared"] for item in semantics)
        mask_fraction = _sample_mask_fraction(sample)
        mask_changed = math.isfinite(mask_fraction) and mask_fraction > config.conflict_mask_threshold
        changeflag = getattr(sample, "changeflag", None)
        if changeflag is None:
            changeflag = getattr(sample, "metadata", {}).get("changeflag") if hasattr(sample, "metadata") else None
        flag_changed = None if changeflag is None else bool(int(changeflag) if str(changeflag).isdigit() else str(changeflag).casefold() in {"change", "changed", "true", "yes", "1"})
        categories: list[str] = []
        if no_change_all and mask_changed:
            categories.append("all_no_change_captions_mask_changed")
        if changed_all and math.isfinite(mask_fraction) and mask_fraction == 0.0:
            categories.append("all_changed_captions_empty_mask")
        if appeared and disappeared:
            categories.append("appeared_disappeared_caption_contradiction")
        caption_consensus = False if no_change_all else True if changed_all else None
        if flag_changed is not None and caption_consensus is not None and math.isfinite(mask_fraction):
            if len({flag_changed, mask_fraction > 0.0, caption_consensus}) > 1:
                categories.append("changeflag_mask_caption_consensus_disagree")
        if categories:
            affected.append(index)
            for category in categories:
                counts[category] += 1
    counts["affected_pairs"] = len(affected)
    counts["total_pairs"] = len(iterable)
    return dict(counts), affected


def composite_score(metrics: dict[str, Any]) -> float:
    score = (
        0.30 * float(metrics.get("macro_semantic_recall@1", metrics.get("semantic_recall@1", 0.0)))
        + 0.25 * float(metrics.get("macro_semantic_recall@5", metrics.get("semantic_recall@5", 0.0)))
        + 0.20 * float(metrics.get("macro_semantic_recall@10", metrics.get("semantic_recall@10", 0.0)))
        + 0.15 * float(metrics.get("macro_semantic_nDCG@10", metrics.get("semantic_nDCG@10", 0.0)))
        + 0.10 * float(metrics.get("detailed_query_R@5", 0.0))
    )
    return float(score)


def _selection_scores(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "semantic_r1": float(metrics.get("macro_semantic_recall@1", metrics.get("semantic_recall@1", 0.0))),
        "semantic_r5": float(metrics.get("macro_semantic_recall@5", metrics.get("semantic_recall@5", 0.0))),
        "semantic_ndcg10": float(metrics.get("macro_semantic_nDCG@10", metrics.get("semantic_nDCG@10", 0.0))),
        "detailed_r5": float(metrics.get("detailed_query_R@5", 0.0)),
        "macro_semantic": float(metrics.get("macro_semantic_mean", 0.0)),
        "composite": composite_score(metrics),
    }


def _critical_resume_config(config: Stage1NextConfig) -> dict[str, Any]:
    keys = (
        "image_size",
        "output_grid",
        "batch_size",
        "temporal_depth",
        "use_direction_embeddings",
        "use_explicit_change_fusion",
        "trainable_temperature",
        "max_captions_per_pair",
        "caption_frequency_power",
        "text_to_pair_weight",
        "pair_to_text_weight",
        "use_text_adapter",
        "text_adapter_hidden_dim",
        "text_max_length",
        "semantic_soft_target_weight",
        "semantic_teacher_top_k",
        "semantic_teacher_temperature",
        "train_eval_pairs",
        "train_eval_interval",
        "enable_patch_reranker",
        "qcpr_alpha",
        "qcpr_beta",
        "qcpr_local_loss_weight",
        "fused_retrieval_loss_weight",
        "global_embedding_preservation_weight",
        "freeze_global_retrieval",
        "query_segmentation_loss_weight",
        "structured_fna_weight",
    )
    payload = asdict(config)
    return {key: payload[key] for key in keys}


def _validate_resume_config(config: Stage1NextConfig, payload: dict[str, Any]) -> None:
    previous = payload.get("config", {})
    expected = _critical_resume_config(config)
    mismatches = {
        key: (previous.get(key), value)
        for key, value in expected.items()
        if previous.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Resume checkpoint configuration mismatch: {mismatches}")


def run(
    data_root: Path,
    output_dir: Path,
    universat_source: Path,
    universat_checkpoint: Path,
    jina_model: Path,
    batch_size: int,
    epochs: int,
    num_workers: int,
    resume: Path | None = None,
    *,
    initialize_from_v1: Path | None = None,
    temporal_depth: int = 4,
    max_captions_per_pair: int = 2,
    caption_frequency_power: float = 0.5,
    train_eval_pairs: int = 1024,
    train_eval_interval: int = 2,
    similarity_query_chunk_size: int = 16,
    similarity_candidate_chunk_size: int = 32,
    enable_conflict_filtering: bool = False,
    train_manifests: tuple[Path, ...] = (),
    val_manifests: tuple[Path, ...] = (),
    dataset_config: Path | None = None,
    dataset_sampling_weights: tuple[str, ...] = ("levir_mci=0.55", "second_cc=0.45"),
    text_max_length: int = 256,
    early_stopping_patience: int = 4,
    early_stopping_min_improvement: float = 0.002,
    max_steps: int | None = None,
    enable_patch_reranker: bool = False,
    qcpr_architecture_version: str = "v1",
    enable_temporal_explanation_channels: bool = False,
    qcpr_alpha: float = 1.0,
    qcpr_beta: float = 0.25,
    qcpr_local_loss_weight: float = 1.0,
    fused_retrieval_loss_weight: float = 0.25,
    global_embedding_preservation_weight: float = 0.0,
    freeze_global_retrieval: bool = False,
    text_adapter_lr: float = 2e-5,
    query_segmentation_loss_weight: float = 0.2,
    changed_channel_loss_weight: float = 0.2,
    appeared_channel_loss_weight: float = 0.2,
    disappeared_channel_loss_weight: float = 0.2,
    temporal_reversal_consistency_loss_weight: float = 0.05,
    structured_fna_weight: float = 0.0,
    local_margin_loss_weight: float = 0.1,
    local_margin: float = 0.1,
    conditional_instance_loss_weight: float = 0.01,
    structured_auxiliary_loss_weight: float = 0.05,
    grad_clip_norm: float = 1.0,
) -> int:
    device = strict_device("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Stage1NextConfig(
        data_root=str(data_root),
        output_dir=str(output_dir),
        universat_source=str(universat_source),
        universat_checkpoint=str(universat_checkpoint),
        jina_model=str(jina_model),
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers,
        temporal_depth=temporal_depth,
        max_captions_per_pair=max_captions_per_pair,
        caption_frequency_power=caption_frequency_power,
        train_eval_pairs=train_eval_pairs,
        train_eval_interval=train_eval_interval,
        similarity_query_chunk_size=similarity_query_chunk_size,
        similarity_candidate_chunk_size=similarity_candidate_chunk_size,
        enable_conflict_filtering=enable_conflict_filtering,
        train_manifests=tuple(str(path) for path in train_manifests),
        val_manifests=tuple(str(path) for path in val_manifests),
        dataset_config=str(dataset_config) if dataset_config else None,
        dataset_sampling_weights=dataset_sampling_weights,
        text_max_length=text_max_length,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_improvement=early_stopping_min_improvement,
        max_steps=max_steps,
        enable_patch_reranker=enable_patch_reranker,
        qcpr_architecture_version=qcpr_architecture_version,
        enable_temporal_explanation_channels=enable_temporal_explanation_channels,
        qcpr_alpha=qcpr_alpha,
        qcpr_beta=qcpr_beta,
        qcpr_local_loss_weight=qcpr_local_loss_weight,
        fused_retrieval_loss_weight=fused_retrieval_loss_weight,
        global_embedding_preservation_weight=global_embedding_preservation_weight,
        freeze_global_retrieval=freeze_global_retrieval,
        text_adapter_lr=text_adapter_lr,
        query_segmentation_loss_weight=query_segmentation_loss_weight,
        changed_channel_loss_weight=changed_channel_loss_weight,
        appeared_channel_loss_weight=appeared_channel_loss_weight,
        disappeared_channel_loss_weight=disappeared_channel_loss_weight,
        temporal_reversal_consistency_loss_weight=temporal_reversal_consistency_loss_weight,
        structured_fna_weight=structured_fna_weight,
        local_margin_loss_weight=local_margin_loss_weight,
        local_margin=local_margin,
        conditional_instance_loss_weight=conditional_instance_loss_weight,
        structured_auxiliary_loss_weight=structured_auxiliary_loss_weight,
        grad_clip_norm=grad_clip_norm,
    )
    validate_config(config)
    base._set_seed(config.seed)
    train, val = _build_stage1_datasets(config)
    _assert_stage1_disjoint(train, val)
    data_metadata = _stage1_data_metadata(config, train, val)
    conflict_counts, conflict_indices = audit_dataset_conflicts(train, config)
    base._write_json(
        output_dir / "reports" / "dataset_conflict_audit.json",
        {
            "stage1_next": True,
            "default_official_data_preserved": not config.enable_conflict_filtering,
            "filtering_enabled": config.enable_conflict_filtering,
            "counts": conflict_counts,
            "affected_indices": conflict_indices[:10000],
        },
    )
    if config.enable_conflict_filtering and conflict_indices:
        conflict_set = set(conflict_indices)
        train = Subset(train, [index for index in range(len(train)) if index not in conflict_set])
    frequencies = caption_frequencies(train)
    base._write_json(
        output_dir / "run_config.json",
        asdict(config)
        | {
            "stage1_next": True,
            "loss": "semantic_soft_target_text_to_pair",
            "stable_caption_groups": True,
            "resume": str(resume) if resume else None,
            "history_schema": 4,
            "caption_group_statistics": {
                "unique_groups": len(frequencies),
                "caption_rows": int(sum(frequencies.values())),
            },
            "conflict_audit_counts": conflict_counts,
            **data_metadata,
            **run_metadata(),
        },
    )

    val_loader = make_eval_loader(val, config)
    train_eval_loader: DataLoader | None = None
    if config.train_eval_pairs > 0:
        count = min(config.train_eval_pairs, len(train))
        train_eval_dataset: Dataset = Subset(train, range(count))
        train_eval_loader = make_eval_loader(train_eval_dataset, config)

    steps_per_epoch = math.ceil(len(train) / config.batch_size)
    total_steps = min(steps_per_epoch * config.epochs, config.max_steps) if config.max_steps is not None else steps_per_epoch * config.epochs
    model = build_model(config, device)
    initialization_missing: list[str] = []
    initialization_unexpected: list[str] = []
    initialized_from_v1_modules: list[str] = []
    if initialize_from_v1 is not None:
        if resume is not None:
            raise ValueError("resume and initialize_from_v1 are mutually exclusive")
        initialization_payload = torch.load(initialize_from_v1, map_location=device)
        incompatible = model.load_state_dict(initialization_payload["model"], strict=False)
        initialization_missing = list(incompatible.missing_keys)
        initialization_unexpected = list(incompatible.unexpected_keys)
        initialized_from_v1_modules = sorted({name.split(".", 1)[0] for name in initialization_payload["model"]})
    frozen_global_modules: list[str] = []
    if config.freeze_global_retrieval:
        frozen_global_modules = freeze_global_retrieval_modules(model)
    optimizer = make_optimizer(model, config)
    scheduler = base._make_scheduler(optimizer, total_steps, config)

    start_epoch = 0
    resume_batch = 0
    step = 0
    best_scores = {
        "semantic_r1": -1.0,
        "semantic_r5": -1.0,
        "semantic_ndcg10": -1.0,
        "detailed_r5": -1.0,
        "macro_semantic": -1.0,
        "composite": -1.0,
    }
    checkpoint_names = {
        "semantic_r1": "best_semantic_r1.pt",
        "semantic_r5": "best_semantic_r5.pt",
        "semantic_ndcg10": "best_semantic_ndcg10.pt",
        "detailed_r5": "best_detailed_r5.pt",
        "macro_semantic": "best_macro_semantic.pt",
        "composite": "best_composite.pt",
    }
    if resume:
        payload = torch.load(resume, map_location=device)
        if not payload.get("stage1_next"):
            raise RuntimeError("Refusing to resume Stage-1-next from a baseline checkpoint")
        _validate_resume_config(config, payload)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        base._restore_rng_state(payload.get("rng", {}))
        start_epoch = int(payload.get("epoch_index", 0))
        resume_batch = int(payload.get("next_batch_index", 0))
        step = int(payload.get("step", 0))
        best_scores.update({key: float(value) for key, value in payload.get("best_scores", {}).items()})

    named_parameters = dict(model.named_parameters())
    trainable_names = sorted(name for name, parameter in named_parameters.items() if parameter.requires_grad)
    frozen_names = sorted(name for name, parameter in named_parameters.items() if not parameter.requires_grad)
    random_names = sorted(name for name in initialization_missing if name in named_parameters)
    initialization_audit = {
        "resume_checkpoint": str(resume) if resume else None,
        "resume_checkpoint_sha256": file_sha256(resume) if resume else None,
        "initialize_from_v1_checkpoint": str(initialize_from_v1) if initialize_from_v1 else None,
        "initialize_from_v1_sha256": file_sha256(initialize_from_v1) if initialize_from_v1 else None,
        "visual_backbone_checkpoint": path_fingerprint(Path(config.universat_checkpoint)),
        "text_checkpoint": path_fingerprint(Path(config.jina_model)),
        "modules_loaded_from_checkpoint": initialized_from_v1_modules,
        "frozen_global_modules": frozen_global_modules,
        "randomly_initialized_parameters": random_names,
        "frozen_parameters": frozen_names,
        "trainable_parameters": trainable_names,
        "missing_keys": initialization_missing,
        "unexpected_keys": initialization_unexpected,
        "qcpr_architecture_version": config.qcpr_architecture_version,
    }
    optimizer_audit = {
        "groups": [
            {"name": group.get("name"), "lr": group["lr"], "weight_decay": group["weight_decay"], "parameter_names": list(group.get("param_names", []))}
            for group in optimizer.param_groups
        ],
        "trainable_parameter_count": len(trainable_names),
        "covered_exactly_once": len({id(parameter) for group in optimizer.param_groups for parameter in group["params"]}) == len(trainable_names),
    }
    base._write_json(output_dir / "initialization_audit.json", initialization_audit)
    base._write_json(output_dir / "optimizer_parameter_audit.json", optimizer_audit)
    base._write_json(output_dir / "resolved_config.json", asdict(config))
    (output_dir / "git_commit.txt").write_text(run_metadata()["git_commit"] + "\n", encoding="utf-8")
    base._write_json(output_dir / "environment_summary.json", {"hostname": os.uname().nodename, "python": os.sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)})
    base._write_json(output_dir / "training_summary.json", {"status": "RUNNING", "step": step, "stage1_next": True, "qcpr_architecture_version": config.qcpr_architecture_version})

    history_path = output_dir / "metrics_history.jsonl"
    epochs_since_composite = 0
    temporal_supervised_pairs_total = 0
    for epoch in range(start_epoch, config.epochs):
        train_loader = make_train_loader(train, config, frequencies, epoch)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        epoch_pair_count = 0
        grad_norms: list[float] = []
        module_grad_norm_history: dict[str, list[float]] = {}
        clipped_steps = 0

        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            model.train()
            batch = base._move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with base._amp_context(device, config.use_bf16):
                output = model(
                    batch["images"],
                    batch["captions"],
                    batch["caption_to_pair"],
                    batch["temporal_valid_mask"],
                )
                retrieval_selection = retrieval_supervision_selection(batch, device)
                pair_retrieval_mask = retrieval_selection["pair_mask"]
                caption_retrieval_mask = retrieval_selection["caption_mask"]
                if torch.any(pair_retrieval_mask) and torch.any(caption_retrieval_mask):
                    selected_pairs = retrieval_selection["selected_pairs"]
                    selected_queries = retrieval_selection["selected_queries"]
                    selected_captions = [batch["captions"][index] for index in selected_queries.tolist()]
                    selected_mapping = retrieval_selection["selected_mapping"]
                    selected_groups = stable_caption_group_ids(selected_captions, device=device)
                    selected_final_scores = (
                        output.final_scores[selected_queries][:, selected_pairs]
                        if output.final_scores is not None and config.enable_patch_reranker
                        else None
                    )
                    global_retrieval_loss, loss_diagnostics = semantic_text_to_pair_set_loss(
                        output.pair_embedding[selected_pairs],
                        output.text_embedding[selected_queries],
                        output.teacher_text_embedding[selected_queries],
                        selected_captions,
                        selected_mapping,
                        selected_groups,
                        logit_scale=model.retrieval_head.similarity_scale(),
                        text_to_pair_weight=config.text_to_pair_weight,
                        pair_to_text_weight=config.pair_to_text_weight,
                        semantic_soft_target_weight=config.semantic_soft_target_weight,
                        semantic_teacher_top_k=config.semantic_teacher_top_k,
                        semantic_teacher_temperature=config.semantic_teacher_temperature,
                        structured_fna_weight=config.structured_fna_weight,
                        return_diagnostics=True,
                    )
                    fused_retrieval_loss = global_retrieval_loss.new_zeros(())
                    if selected_final_scores is not None:
                        fused_retrieval_loss, fused_loss_diagnostics = semantic_text_to_pair_set_loss(
                            output.pair_embedding[selected_pairs],
                            output.text_embedding[selected_queries],
                            output.teacher_text_embedding[selected_queries],
                            selected_captions,
                            selected_mapping,
                            selected_groups,
                            logit_scale=model.retrieval_head.similarity_scale(),
                            text_to_pair_weight=config.text_to_pair_weight,
                            pair_to_text_weight=config.pair_to_text_weight,
                            semantic_soft_target_weight=config.semantic_soft_target_weight,
                            semantic_teacher_top_k=config.semantic_teacher_top_k,
                            semantic_teacher_temperature=config.semantic_teacher_temperature,
                            logits_text_to_pair=selected_final_scores * model.retrieval_head.similarity_scale(),
                            structured_fna_weight=config.structured_fna_weight,
                            return_diagnostics=True,
                        )
                        loss_diagnostics.update(_prefix_metrics(fused_loss_diagnostics, "fused_"))
                    retrieval_loss = global_retrieval_loss + config.fused_retrieval_loss_weight * fused_retrieval_loss
                    embedding_preservation_loss = 1.0 - F.cosine_similarity(
                        output.text_embedding[selected_queries], output.teacher_text_embedding[selected_queries], dim=-1
                    ).mean()
                    local_margin_loss = retrieval_loss.new_zeros(())
                    conditional_identity_loss = retrieval_loss.new_zeros(())
                    structured_auxiliary_loss = retrieval_loss.new_zeros(())
                    if config.qcpr_architecture_version == "v2" and output.local_scores is not None:
                        selected_local_scores = output.local_scores[selected_queries][:, selected_pairs]
                        broad_semantic_positive_mask = structured_fna_relevance_matrix(
                            selected_captions,
                            selected_mapping,
                            selected_groups,
                            pair_count=len(selected_pairs),
                        ) > 0
                        broad_semantic_positive_mask |= semantic_teacher_relevance_matrix(
                            output.teacher_text_embedding[selected_queries],
                            selected_captions,
                            selected_mapping,
                            selected_groups,
                            pair_count=len(selected_pairs),
                            top_k=0,
                        ) > 0
                        local_margin_loss, local_margin_diagnostics = local_positive_negative_margin_loss(
                            selected_local_scores,
                            selected_mapping,
                            selected_captions,
                            margin=config.local_margin,
                            latent_positive_mask=broad_semantic_positive_mask,
                        )
                        conditional_identity_loss = conditional_instance_discrimination_loss(
                            selected_final_scores if selected_final_scores is not None else selected_local_scores,
                            selected_mapping,
                            broad_semantic_positive_mask,
                        )
                        loss_diagnostics.update(local_margin_diagnostics)
                        auxiliary_names = {
                            "S_object": output.object_scores,
                            "S_direction": output.direction_scores,
                            "S_location": output.location_scores,
                            "S_count": output.count_scores,
                            "S_relation": output.relation_scores,
                        }
                        if all(value is not None for value in auxiliary_names.values()):
                            structured_auxiliary_loss, auxiliary_diagnostics = structured_auxiliary_evidence_loss(
                                {name: value[selected_queries][:, selected_pairs] for name, value in auxiliary_names.items()},
                                selected_mapping,
                                selected_captions,
                            )
                            loss_diagnostics.update(auxiliary_diagnostics)
                else:
                    retrieval_loss = output.pair_embedding.sum() * 0.0
                    global_retrieval_loss = retrieval_loss
                    fused_retrieval_loss = retrieval_loss
                    local_margin_loss = retrieval_loss.new_zeros(())
                    conditional_identity_loss = retrieval_loss.new_zeros(())
                    structured_auxiliary_loss = retrieval_loss.new_zeros(())
                    loss_diagnostics = {
                        "retrieval_supervised_pairs": 0,
                        "retrieval_supervised_queries": 0,
                        "structured_fna_weight": float(config.structured_fna_weight),
                    }
                loss_diagnostics["retrieval_supervised_pairs"] = int(pair_retrieval_mask.sum().item())
                loss_diagnostics["retrieval_supervised_queries"] = int(caption_retrieval_mask.sum().item())
                segmentation = {
                    "query_specific_segmentation_loss": retrieval_loss.new_zeros(()),
                    "generic_change_segmentation_loss": retrieval_loss.new_zeros(()),
                    "total_segmentation_loss": retrieval_loss.new_zeros(()),
                    "segmentation_supervised_pairs": 0,
                    "query_specific_supervised_pairs": 0,
                    "generic_supervised_pairs": 0,
                    "mean_segmentation_weight": 0.0,
                }
                if config.enable_patch_reranker:
                    if output.query_mask_logits is None:
                        raise RuntimeError("QCPR is enabled but query_mask_logits are unavailable")
                    segmentation = segmentation_loss_components(
                        output.query_mask_logits,
                        batch["caption_to_pair"],
                        batch["masks"],
                        batch["segmentation_target_kinds"],
                        batch["segmentation_weights"],
                    )
                total_segmentation_loss = segmentation["total_segmentation_loss"]
                temporal_losses = {
                    "changed_channel_loss": retrieval_loss.new_zeros(()), "appeared_channel_loss": retrieval_loss.new_zeros(()),
                    "disappeared_channel_loss": retrieval_loss.new_zeros(()), "temporal_reversal_consistency_loss": retrieval_loss.new_zeros(()),
                    "temporal_supervised_pairs": 0,
                }
                if config.enable_temporal_explanation_channels:
                    if config.qcpr_architecture_version != "v2" or output.temporal_explanation_logits is None:
                        raise RuntimeError("Temporal explanation supervision requires QCPR v2")
                    reverse_output = model(batch["images"].flip(1), batch["captions"], batch["caption_to_pair"], batch["temporal_valid_mask"])
                    temporal_losses = temporal_channel_loss_components(
                        output.temporal_explanation_logits, batch["masks"], batch["changed_masks"], batch["segmentation_target_kinds"],
                        batch["segmentation_weights"], batch["change_types"], reverse_output.temporal_explanation_logits,
                    )
                    temporal_supervised_pairs_total += int(temporal_losses["temporal_supervised_pairs"])
                total_localization_loss = (
                    config.query_segmentation_loss_weight * total_segmentation_loss
                    + config.changed_channel_loss_weight * temporal_losses["changed_channel_loss"]
                    + config.appeared_channel_loss_weight * temporal_losses["appeared_channel_loss"]
                    + config.disappeared_channel_loss_weight * temporal_losses["disappeared_channel_loss"]
                    + config.temporal_reversal_consistency_loss_weight * temporal_losses["temporal_reversal_consistency_loss"]
                )
                loss = (
                    config.qcpr_local_loss_weight * retrieval_loss
                    + config.local_margin_loss_weight * local_margin_loss
                    + config.conditional_instance_loss_weight * conditional_identity_loss
                    + config.structured_auxiliary_loss_weight * structured_auxiliary_loss
                    + config.global_embedding_preservation_weight * embedding_preservation_loss
                    + total_localization_loss
                )
                loss_diagnostics.update(
                    {
                        "retrieval_loss": float(retrieval_loss.detach().cpu()),
                        "global_retrieval_loss": float(global_retrieval_loss.detach().cpu()),
                        "fused_retrieval_loss": float(fused_retrieval_loss.detach().cpu()),
                        "fused_retrieval_loss_weight": float(config.fused_retrieval_loss_weight),
                        "global_embedding_preservation_loss": float(embedding_preservation_loss.detach().cpu()),
                        "global_embedding_preservation_weight": float(config.global_embedding_preservation_weight),
                        "qcpr_local_loss": float(retrieval_loss.detach().cpu()) if config.enable_patch_reranker else 0.0,
                        "local_positive_negative_margin_loss": float(local_margin_loss.detach().cpu()),
                        "conditional_instance_discrimination_loss": float(conditional_identity_loss.detach().cpu()),
                        "structured_auxiliary_evidence_loss": float(structured_auxiliary_loss.detach().cpu()),
                        "query_segmentation_loss": float(segmentation["query_specific_segmentation_loss"].detach().cpu()),
                        "query_specific_segmentation_loss": float(segmentation["query_specific_segmentation_loss"].detach().cpu()),
                        "generic_change_segmentation_loss": float(segmentation["generic_change_segmentation_loss"].detach().cpu()),
                        "total_segmentation_loss": float(total_segmentation_loss.detach().cpu()),
                        "changed_channel_loss": float(temporal_losses["changed_channel_loss"].detach().cpu()),
                        "appeared_channel_loss": float(temporal_losses["appeared_channel_loss"].detach().cpu()),
                        "disappeared_channel_loss": float(temporal_losses["disappeared_channel_loss"].detach().cpu()),
                        "temporal_reversal_consistency_loss": float(temporal_losses["temporal_reversal_consistency_loss"].detach().cpu()),
                        "total_localization_loss": float(total_localization_loss.detach().cpu()),
                        "temporal_supervised_pairs": int(temporal_losses["temporal_supervised_pairs"]),
                        "segmentation_supervised_pairs": int(segmentation["segmentation_supervised_pairs"]),
                        "query_specific_supervised_pairs": int(segmentation["query_specific_supervised_pairs"]),
                        "generic_supervised_pairs": int(segmentation["generic_supervised_pairs"]),
                        "mean_segmentation_weight": float(segmentation["mean_segmentation_weight"]),
                        "qcpr_score_mode": output.score_mode,
                    }
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {float(loss.detach().cpu())}")
            loss.backward()
            def summarize_gradients(parameters) -> dict[str, float | int | bool]:
                trainable_module_parameters = [parameter for parameter in parameters if parameter.requires_grad]
                gradients = [parameter.grad for parameter in trainable_module_parameters if parameter.grad is not None]
                norm = math.sqrt(sum(float(gradient.detach().float().square().sum().cpu()) for gradient in gradients))
                return {
                    "parameter_count": len(trainable_module_parameters),
                    "gradient_count": len(gradients),
                    "finite": bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
                    "nonzero": any(bool(torch.any(gradient != 0)) for gradient in gradients),
                    "grad_norm_before_clip": norm,
                }

            gradient_groups = {
                module_name: list(module.parameters())
                for module_name, module in model.named_children()
                if any(parameter.requires_grad for parameter in module.parameters())
            }
            if config.enable_patch_reranker and model.patch_reranker is not None and config.qcpr_architecture_version == "v2":
                reranker = model.patch_reranker
                gradient_groups.update({
                    "qcpr_temporal_descriptor": list(reranker.temporal_descriptor_mlp.parameters()),
                    "qcpr_token_projection": list(reranker.token_projection.parameters()),
                    "qcpr_interaction": list(reranker.interaction_mlp.parameters()),
                    "qcpr_temporal_channel": list(reranker.temporal_channel_head.parameters()),
                    "qcpr_fusion_calibration": [reranker.fusion_logits, reranker.branch_log_scales],
                })
            gradient_modules = {}
            for module_name, parameters in gradient_groups.items():
                trainable_module_parameters = [parameter for parameter in parameters if parameter.requires_grad]
                if not trainable_module_parameters:
                    continue
                gradient_modules[module_name] = summarize_gradients(trainable_module_parameters)
            module_gradient_norms = {
                f"grad_norm_{module_name}": float(details["grad_norm_before_clip"])
                for module_name, details in gradient_modules.items()
            }
            for name, value in module_gradient_norms.items():
                module_grad_norm_history.setdefault(name, []).append(value)
            base._write_json(output_dir / "gradient_flow_audit.json", {"step": step, "modules": gradient_modules})
            patch_projector_grad_present = False
            query_mask_head_grad_present = False
            if config.enable_patch_reranker and model.patch_reranker is not None:
                patch_projector_grad_present = any(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all() and bool(torch.any(parameter.grad != 0))
                    for parameter in model.patch_reranker.patch_projector.parameters()
                )
                query_mask_head_grad_present = any(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all() and bool(torch.any(parameter.grad != 0))
                    for parameter in model.patch_reranker.query_mask_head.parameters()
                )
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                trainable_stage1_parameters(model),
                config.grad_clip_norm,
            )
            grad_norm = float(torch.as_tensor(grad_norm_tensor).detach().cpu())
            grad_norms.append(grad_norm)
            clipped_steps += int(grad_norm > config.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            step += 1

            loss_value = float(loss.detach().cpu())
            pair_count = int(batch["images"].shape[0])
            epoch_loss_sum += loss_value
            epoch_steps += 1
            epoch_pair_count += pair_count
            _append_history(
                history_path,
                {
                    "record_type": "train_step",
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": loss_value,
                    "batch_pairs": pair_count,
                    "captions": len(batch["captions"]),
                    "grad_norm_before_clip": grad_norm,
                    "gradient_was_clipped": grad_norm > config.grad_clip_norm,
                    "grad_clip_norm": config.grad_clip_norm,
                    "patch_projector_grad_present": patch_projector_grad_present,
                    "query_mask_head_grad_present": query_mask_head_grad_present,
                    "logit_scale": float(model.retrieval_head.similarity_scale().detach().cpu()),
                    "effective_temperature": float(1.0 / model.retrieval_head.similarity_scale().detach().cpu()),
                    **loss_diagnostics,
                    **module_gradient_norms,
                    "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                },
            )
            if config.checkpoint_interval_steps > 0 and step % config.checkpoint_interval_steps == 0:
                save_checkpoint(
                    output_dir / f"step_{step}.pt",
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch_index=epoch,
                    next_batch_index=batch_index + 1,
                    step=step,
                    best_scores=best_scores,
                    metrics={},
                    temporal_supervised_pairs=temporal_supervised_pairs_total,
                )
            if config.max_steps is not None and step >= config.max_steps:
                break

        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        training_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        training_peak_reserved = int(torch.cuda.max_memory_reserved(device))

        validation_started = time.perf_counter()
        metrics = relevance_aware_retrieval_metrics(model, val_loader, device, config)
        validation_seconds = time.perf_counter() - validation_started
        train_eval_metrics: dict[str, Any] = {}
        if (
            train_eval_loader is not None
            and config.train_eval_interval > 0
            and ((epoch + 1) % config.train_eval_interval == 0 or epoch + 1 == config.epochs)
        ):
            train_eval_metrics = _prefix_metrics(
                relevance_aware_retrieval_metrics(model, train_eval_loader, device, config),
                "train_subset_",
            )

        grad_tensor = torch.tensor(grad_norms, dtype=torch.float32)
        module_grad_summary: dict[str, float] = {}
        for name, values in module_grad_norm_history.items():
            value_tensor = torch.tensor(values, dtype=torch.float32)
            module_grad_summary[f"{name}_mean"] = float(value_tensor.mean().item())
            module_grad_summary[f"{name}_p50"] = float(torch.quantile(value_tensor, 0.5).item())
            module_grad_summary[f"{name}_p90"] = float(torch.quantile(value_tensor, 0.9).item())
            module_grad_summary[f"{name}_max"] = float(value_tensor.max().item())
        epoch_record: dict[str, Any] = {
            "record_type": "epoch",
            "epoch": epoch + 1,
            "step": step,
            "train_loss_mean": epoch_loss_sum / max(epoch_steps, 1),
            "train_steps": epoch_steps,
            "train_pair_count": epoch_pair_count,
            "train_seconds": train_seconds,
            "train_pairs_per_second": epoch_pair_count / max(train_seconds, 1e-12),
            "validation_seconds": validation_seconds,
            "training_peak_allocated_vram_bytes": training_peak_allocated,
            "training_peak_reserved_vram_bytes": training_peak_reserved,
            "grad_norm_mean": float(grad_tensor.mean().item()) if grad_tensor.numel() else 0.0,
            "grad_norm_p50": float(torch.quantile(grad_tensor, 0.5).item()) if grad_tensor.numel() else 0.0,
            "grad_norm_p90": float(torch.quantile(grad_tensor, 0.9).item()) if grad_tensor.numel() else 0.0,
            "grad_norm_max": float(grad_tensor.max().item()) if grad_tensor.numel() else 0.0,
            "gradient_clipping_fraction": clipped_steps / max(epoch_steps, 1),
            **module_grad_summary,
            "logit_scale": float(model.retrieval_head.similarity_scale().detach().cpu()),
            "effective_temperature": float(1.0 / model.retrieval_head.similarity_scale().detach().cpu()),
            "optimizer_groups": [
                {
                    "name": group.get("name", f"group_{index}"),
                    "lr": float(group["lr"]),
                    "weight_decay": float(group["weight_decay"]),
                    "param_count": len(group["params"]),
                    "param_names": group.get("param_names", []),
                }
                for index, group in enumerate(optimizer.param_groups)
            ],
            "dataset_sizes": {"train": len(train), "validation": len(val)},
            **data_metadata,
            "caption_group_statistics": {"unique_groups": len(frequencies), "caption_rows": int(sum(frequencies.values()))},
            "conflict_audit_counts": conflict_counts,
            **metrics,
            **train_eval_metrics,
        }
        epoch_record["composite_score"] = composite_score(epoch_record)
        _append_history(history_path, epoch_record)
        base._write_json(output_dir / "latest_metrics.json", epoch_record)

        selection = _selection_scores(epoch_record)
        composite_improved = selection["composite"] > best_scores["composite"] + config.early_stopping_min_improvement
        for name, score in selection.items():
            if score > best_scores[name] + (config.early_stopping_min_improvement if name == "composite" else 0.0):
                best_scores[name] = score
                checkpoint_name = checkpoint_names[name]
                save_checkpoint(
                    output_dir / checkpoint_name,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch_index=epoch + 1,
                    next_batch_index=0,
                    step=step,
                    best_scores=best_scores,
                    metrics=epoch_record,
                    selection_metric=name,
                    selection_value=score,
                    temporal_supervised_pairs=temporal_supervised_pairs_total,
                )
                _append_history(
                    history_path,
                    {
                        "record_type": "checkpoint_event",
                        "epoch": epoch + 1,
                        "step": step,
                        "checkpoint": checkpoint_name,
                        "selection_metric": name,
                        "selection_value": score,
                    },
                )
                if name == "composite":
                    save_checkpoint(
                        output_dir / "best_retrieval.pt",
                        model,
                        optimizer,
                        scheduler,
                        config,
                        epoch_index=epoch + 1,
                        next_batch_index=0,
                        step=step,
                        best_scores=best_scores,
                        metrics=epoch_record,
                        selection_metric="composite",
                        selection_value=score,
                        temporal_supervised_pairs=temporal_supervised_pairs_total,
                    )
                    _append_history(
                        history_path,
                        {
                            "record_type": "checkpoint_event",
                            "epoch": epoch + 1,
                            "step": step,
                            "checkpoint": "best_retrieval.pt",
                            "selection_metric": "composite",
                            "selection_value": score,
                        },
                    )

        save_checkpoint(
            output_dir / "last_retrieval.pt",
            model,
            optimizer,
            scheduler,
            config,
            epoch_index=epoch + 1,
            next_batch_index=0,
            step=step,
            best_scores=best_scores,
            metrics=epoch_record,
            temporal_supervised_pairs=temporal_supervised_pairs_total,
        )
        base._write_json(
            output_dir / "training_report.json",
            {
                "stage1_next": True,
                "status": (
                    "BOUNDED_COMPLETED"
                    if config.max_steps is not None and step >= config.max_steps
                    else "RUNNING" if epoch + 1 < config.epochs else "COMPLETED"
                ),
                "completed_epochs": epoch + 1,
                "step": step,
                "best_scores": best_scores,
                "latest_metrics": epoch_record,
                **data_metadata,
                **run_metadata(),
            },
        )
        resume_batch = 0
        epochs_since_composite = 0 if composite_improved else epochs_since_composite + 1
        if config.early_stopping_patience and epochs_since_composite >= config.early_stopping_patience:
            _append_history(
                history_path,
                {
                    "record_type": "early_stopping",
                    "epoch": epoch + 1,
                    "step": step,
                    "metric": "composite",
                    "patience": config.early_stopping_patience,
                    "minimum_improvement": config.early_stopping_min_improvement,
                },
            )
            break
        if config.max_steps is not None and step >= config.max_steps:
            break

    _append_history(
        history_path,
        {
            "record_type": "final_summary",
            "step": step,
            "best_scores": best_scores,
            "stage1_next": True,
            **run_metadata(),
        },
    )
    base._write_json(output_dir / "training_summary.json", {"status": "BOUNDED_COMPLETED" if config.max_steps is not None and step >= config.max_steps else "COMPLETED", "step": step, "best_scores": best_scores, "stage1_next": True, "qcpr_architecture_version": config.qcpr_architecture_version, **run_metadata()})
    return 0
