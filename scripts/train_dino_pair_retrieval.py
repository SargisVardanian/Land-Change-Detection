from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset

from land_change_detection.losses.retrieval_losses import (
    asymmetric_caption_pair_loss,
    soft_histogram_contrastive_loss,
    supervised_contrastive_loss,
)
from land_change_detection.models.dino_change_retriever import DINOChangeRetriever, DINOChangeRetrieverConfig
from land_change_detection.retrieval_cache import IndexedShardReader
from land_change_detection.retrieval_baselines import preset_by_name, preset_names
from land_change_detection.run_metadata import jsonl_fingerprint, locate_storage_inventory, path_fingerprint, safe_git_commit
from land_change_detection.semantic_transitions import transition_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train retrieval-first DINO/simple-patch change retriever.")
    parser.add_argument("--preset", choices=preset_names(), default=None)
    parser.add_argument("--levir-manifest", type=Path, default=None)
    parser.add_argument("--eval-levir-manifest", type=Path, default=None)
    parser.add_argument("--pair-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-pair", type=float, default=0.2)
    parser.add_argument("--visual-backbone", choices=("simple_patch", "dinov2"), default="simple_patch")
    parser.add_argument("--text-backbone", choices=("simple_text", "remoteclip", "hf_remoteclip", "openclip"), default="remoteclip")
    parser.add_argument("--pair-feature-mode", choices=("t2_only", "signed_delta", "change_fusion"), default="change_fusion")
    parser.add_argument("--dinov2-model-path", type=Path, default=None)
    parser.add_argument("--remoteclip-arch", default="ViT-B-32")
    parser.add_argument("--remoteclip-checkpoint", type=Path, default=None)
    parser.add_argument("--hf-remoteclip-model-path", type=Path, default=None)
    parser.add_argument("--openclip-model-name", default="ViT-B-32")
    parser.add_argument("--openclip-pretrained", default=None)
    parser.add_argument("--levir-pair-cache-index", type=Path, default=None)
    parser.add_argument("--levir-text-cache-index", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--pair-loss", choices=("supervised", "soft"), default="supervised")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RetrievalSample:
    sample_id: str
    pair_id: str
    split: str
    before_path: str
    after_path: str
    caption: str | None
    transition_label: str
    dominant_transition: str | None
    transition_histogram: list[float] | None
    source: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_asset_path(raw_path: str, project_root: Path | None) -> str:
    path = Path(raw_path)
    if path.is_absolute() or project_root is None:
        return str(path)
    return str((project_root / path).resolve())


def load_retrieval_samples(levir_manifest: Path | None, pair_manifests: list[Path], project_root: Path | None = None) -> list[RetrievalSample]:
    samples: list[RetrievalSample] = []
    if levir_manifest is not None:
        for row in _read_jsonl(levir_manifest):
            samples.append(
                RetrievalSample(
                    sample_id=str(row["sample_id"]),
                    pair_id=str(row.get("pair_id") or row["sample_id"]),
                    split=str(row.get("split") or "unknown"),
                    before_path=_resolve_asset_path(str(row["before_path"]), project_root),
                    after_path=_resolve_asset_path(str(row["after_path"]), project_root),
                    caption=str(row.get("caption") or ""),
                    transition_label=str(row.get("metadata", {}).get("transition_label") or row.get("sample_id")),
                    dominant_transition=str(row.get("metadata", {}).get("transition_label") or row.get("sample_id")),
                    transition_histogram=None,
                    source="levir",
                )
            )
    for manifest in pair_manifests:
        for row in _read_jsonl(manifest):
            samples.append(
                RetrievalSample(
                    sample_id=str(row["sample_id"]),
                    pair_id=str(row.get("pair_id") or row["sample_id"]),
                    split=str(row.get("split") or "unknown"),
                    before_path=_resolve_asset_path(str(row["before_path"]), project_root),
                    after_path=_resolve_asset_path(str(row["after_path"]), project_root),
                    caption=None,
                    transition_label=str(row.get("dominant_transition") or row.get("sample_id")),
                    dominant_transition=str(row.get("dominant_transition") or row.get("sample_id")),
                    transition_histogram=[float(value) for value in row.get("transition_histogram", [])] or None,
                    source=str(row.get("dataset_name", "pair")),
                )
            )
    return samples


class PairRetrievalDataset(Dataset):
    def __init__(self, samples: list[RetrievalSample], image_size: int):
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "sample_id": sample.sample_id,
            "pair_id": sample.pair_id,
            "split": sample.split,
            "before": self._load_rgb(sample.before_path),
            "after": self._load_rgb(sample.after_path),
            "caption": sample.caption,
            "transition_label": sample.transition_label,
            "dominant_transition": sample.dominant_transition,
            "transition_histogram": sample.transition_histogram,
            "source": sample.source,
        }


def sample_positive_group_key(sample: RetrievalSample) -> tuple[str, str] | None:
    if sample.caption:
        return ("caption_pair", sample.pair_id)
    if sample.transition_histogram and sample.dominant_transition:
        return ("transition", sample.dominant_transition)
    return None


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    histograms = [row["transition_histogram"] for row in batch]
    return {
        "sample_id": [row["sample_id"] for row in batch],
        "pair_id": [row["pair_id"] for row in batch],
        "split": [row["split"] for row in batch],
        "before": torch.stack([row["before"] for row in batch]),
        "after": torch.stack([row["after"] for row in batch]),
        "caption": [row["caption"] for row in batch],
        "transition_label": [row["transition_label"] for row in batch],
        "dominant_transition": [row["dominant_transition"] for row in batch],
        "transition_histogram": histograms,
        "source": [row["source"] for row in batch],
    }


class PositiveAwareBatchSampler(BatchSampler):
    def __init__(self, samples: list[RetrievalSample], batch_size: int, shuffle: bool, seed: int):
        self.samples = samples
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed if not self.shuffle else random.randint(0, 10**9))
        order = list(range(len(self.samples)))
        if self.shuffle:
            rng.shuffle(order)
        group_to_indices: dict[tuple[str, str], list[int]] = {}
        sample_to_group: dict[int, tuple[str, str]] = {}
        for index, sample in enumerate(self.samples):
            group_key = sample_positive_group_key(sample)
            if group_key is None:
                continue
            sample_to_group[index] = group_key
            group_to_indices.setdefault(group_key, []).append(index)
        if self.shuffle:
            for indices in group_to_indices.values():
                rng.shuffle(indices)

        unused = set(order)
        cursor = 0
        while unused:
            batch: list[int] = []

            def take_next_unused() -> int | None:
                nonlocal cursor
                while cursor < len(order):
                    candidate = order[cursor]
                    cursor += 1
                    if candidate in unused:
                        unused.remove(candidate)
                        return candidate
                return None

            anchor = take_next_unused()
            if anchor is None:
                break
            batch.append(anchor)

            anchor_group = sample_to_group.get(anchor)
            if anchor_group is not None:
                for candidate in group_to_indices.get(anchor_group, []):
                    if candidate in unused:
                        unused.remove(candidate)
                        batch.append(candidate)
                        break

            while len(batch) < self.batch_size and unused:
                candidate = take_next_unused()
                if candidate is None:
                    break
                batch.append(candidate)
                if len(batch) >= self.batch_size:
                    break
                candidate_group = sample_to_group.get(candidate)
                if candidate_group is None:
                    continue
                for partner in group_to_indices.get(candidate_group, []):
                    if partner in unused:
                        unused.remove(partner)
                        batch.append(partner)
                        break

            yield batch[: self.batch_size]

    def __len__(self) -> int:
        return max(1, (len(self.samples) + self.batch_size - 1) // self.batch_size)


class CaptionPairBatchSampler(BatchSampler):
    def __init__(self, samples: list[RetrievalSample], batch_size: int, shuffle: bool, seed: int):
        self.samples = samples
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        grouped: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            grouped.setdefault(sample.pair_id, []).append(index)
        self._pair_groups = grouped
        self._pair_ids = list(grouped)

    def __iter__(self):
        pair_ids = list(self._pair_ids)
        rng = random.Random(self.seed if not self.shuffle else random.randint(0, 10**9))
        if self.shuffle:
            rng.shuffle(pair_ids)
        for start in range(0, len(pair_ids), self.batch_size):
            batch_pair_ids = pair_ids[start : start + self.batch_size]
            batch: list[int] = []
            for pair_id in batch_pair_ids:
                indices = list(self._pair_groups[pair_id])
                if self.shuffle:
                    rng.shuffle(indices)
                batch.extend(indices)
            yield batch

    def __len__(self) -> int:
        return max(1, (len(self._pair_ids) + self.batch_size - 1) // self.batch_size)


class AlternatingTaskLoader:
    def __init__(self, loaders: list[DataLoader]):
        self.loaders = loaders

    def __iter__(self):
        iterators = [iter(loader) for loader in self.loaders]
        active = [True] * len(iterators)
        while any(active):
            for index, iterator in enumerate(iterators):
                if not active[index]:
                    continue
                try:
                    yield next(iterator)
                except StopIteration:
                    active[index] = False

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders)


class CachedLevirFeatures:
    def __init__(self, pair_index: Path | None, text_index: Path | None):
        self.pair_reader = IndexedShardReader(pair_index) if pair_index is not None else None
        self.text_reader = IndexedShardReader(text_index) if text_index is not None else None

    @property
    def enabled(self) -> bool:
        return self.pair_reader is not None and self.text_reader is not None

    def pair_tokens(self, pair_ids: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.pair_reader is not None
        before_rows = []
        after_rows = []
        pair_index = self.pair_reader.index["pairs"]
        for pair_id in pair_ids:
            payload = pair_index[pair_id]
            before_rows.append(self.pair_reader.get(payload["shard"], payload["before_key"]))
            after_rows.append(self.pair_reader.get(payload["shard"], payload["after_key"]))
        return torch.stack(before_rows).to(device=device, dtype=torch.float32), torch.stack(after_rows).to(
            device=device, dtype=torch.float32
        )

    def text_features(self, sample_ids: list[str], device: torch.device) -> torch.Tensor:
        assert self.text_reader is not None
        caption_index = self.text_reader.index["captions"]
        rows = []
        for sample_id in sample_ids:
            payload = caption_index[sample_id]
            rows.append(self.text_reader.get(payload["shard"], payload["key"]))
        return torch.stack(rows).to(device=device, dtype=torch.float32)


def _build_subset_dataloader(samples: list[RetrievalSample], args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = PairRetrievalDataset(samples, args.image_size)
    batch_size = min(args.batch_size, len(samples))
    seed = int(getattr(args, "seed", 7))
    if samples and all(sample.caption for sample in samples):
        batch_sampler = CaptionPairBatchSampler(samples, batch_size=batch_size, shuffle=shuffle, seed=seed)
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, collate_fn=collate_batch)
    if shuffle:
        batch_sampler = PositiveAwareBatchSampler(samples, batch_size=batch_size, shuffle=True, seed=seed)
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, collate_fn=collate_batch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)


def build_dataloader(samples: list[RetrievalSample], args: argparse.Namespace, shuffle: bool) -> DataLoader | AlternatingTaskLoader:
    if args.max_train_samples is not None:
        kept_pair_ids: list[str] = []
        kept = set()
        for sample in samples:
            if sample.pair_id in kept:
                continue
            kept.add(sample.pair_id)
            kept_pair_ids.append(sample.pair_id)
            if len(kept_pair_ids) >= args.max_train_samples:
                break
        selected_pair_ids = set(kept_pair_ids)
        samples = [sample for sample in samples if sample.pair_id in selected_pair_ids]
    caption_samples = [sample for sample in samples if sample.caption]
    transition_samples = [sample for sample in samples if sample.transition_histogram]
    loaders: list[DataLoader] = []
    if caption_samples:
        loaders.append(_build_subset_dataloader(caption_samples, args, shuffle=shuffle))
    if transition_samples:
        loaders.append(_build_subset_dataloader(transition_samples, args, shuffle=shuffle))
    if not loaders:
        raise ValueError("No caption or transition samples available for dataloader construction.")
    if len(loaders) == 1:
        return loaders[0]
    return AlternatingTaskLoader(loaders)


def choose_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def json_safe_config(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, list):
            payload[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            payload[key] = value
    return payload


def apply_preset(args: argparse.Namespace) -> tuple[argparse.Namespace, dict[str, Any] | None]:
    if not args.preset:
        return args, None
    preset = preset_by_name(args.preset)
    if not preset.supported:
        raise SystemExit(
            f"Preset '{preset.name}' is registered but not implemented yet. Notes: {', '.join(preset.notes)}"
        )
    args.visual_backbone = preset.visual_backbone
    args.text_backbone = preset.text_backbone
    args.pair_feature_mode = preset.pair_feature_mode
    return args, preset.to_dict()


def transition_histograms_to_tensor(histograms: list[list[float] | None], device: torch.device) -> tuple[torch.Tensor | None, list[int]]:
    valid_indices = [index for index, row in enumerate(histograms) if row]
    if not valid_indices:
        return None, []
    valid_rows = [histograms[index] for index in valid_indices]
    tensor = torch.tensor(valid_rows, dtype=torch.float32, device=device)
    tensor = tensor / tensor.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return tensor, valid_indices


def build_positive_mask(group_ids: list[str], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[left == right for right in group_ids] for left in group_ids],
        dtype=torch.bool,
        device=device,
    )


def any_relevant_in_top_k(
    ranking: list[int],
    *,
    k: int,
    is_relevant: Any,
) -> bool:
    return any(is_relevant(candidate) for candidate in ranking[:k])


def _average_precision(relevances: list[bool]) -> float:
    hits = 0
    precision_sum = 0.0
    for rank_index, relevant in enumerate(relevances, start=1):
        if relevant:
            hits += 1
            precision_sum += hits / rank_index
    return precision_sum / hits if hits else 0.0


def _ndcg_at_k(grades: list[float], k: int) -> float:
    dcg = sum(((2.0**grade) - 1.0) / log2(rank + 2.0) for rank, grade in enumerate(grades[:k]))
    ideal = sorted(grades, reverse=True)
    idcg = sum(((2.0**grade) - 1.0) / log2(rank + 2.0) for rank, grade in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def validate_pair_id_split_integrity(samples: list[RetrievalSample]) -> None:
    split_to_pairs: dict[str, set[str]] = {}
    for sample in samples:
        split_to_pairs.setdefault(sample.split, set()).add(sample.pair_id)
    known_splits = {split: pair_ids for split, pair_ids in split_to_pairs.items() if split != "unknown"}
    if not known_splits:
        return
    seen: set[str] = set()
    leaked: set[str] = set()
    for pair_ids in known_splits.values():
        leaked.update(seen.intersection(pair_ids))
        seen.update(pair_ids)
    if leaked:
        leaked_preview = ", ".join(sorted(leaked)[:10])
        raise ValueError(f"Detected train/val/test pair_id leakage: {leaked_preview}")


def validate_pair_id_disjoint_sets(train_samples: list[RetrievalSample], eval_samples: list[RetrievalSample]) -> None:
    train_pair_ids = {sample.pair_id for sample in train_samples}
    eval_pair_ids = {sample.pair_id for sample in eval_samples}
    leaked = sorted(train_pair_ids.intersection(eval_pair_ids))
    if leaked:
        leaked_preview = ", ".join(leaked[:10])
        raise ValueError(f"Detected pair_id overlap between train and eval manifests: {leaked_preview}")


def unique_pair_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    pair_to_index: dict[str, int] = {}
    unique_before: list[torch.Tensor] = []
    unique_after: list[torch.Tensor] = []
    unique_pair_ids: list[str] = []
    caption_to_pair: list[int] = []
    for row_index, pair_id in enumerate(batch["pair_id"]):
        if pair_id not in pair_to_index:
            pair_to_index[pair_id] = len(unique_pair_ids)
            unique_pair_ids.append(pair_id)
            unique_before.append(batch["before"][row_index])
            unique_after.append(batch["after"][row_index])
        caption_to_pair.append(pair_to_index[pair_id])
    before = torch.stack(unique_before).to(device)
    after = torch.stack(unique_after).to(device)
    mapping = torch.tensor(caption_to_pair, dtype=torch.long, device=device)
    return before, after, unique_pair_ids, mapping


def compute_retrieval_metrics(
    embeddings: torch.Tensor,
    sample_ids: list[str],
    dominant_transitions: list[str | None],
    histograms: list[list[float] | None],
    text_query_embeddings: torch.Tensor | None,
    text_query_sample_ids: list[str],
    text_query_transition_labels: list[str] | None = None,
    reversed_embeddings: torch.Tensor | None = None,
) -> dict[str, float]:
    text_recalls = {1: 0, 5: 0, 10: 0}
    text_map_scores: list[float] = []
    text_ranks: list[int] = []
    reversed_pair_hits = 0
    reversed_pair_total = 0
    pair_to_text_recalls = {1: 0, 5: 0, 10: 0}
    pair_to_text_ranks: list[int] = []
    if text_query_embeddings is not None and text_query_sample_ids:
        text_similarities = text_query_embeddings @ embeddings.transpose(0, 1)
        reversed_similarities = text_query_embeddings @ reversed_embeddings.transpose(0, 1) if reversed_embeddings is not None else None
        for query_index, query_sample_id in enumerate(text_query_sample_ids):
            ranking = torch.argsort(text_similarities[query_index], descending=True).tolist()
            relevances = [sample_ids[candidate] == query_sample_id for candidate in ranking]
            for k in text_recalls:
                text_recalls[k] += 1 if any(relevances[:k]) else 0
            first_rank = next((rank for rank, relevant in enumerate(relevances, start=1) if relevant), len(ranking) + 1)
            text_ranks.append(first_rank)
            text_map_scores.append(_average_precision(relevances))
            if reversed_similarities is not None:
                original_score = max(
                    float(text_similarities[query_index, candidate].item())
                    for candidate, pair_id in enumerate(sample_ids)
                    if pair_id == query_sample_id
                )
                reversed_score = max(
                    float(reversed_similarities[query_index, candidate].item())
                    for candidate, pair_id in enumerate(sample_ids)
                    if pair_id == query_sample_id
                )
                reversed_pair_total += 1
                if original_score > reversed_score:
                    reversed_pair_hits += 1

        pair_to_text_scores = embeddings @ text_query_embeddings.transpose(0, 1)
        pair_to_caption_ids: dict[str, list[int]] = {}
        for caption_index, pair_id in enumerate(text_query_sample_ids):
            pair_to_caption_ids.setdefault(pair_id, []).append(caption_index)
        for pair_index, pair_id in enumerate(sample_ids):
            positive_caption_indices = pair_to_caption_ids.get(pair_id, [])
            if not positive_caption_indices:
                continue
            ranking = torch.argsort(pair_to_text_scores[pair_index], descending=True).tolist()
            relevances = [candidate in positive_caption_indices for candidate in ranking]
            for k in pair_to_text_recalls:
                pair_to_text_recalls[k] += 1 if any(relevances[:k]) else 0
            first_rank = next((rank for rank, relevant in enumerate(relevances, start=1) if relevant), len(ranking) + 1)
            pair_to_text_ranks.append(first_rank)

    pair_similarities = embeddings @ embeddings.transpose(0, 1)
    hist_sims: list[float] = []
    pair_only_indices = [index for index, histogram in enumerate(histograms) if histogram]
    pair_query_indices = [
        index
        for index in pair_only_indices
        if any(
            candidate != index and histograms[candidate] and dominant_transitions[candidate] == dominant_transitions[index]
            for candidate in pair_only_indices
        )
    ]
    pair_recalls = {1: 0, 5: 0, 10: 0}
    pair_mrr = 0.0
    pair_top1_transition_hits = 0
    pair_ndcg_scores: list[float] = []
    pair_ap_scores: list[float] = []

    for index in pair_query_indices:
        ranking = torch.argsort(pair_similarities[index], descending=True).tolist()
        ranking = [candidate for candidate in ranking if candidate != index]
        binary_relevances = [
            bool(histograms[candidate] and dominant_transitions[candidate] == dominant_transitions[index]) for candidate in ranking
        ]
        graded_relevances = [
            transition_similarity(np.asarray(histograms[index], dtype=np.float64), np.asarray(histograms[candidate], dtype=np.float64))
            if histograms[candidate]
            else 0.0
            for candidate in ranking
        ]
        for k in pair_recalls:
            pair_recalls[k] += 1 if any(binary_relevances[:k]) else 0
        first_relevant_rank = next((rank for rank, relevant in enumerate(binary_relevances, start=1) if relevant), None)
        if first_relevant_rank is not None:
            pair_mrr += 1.0 / first_relevant_rank
        pair_ap_scores.append(_average_precision(binary_relevances))
        pair_ndcg_scores.append(_ndcg_at_k(graded_relevances, 10))
        top_k = [candidate for candidate in ranking[:5] if histograms[candidate]]
        if top_k:
            hist_sims.append(
                float(
                    np.mean(
                        [
                            transition_similarity(
                                np.asarray(histograms[index], dtype=np.float64),
                                np.asarray(histograms[candidate], dtype=np.float64),
                            )
                            for candidate in top_k
                        ]
                    )
                )
            )
        if ranking and histograms[ranking[0]] and dominant_transitions[ranking[0]] == dominant_transitions[index]:
            pair_top1_transition_hits += 1

    text_total = max(len(text_query_sample_ids), 1)
    pair_total = max(len(pair_query_indices), 1)
    pair_to_text_total = max(len(pair_to_text_ranks), 1)
    text_mrr = sum(1.0 / rank for rank in text_ranks) / len(text_ranks) if text_ranks else 0.0
    pair_to_text_mrr = sum(1.0 / rank for rank in pair_to_text_ranks) / len(pair_to_text_ranks) if pair_to_text_ranks else 0.0
    median_rank = float(np.median(text_ranks)) if text_ranks else 0.0
    metrics = {
        "recall@1": text_recalls[1] / text_total if text_query_sample_ids else 0.0,
        "recall@5": text_recalls[5] / text_total if text_query_sample_ids else 0.0,
        "recall@10": text_recalls[10] / text_total if text_query_sample_ids else 0.0,
        "mAP": sum(text_map_scores) / max(len(text_map_scores), 1),
        "median_rank": median_rank,
        "pair_to_text_recall@1": pair_to_text_recalls[1] / pair_to_text_total if pair_to_text_ranks else 0.0,
        "pair_to_text_recall@5": pair_to_text_recalls[5] / pair_to_text_total if pair_to_text_ranks else 0.0,
        "pair_to_text_recall@10": pair_to_text_recalls[10] / pair_to_text_total if pair_to_text_ranks else 0.0,
        "pair_to_text_MRR": pair_to_text_mrr,
        "MRR": text_mrr,
    }
    if reversed_pair_total:
        metrics["reversed_pair_sanity_accuracy"] = reversed_pair_hits / reversed_pair_total
    if pair_query_indices:
        metrics.update(
            {
                "mean_transition_similarity_top5": sum(hist_sims) / max(len(hist_sims), 1) if hist_sims else 0.0,
                "transition_recall@1": pair_recalls[1] / pair_total,
                "transition_recall@5": pair_recalls[5] / pair_total,
                "transition_recall@10": pair_recalls[10] / pair_total,
                "transition_MRR": pair_mrr / pair_total,
                "transition_top1_hit_rate": pair_top1_transition_hits / pair_total,
                "transition_nDCG@10": sum(pair_ndcg_scores) / max(len(pair_ndcg_scores), 1),
                "pair_mAP": sum(pair_ap_scores) / max(len(pair_ap_scores), 1),
                "pair_sample_fraction": len(pair_only_indices) / max(len(sample_ids), 1),
            }
        )
    return metrics


def run_epoch(
    model: DINOChangeRetriever,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    cached_features: CachedLevirFeatures | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows: list[dict[str, float]] = []
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[str] = []
    all_dominant_transitions: list[str | None] = []
    all_histograms: list[list[float]] = []
    all_sources: list[str] = []
    all_text_embeddings: list[torch.Tensor] = []
    all_text_query_ids: list[str] = []
    all_reversed_embeddings: list[torch.Tensor] = []
    anchor_positive_ratios: list[float] = []

    if training:
        optimizer.zero_grad(set_to_none=True)

    total_steps = len(loader)
    amp_enabled = bool(getattr(args, "mixed_precision", False) and device.type == "cuda")

    for step_index, batch in enumerate(loader, start=1):
        before = batch["before"].to(device)
        after = batch["after"].to(device)
        captions = [caption if isinstance(caption, str) and caption else "" for caption in batch["caption"]]
        label_list = [str(label) for label in batch["transition_label"]]
        has_text = any(caption.strip() for caption in captions)

        grad_context = nullcontext() if training else torch.no_grad()
        with grad_context:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                pair_before = before
                pair_after = after
                unique_pair_ids = list(batch["pair_id"])
                if has_text:
                    pair_before, pair_after, unique_pair_ids, caption_to_pair = unique_pair_batch(batch, device)
                else:
                    caption_to_pair = None
                if has_text and cached_features is not None and cached_features.enabled:
                    cached_before, cached_after = cached_features.pair_tokens(unique_pair_ids, device)
                    cached_text = cached_features.text_features(batch["sample_id"], device)
                    outputs = model.forward_from_visual_tokens(cached_before, cached_after, text_features=cached_text)
                else:
                    outputs = model(pair_before, pair_after, captions if has_text else None)
                loss = outputs["change_embedding"].sum() * 0.0
                metrics_row: dict[str, float] = {}

                if has_text:
                    assert caption_to_pair is not None
                    text_loss, text_stats = asymmetric_caption_pair_loss(
                        outputs["change_embedding"],
                        outputs["text_embedding"],
                        caption_to_pair,
                    )
                    loss = loss + text_loss
                    metrics_row.update(text_stats)
                    metrics_row["temperature"] = 0.07
                    captions_per_pair: dict[str, int] = {}
                    for pair_id in batch["pair_id"]:
                        captions_per_pair[pair_id] = captions_per_pair.get(pair_id, 0) + 1
                    distribution = list(captions_per_pair.values())
                    metrics_row["unique_pairs_in_batch"] = float(len(unique_pair_ids))
                    metrics_row["caption_rows_in_batch"] = float(len(captions))
                    metrics_row["captions_per_pair_mean"] = float(sum(distribution) / max(len(distribution), 1))
                    metrics_row["captions_per_pair_max"] = float(max(distribution, default=0))
                    anchor_positive_ratios.append(float(text_stats["anchors_with_positive_ratio"]))
                    all_text_embeddings.append(outputs["text_embedding"].detach().cpu())
                    all_text_query_ids.extend(batch["pair_id"])

                hist_tensor, hist_indices = transition_histograms_to_tensor(batch["transition_histogram"], device)
                if hist_tensor is not None:
                    pair_embeddings = outputs["change_embedding"][hist_indices]
                    pair_labels = [label_list[index] for index in hist_indices]
                    pair_positive_mask = build_positive_mask(pair_labels, device)
                    anchor_positive_ratios.append(float((pair_positive_mask.sum(dim=1) > 1).float().mean().item()))
                    if args.pair_loss == "supervised":
                        pair_loss = supervised_contrastive_loss(pair_embeddings, pair_labels)
                    else:
                        pair_loss = soft_histogram_contrastive_loss(pair_embeddings, hist_tensor)
                    loss = loss + args.lambda_pair * pair_loss
                    metrics_row["pair_loss"] = float(pair_loss.item())

            with torch.no_grad():
                if has_text and cached_features is not None and cached_features.enabled:
                    cached_before, cached_after = cached_features.pair_tokens(unique_pair_ids, device)
                    reversed_outputs = model.forward_from_visual_tokens(cached_after, cached_before, None)
                else:
                    reversed_outputs = model(pair_after, pair_before, None) if has_text else model(after, before, None)

        if optimizer is not None:
            scaled_loss = loss / max(getattr(args, "grad_accum_steps", 1), 1)
            if scaler is not None and amp_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            should_step = step_index % max(getattr(args, "grad_accum_steps", 1), 1) == 0 or step_index == total_steps
            if should_step:
                if scaler is not None and amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        metrics_row["loss"] = float(loss.item())
        rows.append(metrics_row)
        all_embeddings.append(outputs["change_embedding"].detach().cpu())
        all_reversed_embeddings.append(reversed_outputs["change_embedding"].detach().cpu())
        if has_text:
            pair_first_index: dict[str, int] = {}
            for row_index, pair_id in enumerate(batch["pair_id"]):
                pair_first_index.setdefault(pair_id, row_index)
            for pair_id in unique_pair_ids:
                row_index = pair_first_index[pair_id]
                all_labels.append(pair_id)
                all_dominant_transitions.append(batch["dominant_transition"][row_index])
                all_histograms.append(batch["transition_histogram"][row_index])
                all_sources.append(batch["source"][row_index])
        else:
            all_labels.extend(batch["pair_id"])
            all_dominant_transitions.extend(batch["dominant_transition"])
            all_histograms.extend(batch["transition_histogram"])
            all_sources.extend(batch["source"])

    embeddings = torch.cat(all_embeddings, dim=0)
    reversed_embeddings = torch.cat(all_reversed_embeddings, dim=0)
    text_embeddings = torch.cat(all_text_embeddings, dim=0) if all_text_embeddings else None
    mean_keys = {
        "loss",
        "text_to_pair_loss",
        "pair_to_text_loss",
        "pair_loss",
        "unique_pairs_in_batch",
        "caption_rows_in_batch",
        "captions_per_pair_mean",
        "captions_per_pair_max",
        "temperature",
    }
    mean_row = {key: sum(row.get(key, 0.0) for row in rows) / max(len(rows), 1) for key in mean_keys}
    mean_row.update(
        compute_retrieval_metrics(
            embeddings,
            all_labels,
            all_dominant_transitions,
            all_histograms,
            text_embeddings,
            all_text_query_ids,
            None,
            reversed_embeddings,
        )
    )
    mean_row["anchors_with_positive_ratio"] = (
        sum(anchor_positive_ratios) / len(anchor_positive_ratios) if anchor_positive_ratios else 0.0
    )
    if training and anchor_positive_ratios and mean_row["anchors_with_positive_ratio"] < 0.5:
        print(
            f"Warning: anchors_with_positive_ratio is low ({mean_row['anchors_with_positive_ratio']:.3f}); "
            "consider larger or more source-aware batches."
        )
    return mean_row


def main() -> int:
    args = parse_args()
    args, preset_payload = apply_preset(args)
    set_seed(args.seed)
    train_samples = load_retrieval_samples(args.levir_manifest, list(args.pair_manifest), args.project_root)
    if not train_samples:
        raise SystemExit("No retrieval samples were loaded.")
    validate_pair_id_split_integrity(train_samples)
    eval_samples = load_retrieval_samples(args.eval_levir_manifest, [], args.project_root) if args.eval_levir_manifest else list(train_samples)
    if not eval_samples:
        raise SystemExit("No evaluation samples were loaded.")
    validate_pair_id_split_integrity(eval_samples)
    if args.eval_levir_manifest is not None:
        validate_pair_id_disjoint_sets(train_samples, eval_samples)
    device = choose_device(args.device)
    model = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
                visual_backbone=args.visual_backbone,
                text_backbone=args.text_backbone,
                pair_feature_mode=args.pair_feature_mode,
                dinov2_model_path=str(args.dinov2_model_path) if args.dinov2_model_path else None,
                remoteclip_arch=args.remoteclip_arch,
                remoteclip_checkpoint=str(args.remoteclip_checkpoint) if args.remoteclip_checkpoint else None,
                hf_remoteclip_model_path=str(args.hf_remoteclip_model_path) if args.hf_remoteclip_model_path else None,
                openclip_model_name=args.openclip_model_name,
                openclip_pretrained=args.openclip_pretrained,
                local_files_only=args.local_files_only,
                image_size=args.image_size,
            )
    ).to(device)
    train_loader = build_dataloader(train_samples, args, shuffle=True)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.max_train_samples = args.max_eval_samples
    eval_loader = build_dataloader(eval_samples, eval_args, shuffle=False)
    cached_features = CachedLevirFeatures(args.levir_pair_cache_index, args.levir_text_cache_index)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.mixed_precision and device.type == "cuda"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_metric = -1.0
    config_payload = json_safe_config(args)
    run_metadata = {
        "preset": preset_payload,
        "git_commit": safe_git_commit(Path(__file__).resolve().parents[1]),
        "storage_inventory": locate_storage_inventory(args.project_root),
        "model_fingerprints": {
            "dinov2_model_path": path_fingerprint(args.dinov2_model_path),
            "remoteclip_checkpoint": path_fingerprint(args.remoteclip_checkpoint),
            "hf_remoteclip_model_path": path_fingerprint(args.hf_remoteclip_model_path),
            "levir_pair_cache_index": path_fingerprint(args.levir_pair_cache_index),
            "levir_text_cache_index": path_fingerprint(args.levir_text_cache_index),
        },
        "dataset_versions": {
            "levir_manifest": jsonl_fingerprint(args.levir_manifest),
            "eval_levir_manifest": jsonl_fingerprint(args.eval_levir_manifest),
            "pair_manifests": [jsonl_fingerprint(path) for path in args.pair_manifest],
        },
        "exact_split": {
            "levir_manifest_path": str(args.levir_manifest) if args.levir_manifest else None,
            "eval_levir_manifest_path": str(args.eval_levir_manifest) if args.eval_levir_manifest else None,
            "pair_manifest_paths": [str(path) for path in args.pair_manifest],
            "train_num_samples": len(train_samples),
            "train_num_unique_pairs": len({sample.pair_id for sample in train_samples}),
            "eval_num_samples": len(eval_samples),
            "eval_num_unique_pairs": len({sample.pair_id for sample in eval_samples}),
            "train_split_pair_counts": {
                split: len({sample.pair_id for sample in train_samples if sample.split == split})
                for split in sorted({sample.split for sample in train_samples})
            },
            "eval_split_pair_counts": {
                split: len({sample.pair_id for sample in eval_samples if sample.split == split})
                for split in sorted({sample.split for sample in eval_samples})
            },
        },
    }
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, args, device, scaler=scaler, cached_features=cached_features)
        eval_metrics = run_epoch(model, eval_loader, None, eval_args, device, cached_features=cached_features)
        row = {"epoch": epoch, "train": train_metrics, "eval": eval_metrics}
        history.append(row)
        score = float(eval_metrics.get("recall@5", 0.0) + eval_metrics.get("mAP", 0.0))
        torch.save({"model_state": model.state_dict(), "config": config_payload, "run_metadata": run_metadata, "metrics": row}, args.output_dir / "last.pt")
        if score > best_metric:
            best_metric = score
            torch.save({"model_state": model.state_dict(), "config": config_payload, "run_metadata": run_metadata, "metrics": row}, args.output_dir / "best.pt")
        print(json.dumps(row, indent=2))
    (args.output_dir / "metrics_history.json").write_text(
        json.dumps({"history": history, "config": config_payload, "run_metadata": run_metadata}, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
