from __future__ import annotations

import argparse
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
    soft_histogram_contrastive_loss,
    supervised_contrastive_loss,
    symmetric_infonce_loss,
)
from land_change_detection.models.dino_change_retriever import DINOChangeRetriever, DINOChangeRetrieverConfig
from land_change_detection.semantic_transitions import transition_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train retrieval-first DINO/simple-patch change retriever.")
    parser.add_argument("--levir-manifest", type=Path, default=None)
    parser.add_argument("--pair-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-pair", type=float, default=0.2)
    parser.add_argument("--visual-backbone", choices=("simple_patch", "dinov2"), default="simple_patch")
    parser.add_argument("--text-backbone", choices=("simple_text", "remoteclip", "openclip"), default="remoteclip")
    parser.add_argument("--dinov2-model-path", type=Path, default=None)
    parser.add_argument("--remoteclip-model-path", type=Path, default=None)
    parser.add_argument("--openclip-model-name", default="ViT-B-32")
    parser.add_argument("--openclip-pretrained", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--pair-loss", choices=("supervised", "soft"), default="supervised")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
    before_path: str
    after_path: str
    caption: str | None
    transition_label: str
    dominant_transition: str | None
    transition_histogram: list[float] | None
    source: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_retrieval_samples(levir_manifest: Path | None, pair_manifests: list[Path]) -> list[RetrievalSample]:
    samples: list[RetrievalSample] = []
    if levir_manifest is not None:
        for row in _read_jsonl(levir_manifest):
            samples.append(
                RetrievalSample(
                    sample_id=str(row["sample_id"]),
                    before_path=str(row["before_path"]),
                    after_path=str(row["after_path"]),
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
                    before_path=str(row["before_path"]),
                    after_path=str(row["after_path"]),
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
        return ("caption_pair", sample.sample_id)
    if sample.transition_histogram and sample.dominant_transition:
        return ("transition", sample.dominant_transition)
    return None


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    histograms = [row["transition_histogram"] for row in batch]
    return {
        "sample_id": [row["sample_id"] for row in batch],
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


def _build_subset_dataloader(samples: list[RetrievalSample], args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = PairRetrievalDataset(samples, args.image_size)
    batch_size = min(args.batch_size, len(samples))
    if shuffle:
        batch_sampler = PositiveAwareBatchSampler(samples, batch_size=batch_size, shuffle=True, seed=args.seed)
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, collate_fn=collate_batch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)


def build_dataloader(samples: list[RetrievalSample], args: argparse.Namespace, shuffle: bool) -> DataLoader | AlternatingTaskLoader:
    if args.max_train_samples is not None:
        samples = samples[: args.max_train_samples]
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


def compute_retrieval_metrics(
    embeddings: torch.Tensor,
    sample_ids: list[str],
    dominant_transitions: list[str | None],
    histograms: list[list[float] | None],
    text_query_embeddings: torch.Tensor | None,
    text_query_sample_ids: list[str],
) -> dict[str, float]:
    text_recalls = {1: 0, 5: 0, 10: 0}
    text_map_scores: list[float] = []
    text_ranks: list[int] = []
    if text_query_embeddings is not None and text_query_sample_ids:
        text_similarities = text_query_embeddings @ embeddings.transpose(0, 1)
        for query_index, query_sample_id in enumerate(text_query_sample_ids):
            ranking = torch.argsort(text_similarities[query_index], descending=True).tolist()
            relevances = [sample_ids[candidate] == query_sample_id for candidate in ranking]
            for k in text_recalls:
                text_recalls[k] += 1 if any(relevances[:k]) else 0
            first_rank = next((rank for rank, relevant in enumerate(relevances, start=1) if relevant), len(ranking) + 1)
            text_ranks.append(first_rank)
            text_map_scores.append(_average_precision(relevances))

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
    text_mrr = sum(1.0 / rank for rank in text_ranks) / len(text_ranks) if text_ranks else 0.0
    median_rank = float(np.median(text_ranks)) if text_ranks else 0.0
    return {
        "recall@1": text_recalls[1] / text_total if text_query_sample_ids else 0.0,
        "recall@5": text_recalls[5] / text_total if text_query_sample_ids else 0.0,
        "recall@10": text_recalls[10] / text_total if text_query_sample_ids else 0.0,
        "mAP": sum(text_map_scores) / max(len(text_map_scores), 1),
        "median_rank": median_rank,
        "mean_transition_similarity_top5": sum(hist_sims) / max(len(hist_sims), 1) if hist_sims else 0.0,
        "MRR": text_mrr,
        "transition_recall@1": pair_recalls[1] / pair_total if pair_query_indices else 0.0,
        "transition_recall@5": pair_recalls[5] / pair_total if pair_query_indices else 0.0,
        "transition_recall@10": pair_recalls[10] / pair_total if pair_query_indices else 0.0,
        "transition_MRR": pair_mrr / pair_total if pair_query_indices else 0.0,
        "transition_top1_hit_rate": pair_top1_transition_hits / pair_total if pair_query_indices else 0.0,
        "transition_nDCG@10": sum(pair_ndcg_scores) / max(len(pair_ndcg_scores), 1),
        "directionality_accuracy": pair_top1_transition_hits / pair_total if pair_query_indices else 0.0,
        "pair_mAP": sum(pair_ap_scores) / max(len(pair_ap_scores), 1),
        "pair_sample_fraction": len(pair_only_indices) / max(len(sample_ids), 1),
    }


def run_epoch(
    model: DINOChangeRetriever,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
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
    anchor_positive_ratios: list[float] = []

    for batch in loader:
        before = batch["before"].to(device)
        after = batch["after"].to(device)
        captions = [caption if isinstance(caption, str) and caption else "" for caption in batch["caption"]]
        label_list = [str(label) for label in batch["transition_label"]]
        has_text = any(caption.strip() for caption in captions)
        outputs = model(before, after, captions if has_text else None)
        loss = outputs["change_embedding"].sum() * 0.0
        metrics_row: dict[str, float] = {}

        if has_text:
            text_mask = [index for index, caption in enumerate(captions) if caption.strip()]
            if text_mask:
                img_emb = outputs["change_embedding"][text_mask]
                txt_emb = outputs["text_embedding"][text_mask]
                text_group_ids = [batch["sample_id"][index] for index in text_mask]
                positive_mask = build_positive_mask(text_group_ids, device)
                anchor_positive_ratios.append(float((positive_mask.sum(dim=1) > 1).float().mean().item()))
                text_loss = symmetric_infonce_loss(img_emb, txt_emb, positive_mask=positive_mask)
                loss = loss + text_loss
                metrics_row["text_loss"] = float(text_loss.item())
                all_text_embeddings.append(txt_emb.detach().cpu())
                all_text_query_ids.extend(text_group_ids)

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

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        metrics_row["loss"] = float(loss.item())
        rows.append(metrics_row)
        all_embeddings.append(outputs["change_embedding"].detach().cpu())
        all_labels.extend(batch["sample_id"])
        all_dominant_transitions.extend(batch["dominant_transition"])
        all_histograms.extend(batch["transition_histogram"])
        all_sources.extend(batch["source"])

    embeddings = torch.cat(all_embeddings, dim=0)
    text_embeddings = torch.cat(all_text_embeddings, dim=0) if all_text_embeddings else None
    mean_row = {key: sum(row.get(key, 0.0) for row in rows) / max(len(rows), 1) for key in {"loss", "text_loss", "pair_loss"}}
    mean_row.update(
        compute_retrieval_metrics(
            embeddings,
            all_labels,
            all_dominant_transitions,
            all_histograms,
            text_embeddings,
            all_text_query_ids,
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
    set_seed(args.seed)
    samples = load_retrieval_samples(args.levir_manifest, list(args.pair_manifest))
    if not samples:
        raise SystemExit("No retrieval samples were loaded.")
    device = choose_device(args.device)
    model = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
                visual_backbone=args.visual_backbone,
                text_backbone=args.text_backbone,
                dinov2_model_path=str(args.dinov2_model_path) if args.dinov2_model_path else None,
                remoteclip_model_path=str(args.remoteclip_model_path) if args.remoteclip_model_path else None,
                openclip_model_name=args.openclip_model_name,
                openclip_pretrained=args.openclip_pretrained,
                local_files_only=args.local_files_only,
                image_size=args.image_size,
            )
    ).to(device)
    loader = build_dataloader(samples, args, shuffle=True)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_metric = -1.0
    config_payload = json_safe_config(args)
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loader, optimizer, args, device)
        eval_metrics = run_epoch(model, loader, None, args, device)
        row = {"epoch": epoch, "train": train_metrics, "eval": eval_metrics}
        history.append(row)
        score = float(eval_metrics.get("recall@5", 0.0) + eval_metrics.get("mAP", 0.0))
        torch.save({"model_state": model.state_dict(), "config": config_payload, "metrics": row}, args.output_dir / "last.pt")
        if score > best_metric:
            best_metric = score
            torch.save({"model_state": model.state_dict(), "config": config_payload, "metrics": row}, args.output_dir / "best.pt")
        print(json.dumps(row, indent=2))
    (args.output_dir / "metrics_history.json").write_text(json.dumps({"history": history}, indent=2), encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
