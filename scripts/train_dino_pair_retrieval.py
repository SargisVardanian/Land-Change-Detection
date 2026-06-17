from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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
    parser.add_argument("--dinov2-model-path", type=Path, default=None)
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
            "transition_histogram": sample.transition_histogram,
            "source": sample.source,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    histograms = [row["transition_histogram"] for row in batch]
    return {
        "sample_id": [row["sample_id"] for row in batch],
        "before": torch.stack([row["before"] for row in batch]),
        "after": torch.stack([row["after"] for row in batch]),
        "caption": [row["caption"] for row in batch],
        "transition_label": [row["transition_label"] for row in batch],
        "transition_histogram": histograms,
        "source": [row["source"] for row in batch],
    }


def build_dataloader(samples: list[RetrievalSample], args: argparse.Namespace, shuffle: bool) -> DataLoader:
    if args.max_train_samples is not None:
        samples = samples[: args.max_train_samples]
    return DataLoader(
        PairRetrievalDataset(samples, args.image_size),
        batch_size=min(args.batch_size, len(samples)),
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )


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
    all_histograms: list[list[float]] = []
    all_text_embeddings: list[torch.Tensor] = []
    text_batches = 0

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
                text_loss = symmetric_infonce_loss(img_emb, txt_emb)
                loss = loss + text_loss
                metrics_row["text_loss"] = float(text_loss.item())
                all_text_embeddings.append(txt_emb.detach().cpu())
                text_batches += 1

        hist_tensor, hist_indices = transition_histograms_to_tensor(batch["transition_histogram"], device)
        if hist_tensor is not None:
            pair_embeddings = outputs["change_embedding"][hist_indices]
            pair_labels = [label_list[index] for index in hist_indices]
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
        all_labels.extend(label_list)
        all_histograms.extend(batch["transition_histogram"])

    embeddings = torch.cat(all_embeddings, dim=0)
    similarities = embeddings @ embeddings.transpose(0, 1)
    recalls = {1: 0, 5: 0, 10: 0}
    pair_map_scores: list[float] = []
    hist_sims: list[float] = []
    text_mrr = 0.0

    for index in range(similarities.shape[0]):
        ranking = torch.argsort(similarities[index], descending=True).tolist()
        ranking = [candidate for candidate in ranking if candidate != index]
        relevances = [all_labels[candidate] == all_labels[index] for candidate in ranking]
        for k in recalls:
            recalls[k] += 1 if any(relevances[:k]) else 0
        hits = 0
        precision_sum = 0.0
        for rank_index, relevant in enumerate(relevances, start=1):
            if relevant:
                hits += 1
                precision_sum += hits / rank_index
                if hits == 1:
                    text_mrr += 1.0 / rank_index
        pair_map_scores.append(precision_sum / hits if hits else 0.0)
        if all_histograms[index]:
            ref = np.asarray(all_histograms[index], dtype=np.float64)
            top_k = [candidate for candidate in ranking[:5] if all_histograms[candidate]]
            if top_k:
                hist_sims.append(
                    float(np.mean([transition_similarity(ref, np.asarray(all_histograms[candidate], dtype=np.float64)) for candidate in top_k]))
                )

    total = max(len(all_labels), 1)
    mean_row = {key: sum(row.get(key, 0.0) for row in rows) / max(len(rows), 1) for key in {"loss", "text_loss", "pair_loss"}}
    mean_row.update(
        {
            "recall@1": recalls[1] / total,
            "recall@5": recalls[5] / total,
            "recall@10": recalls[10] / total,
            "mAP": sum(pair_map_scores) / max(len(pair_map_scores), 1),
            "mean_transition_similarity_top5": sum(hist_sims) / max(len(hist_sims), 1) if hist_sims else 0.0,
            "MRR": text_mrr / total if text_batches else 0.0,
        }
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
            dinov2_model_path=str(args.dinov2_model_path) if args.dinov2_model_path else None,
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
