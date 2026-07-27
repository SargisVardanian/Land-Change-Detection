from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from land_change_detection.levir_mci import LevirMciSample, discover_levir_mci_samples
from land_change_detection.models.levir_binary_retrieval import (
    LevirBinaryRetrievalConfig,
    LevirBinaryRetrievalModel,
    dice_loss_from_logits,
    retrieval_metrics,
    segmentation_metrics,
    symmetric_contrastive_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first LEVIR-MCI binary segmentation + text retrieval baseline.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LevirTorchDataset(Dataset):
    def __init__(self, samples: list[LevirMciSample], image_size: int):
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _load_mask(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("L").resize((self.image_size, self.image_size))
        array = (np.asarray(image, dtype=np.float32) > 0).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        return {
            "sample_id": sample.sample_id,
            "before": self._load_rgb(sample.image_before),
            "after": self._load_rgb(sample.image_after),
            "mask": self._load_mask(sample.binary_change_mask),
            "caption": sample.caption or "no change description available",
            "split": sample.split,
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "sample_id": [str(row["sample_id"]) for row in batch],
        "before": torch.stack([row["before"] for row in batch]),  # type: ignore[list-item]
        "after": torch.stack([row["after"] for row in batch]),  # type: ignore[list-item]
        "mask": torch.stack([row["mask"] for row in batch]),  # type: ignore[list-item]
        "caption": [str(row["caption"]) for row in batch],
        "split": [str(row["split"]) for row in batch],
    }


def split_samples(samples: list[LevirMciSample], split_name: str) -> list[LevirMciSample]:
    return [sample for sample in samples if sample.split == split_name]


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    samples = discover_levir_mci_samples(args.data_root)
    train_samples = split_samples(samples, args.train_split)
    eval_samples = split_samples(samples, args.eval_split)
    if not train_samples:
        raise ValueError(f"No training samples found for split={args.train_split}")
    if args.max_train_samples is not None:
        train_samples = train_samples[: args.max_train_samples]
    if not eval_samples:
        eval_samples = train_samples

    train_loader = DataLoader(
        LevirTorchDataset(train_samples, args.image_size),
        batch_size=min(args.batch_size, len(train_samples)),
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    eval_loader = DataLoader(
        LevirTorchDataset(eval_samples, args.image_size),
        batch_size=min(args.batch_size, len(eval_samples)),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    return train_loader, eval_loader


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    seg_metrics: list[dict[str, float]] = []
    all_change_embeddings: list[torch.Tensor] = []
    all_text_embeddings: list[torch.Tensor] = []

    for batch in loader:
        before = batch["before"].to(device)  # type: ignore[assignment]
        after = batch["after"].to(device)  # type: ignore[assignment]
        mask = batch["mask"].to(device)  # type: ignore[assignment]
        captions = batch["caption"]  # type: ignore[assignment]
        outputs = model(before, after, captions)
        mask_loss = nn.BCEWithLogitsLoss()(outputs["mask_logits"], mask)
        dice_loss = dice_loss_from_logits(outputs["mask_logits"], mask)
        contrastive = symmetric_contrastive_loss(outputs["change_embedding"], outputs["text_embedding"])
        loss = mask_loss + dice_loss + 0.2 * contrastive

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))
        seg_metrics.append(segmentation_metrics(outputs["mask_logits"].detach(), mask.detach()))
        all_change_embeddings.append(outputs["change_embedding"].detach().cpu())
        all_text_embeddings.append(outputs["text_embedding"].detach().cpu())

    mean_seg = {
        key: sum(metric[key] for metric in seg_metrics) / max(len(seg_metrics), 1)
        for key in ("dice", "iou", "precision", "recall")
    }
    retrieval = retrieval_metrics(torch.cat(all_change_embeddings, dim=0), torch.cat(all_text_embeddings, dim=0))
    return {"loss": sum(losses) / max(len(losses), 1), **mean_seg, **retrieval}


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, eval_loader = build_dataloaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LevirBinaryRetrievalModel(LevirBinaryRetrievalConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, object]] = []
    best_dice = -1.0
    best_path = args.output_dir / "best.pt"
    latest_path = args.output_dir / "last.pt"
    checkpoint_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            eval_metrics = run_epoch(model, eval_loader, None, device)
        row = {"epoch": epoch, "train": train_metrics, "eval": eval_metrics}
        history.append(row)
        if eval_metrics["dice"] > best_dice:
            best_dice = eval_metrics["dice"]
            torch.save({"model_state": model.state_dict(), "config": checkpoint_config, "best_eval": eval_metrics}, best_path)
        if epoch % args.save_every == 0:
            torch.save({"model_state": model.state_dict(), "config": checkpoint_config, "latest_eval": eval_metrics}, latest_path)
        print(json.dumps(row, indent=2))

    summary = {"best_eval_dice": best_dice, "history": history}
    (args.output_dir / "metrics_history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
