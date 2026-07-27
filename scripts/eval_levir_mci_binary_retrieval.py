from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_levir_mci_binary_retrieval import build_dataloaders, run_epoch
from land_change_detection.models.levir_binary_retrieval import LevirBinaryRetrievalConfig, LevirBinaryRetrievalModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained LEVIR-MCI binary segmentation + retrieval checkpoint.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--max-train-samples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, eval_loader = build_dataloaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = LevirBinaryRetrievalModel(LevirBinaryRetrievalConfig()).to(device)
    model.load_state_dict(checkpoint["model_state"])
    with torch.no_grad():
        metrics = run_epoch(model, eval_loader, None, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
