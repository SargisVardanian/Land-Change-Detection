from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local-files-only DINOv2 smoke test.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = AutoModel.from_pretrained(str(args.model_path), local_files_only=True).to(device)
    model.eval()
    dummy = torch.randn(args.batch_size, 3, 224, 224, device=device)
    with torch.no_grad():
        outputs = model(pixel_values=dummy)
    hidden = outputs.last_hidden_state
    cls = hidden[:, :1]
    patches = hidden[:, 1:]
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device name: {torch.cuda.get_device_name(device)}")
    print(f"last_hidden_state shape: {tuple(hidden.shape)}")
    print(f"CLS token shape: {tuple(cls.shape)}")
    print(f"patch token shape: {tuple(patches.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
