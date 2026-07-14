#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the exact canonical QCPR v3 soft mask logits")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--candidate-index", type=int, default=0)
    args = parser.parse_args()
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    logits = artifact["decoded_mask_logits"][args.query_index, args.candidate_index]
    probability = logits.sigmoid().float().numpy()
    heat = np.zeros((*probability.shape, 4), dtype=np.uint8)
    heat[..., 0] = (probability * 255).astype(np.uint8)
    heat[..., 1] = ((1 - probability) * 64).astype(np.uint8)
    heat[..., 3] = (probability * 220).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(heat, "RGBA").save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
