from __future__ import annotations

import argparse
from pathlib import Path


TEMPLATE = """# Minimal TerraTorch/Prithvi runtime template
model_name: prithvi-eo-2.0-300m-tl
task: semantic_change
num_input_channels: 6
num_classes: 12
backbone: prithvi_eo_v2
head: semantic_change_unet
checkpoint_hint: artifacts/models/semantic/Prithvi-EO-2.0-300M-TL/model.pt
notes: update this template to match the real TerraTorch task/head contract on YSU-HPC
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a minimal TerraTorch config template for Prithvi experiments.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(TEMPLATE, encoding="utf-8")
    print(f"Wrote template: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
