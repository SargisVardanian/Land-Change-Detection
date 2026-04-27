from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_TEXT_MODEL = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
DEFAULT_VLM_MODEL = "mlx-community/Qwen3-VL-4B-Thinking-3bit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_TEXT_MODEL,
        help="HF repo id of an MLX or MLX-VLM model",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("artifacts/models"),
    )
    args = parser.parse_args()

    out_dir = args.local_dir / args.model.replace("/", "__")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=args.model, local_dir=out_dir, local_dir_use_symlinks=False)
    print(path)


if __name__ == "__main__":
    main()
