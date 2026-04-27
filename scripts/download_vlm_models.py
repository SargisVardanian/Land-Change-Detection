from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "Qwen/Qwen3-VL-4B-Thinking": Path("artifacts/models/vlm/Qwen3-VL-4B-Thinking"),
}


def main() -> None:
    for repo_id, out_dir in MODELS.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} -> {out_dir}")
        snapshot_download(repo_id=repo_id, local_dir=out_dir)
        print(f"Done: {repo_id}")


if __name__ == "__main__":
    main()
