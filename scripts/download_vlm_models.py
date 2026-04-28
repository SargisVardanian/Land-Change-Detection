from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct": Path("artifacts/models/vlm/remote-sensing-Qwen2-VL-2B-Instruct"),
    "AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct": Path("artifacts/models/vlm/remote-sensing-Qwen2.5-VL-3B-Instruct"),
}


def main() -> None:
    for repo_id, out_dir in MODELS.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} -> {out_dir}")
        snapshot_download(repo_id=repo_id, local_dir=out_dir)
        print(f"Done: {repo_id}")


if __name__ == "__main__":
    main()
