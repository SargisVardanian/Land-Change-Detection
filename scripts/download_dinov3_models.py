from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "timm/vit_large_patch16_dinov3.sat493m": Path("artifacts/models/features/timm-vit_large_patch16_dinov3.sat493m"),
}


def main() -> None:
    for repo_id, out_dir in MODELS.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} -> {out_dir}")
        snapshot_download(repo_id=repo_id, local_dir=out_dir)
        print(f"Done: {repo_id}")


if __name__ == "__main__":
    main()
