from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "akshaydudhane/EarthDial_4B_RGB"
TARGET_DIR = Path("artifacts/models/vlm/EarthDial_4B_RGB")


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {REPO_ID} to {TARGET_DIR}")


if __name__ == "__main__":
    main()
