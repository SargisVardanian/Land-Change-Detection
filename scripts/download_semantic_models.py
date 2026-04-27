from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "mfaytin/mask2former-satellite": Path("artifacts/models/semantic/mask2former-satellite"),
    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL": Path("artifacts/models/semantic/Prithvi-EO-2.0-300M-TL"),
    "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL": Path("artifacts/models/semantic/Prithvi-EO-2.0-600M-TL"),
}


def main() -> None:
    for repo_id, out_dir in MODELS.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} -> {out_dir}")
        snapshot_download(repo_id=repo_id, local_dir=out_dir)
        print(f"Done: {repo_id}")


if __name__ == "__main__":
    main()
