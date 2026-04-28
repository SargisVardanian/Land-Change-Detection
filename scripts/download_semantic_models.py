from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


RUNTIME_MODELS = {
    "mfaytin/mask2former-satellite": Path("artifacts/models/semantic/mask2former-satellite"),
}

RESEARCH_MODELS = {
    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL": Path("artifacts/models/semantic/Prithvi-EO-2.0-300M-TL"),
    "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL": Path("artifacts/models/semantic/Prithvi-EO-2.0-600M-TL"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download semantic model artifacts.")
    parser.add_argument(
        "--include-research",
        action="store_true",
        help="Also download Prithvi research checkpoints. These are not required for the current Streamlit app.",
    )
    args = parser.parse_args()

    models = dict(RUNTIME_MODELS)
    if args.include_research:
        models.update(RESEARCH_MODELS)

    for repo_id, out_dir in models.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} -> {out_dir}")
        snapshot_download(repo_id=repo_id, local_dir=out_dir)
        print(f"Done: {repo_id}")


if __name__ == "__main__":
    main()
