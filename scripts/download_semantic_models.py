from __future__ import annotations

import argparse
import os
from pathlib import Path


RUNTIME_MODELS = {
    "mfaytin/mask2former-satellite": "mask2former-satellite",
}

RESEARCH_MODELS = {
    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL": "Prithvi-EO-2.0-300M-TL",
    "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL": "Prithvi-EO-2.0-600M-TL",
}


def build_model_targets(output_root: Path, include_research: bool) -> dict[str, Path]:
    models = dict(RUNTIME_MODELS)
    if include_research:
        models.update(RESEARCH_MODELS)
    return {repo_id: output_root / folder_name for repo_id, folder_name in models.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download semantic model artifacts.")
    parser.add_argument(
        "--include-research",
        action="store_true",
        help="Also download Prithvi research checkpoints. These are not required for the current Streamlit app.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("SEMANTIC_MODEL_ROOT", "artifacts/models/semantic")),
        help="Directory where model snapshots should be stored.",
    )
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    for repo_id, out_dir in build_model_targets(args.output_root, args.include_research).items():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} -> {out_dir}")
        snapshot_download(repo_id=repo_id, local_dir=out_dir)
        print(f"Done: {repo_id}")


if __name__ == "__main__":
    main()
