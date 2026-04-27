from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


HF_BASE = (
    "https://hf.co/datasets/hkristen/oscd/resolve/"
    "4958d786c1389ede1511d91a6ecf1a75c4074933"
)
DOWNLOADS = {
    "images": {
        "filename": "Onera Satellite Change Detection dataset - Images.zip",
        "url": f"{HF_BASE}/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Images.zip",
        "fallback_url": "https://partage.imt.fr/index.php/s/gKRaWgRnLMfwMGo/download",
        "extract_to": "images",
    },
    "train_labels": {
        "filename": "Onera Satellite Change Detection dataset - Train Labels.zip",
        "url": f"{HF_BASE}/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Train%20Labels.zip",
        "fallback_url": "https://partage.mines-telecom.fr/index.php/s/2D6n03k58ygBSpu/download",
        "extract_to": "train_labels",
    },
    "test_labels": {
        "filename": "Onera Satellite Change Detection dataset - Test Labels.zip",
        "url": f"{HF_BASE}/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Test%20Labels.zip",
        "fallback_url": "https://partage.imt.fr/index.php/s/gpStKn4Mpgfnr63/download",
        "extract_to": "test_labels",
    },
}


def download_file(url: str, destination: Path, *, verify_tls: bool = True) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120, verify=verify_tls) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with destination.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))


def extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/oscd"),
        help="Target directory for OSCD",
    )
    parser.add_argument(
        "--component",
        choices=["all", *DOWNLOADS.keys()],
        default="all",
        help="Only download part of the dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract files even if they already exist",
    )
    args = parser.parse_args()

    keys = list(DOWNLOADS) if args.component == "all" else [args.component]
    for key in keys:
        item = DOWNLOADS[key]
        archive_path = args.root / item["filename"]
        extract_dir = args.root / item["extract_to"]

        if args.force and extract_dir.exists():
            shutil.rmtree(extract_dir)
        if args.force and archive_path.exists():
            archive_path.unlink()

        if not archive_path.exists():
            try:
                download_file(item["url"], archive_path)
            except Exception:
                verify_tls = "mines-telecom" not in item["fallback_url"]
                download_file(item["fallback_url"], archive_path, verify_tls=verify_tls)

        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            extract_zip(archive_path, extract_dir)

        print(f"{key}: ready at {extract_dir}")


if __name__ == "__main__":
    main()
