#!/usr/bin/env python3
"""Acquire RSRCC physical assets through the official HF Dataset Viewer.

The source metadata are joined to Dataset Viewer rows by split/order and an
exact normalized-text check.  Asset URLs are signed URLs returned by the
official Dataset Viewer API.  This script only acquires and hashes physical
images; generated RSRCC text remains evaluation-only and training-disabled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DATASET = "google/RSRCC"
VIEWER_BASE = "https://datasets-server.huggingface.co"
VIEWER_SPLITS = {"train": "train", "val": "validation", "test": "test"}
DEFAULT_PAGE_LENGTH = 100
USER_AGENT = "qcpr-rsrcc-dataset-viewer-acquisition/1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe RSRCC relative path: {value!r}")
    return path


def read_metadata_rows(root: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for local_split in VIEWER_SPLITS:
        path = root / f"{local_split}_metadata.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = []
            for row in csv.DictReader(handle):
                before = str(row.get("before_file_name") or "").strip()
                after = str(row.get("after_file_name") or "").strip()
                if not before or not after:
                    continue
                rows.append({
                    "before": before,
                    "after": after,
                    "text": str(row.get("text") or ""),
                })
        result[local_split] = rows
    return result


def fetch_json(url: str, *, attempts: int = 7) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 >= attempts:
                break
            retry_after = exc.headers.get("Retry-After")
            try:
                wait = min(120, max(1, int(retry_after))) if retry_after else min(120, 2 ** attempt)
            except ValueError:
                wait = min(120, 2 ** attempt)
            time.sleep(wait)
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Dataset Viewer request failed: {url}: {last_error}") from last_error


def viewer_page(viewer_split: str, offset: int, length: int) -> dict[str, Any]:
    query = urlencode({
        "dataset": DATASET,
        "config": "default",
        "split": viewer_split,
        "offset": offset,
        "length": length,
    })
    return fetch_json(f"{VIEWER_BASE}/rows?{query}")


def _image_value(row: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    value = row.get(side)
    return value if isinstance(value, Mapping) else {}


def _viewer_pages(
    viewer_split: str,
    *,
    page_fetch: Callable[[str, int, int], Mapping[str, Any]],
    page_length: int,
    page_workers: int,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Fetch all viewer pages deterministically, parallelizing known totals."""
    first = page_fetch(viewer_split, 0, page_length)
    first_rows = first.get("rows") if isinstance(first, Mapping) else None
    if not isinstance(first_rows, list) or not first_rows:
        return []
    total_raw = first.get("num_rows_total") if isinstance(first, Mapping) else None
    try:
        total = int(total_raw) if total_raw is not None else None
    except (TypeError, ValueError):
        total = None
    if total is not None and total >= len(first_rows):
        offsets = list(range(0, total, page_length))
        payloads: dict[int, Mapping[str, Any]] = {0: first}
        pending = offsets[1:]
        if pending:
            with ThreadPoolExecutor(max_workers=max(1, page_workers)) as executor:
                futures = {
                    offset: executor.submit(page_fetch, viewer_split, offset, page_length)
                    for offset in pending
                }
                for offset in pending:
                    payloads[offset] = futures[offset].result()
        return [(offset, payloads[offset]) for offset in offsets]
    pages: list[tuple[int, Mapping[str, Any]]] = [(0, first)]
    offset = len(first_rows)
    while len(first_rows) >= page_length:
        payload = page_fetch(viewer_split, offset, page_length)
        rows = payload.get("rows") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list) or not rows:
            break
        pages.append((offset, payload))
        offset += len(rows)
        first_rows = rows
        if len(rows) < page_length:
            break
    return pages


def resolve_viewer_assets(
    metadata: Mapping[str, list[Mapping[str, str]]],
    *,
    page_fetch: Callable[[str, int, int], Mapping[str, Any]] = viewer_page,
    page_length: int = DEFAULT_PAGE_LENGTH,
    page_workers: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assets: dict[tuple[str, str], dict[str, Any]] = {}
    audit: dict[str, Any] = {
        "dataset": DATASET,
        "viewer_base": VIEWER_BASE,
        "page_length": page_length,
        "page_workers": page_workers,
        "pages": {},
        "row_order_mismatches": [],
        "text_mismatches": [],
        "missing_viewer_rows": [],
        "extra_viewer_rows": [],
        "duplicate_asset_url_conflicts": [],
    }
    for local_split, viewer_split in VIEWER_SPLITS.items():
        local_rows = list(metadata[local_split])
        pages = _viewer_pages(
            viewer_split,
            page_fetch=page_fetch,
            page_length=page_length,
            page_workers=page_workers,
        )
        page_count = 0
        offset = 0
        for page_offset, payload in pages:
            rows = payload.get("rows") if isinstance(payload, Mapping) else None
            if not isinstance(rows, list) or not rows:
                continue
            page_count += 1
            for position, wrapper in enumerate(rows):
                if not isinstance(wrapper, Mapping):
                    continue
                viewer_index = int(wrapper.get("row_idx", page_offset + position))
                local_index = page_offset + position
                if viewer_index != local_index:
                    audit["row_order_mismatches"].append({
                        "local_split": local_split,
                        "viewer_split": viewer_split,
                        "expected": local_index,
                        "observed": viewer_index,
                    })
                if local_index >= len(local_rows):
                    audit["extra_viewer_rows"].append({
                        "local_split": local_split,
                        "viewer_split": viewer_split,
                        "viewer_row_idx": viewer_index,
                    })
                    continue
                local_row = local_rows[local_index]
                viewer_row = wrapper.get("row") if isinstance(wrapper.get("row"), Mapping) else {}
                if normalize_text(local_row.get("text")) != normalize_text(viewer_row.get("text")):
                    audit["text_mismatches"].append({
                        "local_split": local_split,
                        "viewer_split": viewer_split,
                        "row_idx": viewer_index,
                    })
                for side in ("before", "after"):
                    image = _image_value(viewer_row, side)
                    relative = str(local_row[side])
                    key = (local_split, relative)
                    entry = {
                        "local_split": local_split,
                        "viewer_split": viewer_split,
                        "relative": relative,
                        "side": side,
                        "viewer_row_idx": viewer_index,
                        "asset_url": str(image.get("src") or ""),
                        "width": image.get("width"),
                        "height": image.get("height"),
                    }
                    previous = assets.get(key)
                    if previous is None:
                        assets[key] = entry
                    elif previous.get("asset_url") != entry.get("asset_url"):
                        audit["duplicate_asset_url_conflicts"].append({
                            "local_split": local_split,
                            "relative": relative,
                            "first_row_idx": previous.get("viewer_row_idx"),
                            "second_row_idx": viewer_index,
                        })
            offset = max(offset, page_offset + len(rows))
        if offset < len(local_rows):
            audit["missing_viewer_rows"].append({
                "local_split": local_split,
                "viewer_split": viewer_split,
                "metadata_rows": len(local_rows),
                "viewer_rows": offset,
            })
        audit["pages"][viewer_split] = {
            "page_count": page_count,
            "metadata_rows": len(local_rows),
            "viewer_rows": offset,
        }
    audit["alignment_ok"] = not any(
        audit[key]
        for key in ("row_order_mismatches", "text_mismatches", "missing_viewer_rows", "extra_viewer_rows")
    )
    return sorted(assets.values(), key=lambda row: (str(row["local_split"]), str(row["relative"]), str(row["side"]))), audit

def _download_one(entry: Mapping[str, Any], repository_root: Path) -> dict[str, Any]:
    local_split = str(entry["local_split"])
    relative = safe_relative(str(entry["relative"]))
    target = repository_root / local_split / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    result = dict(entry)
    result["path"] = str(target)
    if target.is_file() and target.stat().st_size > 0:
        result.update({
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "downloaded": False,
            "acquisition": "preexisting",
        })
        return result
    url = str(entry.get("asset_url") or "")
    if not url:
        result.update({"bytes": None, "sha256": None, "downloaded": False, "error": "MISSING_SIGNED_ASSET_URL"})
        return result
    temporary = target.with_name(target.name + f".viewer.part.{os.getpid()}")
    last_error: Exception | None = None
    for attempt in range(7):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(block)
                    digest.update(block)
                    size += len(block)
            if size <= 0:
                raise RuntimeError("empty asset response")
            os.replace(temporary, target)
            result.update({
                "bytes": size,
                "sha256": digest.hexdigest(),
                "downloaded": True,
                "acquisition": "dataset_viewer_signed_url",
            })
            return result
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 >= 7:
                break
            time.sleep(min(120, 2 ** attempt))
        except (URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 >= 7:
                break
            time.sleep(min(60, 2 ** attempt))
        finally:
            if temporary.exists() and not target.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass
    result.update({
        "bytes": None,
        "sha256": None,
        "downloaded": False,
        "error": f"{type(last_error).__name__}: {last_error}",
    })
    return result


def acquire_assets(entries: list[dict[str, Any]], repository_root: Path, workers: int) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_download_one, entry, repository_root) for entry in entries]
        return sorted((future.result() for future in as_completed(futures)), key=lambda row: (str(row["local_split"]), str(row["relative"]), str(row["side"])))


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")



def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON object in {path}: {value!r}")
                rows.append(value)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--input-report", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--page-workers", type=int, default=1)
    parser.add_argument("--page-length", type=int, default=DEFAULT_PAGE_LENGTH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.page_length <= 100:
        raise SystemExit("--page-length must be between 1 and 100")
    if args.page_workers < 1:
        raise SystemExit("--page-workers must be at least 1")
    if bool(args.input_manifest) != bool(args.input_report):
        raise SystemExit("--input-manifest and --input-report must be provided together")
    input_report = {}
    if args.input_manifest:
        entries = read_jsonl(args.input_manifest)
        input_report = json.loads(args.input_report.read_text(encoding="utf-8"))
        alignment = input_report.get("alignment") if isinstance(input_report, dict) else None
        if not isinstance(alignment, dict) or not alignment.get("alignment_ok"):
            raise SystemExit(json.dumps({"status": "REFUSE_DOWNLOAD_INPUT_ALIGNMENT_FAILURE", "alignment": alignment}, sort_keys=True))
        metadata = {}
    else:
        metadata = read_metadata_rows(args.metadata_root)
        entries, alignment = resolve_viewer_assets(metadata, page_length=args.page_length, page_workers=args.page_workers)
    if args.download and not alignment["alignment_ok"]:
        raise SystemExit(json.dumps({"status": "REFUSE_DOWNLOAD_ALIGNMENT_FAILURE", "alignment": alignment}, sort_keys=True))
    if args.download:
        entries = acquire_assets(entries, args.repository_root, args.workers)
    else:
        resolved_entries = []
        for original in entries:
            entry = dict(original)
            target = args.repository_root / str(entry["local_split"]) / safe_relative(str(entry["relative"]))
            entry["path"] = str(target)
            if target.is_file() and target.stat().st_size > 0:
                entry.update({"bytes": target.stat().st_size, "sha256": sha256_file(target), "downloaded": False, "acquisition": "preexisting"})
            else:
                entry.update({"bytes": None, "sha256": None, "downloaded": False, "error": "MISSING_LOCAL_ASSET"})
            resolved_entries.append(entry)
        entries = resolved_entries
    manifest = args.manifest or args.output.with_name("rsrcc_viewer_asset_manifest.jsonl")
    write_jsonl(manifest, entries)
    missing = [row for row in entries if not row.get("sha256")]
    report = {
        "schema_version": "qcpr-rsrcc-dataset-viewer-acquisition-v1",
        "status": (
            "VIEWER_ASSET_ACQUISITION_COMPLETE"
            if args.download and not missing
            else "VIEWER_ASSET_ACQUISITION_INCOMPLETE"
            if args.download
            else "VIEWER_MANIFEST_RESOLVED_DOWNLOAD_PENDING"
        ),
        "dataset": DATASET,
        "viewer_base": VIEWER_BASE,
        "metadata_root": str(args.metadata_root),
        "repository_root": str(args.repository_root),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "download_mode": bool(args.download),
        "workers": args.workers,
        "page_workers": args.page_workers,
        "input_manifest": str(args.input_manifest) if args.input_manifest else None,
        "input_report": str(args.input_report) if args.input_report else None,
        "metadata_row_counts": (
            {split: len(rows) for split, rows in metadata.items()}
            if metadata
            else input_report.get("metadata_row_counts")
        ),
        "metadata_row_count": (
            sum(len(rows) for rows in metadata.values())
            if metadata
            else input_report.get("metadata_row_count")
        ),
        "unique_asset_count": len(entries),
        "available_asset_count": sum(bool(row.get("sha256")) for row in entries),
        "missing_asset_count": len(missing),
        "downloaded_asset_count": sum(bool(row.get("downloaded")) for row in entries),
        "preexisting_asset_count": sum(row.get("acquisition") == "preexisting" for row in entries),
        "alignment": alignment,
        "training_enabled": False,
        "text_provenance": "official RSRCC generated language annotations; evaluation-only until independent factual review",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if not missing else (2 if args.download else 0)


if __name__ == "__main__":
    raise SystemExit(main())
