from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.change_retrieval_datasets import discover_change_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap indexed dataset assets for LEVIR-MCI, SECOND-CC, and LEVIR-CC retrieval experiments."
    )
    parser.add_argument("--project-root", type=Path, required=True, help="Root like /data/$USER/rs_change_project")
    parser.add_argument(
        "--levir-cc-captions",
        type=Path,
        help="Optional explicit captions JSON for LEVIR-CC. If omitted, the script searches under datasets/raw/LEVIR-CC.",
    )
    parser.add_argument("--feature-dim", type=int, default=16)
    return parser.parse_args()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def _find_caption_json(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*.json"))
    for path in candidates:
        lowered = path.name.lower()
        if "caption" in lowered or "levir" in lowered or "change" in lowered or "label" in lowered:
            return path
    return candidates[0] if candidates else None


def _hashed_features(text: str, dims: int) -> list[float]:
    values = [0.0] * dims
    for index, token in enumerate(text.lower().split()):
        slot = (sum(ord(ch) for ch in token) + index) % dims
        values[slot] += 1.0
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


def _load_caption_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if "images" in payload and isinstance(payload["images"], list):
            return [dict(row) for row in payload["images"] if isinstance(row, dict)]
        if "items" in payload and isinstance(payload["items"], list):
            return [dict(row) for row in payload["items"] if isinstance(row, dict)]
        if all(isinstance(value, str) for value in payload.values()):
            return [{"id": key, "caption": value} for key, value in payload.items()]
    raise ValueError(f"Unsupported caption JSON format: {path}")


def _build_text_manifest(captions_json: Path, output_path: Path, feature_dim: int) -> int:
    rows = _load_caption_rows(captions_json)
    grouped: dict[str, list[str]] = {}
    normalized_rows: list[dict] = []
    for row in rows:
        item_id = str(row.get("id") or row.get("image_id") or row.get("sample_id") or row.get("filename") or len(normalized_rows))
        caption = str(row.get("caption") or row.get("text") or row.get("change_caption") or "")
        label = str(row.get("transition_label") or row.get("class") or row.get("change_type") or "unknown")
        normalized = {"item_id": item_id, "caption": caption, "transition_label": label}
        normalized_rows.append(normalized)
        grouped.setdefault(label, []).append(item_id)

    output_rows = []
    for row in normalized_rows:
        item_id = row["item_id"]
        label = row["transition_label"]
        positives = [candidate for candidate in grouped[label] if candidate != item_id]
        negatives = [candidate for other_label, ids in grouped.items() if other_label != label for candidate in ids][:32]
        output_rows.append(
            {
                "item_id": item_id,
                "mode": "text_bitemporal",
                "features": _hashed_features(row["caption"], feature_dim),
                "text": row["caption"],
                "transition_label": label,
                "positives": positives,
                "negatives": negatives,
                "metadata": {"source": str(captions_json)},
            }
        )
    _write_jsonl(output_path, output_rows)
    return len(output_rows)


def _write_preview_manifest(project_root: Path, levir_samples: list[dict], second_samples: list[dict]) -> Path:
    preview_manifest = project_root / "indexes" / "preview_samples.json"
    payload = {
        "LEVIR-MCI": levir_samples[0] if levir_samples else None,
        "SECOND-CC": second_samples[0] if second_samples else None,
    }
    preview_manifest.parent.mkdir(parents=True, exist_ok=True)
    preview_manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return preview_manifest


def main() -> int:
    args = parse_args()
    project_root = args.project_root
    raw_root = project_root / "datasets" / "raw"
    indexes_root = project_root / "indexes"

    levir_mci_root = raw_root / "LEVIR-MCI"
    second_cc_root = raw_root / "SECOND-CC"
    levir_cc_root = raw_root / "LEVIR-CC"

    levir_mci_samples = [sample.to_dict() for sample in discover_change_samples(levir_mci_root, "LEVIR-MCI")] if levir_mci_root.exists() else []
    second_cc_samples = [sample.to_dict() for sample in discover_change_samples(second_cc_root, "SECOND-CC")] if second_cc_root.exists() else []

    if levir_mci_samples:
        _write_jsonl(indexes_root / "levir_mci_samples.jsonl", levir_mci_samples)
        print(f"Indexed LEVIR-MCI samples: {len(levir_mci_samples)}")
    else:
        print("LEVIR-MCI not indexed: folder missing or no before/after pairs discovered.")

    if second_cc_samples:
        _write_jsonl(indexes_root / "second_cc_samples.jsonl", second_cc_samples)
        print(f"Indexed SECOND-CC samples: {len(second_cc_samples)}")
    else:
        print("SECOND-CC not indexed: folder missing or no before/after pairs discovered.")

    preview_manifest = _write_preview_manifest(project_root, levir_mci_samples, second_cc_samples)
    print(f"Wrote preview manifest: {preview_manifest}")

    captions_json = args.levir_cc_captions or (_find_caption_json(levir_cc_root) if levir_cc_root.exists() else None)
    if captions_json is not None and captions_json.exists():
        rows = _build_text_manifest(captions_json, indexes_root / "levir_cc_text_manifest.jsonl", args.feature_dim)
        print(f"Built LEVIR-CC text retrieval manifest: {rows} rows")
    else:
        print("LEVIR-CC text manifest skipped: captions JSON not found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
