from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert indexed change samples into a lightweight retrieval training manifest.")
    parser.add_argument("--index", type=Path, required=True, help="Input JSONL from index_change_retrieval_dataset.py")
    parser.add_argument("--output", type=Path, required=True, help="Output retrieval training manifest JSONL")
    parser.add_argument("--mode", default="pair_analog", choices=("pair_analog", "transition_conditioned"))
    parser.add_argument("--feature-dim", type=int, default=16)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _transition_label(row: dict) -> str:
    metadata = dict(row.get("metadata", {}))
    if metadata.get("transition_label"):
        return str(metadata["transition_label"])
    if metadata.get("transition_hint"):
        return str(metadata["transition_hint"])
    caption = str(row.get("caption") or "").strip().lower()
    if caption:
        return caption
    return str(row.get("dataset_name", "unknown"))


def _hash_features(text: str, dims: int) -> list[float]:
    values = [0.0] * dims
    for index, token in enumerate(text.lower().split()):
        values[(sum(ord(ch) for ch in token) + index) % dims] += 1.0
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


def main() -> int:
    args = parse_args()
    rows = _read_rows(args.index)
    grouped: dict[str, list[str]] = defaultdict(list)
    normalized: list[dict] = []

    for row in rows:
        item_id = str(row["sample_id"])
        text = str(row.get("caption") or row.get("sample_id") or "")
        label = _transition_label(row)
        payload = {
            "item_id": item_id,
            "text": text,
            "label": label,
            "before_path": row.get("before_path"),
            "after_path": row.get("after_path"),
            "mask_path": row.get("mask_path"),
            "metadata": dict(row.get("metadata", {})),
        }
        normalized.append(payload)
        grouped[label].append(item_id)

    output_rows = []
    for row in normalized:
        positives = [candidate for candidate in grouped[row["label"]] if candidate != row["item_id"]]
        negatives = [candidate for label, ids in grouped.items() if label != row["label"] for candidate in ids][:32]
        features_source = " | ".join(
            value for value in [row["text"], row["before_path"], row["after_path"], row["mask_path"]] if value
        )
        output_rows.append(
            {
                "item_id": row["item_id"],
                "mode": args.mode,
                "features": _hash_features(features_source, args.feature_dim),
                "text": row["text"],
                "transition_label": row["label"],
                "positives": positives,
                "negatives": negatives,
                "metadata": row["metadata"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows), encoding="utf-8")
    print(f"Wrote {len(output_rows)} training manifest rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
