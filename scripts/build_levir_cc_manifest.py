from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny text-to-change retrieval manifest from LEVIR-CC style captions.")
    parser.add_argument("--captions-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, default=16)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "images" in payload and isinstance(payload["images"], list):
            return [dict(row) for row in payload["images"] if isinstance(row, dict)]
        if "items" in payload and isinstance(payload["items"], list):
            return [dict(row) for row in payload["items"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    raise ValueError("Unsupported captions format. Expected a list, or a dict with 'images'/'items'.")


def _text_features(text: str, feature_dim: int) -> list[float]:
    values = [0.0] * feature_dim
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = digest[0] % feature_dim
        values[index] += 1.0
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


def _group_key(row: dict) -> str:
    return str(row.get("transition_label") or row.get("class") or row.get("change_type") or row.get("category") or "unknown")


def main() -> int:
    args = parse_args()
    rows = _load_rows(args.captions_json)
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        item_id = str(row.get("id") or row.get("image_id") or row.get("sample_id") or row.get("filename") or len(grouped))
        grouped[_group_key(row)].append(item_id)

    output_rows = []
    for row in rows:
        item_id = str(row.get("id") or row.get("image_id") or row.get("sample_id") or row.get("filename") or len(output_rows))
        text = str(row.get("caption") or row.get("text") or row.get("change_caption") or "")
        label = _group_key(row)
        positives = [candidate for candidate in grouped[label] if candidate != item_id]
        negatives = [candidate for other_label, ids in grouped.items() if other_label != label for candidate in ids][:16]
        output_rows.append(
            {
                "item_id": item_id,
                "mode": "text_bitemporal",
                "features": _text_features(text, args.feature_dim),
                "text": text,
                "transition_label": label,
                "positives": positives,
                "negatives": negatives,
                "metadata": {"source": str(args.captions_json)},
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows), encoding="utf-8")
    print(f"Wrote {len(output_rows)} retrieval manifest rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
