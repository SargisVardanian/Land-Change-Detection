from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small LEVIR-CC text retrieval index with FAISS if available.")
    parser.add_argument("--manifest", type=Path, required=True, help="Input text retrieval manifest JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for index artifacts.")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _normalized_matrix(rows: list[dict]):
    import numpy as np

    matrix = np.array([row["features"] for row in rows], dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def main() -> int:
    args = parse_args()
    import numpy as np

    rows = _load_rows(args.manifest)
    matrix = _normalized_matrix(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "count": len(rows),
        "dim": int(matrix.shape[1]) if len(rows) else 0,
        "items": [
            {
                "item_id": row["item_id"],
                "text": row.get("text"),
                "transition_label": row.get("transition_label"),
            }
            for row in rows
        ],
    }

    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(args.output_dir / "levir_cc_text.index"))
        metadata["backend"] = "faiss"
    except Exception:
        np.save(args.output_dir / "levir_cc_text_vectors.npy", matrix)
        metadata["backend"] = "numpy"

    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Built LEVIR-CC text index with backend={metadata['backend']} -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
