from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from land_change_detection.manual_dataset_imports import canonical_import_outputs, directory_inventory, ensure_symlink, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and import a manually downloaded QAG-360K dataset.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    return parser.parse_args()


def _annotation_files(root: Path) -> list[Path]:
    return sorted([path for path in root.rglob("*.jsonl") if path.is_file()] + [path for path in root.rglob("*.json") if path.is_file()])


def _resolve_optional_path(root: Path, value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    resolved = path if path.is_absolute() else root / path
    return str(resolved) if resolved.exists() else None


def _extract_rows(path: Path, root: Path) -> list[dict[str, Any]]:
    payload = path.read_text(encoding="utf-8").splitlines() if path.suffix == ".jsonl" else [path.read_text(encoding="utf-8")]
    rows: list[dict[str, Any]] = []
    for chunk in payload:
        data = json.loads(chunk)
        candidates = data if isinstance(data, list) else data.get("items") or data.get("annotations") or data.get("data") or [data]
        for row in candidates:
            if not isinstance(row, dict):
                continue
            question = row.get("question") or row.get("query") or row.get("prompt") or row.get("text")
            answer = row.get("answer") or row.get("response") or row.get("label")
            image_path = _resolve_optional_path(root, row.get("image") or row.get("image_path") or row.get("filename"))
            mask_path = _resolve_optional_path(root, row.get("mask") or row.get("mask_path") or row.get("segmentation"))
            if not question and not answer and not image_path and not mask_path:
                continue
            item_id = str(row.get("id") or row.get("question_id") or row.get("sample_id") or f"{path.stem}:{len(rows)}")
            rows.append(
                {
                    "item_id": item_id,
                    "dataset_name": "QAG-360K",
                    "question": str(question) if question else "",
                    "answer": str(answer) if answer else "",
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "split": row.get("split") or row.get("subset") or "unknown",
                    "metadata": {"annotation_source": str(path.relative_to(root))},
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    root = args.root or args.project_root / "datasets" / "manual" / "QAG-360K"
    if not root.exists():
        raise SystemExit(f"Manual QAG-360K root not found: {root}")
    rows: list[dict[str, Any]] = []
    for path in _annotation_files(root):
        rows.extend(_extract_rows(path, root))
    if not rows:
        raise SystemExit(f"No QAG-360K annotation rows discovered under {root}")
    outputs = canonical_import_outputs(args.project_root, "qag360k")
    manifest = write_jsonl(outputs["sample_manifest"], rows)
    raw_link = ensure_symlink(args.project_root / "datasets" / "raw" / "QAG-360K", root)
    payload = {
        "dataset_name": "QAG-360K",
        "manual_root": str(root),
        "raw_link": str(raw_link),
        "inventory": directory_inventory(root),
        "sample_manifest": str(manifest),
        "num_rows": len(rows),
    }
    write_json(root / "IMPORT_PROVENANCE.json", payload)
    write_json(outputs["report"], payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
