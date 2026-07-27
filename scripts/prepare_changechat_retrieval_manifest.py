from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from land_change_detection.temporal_caption_manifest import audit_manifest_rows, make_manifest_row, write_jsonl


DECLARATIVE_FIELDS = (
    "change_caption",
    "caption",
    "description",
    "change_description",
    "localization_description",
    "quantification_description",
    "gpt_assisted_change_description",
    "gpt_description",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional ChangeChat-to-retrieval manifest adapter; disabled unless run explicitly.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=None)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    for key in ("data", "rows", "annotations", "items"):
        if isinstance(payload.get(key), list):
            return [row for row in payload[key] if isinstance(row, dict)]
    raise ValueError(f"Unsupported ChangeChat annotation shape: {path}")


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _declarative_captions(row: dict[str, Any]) -> list[tuple[str, str]]:
    captions: list[tuple[str, str]] = []
    for field in DECLARATIVE_FIELDS:
        value = row.get(field)
        values = value if isinstance(value, list) else [value]
        for item in values:
            text = str(item or "").strip()
            if text and not text.endswith("?"):
                captions.append((text, field))
    for qa in row.get("conversations", row.get("qa", [])) or []:
        if not isinstance(qa, dict):
            continue
        answer = str(_pick(qa, "answer", "a", "response") or "").strip()
        task = str(_pick(qa, "task", "type", "category") or "").casefold()
        if answer and not answer.endswith("?") and any(label in task for label in ("caption", "localization", "quantification", "description")):
            captions.append((answer, f"conversation:{task}"))
    deduped: dict[str, str] = {}
    for text, provenance in captions:
        deduped.setdefault(text, provenance)
    return list(deduped.items())


def build_rows(root: Path, annotations: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _rows(annotations):
        captions_with_provenance = _declarative_captions(raw)
        if not captions_with_provenance:
            continue
        split = str(_pick(raw, "split", "subset") or "train").casefold()
        if split == "validation":
            split = "val"
        original_id = str(_pick(raw, "original_id", "pair_id", "sample_id", "id", "image_id") or "").strip()
        t1 = _resolve(root, _pick(raw, "t1_path", "before_path", "image_before", "pre_image"))
        t2 = _resolve(root, _pick(raw, "t2_path", "after_path", "image_after", "post_image"))
        provenance = sorted({source for _, source in captions_with_provenance})
        rows.append(
            make_manifest_row(
                dataset_name="changechat",
                split=split,
                original_id=original_id,
                t1_path=t1,
                t2_path=t2,
                captions=[caption for caption, _ in captions_with_provenance],
                caption_source="model_generated",
                mask_path=_resolve(root, raw["mask_path"]) if raw.get("mask_path") else None,
                source_metadata={
                    "dataset": "ChangeChat",
                    "adapter": "declarative_retrieval_descriptions_only",
                    "caption_provenance": provenance,
                    "annotation_file": str(annotations),
                },
            )
        )
    return rows


def main() -> int:
    args = parse_args()
    rows = build_rows(args.root, args.annotations)
    if args.expected_pairs is not None and len(rows) != args.expected_pairs:
        raise SystemExit(f"Expected {args.expected_pairs} ChangeChat pairs, found {len(rows)}")
    write_jsonl(args.output, rows)
    report = audit_manifest_rows(rows)
    args.audit_report.parent.mkdir(parents=True, exist_ok=True)
    args.audit_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(f"Manifest audit failed: {report['errors'][:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
