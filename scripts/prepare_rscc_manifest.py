from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from land_change_detection.temporal_caption_manifest import audit_manifest_rows, make_manifest_row, write_jsonl


GENERATED_MARKERS = {"model", "generated", "model_generated", "gpt", "chatgpt", "assistant", "synthetic"}
HUMAN_MARKERS = {"human", "human_subset", "annotator", "expert", "manual"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an official RSCC/xBD temporal-caption retrieval manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--caption-policy", choices=("human_subset_only", "model_generated_only", "all"), default="human_subset_only")
    parser.add_argument("--expected-pairs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    return parser.parse_args()


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"RSCC annotation line {line_number} is not an object")
            rows.append(payload)
    return rows


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _caption_source(row: dict[str, Any]) -> str:
    raw = str(_pick(row, "caption_source", "source", "provenance", "caption_provenance", "answer_source") or "").casefold()
    if raw in HUMAN_MARKERS or bool(row.get("human_subset")):
        return "human"
    if raw in GENERATED_MARKERS or any(marker in raw for marker in GENERATED_MARKERS):
        return "model_generated"
    if bool(row.get("is_generated")) or bool(row.get("generated")):
        return "model_generated"
    return "human" if bool(row.get("is_human", True)) else "model_generated"


def _captions(row: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("captions", "caption", "description", "answer", "change_description", "retrieval_caption"):
        value = row.get(key)
        if value not in (None, ""):
            values.append(value)
    captions: list[str] = []
    for value in values:
        if isinstance(value, list):
            captions.extend(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value).strip()
            if text and not text.endswith("?"):
                captions.append(text)
    return captions


def _split(row: dict[str, Any]) -> str:
    value = str(_pick(row, "split", "subset", "partition") or "train").casefold()
    mapping = {"valid": "val", "validation": "val", "dev": "val"}
    value = mapping.get(value, value)
    if value not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported RSCC split {value!r}")
    return value


def _resolve(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _find_xbd_pair(root: Path, row: dict[str, Any], split: str) -> tuple[Path, Path]:
    before = _resolve(root, _pick(row, "t1_path", "before_path", "pre_path", "pre_image", "image_before"))
    after = _resolve(root, _pick(row, "t2_path", "after_path", "post_path", "post_image", "image_after"))
    if before and after:
        return before, after
    image_id = str(_pick(row, "image_id", "pair_id", "sample_id", "id", "xbd_id") or "").strip()
    if not image_id:
        raise ValueError("RSCC row missing image/pair id")
    candidates = [
        (root / split / "images" / f"{image_id}_pre_disaster.png", root / split / "images" / f"{image_id}_post_disaster.png"),
        (root / split / "images" / f"{image_id}_pre_disaster.tif", root / split / "images" / f"{image_id}_post_disaster.tif"),
        (root / split / "A" / f"{image_id}.png", root / split / "B" / f"{image_id}.png"),
        (root / split / "rgb" / "A" / f"{image_id}.png", root / split / "rgb" / "B" / f"{image_id}.png"),
    ]
    for before_candidate, after_candidate in candidates:
        if before_candidate.exists() and after_candidate.exists():
            return before_candidate, after_candidate
    raise FileNotFoundError(f"Could not resolve xBD image pair for RSCC id {image_id!r}")


def build_rows(root: Path, annotations: Path, caption_policy: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    captions_by_key: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    source_by_key: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in _jsonl_rows(annotations):
        source = _caption_source(row)
        if caption_policy == "human_subset_only" and source != "human":
            continue
        if caption_policy == "model_generated_only" and source != "model_generated":
            continue
        split = _split(row)
        original_id = str(_pick(row, "original_id", "pair_id", "sample_id", "image_id", "id", "xbd_id") or "").strip()
        if not original_id:
            raise ValueError("RSCC row missing original id")
        before, after = _find_xbd_pair(root, row, split)
        key = (split, original_id, source if caption_policy != "all" else "mixed")
        grouped.setdefault(
            key,
            {
                "split": split,
                "original_id": original_id,
                "before": before,
                "after": after,
                "metadata": {
                    "dataset": "RSCC",
                    "annotation_file": str(annotations),
                    "event": _pick(row, "event", "event_type"),
                    "disaster": _pick(row, "disaster", "disaster_name", "hazard"),
                    "license_family": _pick(row, "license", "license_family") or "xBD",
                    "caption_policy": caption_policy,
                },
            },
        )
        captions_by_key[key].extend(_captions(row))
        source_by_key[key].add(source)

    manifest_rows: list[dict[str, Any]] = []
    for key, payload in sorted(grouped.items()):
        captions = list(dict.fromkeys(captions_by_key[key]))
        if not captions:
            continue
        sources = source_by_key[key]
        caption_source = "model_generated" if sources == {"model_generated"} else "human"
        if "model_generated" in sources and "human" in sources:
            caption_source = "semantic_template"
        row = make_manifest_row(
            dataset_name="rscc",
            split=payload["split"],
            original_id=payload["original_id"],
            t1_path=payload["before"],
            t2_path=payload["after"],
            captions=captions,
            caption_source=caption_source,
            source_metadata=payload["metadata"] | {"caption_sources_observed": sorted(sources)},
        )
        manifest_rows.append(row)
    return manifest_rows


def main() -> int:
    args = parse_args()
    rows = build_rows(args.root, args.annotations, args.caption_policy)
    if len(rows) != args.expected_pairs:
        raise SystemExit(f"Expected {args.expected_pairs} RSCC pairs, found {len(rows)}")
    write_jsonl(args.output, rows)
    report = audit_manifest_rows(rows)
    args.audit_report.parent.mkdir(parents=True, exist_ok=True)
    args.audit_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(f"Manifest audit failed: {report['errors'][:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
