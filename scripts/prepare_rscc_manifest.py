from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from land_change_detection.temporal_caption_manifest import audit_manifest_rows, make_manifest_row, write_jsonl


GENERATED_MARKERS = {"model", "generated", "model_generated", "gpt", "chatgpt", "assistant", "synthetic", "qvq", "qvq-max", "qvq_max"}
QVQ_GENERATORS = {"qvq-max", "qvq_max", "qvq max", "qvqmax"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an official RSCC/xBD temporal-caption retrieval manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--caption-policy", choices=("qvq_ground_truth_only", "model_generated_only", "all"), default="qvq_ground_truth_only")
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


def _is_qvq_ground_truth(row: dict[str, Any], annotations: Path) -> bool:
    generator = str(_pick(row, "caption_generator", "generator", "model", "source_model") or "").casefold()
    source = str(_pick(row, "caption_source", "source", "provenance", "caption_provenance", "answer_source") or "").casefold()
    subset = str(_pick(row, "benchmark_subset", "subset_name", "dataset_subset") or "").casefold()
    return bool(
        row.get("benchmark_ground_truth")
        or "qvq" in annotations.stem.casefold()
        or generator in QVQ_GENERATORS
        or "qvq" in source
        or subset == "rscc_xbd_988"
    )


def _caption_kind(row: dict[str, Any], annotations: Path) -> str:
    if _is_qvq_ground_truth(row, annotations):
        return "qvq_ground_truth"
    raw = str(_pick(row, "caption_source", "source", "provenance", "caption_provenance", "answer_source") or "").casefold()
    if raw in GENERATED_MARKERS or any(marker in raw for marker in GENERATED_MARKERS):
        return "full_model_generated"
    if bool(row.get("is_generated")) or bool(row.get("generated")):
        return "full_model_generated"
    return "full_model_generated"


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
    value = str(_pick(row, "split", "subset", "partition") or "test").casefold()
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
        kind = _caption_kind(row, annotations)
        if caption_policy == "qvq_ground_truth_only" and kind != "qvq_ground_truth":
            continue
        if caption_policy == "model_generated_only" and kind == "qvq_ground_truth":
            continue
        split = _split(row)
        original_id = str(_pick(row, "original_id", "pair_id", "sample_id", "image_id", "id", "xbd_id") or "").strip()
        if not original_id:
            raise ValueError("RSCC row missing original id")
        before, after = _find_xbd_pair(root, row, split)
        key = (split, original_id, kind)
        metadata = {
            "dataset": "RSCC",
            "annotation_file": str(annotations),
            "event": _pick(row, "event", "event_type"),
            "disaster": _pick(row, "disaster", "disaster_name", "hazard"),
            "license_family": "xBD",
            "caption_policy": caption_policy,
            "caption_kind": kind,
        }
        if kind == "qvq_ground_truth":
            metadata.update(
                {
                    "caption_generator": "QvQ-Max",
                    "benchmark_ground_truth": True,
                    "benchmark_subset": "rscc_xbd_988",
                    "training_default_enabled": False,
                    "research_only": True,
                }
            )
        else:
            metadata.update(
                {
                    "caption_generator": _pick(row, "caption_generator", "generator", "model", "source_model") or "model_generated",
                    "benchmark_ground_truth": False,
                    "training_default_enabled": False,
                    "research_only": True,
                }
            )
        grouped.setdefault(
            key,
            {
                "split": split,
                "original_id": original_id,
                "before": before,
                "after": after,
                "metadata": metadata,
            },
        )
        captions_by_key[key].extend(_captions(row))
        source_by_key[key].add(kind)

    manifest_rows: list[dict[str, Any]] = []
    for key, payload in sorted(grouped.items()):
        captions = list(dict.fromkeys(captions_by_key[key]))
        if not captions:
            continue
        row = make_manifest_row(
            dataset_name="rscc",
            split=payload["split"],
            original_id=payload["original_id"],
            t1_path=payload["before"],
            t2_path=payload["after"],
            captions=captions,
            caption_source="model_generated",
            source_metadata=payload["metadata"] | {"caption_sources_observed": sorted(source_by_key[key])},
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
