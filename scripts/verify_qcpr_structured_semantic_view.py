#!/usr/bin/env python3
"""Independently verify the source-verified S2Looking semantic view.

This verifier intentionally re-derives the eligible relation rows from the
raw pair-level source manifest instead of importing the view builder.  It
proves official coarse-label mapping, dense-target polarity, same-split
multi-positive groups, and the mask-free output contract.  It does not claim
human review of natural-language captions.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


EXPECTED_MAPPING = {
    "official_source_order": "Image2_to_Image1",
    "label1": "appeared",
    "label2": "disappeared",
}
RELATION_TEXT = {
    "appeared": "new buildings appeared",
    "disappeared": "buildings were demolished",
}
SPLITS = ("train", "development", "test")
FORBIDDEN_OUTPUT_KEYS = (
    "mask_path",
    "semantic_t1_path",
    "semantic_t2_path",
    "label_path",
    "dense_path",
    "semantic_map",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_split(value: Any) -> str:
    value = str(value or "")
    return "development" if value in {"val", "validation", "dev"} else value


def actual_nonempty(mask_path: Path) -> bool:
    with Image.open(mask_path) as image:
        return bool(image.convert("L").getbbox())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha", default=None)
    args = parser.parse_args()

    structured_dir = args.structured_dir
    audit = read_json(structured_dir / "structured_semantic_audit.json")
    source_path = Path(str(audit["input"]))
    source_rows = read_jsonl(source_path)
    source_pair_ids: set[str] = set()
    group_pairs: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    eligible: set[tuple[str, str, str]] = set()
    mapping_values: set[tuple[tuple[str, str], ...]] = set()
    rejection_counts: collections.Counter[str] = collections.Counter()

    for source_row in source_rows:
        pair_id = str(source_row.get("pair_id") or "")
        if not pair_id or pair_id in source_pair_ids:
            raise SystemExit(f"duplicate or missing source pair_id: {pair_id}")
        source_pair_ids.add(pair_id)
        if source_row.get("dataset_name") != "s2looking":
            raise SystemExit(f"unexpected source dataset for {pair_id}")
        split = normalized_split(source_row.get("split"))
        if split not in SPLITS:
            raise SystemExit(f"unsupported source split for {pair_id}: {split}")
        metadata = source_row.get("source_metadata") or {}
        mapping = dict(metadata.get("official_label_mapping") or {})
        mapping["official_source_order"] = metadata.get("official_source_order")
        mapping_values.add(tuple(sorted((str(key), str(value)) for key, value in mapping.items())))
        if mapping != EXPECTED_MAPPING:
            raise SystemExit(f"official label mapping mismatch for {pair_id}: {mapping}")
        targets = {str(target.get("direction")): target for target in source_row.get("directional_targets", [])}
        if set(targets) != set(RELATION_TEXT):
            raise SystemExit(f"incomplete directional targets for {pair_id}")
        for direction, target in targets.items():
            mask_path = Path(str(target.get("mask_path") or ""))
            if not mask_path.is_file():
                raise SystemExit(f"missing dense target for {pair_id}/{direction}")
            actual = actual_nonempty(mask_path)
            if actual != bool(target.get("has_visual_target")):
                raise SystemExit(f"dense polarity mismatch for {pair_id}/{direction}")
            if actual:
                group_pairs[(split, direction)].add(pair_id)

    expected_group_pairs = {
        key: values for key, values in group_pairs.items() if len(values) >= 2
    }
    for split, direction in expected_group_pairs:
        for pair_id in expected_group_pairs[(split, direction)]:
            eligible.add((split, direction, pair_id))

    actual_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    mask_free = True
    for split in SPLITS:
        for row in read_jsonl(structured_dir / f"retrieval_semantic_structured_{split}.jsonl"):
            encoded = json.dumps(row, sort_keys=True, ensure_ascii=False).casefold()
            if any(token in encoded for token in FORBIDDEN_OUTPUT_KEYS):
                mask_free = False
            pair_id = str(row.get("canonical_pair_id") or "")
            group_id = str(row.get("semantic_group_id") or "")
            direction = group_id.rsplit(":", 1)[-1]
            key = (str(row.get("split") or ""), direction, pair_id)
            if key in actual_rows:
                raise SystemExit(f"duplicate output semantic row: {key}")
            actual_rows[key] = row
            if row.get("schema_version") != "temporal-caption-manifest-v1":
                raise SystemExit(f"unexpected output schema for {key}")
            if row.get("verification_status") != "structured_source_verified" or not row.get("training_enabled"):
                raise SystemExit(f"output row is not source-verified/training-enabled: {key}")
            if row.get("text") != RELATION_TEXT.get(direction):
                raise SystemExit(f"unexpected relation text for {key}")
            if row.get("semantic_group_pair_count") != len(expected_group_pairs.get((split, direction), ())):
                raise SystemExit(f"group count mismatch for {key}")

    if set(actual_rows) != eligible:
        missing = sorted(eligible - set(actual_rows))[:5]
        extra = sorted(set(actual_rows) - eligible)[:5]
        raise SystemExit(f"eligible/output relation mismatch; missing={missing}, extra={extra}")
    if not mask_free:
        raise SystemExit("dense/mask path leaked into structured semantic output")

    registry_rows = read_jsonl(structured_dir / "semantic_group_registry_structured.jsonl")
    registry_keys = {(str(row.get("split")), str(row.get("semantic_group_id"))) for row in registry_rows}
    expected_registry_keys = {(split, f"s2looking:official_relation:{direction}") for split, direction in expected_group_pairs}
    if registry_keys != expected_registry_keys:
        raise SystemExit(f"semantic group registry mismatch: expected={expected_registry_keys}, actual={registry_keys}")
    for row in registry_rows:
        split = str(row.get("split"))
        direction = str(row.get("semantic_group_id")).rsplit(":", 1)[-1]
        expected_pairs = sorted(expected_group_pairs[(split, direction)])
        if sorted(row.get("pair_ids") or []) != expected_pairs or row.get("pair_count") != len(expected_pairs):
            raise SystemExit(f"semantic group pair membership mismatch: {row}")
        if row.get("verification_status") != "structured_source_verified" or not row.get("training_enabled"):
            raise SystemExit(f"semantic group is not verified/training-enabled: {row}")

    output_hashes = {
        path.name: sha256(path)
        for path in [structured_dir / f"retrieval_semantic_structured_{split}.jsonl" for split in SPLITS]
        + [structured_dir / "semantic_group_registry_structured.jsonl"]
    }
    result = {
        "schema_version": "qcpr-stage2-independent-structured-semantic-verification-v1",
        "status": "INDEPENDENT_STRUCTURED_SEMANTIC_VERIFIED",
        "code_sha": args.code_sha,
        "source": "S2Looking",
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": sha256(source_path),
        "official_mapping": EXPECTED_MAPPING,
        "observed_mapping_values": [dict(values) for values in sorted(mapping_values)],
        "source_pair_count": len(source_rows),
        "verified_rows": len(actual_rows),
        "verified_split_counts": dict(collections.Counter(split for split, _, _ in actual_rows)),
        "verified_group_count": len(expected_group_pairs),
        "group_size_distribution": dict(collections.Counter(len(values) for values in expected_group_pairs.values())),
        "mask_free_output": mask_free,
        "dense_target_polarity_rechecked": True,
        "output_hashes": output_hashes,
        "human_caption_review": "not_claimed; deterministic official coarse relation only",
        "training_enabled_scope": "coarse_official_relation_semantic_view_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "verified_rows", "verified_split_counts", "verified_group_count", "mask_free_output")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
