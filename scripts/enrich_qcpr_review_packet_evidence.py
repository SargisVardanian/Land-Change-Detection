#!/usr/bin/env python3
"""Materialize physical and semantic-neighbour evidence for retrieval review packets.

This script only enriches review evidence.  It never writes reviewer decisions,
never enables training, and refuses to overwrite an existing output directory.
The source repair output is cloned with hardlinks so the original scratch
candidate remains unchanged; enriched packet files are copied as new inodes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from qcpr_data.queries.purpose import attribute_similarity, extract_visual_attributes, semantic_signature


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def split_name(value: Any) -> str:
    value = str(value or "")
    return {"val": "development", "validation": "development", "dev": "development"}.get(value, value or "unknown")


def item_paths(item: Mapping[str, Any]) -> tuple[str, str]:
    frames = item.get("frames") if isinstance(item.get("frames"), list) else []
    paths = [str(frame.get("path") or "") for frame in frames]
    return (paths[0] if paths else "", paths[1] if len(paths) > 1 else "")


def item_source(item: Mapping[str, Any]) -> str:
    provenance = item.get("provenance") if isinstance(item.get("provenance"), Mapping) else {}
    return str(item.get("source") or provenance.get("source_dataset") or "unknown")


def clone_with_hardlinks(source: Path, target: Path) -> None:
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing output: {target}")
    shutil.copytree(source, target, copy_function=os.link)


def replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.write_text(text, encoding="utf-8")


def attrs_for_row(row: Mapping[str, Any]) -> dict[str, Any]:
    attrs = row.get("attributes")
    if isinstance(attrs, Mapping):
        return dict(attrs)
    attrs = row.get("candidate_attributes")
    if isinstance(attrs, Mapping) and "objects" in attrs:
        return dict(attrs)
    return extract_visual_attributes(str(row.get("text") or row.get("query_text") or ""))


def control_pair(item_id: str, item: Mapping[str, Any], *, method: str, grade: int | None = None) -> dict[str, Any]:
    t1, t2 = item_paths(item)
    result: dict[str, Any] = {
        "item_id": item_id,
        "t1_path": t1,
        "t2_path": t2,
        "selection_method": method,
    }
    if grade is not None:
        result["attribute_similarity_grade"] = int(grade)
    return result


def build_same_signature_controls(
    *,
    split: str,
    signature: str,
    target_item_id: str,
    by_group: Mapping[tuple[str, str], set[str]],
    items: Mapping[str, Mapping[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for item_id in sorted(by_group.get((split, signature), set())):
        if item_id == target_item_id:
            continue
        t1, t2 = item_paths(items.get(item_id, {}))
        if not (t1 and t2):
            continue
        controls.append(control_pair(item_id, items[item_id], method="same_semantic_signature"))
        if len(controls) >= limit:
            break
    return controls


def build_attribute_fallback_controls(
    *,
    split: str,
    query_attrs: Mapping[str, Any],
    target_item_id: str,
    item_attributes: Mapping[tuple[str, str], list[Mapping[str, Any]]],
    items: Mapping[str, Mapping[str, Any]],
    existing_ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    scored: list[tuple[int, int, str]] = []
    for (candidate_split, item_id), candidates in item_attributes.items():
        if candidate_split != split or item_id == target_item_id or item_id in existing_ids:
            continue
        t1, t2 = item_paths(items.get(item_id, {}))
        if not (t1 and t2):
            continue
        best_grade = 0
        best_overlap = 0
        for candidate_attrs in candidates:
            similarity = attribute_similarity(query_attrs, candidate_attrs)
            grade = int(similarity["grade"])
            overlap = sum(bool(value) for value in similarity["matches"].values())
            best_grade = max(best_grade, grade)
            best_overlap = max(best_overlap, overlap)
        if best_grade >= 1:
            scored.append((best_grade, best_overlap, item_id))
    scored.sort(key=lambda value: (-value[0], -value[1], value[2]))
    return [
        control_pair(item_id, items[item_id], method="attribute_overlap_fallback", grade=grade)
        for grade, _overlap, item_id in scored[:limit]
    ]


def enrich_packet(
    rows: list[dict[str, Any]],
    *,
    purpose: str,
    items: Mapping[str, Mapping[str, Any]],
    by_group: Mapping[tuple[str, str], set[str]],
    item_attributes: Mapping[tuple[str, str], list[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    same_signature_rows = 0
    fallback_rows = 0
    missing_true_pair_rows = 0
    missing_neighbour_rows = 0
    decisions_completed = 0

    for row in rows:
        true_item_id = str(row.get("true_item_id") or row.get("source_item_id") or "")
        item = items.get(true_item_id, {})
        item_t1, item_t2 = item_paths(item)
        t1_path = str(row.get("t1_path") or item_t1 or "")
        t2_path = str(row.get("t2_path") or item_t2 or "")
        physical_available = bool(t1_path and t2_path and Path(t1_path).is_file() and Path(t2_path).is_file())
        if not physical_available:
            missing_true_pair_rows += 1

        text = str(row.get("text") or "")
        attrs = attrs_for_row(row)
        signature = str(row.get("semantic_signature") or semantic_signature(attrs))
        split = split_name(row.get("split") or item.get("split"))
        same = [] if purpose == "generic_no_change" else build_same_signature_controls(
            split=split,
            signature=signature,
            target_item_id=true_item_id,
            by_group=by_group,
            items=items,
            limit=12,
        )
        controls = list(same)
        selection_method = "same_semantic_signature" if same else "none_available"
        if same:
            same_signature_rows += 1
        if not same and purpose != "generic_no_change":
            fallback = build_attribute_fallback_controls(
                split=split,
                query_attrs=attrs,
                target_item_id=true_item_id,
                item_attributes=item_attributes,
                items=items,
                existing_ids={true_item_id},
                limit=12,
            )
            controls.extend(fallback)
            if fallback:
                selection_method = "attribute_overlap_fallback"
                fallback_rows += 1
        if purpose != "generic_no_change" and not controls:
            missing_neighbour_rows += 1

        result = dict(row)
        result.update({
            "source_dataset": str(row.get("source_dataset") or item_source(item)),
            "split": split,
            "true_item_id": true_item_id,
            "t1_path": t1_path,
            "t2_path": t2_path,
            "semantic_signature": signature,
            "semantic_neighbour_pairs": controls,
            "review_evidence": {
                "physical_pair_available": physical_available,
                "semantic_neighbour_evidence_available": bool(controls),
                "semantic_neighbour_selection": selection_method,
                "paths_from_canonical_physical_registry": bool(item_t1 and item_t2),
                "masks_used_for_text": False,
            },
            "training_enabled": False,
        })
        classifier = dict(result.get("classifier_evidence") or {})
        classifier.setdefault("positive_set_size", len(row.get("positive_item_ids") or []) or 1)
        classifier["semantic_neighbour_control_count"] = len(controls)
        result["classifier_evidence"] = classifier
        if result.get("review_decision") is not None:
            decisions_completed += 1
        enriched.append(result)

    return enriched, {
        "rows": len(rows),
        "true_pair_paths_present": len(rows) - missing_true_pair_rows,
        "missing_true_pair_rows": missing_true_pair_rows,
        "same_signature_control_rows": same_signature_rows,
        "attribute_overlap_fallback_rows": fallback_rows,
        "semantic_neighbour_control_rows": len(rows) - missing_neighbour_rows if purpose != "generic_no_change" else 0,
        "missing_neighbour_evidence_rows": missing_neighbour_rows,
        "decisions_completed": decisions_completed,
        "training_enabled": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-release", type=Path, required=True)
    parser.add_argument("--repair-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.current_release.is_dir() or not args.repair_output.is_dir():
        raise SystemExit("current release and repair output must be directories")
    clone_with_hardlinks(args.repair_output, args.output_dir)

    items = {
        str(row["item_id"]): row
        for row in read_jsonl(args.current_release / "registries/physical_items.jsonl")
        if row.get("item_id")
    }
    by_group: dict[tuple[str, str], set[str]] = defaultdict(set)
    item_attributes: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.current_release / "registries/query_purpose_registry.jsonl"):
        item_id = str(row.get("source_item_id") or "")
        signature = str(row.get("semantic_signature") or "")
        split = split_name(row.get("split"))
        attrs = attrs_for_row(row)
        if item_id and signature and attrs.get("has_specific_visual_claim"):
            by_group[(split, signature)].add(item_id)
            item_attributes[(split, item_id)].append(attrs)

    packet_dir = args.output_dir / "source_reports/review_samples"
    evidence: dict[str, Any] = {
        "schema_version": "qcpr-retrieval-review-evidence-v1",
        "status": "PASS_REVIEW_EVIDENCE_MATERIALIZED_HUMAN_DECISIONS_PENDING",
        "training_enabled": False,
        "source_release": str(args.current_release),
        "source_repair_output": str(args.repair_output),
        "populations": {},
    }
    for packet_path in sorted(packet_dir.glob("*_300.jsonl")):
        purpose = packet_path.stem.rsplit("_300", 1)[0]
        rows = read_jsonl(packet_path)
        enriched, audit = enrich_packet(
            rows,
            purpose=purpose,
            items=items,
            by_group=by_group,
            item_attributes=item_attributes,
        )
        replace_text(
            packet_path,
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in enriched),
        )
        evidence["populations"][purpose] = audit

    status_path = args.output_dir / "source_reports/human_review_packet_status.json"
    status = read_json(status_path)
    for purpose, audit in evidence["populations"].items():
        if purpose in status.get("packets", {}):
            status["packets"][purpose].update({
                "true_pair_paths_present": audit["true_pair_paths_present"],
                "semantic_neighbour_control_rows": audit["semantic_neighbour_control_rows"],
                "missing_true_pair_rows": audit["missing_true_pair_rows"],
                "missing_neighbour_evidence_rows": audit["missing_neighbour_evidence_rows"],
            })
    status["review_evidence_status"] = evidence["status"]
    status["training_enabled"] = False
    replace_text(status_path, json.dumps(status, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    evidence_path = args.output_dir / "source_reports/review_packet_evidence_audit.json"
    replace_text(evidence_path, json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    calibration_path = args.output_dir / "audits/retrieval_identifiability_calibration.json"
    calibration = read_json(calibration_path)
    calibration["review_evidence"] = {
        "status": evidence["status"],
        "all_true_pair_paths_present": all(
            value["missing_true_pair_rows"] == 0 for value in evidence["populations"].values()
        ),
        "exact_neighbour_control_gate": (
            evidence["populations"].get("exact_discriminative", {}).get("missing_neighbour_evidence_rows", 1) == 0
        ),
        "reviewer_decisions_completed": sum(
            value["decisions_completed"] for value in evidence["populations"].values()
        ),
    }
    replace_text(calibration_path, json.dumps(calibration, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
