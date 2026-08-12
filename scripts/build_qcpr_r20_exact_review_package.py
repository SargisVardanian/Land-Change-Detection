#!/usr/bin/env python3
"""Build the decision-free QCPR r20 exact-supervision human-review package.

The package is deliberately not a training release.  It joins the immutable
r19g calibration packets to native T1/T2 image paths, and adds deterministic
candidate neighbours selected from source attributes/collision groups only.
No model scores, rankings, masks, event IDs, or source names are shown in the
review sheets.  Automatic selection is provenance for sampling only; it never
creates a relevance label or promotes a query to exact training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRATA = (
    "exact_discriminative",
    "semantic_multi_positive",
    "generic_no_change",
    "stable_scene_specific",
    "localized_direction",
)
DECISIONS = (
    "EXACT",
    "SEMANTIC_ONLY",
    "AMBIGUOUS_IGNORE",
    "REWRITE_REQUIRED",
    "REJECT",
)
ATTR_FIELDS = (
    "changed_object",
    "change_type",
    "change_direction",
    "spatial_relation",
    "surface_type",
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != "artifact_sha256"}
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def values(attrs: dict[str, Any], field: str) -> set[str]:
    value = attrs.get(field, [])
    if isinstance(value, list):
        return {str(item).casefold() for item in value if item is not None}
    return {str(value).casefold()} if value not in (None, "") else set()


def attrs(row: dict[str, Any]) -> dict[str, set[str]]:
    source = row.get("attributes") or {}
    return {field: values(source, field) for field in ATTR_FIELDS}


def attrs_to_json(row_attrs: dict[str, set[str]]) -> dict[str, list[str]]:
    return {field: sorted(row_attrs.get(field, set())) for field in ATTR_FIELDS}


def overlap(query_attrs: dict[str, set[str]], candidate_attrs: dict[str, set[str]]) -> int:
    weights = {
        "changed_object": 4,
        "change_type": 4,
        "change_direction": 3,
        "spatial_relation": 2,
        "surface_type": 1,
    }
    return sum(weights[field] for field in ATTR_FIELDS if query_attrs[field] & candidate_attrs[field])


def frame_paths(physical: dict[str, Any]) -> list[str]:
    return [
        str(frame.get("native_path") or frame.get("path"))
        for frame in (physical.get("frames") or [])[:2]
    ]


def source_dataset(row: dict[str, Any]) -> str | None:
    provenance = row.get("provenance") or {}
    return provenance.get("source_dataset") or row.get("source_dataset")


def safe_asset_token(review_sample_id: str) -> str:
    """Return a stable, source-free directory token for a review sample."""
    return hashlib.sha256(review_sample_id.encode("utf-8")).hexdigest()[:24]


def materialize_blind_asset(source_path: str, destination: Path) -> str:
    """Materialize an image under an anonymous reviewer-facing path.

    Hard links avoid duplicating the large native images on the cluster while
    still producing ordinary files that remain usable if the review package
    is copied elsewhere.  Cross-device filesystems fall back to a byte copy.
    The native source path never enters the visible packet.
    """
    source = Path(source_path)
    if not source.is_file():
        raise RuntimeError(f"review image is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"blind asset destination already exists: {destination}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_blind_asset_manifest(asset_root: Path, output_path: Path) -> tuple[int, str]:
    """Write a source-free integrity manifest for reviewer-facing assets."""
    rows: list[dict[str, Any]] = []
    for path in sorted(path for path in asset_root.rglob("*") if path.is_file()):
        rows.append(
            {
                "relative_path": str(path.relative_to(asset_root.parent)),
                "bytes": path.stat().st_size,
                "sha256": file_sha(path),
            }
        )
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return len(rows), file_sha(output_path)


def select_source_balanced_exact_rows(
    query_rows: list[dict[str, Any]],
    seed: int,
    per_source: int = 150,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select one human exact query per physical pair, balanced across core sources.

    The calibration is a query-level review, but selecting at most one caption
    per physical pair prevents a caption family from consuming a stratum.  The
    source allocation is explicit so source-conditioned LEVIR/SECOND rates are
    estimable after adjudication.
    """
    eligible: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in query_rows:
        scope = str(row.get("query_scope") or row.get("query_classification") or "")
        split = str(row.get("split") or "")
        source = source_dataset(row)
        pair_id = str(row.get("source_pair_id") or row.get("source_item_id") or "")
        if (
            scope not in {"exact", "exact_discriminative", "exact_pair"}
            or split != "train"
            or source not in {"levir_mci", "second_cc"}
            or row.get("training_enabled") is not True
            or str(row.get("verification") or "") != "human"
            or not pair_id
            or not str(row.get("text") or "").strip()
        ):
            continue
        eligible[source][pair_id].append(row)

    selected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for source in ("levir_mci", "second_cc"):
        pair_candidates = []
        for pair_id, rows in eligible[source].items():
            chosen = min(
                rows,
                key=lambda row: hashlib.sha256(
                    f"{seed}:exact:{source}:{pair_id}:{row.get('query_id', '')}".encode()
                ).hexdigest(),
            )
            pair_candidates.append(chosen)
        pair_candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:exact:{source}:{row.get('source_pair_id') or row.get('source_item_id')}".encode()
            ).hexdigest(),
        )
        if len(pair_candidates) < per_source:
            raise RuntimeError(
                f"source-balanced exact calibration needs {per_source} unique {source} pairs; "
                f"found {len(pair_candidates)}"
            )
        source_counts[source] = per_source
        selected.extend(pair_candidates[:per_source])
    return selected, source_counts


def review_input_row(
    row: dict[str, Any],
    stratum: str,
    index: int,
) -> dict[str, Any]:
    """Adapt a canonical query row to the review-packet input contract."""
    pair_id = row.get("source_pair_id") or row.get("source_item_id")
    return {
        "review_sample_id": f"{stratum}_300:{index:04d}",
        "query_id": row.get("query_id"),
        "source_pair_id": pair_id,
        "text": row.get("text"),
        "candidate": row,
    }


def build_neighbor_index(
    query_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, set[str]]], dict[str, str], dict[str, set[str]]]:
    pair_attrs: dict[str, dict[str, set[str]]] = defaultdict(lambda: {field: set() for field in ATTR_FIELDS})
    pair_split: dict[str, str] = {}
    pair_source: dict[str, set[str]] = defaultdict(set)
    for row in query_rows:
        pair_id = str(row.get("source_item_id") or row.get("source_pair_id") or "")
        if not pair_id:
            continue
        for field, field_values in attrs(row).items():
            pair_attrs[pair_id][field].update(field_values)
        if row.get("split") is not None:
            pair_split[pair_id] = str(row["split"])
        if source_dataset(row):
            pair_source[pair_id].add(str(source_dataset(row)))
    return dict(pair_attrs), pair_split, dict(pair_source)


def select_neighbors(
    pair_id: str,
    split: str,
    query_attrs: dict[str, set[str]],
    normalized_text: str,
    physical: dict[str, dict[str, Any]],
    pair_attrs: dict[str, dict[str, set[str]]],
    pair_split: dict[str, str],
    collision_groups: dict[str, dict[str, Any]],
    seed: int,
    count: int,
) -> list[dict[str, Any]]:
    collision = collision_groups.get(normalized_text)
    collision_ids = set((collision or {}).get("physical_item_ids", [])) - {pair_id}
    selected: list[tuple[str, str, int]] = []
    for candidate_id in sorted(collision_ids):
        if candidate_id in physical and pair_split.get(candidate_id) == split:
            selected.append((candidate_id, "normalized_text_collision_candidate", 0))
    candidate_ids = [
        candidate_id
        for candidate_id in pair_attrs
        if candidate_id != pair_id
        and candidate_id in physical
        and pair_split.get(candidate_id) == split
        and candidate_id not in collision_ids
    ]
    ranked = []
    for candidate_id in candidate_ids:
        score = overlap(query_attrs, pair_attrs[candidate_id])
        if score > 0:
            tie = hashlib.sha256(f"{seed}:{pair_id}:{candidate_id}".encode()).hexdigest()
            ranked.append((-score, tie, candidate_id))
    ranked.sort()
    for negative_score, _, candidate_id in ranked:
        selected.append((candidate_id, "attribute_overlap_candidate", -negative_score))
    if len(selected) < count:
        remaining = [
            candidate_id
            for candidate_id in physical
            if candidate_id != pair_id
            and pair_split.get(candidate_id) == split
            and candidate_id not in {item[0] for item in selected}
        ]
        remaining.sort(key=lambda item: hashlib.sha256(f"{seed}:{pair_id}:{item}".encode()).hexdigest())
        selected.extend((item, "same_split_fallback_candidate", 0) for item in remaining)
    return [
        {
            "candidate_id": candidate_id,
            "selection_reason": reason,
            "attribute_overlap_score": score,
            "t1_path": frame_paths(physical[candidate_id])[0],
            "t2_path": frame_paths(physical[candidate_id])[1],
            "reviewer_must_not_treat_as_positive": True,
        }
        for candidate_id, reason, score in selected[:count]
        if len(frame_paths(physical[candidate_id])) == 2
    ]


def review_row(
    row: dict[str, Any],
    physical: dict[str, dict[str, Any]],
    pair_attrs: dict[str, dict[str, set[str]]],
    pair_split: dict[str, str],
    collision_groups: dict[str, dict[str, Any]],
    seed: int,
    reviewer_id: str,
    args_output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a blind visible row plus an internal lineage row.

    The visible row intentionally contains no canonical query/pair IDs,
    source labels, source-derived path components, selection scores, or raw
    native paths.  Promotion receives the sibling internal join-ledger later;
    human reviewers receive only the visible row and blind image assets.
    """
    candidate = row.get("candidate") or {}
    pair_id = str(row.get("source_pair_id") or candidate.get("source_item_id") or "")
    if pair_id not in physical:
        raise RuntimeError(f"review pair missing from physical registry: {pair_id}")
    text = str(row.get("text") or candidate.get("text") or "")
    normalized_text = norm((candidate.get("provenance") or {}).get("normalized_text") or text)
    query_attrs = attrs(candidate)
    true_paths = frame_paths(physical[pair_id])
    if len(true_paths) != 2:
        raise RuntimeError(f"review pair does not have two native frames: {pair_id}")
    neighbors = select_neighbors(
        pair_id,
        pair_split.get(pair_id, str(candidate.get("split") or "")),
        query_attrs,
        normalized_text,
        physical,
        pair_attrs,
        pair_split,
        collision_groups,
        seed,
        3,
    )
    # A reviewer-specific order prevents copying a sheet while keeping the
    # underlying candidate set identical for agreement analysis.
    randomizer = random.Random(f"{seed}:{reviewer_id}:{row['review_sample_id']}")
    randomizer.shuffle(neighbors)
    sample_token = safe_asset_token(row["review_sample_id"])
    asset_root = Path("blind_assets") / sample_token / reviewer_id
    true_aliases = {
        "t1_path": str(asset_root / "intended_t1.png"),
        "t2_path": str(asset_root / "intended_t2.png"),
    }
    asset_modes: list[str] = []
    asset_modes.append(materialize_blind_asset(true_paths[0], args_output_dir / true_aliases["t1_path"]))
    asset_modes.append(materialize_blind_asset(true_paths[1], args_output_dir / true_aliases["t2_path"]))
    visible_neighbors = []
    internal_neighbors = []
    for index, neighbor in enumerate(neighbors, start=1):
        candidate_aliases = {
            "t1_path": str(asset_root / f"candidate_{index:02d}_t1.png"),
            "t2_path": str(asset_root / f"candidate_{index:02d}_t2.png"),
        }
        asset_modes.append(
            materialize_blind_asset(neighbor["t1_path"], args_output_dir / candidate_aliases["t1_path"])
        )
        asset_modes.append(
            materialize_blind_asset(neighbor["t2_path"], args_output_dir / candidate_aliases["t2_path"])
        )
        visible_neighbors.append(
            {
                "candidate_slot": f"candidate_{index:02d}",
                **candidate_aliases,
                "reviewer_must_not_treat_as_positive": True,
            }
        )
        internal_neighbors.append(
            {
                "candidate_slot": f"candidate_{index:02d}",
                "candidate_id": neighbor["candidate_id"],
                "selection_reason": neighbor["selection_reason"],
                "attribute_overlap_score": neighbor["attribute_overlap_score"],
                "t1_path": neighbor["t1_path"],
                "t2_path": neighbor["t2_path"],
                "blind_t1_path": candidate_aliases["t1_path"],
                "blind_t2_path": candidate_aliases["t2_path"],
            }
        )
    visible = {
        "schema_version": "qcpr-r20-exact-human-review-row-v2-blind",
        "review_sample_id": row["review_sample_id"],
        "stratum": next((name for name in STRATA if row["review_sample_id"].startswith(name)), None),
        "caption": text,
        "true_pair": {
            **true_aliases,
            "role": "intended_pair",
        },
        "candidate_neighbours": visible_neighbors,
        "model_scores_included": False,
        "rankings_included": False,
        "masks_included": False,
        "event_or_source_metadata_included": False,
        "visible_internal_ids": False,
        "paths_are_blind_aliases": True,
        "candidate_neighbours_are_not_labels": True,
        "review_questions": {
            "caption_accurately_describes_true_pair": "yes_no_uncertain",
            "caption_identifies_true_pair_against_neighbours": "yes_no_uncertain",
            "any_neighbour_fully_consistent_with_caption": "yes_no_uncertain",
            "final_decision": list(DECISIONS),
            "rewrite_if_required": "visible_T1_T2_evidence_only",
        },
        "reviewer_fields": {
            "reviewer_identity": None,
            "reviewed_at": None,
            "independence_attestation": None,
            "caption_accurate": None,
            "identifiable": None,
            "nonexact_candidate_consistent": None,
            "final_decision": None,
            "rewritten_caption": None,
            "confidence": None,
            "notes": None,
        },
    }
    internal = {
        "schema_version": "qcpr-r20-exact-human-review-join-v2",
        "review_sample_id": row["review_sample_id"],
        "query_id": row.get("query_id"),
        "source_pair_id": pair_id,
        "source_dataset": source_dataset(candidate),
        "split": pair_split.get(pair_id, candidate.get("split")),
        "stratum": visible["stratum"],
        "caption": text,
        "caption_normalized": normalized_text,
        "true_native_t1_path": true_paths[0],
        "true_native_t2_path": true_paths[1],
        "true_blind_t1_path": true_aliases["t1_path"],
        "true_blind_t2_path": true_aliases["t2_path"],
        "candidate_neighbours": internal_neighbors,
        "asset_modes": dict(sorted({mode: asset_modes.count(mode) for mode in set(asset_modes)}.items())),
    }
    return visible, internal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-release-sha", required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    release = args.release_dir
    physical_rows = read_jsonl(release / "registries/physical_items.jsonl")
    physical = {str(row["item_id"]): row for row in physical_rows}
    query_rows = read_jsonl(release / "registries/queries.jsonl")
    pair_attrs, pair_split, _ = build_neighbor_index(query_rows)
    collision_groups = {
        str(row.get("normalized_text") or ""): row
        for row in read_jsonl(release / "collision_groups.jsonl")
        if row.get("normalized_text")
    }
    packets_dir = release / "source_reports/human_review_packets"
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    source_selection: dict[str, Any] = {}
    for stratum in STRATA:
        if stratum == "exact_discriminative":
            selected, exact_source_counts = select_source_balanced_exact_rows(query_rows, args.seed)
            packet_rows = [review_input_row(row, stratum, index) for index, row in enumerate(selected)]
            source_hashes[stratum] = file_sha(release / "registries/queries.jsonl")
            source_selection[stratum] = {
                "policy": "150 unique train physical pairs per source; one human source-trusted exact caption per pair; deterministic SHA256 ordering",
                "source_counts": exact_source_counts,
                "source_registry_sha256": source_hashes[stratum],
            }
        else:
            path = packets_dir / f"{stratum}_300.jsonl"
            source_hashes[stratum] = file_sha(path)
            packet_rows = read_jsonl(path)
        if len(packet_rows) != 300:
            raise RuntimeError(f"expected 300 rows in {stratum}, got {len(packet_rows)}")
        for packet_row in packet_rows:
            packet_row = dict(packet_row)
            packet_row["review_sample_id"] = f"{stratum}:{packet_row['review_sample_id'].split(':')[-1]}"
            rows.append(packet_row)
    sample_ids = [row["review_sample_id"] for row in rows]
    if len(sample_ids) != 1500 or len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("calibration sample IDs are not unique at 1500-row grain")

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    reviewer_rows: dict[str, list[dict[str, Any]]] = {}
    internal_ledger: dict[str, dict[str, Any]] = {}
    for reviewer_id in ("reviewer_a", "reviewer_b"):
        visible_rows: list[dict[str, Any]] = []
        for row in rows:
            visible, internal = review_row(
                row,
                physical,
                pair_attrs,
                pair_split,
                collision_groups,
                args.seed,
                reviewer_id,
                args.output_dir,
            )
            visible_rows.append(visible)
            sample_id = str(internal["review_sample_id"])
            if sample_id not in internal_ledger:
                internal_ledger[sample_id] = {
                    key: value
                    for key, value in internal.items()
                    if key not in {"candidate_neighbours", "asset_modes"}
                }
                internal_ledger[sample_id]["reviewer_aliases"] = {}
                internal_ledger[sample_id]["candidate_neighbours_by_reviewer"] = {}
            internal_ledger[sample_id]["reviewer_aliases"][reviewer_id] = {
                "true_blind_t1_path": internal["true_blind_t1_path"],
                "true_blind_t2_path": internal["true_blind_t2_path"],
                "asset_modes": internal["asset_modes"],
            }
            internal_ledger[sample_id]["candidate_neighbours_by_reviewer"][reviewer_id] = internal["candidate_neighbours"]
        reviewer_rows[reviewer_id] = visible_rows
    for output_rows, name in ((reviewer_rows["reviewer_a"], "reviewer_a"), (reviewer_rows["reviewer_b"], "reviewer_b")):
        path = args.output_dir / f"{name}_packet.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in output_rows), encoding="utf-8")
        decision_path = args.output_dir / f"{name}_decision_template.jsonl"
        decision_rows = []
        for row in output_rows:
            decision_rows.append({
                "schema_version": "qcpr-r20-exact-human-decision-v1",
                "review_sample_id": row["review_sample_id"],
                "reviewer_id": name,
                "reviewer_identity": None,
                "reviewer_type": "human_required",
                "reviewed_at": None,
                "independence_attestation": None,
                "caption_accurate": None,
                "identifiable_against_neighbours": None,
                "nonexact_candidate_consistent": None,
                "final_decision": None,
                "allowed_final_decisions": list(DECISIONS),
                "rewritten_caption": None,
                "rewrite_evidence": "visible_T1_T2_only",
                "confidence": None,
                "notes": None,
            })
        decision_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in decision_rows), encoding="utf-8")

    adjudication_rows = []
    for row in rows:
        adjudication_rows.append({
            "schema_version": "qcpr-r20-exact-adjudication-v1",
            "review_sample_id": row["review_sample_id"],
            "reviewer_a_decision": None,
            "reviewer_b_decision": None,
            "final_decision": None,
            "adjudicator_identity": None,
            "adjudicated_at": None,
            "rewritten_caption": None,
            "status": "PENDING_TWO_INDEPENDENT_HUMAN_REVIEWS",
        })
    (args.output_dir / "adjudication_template.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in adjudication_rows), encoding="utf-8"
    )

    join_ledger_path = args.output_dir.parent / "internal_review_join_ledger.jsonl"
    join_ledger_rows = [internal_ledger[sample_id] for sample_id in sorted(internal_ledger)]
    join_ledger_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in join_ledger_rows
        ),
        encoding="utf-8",
    )
    join_ledger_sha256 = file_sha(join_ledger_path)

    readme = f"""# QCPR r20 exact-supervision human review package

Status: **READY_FOR_TWO_INDEPENDENT_HUMAN_REVIEWERS**

This package is derived from immutable r19g release `{args.dataset_release_sha}`.
It contains 1,500 rows: 300 exact-discriminative (source-balanced across the
two core sources, with unique physical pairs), 300 semantic-multi-positive,
300 generic-no-change, 300 stable-scene-specific, and 300 localized-direction.

Review T1 and T2 for the intended pair, then inspect the candidate neighbours.
The visible sheets contain no model scores, rankings, masks, event IDs,
canonical IDs, source labels, or native source paths. Image paths are blind
aliases inside `blind_assets/`; candidate neighbours are sampling aids only
and are never automatically positive or negative. The internal join ledger is
kept outside the reviewer package and is not part of the human review sheet.

For each row answer independently:

1. Does the caption accurately describe the intended T1→T2 pair?
2. Does it identify this pair rather than many alternatives?
3. Is any candidate neighbour also fully consistent with the caption?
4. Choose exactly one: `{", ".join(DECISIONS)}`.

`REWRITE_REQUIRED` captions must be rewritten using visible T1/T2 evidence only.
No masks, coordinates unavailable to a human viewer, event metadata, source names,
or model output may be used. Two distinct human identities, timestamps, and
independence attestations are required. Codex/agent inspection is not human review.

Files:

- `reviewer_a_packet.jsonl`, `reviewer_b_packet.jsonl`: independent visual sheets
- `reviewer_a_decision_template.jsonl`, `reviewer_b_decision_template.jsonl`: decision sheets
- `adjudication_template.jsonl`: final adjudication after both sheets are returned
- `blind_assets/`: reviewer-facing anonymous image files
- `review_package_audit.json`: lineage and hash audit

The Dataset Agent retains `../internal_review_join_ledger.jsonl` for promotion
and source-conditioned analysis; reviewers should not use or receive it.

No r20 training manifest is created by this package. Promotion remains disabled
until human decisions and adjudication pass the contract.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    blind_asset_count, blind_asset_manifest_sha256 = write_blind_asset_manifest(
        args.output_dir / "blind_assets",
        args.output_dir / "blind_assets_manifest.jsonl",
    )
    audit = {
        "schema_version": "qcpr-r20-exact-human-review-package-audit-v1",
        "status": "READY_FOR_TWO_INDEPENDENT_HUMAN_REVIEWERS",
        "producer_agent": "DATASET_AGENT",
        "producer_git_sha": args.producer_git_sha,
        "dataset_release_name": "qcpr_bitemporal_v2_train_20260808_final_r19g",
        "dataset_release_sha": args.dataset_release_sha,
        "created_at": created_at,
        "sample_seed": args.seed,
        "row_count": 1500,
        "stratum_counts": {stratum: 300 for stratum in STRATA},
        "reviewer_count_required": 2,
        "reviewer_decisions_completed": 0,
        "adjudicated_rows": 0,
        "model_scores_included": False,
        "rankings_included": False,
        "masks_included": False,
        "event_or_source_metadata_included": False,
        "automatic_labels_created": False,
        "candidate_neighbours_are_not_relevance_labels": True,
        "source_packet_sha256": source_hashes,
        "source_balance_policy": "exact stratum is source-balanced; source identities are withheld from reviewers",
        "blind_asset_count": blind_asset_count,
        "blind_asset_manifest_sha256": blind_asset_manifest_sha256,
        "r19_train_manifest_sha256": file_sha(release / "exact_core_train.jsonl"),
        "r19_development_manifest_sha256": file_sha(release / "exact_core_development.jsonl"),
        "r19_test_manifest_sha256": file_sha(release / "exact_core_test.jsonl"),
        "r19_immutable_required": True,
        "r20_training_promotion": "DISABLED_PENDING_HUMAN_REVIEW_AND_ADJUDICATION",
        "artifact_hash_definition": "SHA256 of UTF-8 canonical JSON excluding artifact_sha256, with one trailing newline",
    }
    audit["artifact_sha256"] = canonical_sha(audit)
    (args.output_dir / "review_package_audit.json").write_bytes(canonical_bytes(audit))
    source = {
        "schema_version": "qcpr-r20-exact-supervision-source-lineage-v1",
        "dataset_release_sha": args.dataset_release_sha,
        "producer_git_sha": args.producer_git_sha,
        "created_at": created_at,
        "r19_immutable": True,
        "review_package_audit_sha256": file_sha(args.output_dir / "review_package_audit.json"),
        "reviewer_a_packet_sha256": file_sha(args.output_dir / "reviewer_a_packet.jsonl"),
        "reviewer_b_packet_sha256": file_sha(args.output_dir / "reviewer_b_packet.jsonl"),
        "reviewer_a_decision_template_sha256": file_sha(args.output_dir / "reviewer_a_decision_template.jsonl"),
        "reviewer_b_decision_template_sha256": file_sha(args.output_dir / "reviewer_b_decision_template.jsonl"),
        "adjudication_template_sha256": file_sha(args.output_dir / "adjudication_template.jsonl"),
        "blind_asset_manifest_sha256": blind_asset_manifest_sha256,
        "blind_asset_count": blind_asset_count,
        "internal_review_join_ledger_sha256": join_ledger_sha256,
        "exact_source_counts": source_selection["exact_discriminative"]["source_counts"],
        "exact_source_registry_sha256": source_selection["exact_discriminative"]["source_registry_sha256"],
        "status": "PENDING_EXTERNAL_HUMAN_REVIEW",
    }
    source["artifact_sha256"] = canonical_sha(source)
    (args.output_dir.parent / "source_lineage.json").write_bytes(canonical_bytes(source))
    print(json.dumps({
        "status": audit["status"],
        "rows": audit["row_count"],
        "strata": audit["stratum_counts"],
        "model_scores_included": False,
        "reviewer_decisions_completed": 0,
        "audit_artifact_sha256": audit["artifact_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
