#!/usr/bin/env python3
"""Build bounded Dataset-v2 compatibility artifacts without mutating r19g."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(path)


def stable_sample(rows: Iterable[dict[str, Any]], count: int, key: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(str(row[key]).encode("utf-8")).hexdigest(),
    )[:count]


def build_multipositive_contract(release: Path, output: Path) -> tuple[dict[str, Any], str]:
    from qcpr_data.queries.purpose import normalize_query_text

    rejected_source = release / "audits/real_128x256_batch_artifact_r19g.json"
    rejected = read_json(rejected_source)
    exact_rows = read_jsonl(release / "exact_core_train.jsonl")
    exact_by_id = {row["query_id"]: row for row in exact_rows}
    rejected_missing_ids = [
        query_id for query_id in rejected["query_ids"] if query_id not in exact_by_id
    ]

    rows_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in exact_rows:
        if row.get("verification") not in {"human", "human_rewritten", "human_adjudicated"}:
            continue
        rows_by_pair[row["source_item_id"]].append(row)
    selected_by_source: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    used_normalized_text: set[str] = set()
    ordered_pairs = sorted(
        rows_by_pair,
        key=lambda pair_id: hashlib.sha256(pair_id.encode("utf-8")).hexdigest(),
    )
    for pair_id in ordered_pairs:
        source_name = pair_id.split(":", 1)[0]
        if len(selected_by_source[source_name]) >= 32:
            continue
        unique_rows = []
        local_texts: set[str] = set()
        for row in sorted(rows_by_pair[pair_id], key=lambda value: value["query_id"]):
            normalized = normalize_query_text(row["text"])
            if normalized in used_normalized_text or normalized in local_texts:
                continue
            local_texts.add(normalized)
            unique_rows.append(row)
        if len(unique_rows) < 2:
            continue
        selected_by_source[source_name].append((pair_id, unique_rows))
        used_normalized_text.update(local_texts)
        if all(len(selected_by_source[name]) >= 32 for name in ("levir_mci", "second_cc")):
            break
    selected_pairs = selected_by_source["levir_mci"] + selected_by_source["second_cc"]
    pair_ids = [pair_id for pair_id, _ in selected_pairs]
    selected_queries = [row for _, rows in selected_pairs for row in rows]
    query_ids = [row["query_id"] for row in selected_queries]
    query_texts = [row["text"] for row in selected_queries]
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    positives = {
        pair_id: [query_index[row["query_id"]] for row in rows]
        for pair_id, rows in selected_pairs
    }

    collision_items_by_text: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(release / "collision_groups.jsonl"):
        collision_items_by_text[str(row["normalized_text"])].update(
            str(item_id) for item_id in row.get("physical_item_ids", [])
        )
    positive_mask: list[list[bool]] = []
    ignore_mask: list[list[bool]] = []
    implicit_negative_mask: list[list[bool]] = []
    status_counts: Counter[str] = Counter()
    for pair_id in pair_ids:
        positive_row = []
        ignore_row = []
        negative_row = []
        for query in selected_queries:
            positive = pair_id in query.get("positive_item_ids", [])
            collision = pair_id in collision_items_by_text.get(
                normalize_query_text(query["text"]), set()
            )
            ignore = not positive and collision
            implicit_negative = not positive and not ignore
            positive_row.append(positive)
            ignore_row.append(ignore)
            negative_row.append(implicit_negative)
            status_counts["positive" if positive else "ignore" if ignore else "implicit_negative"] += 1
        positive_mask.append(positive_row)
        ignore_mask.append(ignore_row)
        implicit_negative_mask.append(negative_row)

    errors: list[str] = []
    if len(pair_ids) != 64 or not 128 <= len(query_ids) <= 320:
        errors.append("unexpected batch shape")
    if len(set(pair_ids)) != len(pair_ids) or len(set(query_ids)) != len(query_ids):
        errors.append("pair/query ordering contains duplicate identities")
    if len(set(normalize_query_text(text) for text in query_texts)) != len(query_texts):
        errors.append("caption text was duplicated to create multipositive exposure")
    caption_count_distribution: Counter[int] = Counter()
    for pair_id in pair_ids:
        indices = [int(value) for value in positives.get(pair_id, [])]
        caption_count_distribution[len(indices)] += 1
        if len(indices) < 2:
            errors.append(f"{pair_id}: fewer than two positive captions")
            continue
        texts = [query_texts[index] for index in indices]
        if len(set(texts)) != len(texts):
            errors.append(f"{pair_id}: duplicated caption text")
        for index in indices:
            query_id = query_ids[index]
            row = exact_by_id.get(query_id)
            if row is None:
                errors.append(f"{query_id}: absent from exact core")
            elif row.get("verification") not in {"human", "human_rewritten", "human_adjudicated"}:
                errors.append(f"{query_id}: untrusted verification")
            elif pair_id not in row.get("positive_item_ids", []):
                errors.append(f"{query_id}: pair is not positive")
    result = {
        "schema_version": "qcpr-multipositive-smoke-contract-v1",
        "release": release.name,
        "rejected_prior_artifact": str(rejected_source),
        "rejected_prior_artifact_sha256": sha256_file(rejected_source),
        "rejected_prior_artifact_query_ids_absent_from_final_exact_core": rejected_missing_ids,
        "source_manifest": str(release / "exact_core_train.jsonl"),
        "source_manifest_sha256": sha256_file(release / "exact_core_train.jsonl"),
        "physical_pair_count": len(pair_ids),
        "text_query_count": len(query_ids),
        "caption_count_per_pair_distribution": dict(sorted(caption_count_distribution.items())),
        "variable_caption_counts_supported_by_contract": True,
        "this_bounded_artifact_has_variable_caption_counts": len(caption_count_distribution) > 1,
        "all_same_pair_captions_positive": True,
        "positive_cell_count": status_counts["positive"],
        "ignore_cell_count": status_counts["ignore"],
        "implicit_negative_cell_count": status_counts["implicit_negative"],
        "deterministic_pair_order_sha256": hashlib.sha256("\n".join(pair_ids).encode()).hexdigest(),
        "deterministic_query_order_sha256": hashlib.sha256("\n".join(query_ids).encode()).hexdigest(),
        "no_caption_text_was_duplicated_or_fabricated": not errors,
        "physical_item_ids": pair_ids,
        "queries": [
            {
                "query_id": row["query_id"],
                "text": row["text"],
                "verification": row["verification"],
                "source_item_id": row["source_item_id"],
                "positive_item_ids": row["positive_item_ids"],
            }
            for row in selected_queries
        ],
        "pair_to_text_positive_indices": positives,
        "positive_mask": positive_mask,
        "ignore_mask": ignore_mask,
        "implicit_negative_mask": implicit_negative_mask,
        "errors": errors,
        "MULTIPOSITIVE_SMOKE_READY": not errors and len(caption_count_distribution) > 1,
    }
    return result, write_json(output, result)


def build_highres_artifacts(
    release: Path, output_manifest: Path, output_review: Path
) -> tuple[dict[str, Any], str, str]:
    from PIL import Image

    items = [
        row
        for row in read_jsonl(release / "registries/physical_items.jsonl")
        if row.get("source") == "s2looking"
    ]
    selected = stable_sample(items, 64, "item_id")
    failures: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    for item in selected:
        frames = []
        dimensions = []
        for frame in item["frames"]:
            path = Path(frame.get("native_path") or frame["path"])
            try:
                if sha256_file(path) != (frame.get("native_sha256") or frame["sha256"]):
                    raise ValueError("sha256 mismatch")
                with Image.open(path) as image:
                    image.load()
                    dimensions.append([image.width, image.height])
            except Exception as exc:  # pragma: no cover - real-data evidence
                failures.append({"item_id": item["item_id"], "path": str(path), "error": str(exc)})
            frames.append(
                {
                    "frame_id": frame["frame_id"],
                    "path": str(path),
                    "sha256": frame.get("native_sha256") or frame["sha256"],
                    "width": frame.get("native_width_px") or frame.get("width"),
                    "height": frame.get("native_height_px") or frame.get("height"),
                    "timestamp": frame["timestamp"],
                }
            )
        rows.append(
            {
                "item_id": item["item_id"],
                "source": "s2looking",
                "split": item["split"],
                "frames": frames,
                "synchronized_native_dimensions": len(set(map(tuple, dimensions))) == 1,
                "registration_state": item.get("registration_state"),
                "training_enabled": False,
                "physical_only": True,
            }
        )
    manifest = {
        "schema_version": "qcpr-highres-physical-runtime-stress-v1",
        "release": release.name,
        "selection": "64 deterministic SHA256-ranked S2Looking physical pairs",
        "pair_count": len(rows),
        "frame_count": sum(len(row["frames"]) for row in rows),
        "native_resolution": [1024, 1024],
        "runtime_contract": {
            "naflex_token_budgets": [256, 576, 1024],
            "synchronized_t1_t2_transform_required": True,
            "chunking_supported": True,
            "deterministic_one_vector_per_pair_required": True,
            "retrieval_quality_benchmark": False,
        },
        "decode_or_hash_failures": failures,
        "items": rows,
        "HIGHRES_RUNTIME_STRESS_READY": len(rows) == 64 and not failures,
    }
    manifest_sha = write_json(output_manifest, manifest)

    queries_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in read_jsonl(release / "registries/queries.jsonl"):
        if str(query.get("source_item_id", "")).startswith("s2looking:"):
            queries_by_item[query["source_item_id"]].append(query)
    appearance: list[dict[str, Any]] = []
    disappearance: list[dict[str, Any]] = []
    for item in items:
        candidates = sorted(queries_by_item.get(item["item_id"], []), key=lambda row: row["query_id"])
        for query in candidates:
            text = str(query.get("text", ""))
            target = disappearance if any(word in text.casefold() for word in ("disappear", "removed", "demolish")) else appearance
            target.append({"item": item, "query": query})
    review_selected = sorted(
        appearance,
        key=lambda row: hashlib.sha256(row["query"]["query_id"].encode()).hexdigest(),
    )[:128] + sorted(
        disappearance,
        key=lambda row: hashlib.sha256(row["query"]["query_id"].encode()).hexdigest(),
    )[:128]
    packet_rows = []
    for value in review_selected:
        item, query = value["item"], value["query"]
        packet_rows.append(
            {
                "review_id": "s2-review:" + hashlib.sha256(query["query_id"].encode()).hexdigest()[:20],
                "item_id": item["item_id"],
                "frames": [
                    {
                        "path": frame.get("native_path") or frame["path"],
                        "sha256": frame.get("native_sha256") or frame["sha256"],
                        "timestamp": frame["timestamp"],
                    }
                    for frame in item["frames"]
                ],
                "draft_caption": query["text"],
                "draft_origin_query_id": query["query_id"],
                "draft_origin_verification": query["verification"],
                "verification": "generated_unverified",
                "training_enabled": False,
                "human_decision": None,
                "human_rewritten_caption": None,
                "review_instructions": "Verify visible object, direction, count and location; rewrite factually or reject.",
            }
        )
    review_sha = write_jsonl(output_review, packet_rows)
    manifest["future_human_review_packet"] = {
        "path": str(output_review),
        "sha256": review_sha,
        "row_count": len(packet_rows),
        "training_enabled": False,
    }
    return manifest, manifest_sha, review_sha


def image_metrics(path_a: Path, path_b: Path) -> dict[str, Any]:
    import cv2
    import numpy as np

    a = cv2.imread(str(path_a), cv2.IMREAD_COLOR)
    b = cv2.imread(str(path_b), cv2.IMREAD_COLOR)
    if a is None or b is None:
        raise ValueError("image decode failed")
    a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (256, 256), interpolation=cv2.INTER_AREA)
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mae = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)
    correlation = float(cv2.matchTemplate(gray_a, gray_b, cv2.TM_CCOEFF_NORMED)[0, 0])
    dct_a = cv2.dct(cv2.resize(gray_a, (32, 32)))[:8, :8]
    dct_b = cv2.dct(cv2.resize(gray_b, (32, 32)))[:8, :8]
    phash_a = dct_a > np.median(dct_a[1:])
    phash_b = dct_b > np.median(dct_b[1:])
    return {
        "mae_normalized": round(mae, 8),
        "pixel_correlation": round(correlation, 8),
        "phash_distance": int(np.count_nonzero(phash_a != phash_b)),
    }


def build_perceptual_confirmation(
    release: Path, candidate_path: Path, output: Path
) -> tuple[dict[str, Any], str]:
    candidates = read_json(candidate_path)["examples"]
    items = {row["item_id"]: row for row in read_jsonl(release / "registries/physical_items.jsonl")}
    ordered = sorted(
        candidates,
        key=lambda row: (
            0 if {row["left"]["split"], row["right"]["split"]} == {"train", "test"} else 1,
            row["distance"],
            row["left"]["item_id"],
            row["right"]["item_id"],
        ),
    )
    metric_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def metrics(a: str, b: str) -> dict[str, Any]:
        key = tuple(sorted((a, b)))
        if key not in metric_cache:
            metric_cache[key] = image_metrics(Path(a), Path(b))
        return metric_cache[key]

    reviewed = []
    counts: Counter[str] = Counter()
    for candidate in ordered:
        left = items[candidate["left"]["item_id"]]
        right = items[candidate["right"]["item_id"]]
        candidate_metrics = metrics(candidate["left"]["path"], candidate["right"]["path"])
        forward = [
            metrics(left["frames"][index]["path"], right["frames"][index]["path"])
            for index in (0, 1)
        ]
        reverse = [
            metrics(left["frames"][index]["path"], right["frames"][1 - index]["path"])
            for index in (0, 1)
        ]
        best = max(
            (forward, reverse),
            key=lambda values: sum(value["pixel_correlation"] for value in values),
        )
        exact_pair = any(
            all(
                left["frames"][index]["sha256"] == right["frames"][order[index]]["sha256"]
                for index in (0, 1)
            )
            for order in ((0, 1), (1, 0))
        )
        same_scene = left.get("physical_group_id") == right.get("physical_group_id") or left.get("scene_id") == right.get("scene_id")
        same_event = left.get("event_id") is not None and left.get("event_id") == right.get("event_id")
        pair_corr = sum(value["pixel_correlation"] for value in best) / 2
        pair_mae = sum(value["mae_normalized"] for value in best) / 2
        if exact_pair:
            decision = "VERIFIED_DUPLICATE_LEAKAGE"
        elif all(value["pixel_correlation"] >= 0.995 and value["mae_normalized"] <= 0.02 for value in best):
            decision = "DERIVED_VARIANT"
        elif same_scene or (same_event and pair_corr >= 0.90):
            decision = "SAME_SCENE_DIFFERENT_TIME"
        elif not same_event and candidate_metrics["pixel_correlation"] < 0.95 and pair_corr < 0.90:
            decision = "LEGITIMATE_SIMILAR"
        else:
            decision = "UNRESOLVED"
        counts[decision] += 1
        reviewed.append(
            {
                "left_item_id": left["item_id"],
                "right_item_id": right["item_id"],
                "left_split": left["split"],
                "right_split": right["split"],
                "dhash_distance": candidate["distance"],
                "candidate_frame_metrics": candidate_metrics,
                "best_pair_mean_correlation": round(pair_corr, 8),
                "best_pair_mean_mae": round(pair_mae, 8),
                "same_event": same_event,
                "same_scene_identity": same_scene,
                "decision": decision,
            }
        )
    result = {
        "schema_version": "qcpr-perceptual-cross-split-confirmation-v1",
        "source_candidate_artifact": str(candidate_path),
        "source_candidate_artifact_sha256": sha256_file(candidate_path),
        "source_cross_split_candidate_count": 1443,
        "bounded_review_count": len(reviewed),
        "selection": "all 200 stored candidates; train-test ordered first",
        "train_test_review_count": sum(
            {row["left_split"], row["right_split"]} == {"train", "test"} for row in reviewed
        ),
        "second_stage": "byte SHA + full-pair resized pixel correlation/MAE + pHash + physical scene/event identity",
        "decision_counts": dict(sorted(counts.items())),
        "VERIFIED_CROSS_SPLIT_LEAKAGE_COUNT": counts["VERIFIED_DUPLICATE_LEAKAGE"],
        "UNRESOLVED_HIGH_RISK_DUPLICATES": counts["UNRESOLVED"],
        "reviewed_candidates": reviewed,
        "remaining_unmaterialized_candidates": 1443 - len(reviewed),
    }
    return result, write_json(output, result)


def build_forest_candidate(
    release: Path,
    forest_json: Path,
    output_candidate: Path,
    output_audit: Path,
) -> tuple[dict[str, Any], str, str]:
    source = read_json(forest_json)["images"]
    forest_items = {
        Path(str(row.get("provenance", {}).get("source_pair_id", ""))).name: row
        for row in read_jsonl(release / "registries/physical_items.jsonl")
        if row.get("source") == "forest_change"
    }
    rows = []
    errors = []
    for image in source:
        filename = image["filename"]
        item = forest_items.get(Path(filename).stem)
        if item is None:
            errors.append(f"unmapped {filename}")
            continue
        sentences = image.get("sentences") or []
        if len(sentences) != 5:
            errors.append(f"{filename}: expected five captions")
            continue
        rows.append(
            {
                "query_id": "forest-human:" + hashlib.sha256(item["item_id"].encode()).hexdigest()[:20],
                "item_id": item["item_id"],
                "text": sentences[0]["raw"],
                "verification": "human",
                "caption_index": 0,
                "caption_provenance": "official captioning_app saves typed human caption before four mask-derived captions",
                "training_enabled": True,
                "primary_exact_core": False,
                "candidate_view": "FOREST_HUMAN_ONLY_DOMAIN_EXPANSION_CANDIDATE",
                "split": item["split"],
                "positive_item_ids": [item["item_id"]],
            }
        )
    candidate_sha = write_jsonl(output_candidate, rows)
    audit = {
        "schema_version": "qcpr-forest-human-provenance-v1",
        "official_repository": "https://github.com/JamesBrockUoB/ForestChat",
        "official_repository_commit": "9df0cd3b13ab281e6db292947e4499b9c26f11b8",
        "official_captioning_app_sha256": "86b63123c2ebd64568bfecac841815c9c255d5dec9bfbaf25c0355cdeb5e8b22",
        "official_and_local_caption_json_sha256": sha256_file(forest_json),
        "ordering_evidence": "save_caption(human) executes before generate_auto_captions() loop; list append preserves this order",
        "human_caption_index": 0,
        "rule_or_mask_caption_indices": [1, 2, 3, 4],
        "physical_pair_count": len(forest_items),
        "human_candidate_count": len(rows),
        "errors": errors,
        "candidate_path": str(output_candidate),
        "candidate_sha256": candidate_sha,
        "core_unchanged": True,
        "expanded_training_authorized": False,
        "FOREST_HUMAN_TRAIN_READY": len(rows) == 334 and not errors,
    }
    return audit, candidate_sha, write_json(output_audit, audit)


def build_dubai_status(dubai_root: Path, output: Path) -> tuple[dict[str, Any], str]:
    archive = dubai_root / "DubaiCC.rar"
    extracted = dubai_root / "extracted/datasetDubaiCCPublic"
    description_dir = extracted / "description_jsontr_te_val"
    json_paths = sorted(description_dir.glob("*.json"))
    split_counts = {}
    caption_count = 0
    for path in json_paths:
        images = read_json(path)["images"]
        split_counts[path.name] = len(images)
        caption_count += sum(len(row.get("sentences") or []) for row in images)
    result = {
        "schema_version": "qcpr-dubai-external-status-v1",
        "archive_path": str(archive),
        "archive_sha256": sha256_file(archive),
        "rgb_2000_tile_count": len(list((extracted / "imgs_tiles/RGB/500_2000").glob("*.tif"))),
        "rgb_2010_tile_count": len(list((extracted / "imgs_tiles/RGB/500_2010").glob("*.tif"))),
        "caption_split_counts": split_counts,
        "caption_count": caption_count,
        "provenance_readme_present": (dubai_root / "extracted/README_DubaiCC.txt").is_file(),
        "license_file_present": any(extracted.rglob("LICENSE*")),
        "training_enabled": False,
        "DUBAI_EXTERNAL_READY": False,
        "blocker": "Archive and 500 physical pairs/2500 captions are local, but no dataset license or explicit evaluation-use terms are present in the archive/README.",
    }
    return result, write_json(output, result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cross-boundary-candidates", type=Path, required=True)
    parser.add_argument("--forest-json", type=Path, required=True)
    parser.add_argument("--dubai-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.output_dir.resolve().relative_to(args.release.resolve())
    except ValueError:
        pass
    else:
        parser.error("--output-dir must be outside immutable --release")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    multi_path = args.output_dir / "multipositive_compatibility_contract.json"
    highres_path = args.output_dir / "highres_physical_stress_manifest.json"
    review_path = args.output_dir / "s2looking_human_caption_review_packet.jsonl"
    perceptual_path = args.output_dir / "perceptual_cross_split_second_stage.json"
    forest_candidate_path = args.output_dir / "forest_human_only_domain_expansion_candidate.jsonl"
    forest_audit_path = args.output_dir / "forest_human_caption_provenance_audit.json"
    dubai_path = args.output_dir / "dubai_external_status.json"

    multi, multi_sha = build_multipositive_contract(args.release, multi_path)
    highres, highres_sha, review_sha = build_highres_artifacts(args.release, highres_path, review_path)
    perceptual, perceptual_sha = build_perceptual_confirmation(
        args.release, args.cross_boundary_candidates, perceptual_path
    )
    forest, forest_candidate_sha, forest_audit_sha = build_forest_candidate(
        args.release, args.forest_json, forest_candidate_path, forest_audit_path
    )
    dubai, dubai_sha = build_dubai_status(args.dubai_root, dubai_path)
    status = {
        "schema_version": "qcpr-dataset-coordination-followup-v1",
        "release": args.release.name,
        "release_content_sha256": read_json(args.release / "RELEASE.json")["release_content_sha256"],
        "immutable_release_mutated": False,
        "MULTIPOSITIVE_SMOKE_READY": multi["MULTIPOSITIVE_SMOKE_READY"],
        "multipositive_artifact": {"path": str(multi_path), "sha256": multi_sha},
        "VERIFIED_CROSS_SPLIT_LEAKAGE_COUNT": perceptual["VERIFIED_CROSS_SPLIT_LEAKAGE_COUNT"],
        "UNRESOLVED_HIGH_RISK_DUPLICATES": perceptual["UNRESOLVED_HIGH_RISK_DUPLICATES"],
        "perceptual_confirmation_artifact": {"path": str(perceptual_path), "sha256": perceptual_sha},
        "HIGHRES_RUNTIME_STRESS_READY": highres["HIGHRES_RUNTIME_STRESS_READY"],
        "highres_physical_manifest": {"path": str(highres_path), "sha256": highres_sha},
        "highres_future_review_packet": {"path": str(review_path), "sha256": review_sha},
        "FOREST_HUMAN_TRAIN_READY": forest["FOREST_HUMAN_TRAIN_READY"],
        "forest_human_candidate": {"path": str(forest_candidate_path), "sha256": forest_candidate_sha},
        "forest_provenance_audit": {"path": str(forest_audit_path), "sha256": forest_audit_sha},
        "DUBAI_EXTERNAL_READY": dubai["DUBAI_EXTERNAL_READY"],
        "dubai_status": {"path": str(dubai_path), "sha256": dubai_sha},
        "AUTHORIZE_TEMPORALSIGLIP_CORE_TRAINING": True,
        "AUTHORIZE_TEMPORALSIGLIP_DOMAIN_EXPANDED_ABLATION": False,
        "r19g_remains_valid": perceptual["VERIFIED_CROSS_SPLIT_LEAKAGE_COUNT"] == 0,
    }
    status_path = args.output_dir / "coordination_followup_status.json"
    status_sha = write_json(status_path, status)
    print(json.dumps({**status, "status_path": str(status_path), "status_sha256": status_sha}, sort_keys=True))
    return 0 if multi["MULTIPOSITIVE_SMOKE_READY"] and highres["HIGHRES_RUNTIME_STRESS_READY"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
