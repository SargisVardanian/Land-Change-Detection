#!/usr/bin/env python3
"""Build the bounded Forest human-only research extension outside immutable r19g."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_COMMIT = "9df0cd3b13ab281e6db292947e4499b9c26f11b8"
UPSTREAM_README_URL = (
    "https://raw.githubusercontent.com/JamesBrockUoB/ForestChat/"
    f"{UPSTREAM_COMMIT}/README.md"
)
UPSTREAM_LICENSE_URL = (
    "https://raw.githubusercontent.com/JamesBrockUoB/ForestChat/"
    f"{UPSTREAM_COMMIT}/LICENSE"
)
UPSTREAM_APP_URL = (
    "https://raw.githubusercontent.com/JamesBrockUoB/ForestChat/"
    f"{UPSTREAM_COMMIT}/Multi_change/captioning_app.py"
)
UPSTREAM_README_SHA256 = "70d902294e75e8aa1c381d8297c2a0831276f33cf5fccfee73e0676fa866b212"
UPSTREAM_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
UPSTREAM_APP_SHA256 = "86b63123c2ebd64568bfecac841815c9c255d5dec9bfbaf25c0355cdeb5e8b22"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> str:
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


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def numeric_stats(values: list[int]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": round(statistics.fmean(values), 6) if values else None,
        "median": round(statistics.median(values), 6) if values else None,
        "p95": percentile(values, 0.95),
    }


def forest_split_leakage(items: list[dict[str, Any]]) -> dict[str, Any]:
    signatures: dict[str, dict[str, set[str]]] = {
        "physical_item_id": defaultdict(set),
        "physical_group_id": defaultdict(set),
        "frame_sha256": defaultdict(set),
        "pair_frame_sha256": defaultdict(set),
    }
    for item in items:
        split = item["split"]
        signatures["physical_item_id"][item["item_id"]].add(split)
        signatures["physical_group_id"][item["physical_group_id"]].add(split)
        frame_hashes = [frame["sha256"] for frame in item["frames"]]
        for frame_hash in frame_hashes:
            signatures["frame_sha256"][frame_hash].add(split)
        pair_signature = hashlib.sha256("\n".join(sorted(frame_hashes)).encode()).hexdigest()
        signatures["pair_frame_sha256"][pair_signature].add(split)
    collisions = {
        name: sorted(identity for identity, splits in values.items() if len(splits) > 1)
        for name, values in signatures.items()
    }
    return {
        "checks": {name: len(values) == 0 for name, values in collisions.items()},
        "collision_counts": {name: len(values) for name, values in collisions.items()},
        "collision_examples": {name: values[:20] for name, values in collisions.items()},
        "PASS": all(not values for values in collisions.values()),
    }


def forest_query(item: dict[str, Any], image: dict[str, Any]) -> dict[str, Any]:
    text = str(image["sentences"][0]["raw"]).strip()
    query_id = "forest-human:" + hashlib.sha256(item["item_id"].encode()).hexdigest()[:20]
    return {
        "attributes": {},
        "candidate_training_enabled": item["split"] == "train",
        "canonical_query_id": query_id,
        "caption_index": 0,
        "caption_provenance": {
            "human_authored": True,
            "mask_free": True,
            "review_status": "source_human_caption_order_verified",
            "source_dataset": "forest_change",
            "source_revision": UPSTREAM_COMMIT,
            "type": "source_authored",
        },
        "graded_relevance": {item["item_id"]: 3},
        "localized_relation": None,
        "positive_item_ids": [item["item_id"]],
        "provenance": {
            "caption_source": "human",
            "caption_source_index": 0,
            "forest_license_status": "PARTIAL",
            "mask_derived": False,
            "source_dataset": "forest_change",
            "source_pair_id": item["provenance"]["source_pair_id"],
        },
        "query_classification": "domain_expansion_human_pair",
        "query_id": query_id,
        "query_scope": "domain_expansion_human_pair",
        "release_lineage": {
            "core_release_unchanged": True,
            "extension": "CORE_PLUS_FOREST_HUMAN_TRAIN",
        },
        "roles": ["forest_human_only", "domain_expanded_ablation"],
        "source_item_id": item["item_id"],
        "source_pair_id": item["item_id"],
        "split": item["split"],
        "temporal_direction": "t1_to_t2",
        "text": text,
        "training_enabled": item["split"] == "train",
        "training_gate": "domain_expanded_ablation_research_only",
        "verification": "human",
        "view_scope": "forest_human_only_domain_expansion",
        "view_status": "READY_RESEARCH_ONLY" if item["split"] == "train" else "EVAL_ONLY",
    }


def source_id(row: dict[str, Any]) -> str:
    return str(
        row.get("caption_provenance", {}).get("source_dataset")
        or row.get("provenance", {}).get("source_dataset")
        or row["source_item_id"].split(":", 1)[0]
    )


def build_source_controls(
    rows_by_split: dict[str, list[dict[str, Any]]],
    physical_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    records = []
    for split in ("train", "development", "test"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows_by_split[split]:
            grouped[source_id(row)].append(row)
        for source in sorted(grouped):
            rows = grouped[source]
            item_ids = sorted({row["source_item_id"] for row in rows})
            geometry = Counter()
            for item_id in item_ids:
                for frame in physical_by_id[item_id]["frames"]:
                    geometry[(int(frame["width"]), int(frame["height"]))] += 1
            word_lengths = [len(str(row["text"]).split()) for row in rows]
            character_lengths = [len(str(row["text"])) for row in rows]
            records.append(
                {
                    "source_id": source,
                    "split": split,
                    "physical_pair_count": len(item_ids),
                    "query_count": len(rows),
                    "native_frame_geometry_counts": [
                        {"width": width, "height": height, "frame_count": count}
                        for (width, height), count in sorted(geometry.items())
                    ],
                    "caption_word_count": numeric_stats(word_lengths),
                    "caption_character_count": numeric_stats(character_lengths),
                }
            )
    return {
        "schema_version": "qcpr-source-shortcut-controls-v1",
        "scope": "CORE_PLUS_FOREST_HUMAN_TRAIN plus frozen core/Forest development/test",
        "warning": "Source-balanced exposure does not prove removal of source shortcuts.",
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--forest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.output_dir.resolve().relative_to(args.release.resolve())
    except ValueError:
        pass
    else:
        parser.error("--output-dir must be outside immutable --release")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    physical_items = read_jsonl(args.release / "registries/physical_items.jsonl")
    physical_by_id = {item["item_id"]: item for item in physical_items}
    forest_items = [item for item in physical_items if item.get("source") == "forest_change"]
    forest_by_name: dict[str, dict[str, Any]] = {}
    mapping_errors = []
    for item in forest_items:
        name = Path(str(item.get("provenance", {}).get("source_pair_id", ""))).name
        if not name or name in forest_by_name:
            mapping_errors.append(f"duplicate_or_empty_physical_mapping:{name}")
        forest_by_name[name] = item

    source_images = read_json(args.forest_json)["images"]
    forest_rows = []
    caption_errors = []
    rule_caption_count = 0
    for image in source_images:
        filename = str(image["filename"])
        item = forest_by_name.get(Path(filename).stem)
        if item is None:
            caption_errors.append(f"unmapped:{filename}")
            continue
        sentences = image.get("sentences") or []
        if len(sentences) != 5:
            caption_errors.append(f"expected_five_captions:{filename}:{len(sentences)}")
            continue
        if not str(sentences[0].get("raw", "")).strip():
            caption_errors.append(f"empty_human_caption:{filename}")
            continue
        if any(not str(sentences[index].get("raw", "")).strip() for index in (1, 2, 3, 4)):
            caption_errors.append(f"empty_rule_caption:{filename}")
            continue
        rule_caption_count += 4
        forest_rows.append(forest_query(item, image))
    forest_rows.sort(key=lambda row: row["source_item_id"])
    forest_train = [row for row in forest_rows if row["split"] == "train"]
    forest_eval = [row for row in forest_rows if row["split"] in {"development", "test"}]

    core_by_split = {
        split: read_jsonl(args.release / f"exact_core_{split}.jsonl")
        for split in ("train", "development", "test")
    }
    combined_train = [*core_by_split["train"], *forest_train]
    combined_path = args.output_dir / "core_plus_forest_human_train.jsonl"
    forest_train_path = args.output_dir / "forest_human_train.jsonl"
    forest_eval_path = args.output_dir / "forest_human_domain_shift_eval.jsonl"
    combined_sha = write_jsonl(combined_path, combined_train)
    forest_train_sha = write_jsonl(forest_train_path, forest_train)
    forest_eval_sha = write_jsonl(forest_eval_path, forest_eval)

    leakage = forest_split_leakage(forest_items)
    split_counts = Counter(row["split"] for row in forest_rows)
    selected_indices = Counter(row["caption_index"] for row in forest_rows)
    ready = (
        len(forest_items) == 334
        and len(forest_rows) == 334
        and split_counts == {"train": 232, "development": 49, "test": 53}
        and selected_indices == {0: 334}
        and rule_caption_count == 1336
        and not mapping_errors
        and not caption_errors
        and leakage["PASS"]
    )

    provenance_path = args.output_dir / "forest_human_provenance_license_audit.json"
    provenance = {
        "schema_version": "qcpr-forest-human-provenance-license-v2",
        "official_repository": "https://github.com/JamesBrockUoB/ForestChat",
        "official_repository_commit": UPSTREAM_COMMIT,
        "official_caption_json_sha256": sha256_file(args.forest_json),
        "captioning_app": {"url": UPSTREAM_APP_URL, "sha256": UPSTREAM_APP_SHA256},
        "deterministic_caption_order_proof": {
            "human_caption_index": 0,
            "rule_or_mask_caption_indices": [1, 2, 3, 4],
            "app_lines": {
                "append_preserves_order": "388-392",
                "human_saved_first": "598",
                "four_generated_captions_saved_after_human": "600-604",
            },
            "selected_caption_index_counts": dict(sorted(selected_indices.items())),
            "rule_or_mask_caption_count_in_candidate_manifests": 0,
            "source_rule_or_mask_caption_count_excluded": rule_caption_count,
        },
        "license": {
            "LICENSE_STATUS": "PARTIAL",
            "use_scope": "RESEARCH_ONLY_UNTIL_CLARIFIED",
            "readme_url": UPSTREAM_README_URL,
            "readme_sha256": UPSTREAM_README_SHA256,
            "readme_exact_statement": "This repo is distributed under MIT License. The code can be used for academic purposes only.",
            "license_file_url": UPSTREAM_LICENSE_URL,
            "license_file_sha256": UPSTREAM_LICENSE_SHA256,
            "license_file_identity": "Apache License 2.0",
            "github_metadata_detected_spdx_at_check": "Apache-2.0",
            "conflict": "README says MIT plus academic-only; linked LICENSE and GitHub metadata identify Apache-2.0.",
            "interpretation": "No reconciliation inferred; extension remains research-only pending upstream clarification.",
        },
        "forest_split_counts": dict(sorted(split_counts.items())),
        "split_physical_leakage": leakage,
        "mapping_errors": mapping_errors,
        "caption_errors": caption_errors,
        "FOREST_HUMAN_TRAIN_READY": ready,
        "core_unchanged": True,
    }
    provenance_sha = write_json(provenance_path, provenance)

    core_pair_ids = {row["source_item_id"] for row in core_by_split["train"]}
    sampler_path = args.output_dir / "core_plus_forest_human_sampler_proposal.json"
    sampler = {
        "schema_version": "qcpr-core-plus-forest-pair-sampler-v1",
        "purpose": "later matched domain-expanded ablation only",
        "sampling_unit": "physical_pair",
        "without_replacement_within_epoch": True,
        "per_pair_draw_cap_per_epoch": 1,
        "caption_selection": "deterministic epoch rotation within each physical pair",
        "source_balancing_enabled": False,
        "core_unique_pair_count": len(core_pair_ids),
        "forest_unique_train_pair_count": len(forest_train),
        "forest_max_draws_per_epoch": len(forest_train),
        "forest_pair_oversampling_factor": 1.0,
        "warning": "Exposure control does not prove removal of source shortcuts.",
    }
    sampler_sha = write_json(sampler_path, sampler)

    rows_by_split = {
        "train": combined_train,
        "development": [*core_by_split["development"], *[r for r in forest_eval if r["split"] == "development"]],
        "test": [*core_by_split["test"], *[r for r in forest_eval if r["split"] == "test"]],
    }
    source_controls_path = args.output_dir / "core_plus_forest_source_control_metadata.json"
    source_controls_sha = write_json(
        source_controls_path, build_source_controls(rows_by_split, physical_by_id)
    )

    status = {
        "schema_version": "qcpr-forest-human-extension-status-v1",
        "core_release": args.release.name,
        "core_release_content_sha256": read_json(args.release / "RELEASE.json")[
            "release_content_sha256"
        ],
        "core_unchanged": True,
        "FOREST_HUMAN_TRAIN_READY": ready,
        "FOREST_LICENSE_STATUS": "PARTIAL",
        "FOREST_USE_SCOPE": "RESEARCH_ONLY_UNTIL_CLARIFIED",
        "CORE_PLUS_FOREST_HUMAN_TRAIN": {"path": str(combined_path), "sha256": combined_sha},
        "FOREST_HUMAN_TRAIN": {"path": str(forest_train_path), "sha256": forest_train_sha},
        "FOREST_FROZEN_DOMAIN_SHIFT_EVAL": {"path": str(forest_eval_path), "sha256": forest_eval_sha},
        "FOREST_PROVENANCE_LICENSE_AUDIT": {"path": str(provenance_path), "sha256": provenance_sha},
        "FOREST_SAMPLER_PROPOSAL": {"path": str(sampler_path), "sha256": sampler_sha},
        "SOURCE_SHORTCUT_CONTROL_METADATA": {
            "path": str(source_controls_path),
            "sha256": source_controls_sha,
        },
        "AUTHORIZE_TEMPORALSIGLIP_DOMAIN_EXPANDED_ABLATION": ready,
        "authorization_scope": "later matched research-only data ablation; never primary core replacement",
        "HIGHRES_PHYSICAL_READY": True,
        "HIGHRES_RUNTIME_STRESS_READY": True,
        "HIGHRES_EVAL_READY": False,
        "HIGHRES_TRAIN_READY": False,
        "DUBAI_EXTERNAL_READY": False,
        "DUBAI_TRAIN_READY": False,
    }
    status_path = args.output_dir / "forest_human_extension_status.json"
    status_sha = write_json(status_path, status)
    print(json.dumps({**status, "status_path": str(status_path), "status_sha256": status_sha}, sort_keys=True))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
