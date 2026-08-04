#!/usr/bin/env python3
"""Build a conservative, mask-free Forest-Change physical/text pilot."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any
from PIL import Image
from land_change_detection.temporal_caption_manifest import make_manifest_row

ARCHIVE_BYTES = 23_559_661
ARCHIVE_SHA256 = "424931a075f00f8cf21d4d2f622df688de559494844df4876b59bde13d3d855d"
REVISION = "e8b25bf09c85ec85633d1b1b554f7bb23e47724d"
SPLITS = (("train", "train"), ("val", "development"), ("test", "test"))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

def verify_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return int(image.width), int(image.height)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.project_root / "datasets" / "raw" / "Forest-Change"
    archive = raw / "Forest-Change-dataset.zip"
    extracted = raw / "extracted" / "Forest-Change-dataset"
    captions_path = raw / "ForestChatcaptions.json"
    if not archive.is_file() or archive.stat().st_size != ARCHIVE_BYTES or sha256(archive) != ARCHIVE_SHA256:
        raise SystemExit("official Forest-Change archive size/SHA256 mismatch")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise SystemExit("Forest-Change archive integrity failure")
    payload = json.loads(captions_path.read_text(encoding="utf-8"))
    caption_rows = payload.get("images", [])
    captions_by_key = {(str(row["split"]), str(row["filename"])): row for row in caption_rows}
    if len(caption_rows) != 334 or len(captions_by_key) != 334:
        raise SystemExit("expected 334 unique Forest-Change caption rows")
    pairs: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    dense: list[dict[str, Any]] = []
    split_counts: collections.Counter[str] = collections.Counter()
    seen: set[str] = set()
    for official_split, split in SPLITS:
        root = extracted / "images" / official_split
        a_paths = sorted((root / "A").glob("*.png"))
        b_paths = {path.stem: path for path in (root / "B").glob("*.png")}
        label_paths = {path.stem: path for path in (root / "label").glob("*.png")}
        if len(a_paths) != len(b_paths) or len(a_paths) != len(label_paths):
            raise SystemExit(f"A/B/label count mismatch in {official_split}")
        for a_path in a_paths:
            stem = a_path.stem
            b_path, label_path = b_paths.get(stem), label_paths.get(stem)
            item = captions_by_key.get((official_split, a_path.name))
            if b_path is None or label_path is None or item is None:
                raise SystemExit(f"missing aligned Forest-Change item {official_split}/{stem}")
            caption_list = [str(x.get("raw", "")).strip() for x in item.get("sentences", [])]
            caption_list = [x for x in caption_list if x]
            if len(caption_list) != 5:
                raise SystemExit(f"expected five captions for {official_split}/{stem}")
            width, height = verify_image(a_path)
            if verify_image(b_path) != (width, height) or verify_image(label_path) != (width, height):
                raise SystemExit(f"dimension mismatch for {official_split}/{stem}")
            pair_id = f"forest_change:{REVISION}:{split}/{stem}"
            if pair_id in seen:
                raise SystemExit(f"duplicate pair ID {pair_id}")
            seen.add(pair_id)
            a_sha, b_sha = sha256(a_path), sha256(b_path)
            metadata = {
                "source_dataset": "Forest-Change",
                "source_version": REVISION,
                "source_pair_id": f"{official_split}/{stem}",
                "source_scene_group_id": f"forest_change:{REVISION}:{stem}",
                "source_event_id": None,
                "official_split": official_split,
                "license": "official dataset terms; verify before training",
                "sensor": "satellite RGB",
                "gsd": "approximately 30 m/pixel",
                "temporal_interval": "approximately one year",
                "caption_provenance": "mixed human annotation and rule-based generation; unresolved per caption",
                "verification_status": "source_unverified",
                "training_enabled": False,
                "ordered_pair_hash": hashlib.sha256(f"{a_sha}\n{b_sha}".encode()).hexdigest(),
                "order_invariant_pair_hash": hashlib.sha256("\n".join(sorted((a_sha, b_sha))).encode()).hexdigest(),
                "t1_sha256": a_sha,
                "t2_sha256": b_sha,
                "retrieval_role": "physical_pilot_only",
            }
            row = make_manifest_row(
                dataset_name="forest_change", split=split, original_id=f"{official_split}/{stem}",
                t1_path=a_path, t2_path=b_path, captions=caption_list,
                caption_source="model_generated", sensor="satellite RGB",
                spatial_resolution="approximately 30 m/pixel", source_metadata=metadata,
            )
            for forbidden_key in ("mask_path", "semantic_t1_path", "semantic_t2_path", "label_path", "dense_path", "official_label", "semantic_label", "dense_label"):
                row.pop(forbidden_key, None)
            row.update({"pair_id": pair_id, "training_enabled": False, "verification_status": "source_unverified",
                        "is_generated": None, "caption_provenance": "mixed_human_rule_based_unresolved"})
            pairs.append(row)
            dense.append({"schema_version": "qcpr-stage2-dense-label-v1", "canonical_pair_id": pair_id,
                          "source_dataset": "Forest-Change", "split": split, "label_path": str(label_path),
                          "label_kind": "binary_deforestation_change", "training_enabled": False})
            split_counts[split] += 1
            for index, text in enumerate(caption_list):
                texts.append({"schema_version": "qcpr-stage2-text-v1", "caption_id": f"{pair_id}:caption:{index}",
                              "canonical_pair_id": pair_id, "query_id": f"{pair_id}:query:{index}", "text": text,
                              "caption_source": "model_generated", "caption_provenance": "mixed_human_rule_based_unresolved",
                              "generator": None, "is_generated": None, "verification_status": "source_unverified",
                              "query_scope": "exact_candidate", "semantic_group_id": None, "split": split,
                              "training_enabled": False,
                              "reason_training_disabled": "mixed provenance and factual verification unresolved"})
    if len(pairs) != 334 or len(texts) != 1670 or len(dense) != 334:
        raise SystemExit("Forest-Change cardinality contract failed")

    # Official split labels are not accepted as scene-disjoint until shared
    # image components have been computed across both temporal directions.
    parent: dict[str, str] = {}
    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    hashes_by_pair: dict[str, tuple[str, str]] = {}
    for row in pairs:
        metadata = row["source_metadata"]
        hashes_by_pair[str(row["pair_id"])] = (str(metadata["t1_sha256"]), str(metadata["t2_sha256"]))
        union(str(metadata["t1_sha256"]), str(metadata["t2_sha256"]))
    group_members: collections.defaultdict[str, set[str]] = collections.defaultdict(set)
    group_pairs: collections.defaultdict[str, set[str]] = collections.defaultdict(set)
    for pair_id, image_hashes in hashes_by_pair.items():
        root = find(image_hashes[0])
        group_members[root].update(image_hashes)
        group_pairs[root].add(pair_id)
    group_splits: collections.defaultdict[str, set[str]] = collections.defaultdict(set)
    pair_to_group: dict[str, str] = {}
    for group_root, pair_ids in group_pairs.items():
        group_id = f"forest_change:{REVISION}:scene:{min(group_members[group_root])[:16]}"
        for pair_id in pair_ids:
            pair_to_group[pair_id] = group_id
            pair = next(row for row in pairs if str(row["pair_id"]) == pair_id)
            group_splits[group_id].add(str(pair["split"]))
    for row in pairs:
        group_id = pair_to_group[str(row["pair_id"])]
        row["source_metadata"]["source_scene_group_id"] = group_id
        row["source_metadata"]["source_scene_component_size"] = len(group_pairs[next(root for root, ids in group_pairs.items() if str(row["pair_id"]) in ids)])
    for label in dense:
        label["source_scene_group_id"] = pair_to_group[str(label["canonical_pair_id"])]
    for text in texts:
        text["source_scene_group_id"] = pair_to_group[str(text["canonical_pair_id"])]
    leaking_groups = {group_id: sorted(splits) for group_id, splits in group_splits.items() if len(splits) > 1}
    affected_pairs = sorted(pair_id for pair_id, group_id in pair_to_group.items() if group_id in leaking_groups)
    status = "PILOT_HOLD_SPLIT_IMAGE_LEAKAGE" if leaking_groups else "PILOT_READY_TEXT_VERIFICATION_REQUIRED"
    blockers = ["mixed human/rule provenance unresolved", "independent factual review pending"]
    if leaking_groups:
        blockers.insert(0, "official split contains shared-image scene components across train/development/test")
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "forest_change_pair_manifest_mask_free.jsonl", pairs)
    write_jsonl(args.output / "forest_change_text_registry_unverified.jsonl", texts)
    write_jsonl(args.output / "forest_change_dense_label_registry.jsonl", dense)
    write_json(args.output / "forest_change_pilot_audit.json", {
        "status": status, "source": "Forest-Change", "source_revision": REVISION,
        "archive_bytes": archive.stat().st_size, "archive_sha256": sha256(archive), "pair_count": len(pairs),
        "caption_count": len(texts), "dense_label_count": len(dense), "official_split_counts": dict(sorted(split_counts.items())),
        "decode_failure_count": 0, "training_enabled": False,
        "mask_free_manifest": str(args.output / "forest_change_pair_manifest_mask_free.jsonl"),
        "dense_sidecar": str(args.output / "forest_change_dense_label_registry.jsonl"),
        "scene_group_count": len(group_splits), "cross_split_scene_group_count": len(leaking_groups),
        "cross_split_scene_groups": leaking_groups, "affected_pair_count": len(affected_pairs),
        "affected_pair_ids_sample": affected_pairs[:50], "training_blockers": blockers,
        "source_facts": {"native_dimensions": [480, 480], "processed_dimensions": [256, 256],
                         "temporal_interval": "approximately one year", "caption_count_per_pair": 5,
                         "geographic_scope": "tropical and subtropical deforestation fronts"},
    })
    print(json.dumps({"status": status, "pairs": len(pairs), "captions": len(texts), "dense_labels": len(dense),
                      "splits": dict(sorted(split_counts.items())), "scene_groups": len(group_splits),
                      "cross_split_scene_groups": len(leaking_groups), "affected_pairs": len(affected_pairs),
                      "output": str(args.output)}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
