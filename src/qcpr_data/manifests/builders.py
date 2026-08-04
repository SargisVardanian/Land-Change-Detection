"""Deterministic Dataset-v2 release layout writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ..contracts.validation import write_json, write_jsonl

SPLITS = ("train", "development", "test")
RELEASE_MANIFEST_NAMES = tuple(
    f"{view}_{split}.jsonl"
    for view in ("exact", "semantic", "localized", "long_series", "direction", "stable")
    for split in SPLITS
)


def _rows_for_split(rows: Iterable[Mapping[str, Any]], split: str) -> list[Mapping[str, Any]]:
    return sorted((row for row in rows if str(row.get("split")) == split), key=lambda row: str(row.get("query_id")))


def _write_manifest_set(root: Path, view: str, rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for split in SPLITS:
        path = root / "manifests" / f"{view}_{split}.jsonl"
        hashes[str(path.relative_to(root))] = write_jsonl(path, _rows_for_split(rows, split))
    return hashes


def write_release_layout(
    root: Path,
    *,
    items: Iterable[Mapping[str, Any]],
    exact: Iterable[Mapping[str, Any]],
    semantic: Iterable[Mapping[str, Any]],
    localized: Iterable[Mapping[str, Any]],
    long_series: Iterable[Mapping[str, Any]],
    direction: Iterable[Mapping[str, Any]],
    stable: Iterable[Mapping[str, Any]],
    semantic_groups: Iterable[Mapping[str, Any]],
    source_registry: Iterable[Mapping[str, Any]],
    sidecars: Iterable[Mapping[str, Any]],
    licenses: Mapping[str, Any],
    audits: Mapping[str, Any],
    release_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    for directory in ("LICENSES", "registries", "manifests", "evaluation_sidecars", "audits", "schemas", "source_reports"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    items_list = sorted(items, key=lambda row: str(row["item_id"]))
    item_rows = [dict(row) for row in items_list]
    frame_rows = [
        {**dict(frame), "item_id": str(item["item_id"]), "source": str(item["source"])}
        for item in items_list
        for frame in item.get("frames", [])
    ]
    write_jsonl(root / "registries/physical_items.jsonl", item_rows)
    write_jsonl(root / "registries/frames.jsonl", sorted(frame_rows, key=lambda row: str(row["frame_id"])))

    all_queries = []
    manifest_hashes: dict[str, str] = {}
    for view, rows in (
        ("exact", exact),
        ("semantic", semantic),
        ("localized", localized),
        ("long_series", long_series),
        ("direction", direction),
        ("stable", stable),
    ):
        rows_list = [dict(row) for row in rows]
        all_queries.extend(rows_list)
        manifest_hashes.update(_write_manifest_set(root, view, rows_list))
    write_jsonl(root / "registries/queries.jsonl", sorted(all_queries, key=lambda row: str(row["query_id"])))

    write_jsonl(root / "registries/semantic_groups.jsonl", sorted(semantic_groups, key=lambda row: str(row.get("group_id"))))
    write_jsonl(root / "registries/source_registry.jsonl", sorted(source_registry, key=lambda row: str(row.get("source_dataset"))))
    write_jsonl(root / "evaluation_sidecars/dense.jsonl", sorted(sidecars, key=lambda row: str(row.get("item_id"))))

    write_json(root / "LICENSES/records.json", dict(licenses))
    write_json(root / "audits/source_and_view_status.json", dict(audits))
    write_json(root / "schemas/schema_version.json", {"schema_version": "qcpr-dataset-v2-final-v1"})
    release = dict(release_metadata)
    release.update(
        {
            "schema_version": "qcpr-dataset-v2-final-v1",
            "required_manifest_names": list(RELEASE_MANIFEST_NAMES),
            "physical_item_count": len(item_rows),
            "frame_count": len(frame_rows),
            "query_count": len(all_queries),
            "manifest_hashes_before_release_metadata": manifest_hashes,
        }
    )
    write_json(root / "RELEASE.json", release)
    return {"manifest_hashes": manifest_hashes, "physical_item_count": len(item_rows), "frame_count": len(frame_rows), "query_count": len(all_queries)}
