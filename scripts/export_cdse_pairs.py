from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.ingestion import (
    CDSECatalog,
    JsonSceneProvider,
    PairingConfig,
    SceneSearchRequest,
    build_before_after_pairs,
    simulate_patch_exports,
)
from land_change_detection.ingestion.sentinel_hub import PatchExportRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export offline CDSE-style scene and pair manifests.")
    parser.add_argument("--mock-scenes-json", required=True, help="Path to a JSON file containing raw scene records.")
    parser.add_argument("--scenes-output", required=True, help="Output JSONL path for normalized scenes.")
    parser.add_argument("--pairs-output", help="Optional output JSONL path for before/after pairs.")
    parser.add_argument("--cache-dir", help="Optional cache dir used for dry-run local path generation.")
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--max-cloud-percentage", type=float)
    parser.add_argument("--sensor")
    parser.add_argument("--source", default="cdse")
    parser.add_argument("--region-id")
    parser.add_argument("--bbox", nargs=4, type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pair-max-day-delta", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    request = SceneSearchRequest(
        bbox=tuple(args.bbox) if args.bbox else None,
        date_from=args.date_from,
        date_to=args.date_to,
        max_cloud_percentage=args.max_cloud_percentage,
        sensor=args.sensor,
        source=args.source,
        region_id=args.region_id,
        dry_run=bool(args.dry_run),
        limit=args.limit,
    )
    catalog = CDSECatalog(provider=JsonSceneProvider(args.mock_scenes_json))
    scenes = catalog.search(request)
    if args.cache_dir:
        scenes = simulate_patch_exports(scenes, PatchExportRequest(cache_dir=args.cache_dir, dry_run=True))
    write_jsonl(args.scenes_output, [scene.to_dict() for scene in scenes])
    if args.pairs_output:
        pairs = build_before_after_pairs(
            scenes,
            config=PairingConfig(
                max_day_delta=args.pair_max_day_delta,
                max_cloud_percentage=args.max_cloud_percentage,
            ),
        )
        write_jsonl(args.pairs_output, [pair.to_dict() for pair in pairs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
