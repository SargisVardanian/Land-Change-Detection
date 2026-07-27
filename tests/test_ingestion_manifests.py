from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from land_change_detection.ingestion.cache import build_cache_path, ensure_cache_dir
from land_change_detection.ingestion.cdse_catalog import CDSECatalog, normalize_scene_record
from land_change_detection.ingestion.patch_builder import PairingConfig, build_before_after_pairs
from land_change_detection.ingestion.schemas import SceneManifest, SceneSearchRequest
from land_change_detection.ingestion.sentinel_hub import PatchExportRequest, simulate_patch_exports


class FakeSceneProvider:
    def __init__(self, rows):
        self.rows = list(rows)

    def search(self, request):
        return list(self.rows)


def test_normalize_scene_record_and_catalog_filtering():
    provider = FakeSceneProvider(
        [
            {
                "id": "scene-a",
                "date": "2025-04-01",
                "provider": "cdse",
                "collection": "sentinel-2",
                "bbox": [44.4, 40.1, 44.5, 40.2],
                "cloud_cover": 10,
                "bands": ["B02", "B03", "B04"],
                "region_id": "arm-1",
            },
            {
                "id": "scene-b",
                "date": "2025-04-15",
                "provider": "cdse",
                "collection": "sentinel-2",
                "bbox": [44.4, 40.1, 44.5, 40.2],
                "cloud_cover": 35,
                "region_id": "arm-1",
            },
        ]
    )
    catalog = CDSECatalog(provider=provider)

    scenes = catalog.search(
        SceneSearchRequest(
            bbox=(44.4, 40.1, 44.5, 40.2),
            date_from="2025-04-01",
            date_to="2025-04-20",
            max_cloud_percentage=20,
            sensor="sentinel-2",
            region_id="arm-1",
        )
    )

    assert [scene.scene_id for scene in scenes] == ["scene-a"]
    assert scenes[0].month == 4
    assert scenes[0].season == "spring"
    assert scenes[0].to_dict()["bands"] == ["B02", "B03", "B04"]


def test_pair_builder_uses_dates_cloud_and_bbox():
    scene_before = normalize_scene_record(
        {
            "scene_id": "before",
            "acquisition_date": "2025-05-01",
            "source": "cdse",
            "sensor": "sentinel-2",
            "bbox": [1, 2, 3, 4],
            "crs": "EPSG:4326",
            "cloud_percentage": 5,
            "region_id": "arm-a",
        }
    )
    scene_after = normalize_scene_record(
        {
            "scene_id": "after",
            "acquisition_date": "2025-05-10",
            "source": "cdse",
            "sensor": "sentinel-2",
            "bbox": [1, 2, 3, 4],
            "crs": "EPSG:4326",
            "cloud_percentage": 7,
            "region_id": "arm-a",
        }
    )
    scene_other = normalize_scene_record(
        {
            "scene_id": "other-region",
            "acquisition_date": "2025-05-12",
            "source": "cdse",
            "sensor": "sentinel-2",
            "bbox": [9, 9, 10, 10],
            "crs": "EPSG:4326",
            "cloud_percentage": 3,
            "region_id": "arm-b",
        }
    )

    pairs = build_before_after_pairs(
        [scene_other, scene_after, scene_before],
        PairingConfig(max_day_delta=14, max_cloud_percentage=10),
    )

    assert [pair.pair_id for pair in pairs] == ["before__after"]
    assert pairs[0].date_delta_days == 9
    assert pairs[0].metadata["before_season"] == "spring"


def test_simulated_patch_exports_assign_local_paths(tmp_path: Path):
    scene = SceneManifest(
        scene_id="scene-a",
        acquisition_date="2025-06-01",
        source="cdse",
        sensor="Sentinel 2",
        bbox=(1, 2, 3, 4),
        crs="EPSG:4326",
    )

    exported = simulate_patch_exports([scene], PatchExportRequest(cache_dir=tmp_path, dry_run=True))

    assert exported[0].local_paths["default"].endswith(".tif")
    assert exported[0].metadata["patch_export_dry_run"] is True
    assert ensure_cache_dir(tmp_path).exists()
    assert build_cache_path(scene, tmp_path).parent == tmp_path


def test_export_cdse_pairs_cli_writes_jsonl(tmp_path: Path):
    mock_rows = [
        {
            "id": "scene-a",
            "date": "2025-05-01",
            "provider": "cdse",
            "collection": "sentinel-2",
            "bbox": [44.4, 40.1, 44.5, 40.2],
            "cloud_cover": 10,
            "region_id": "arm-1",
        },
        {
            "id": "scene-b",
            "date": "2025-05-20",
            "provider": "cdse",
            "collection": "sentinel-2",
            "bbox": [44.4, 40.1, 44.5, 40.2],
            "cloud_cover": 5,
            "region_id": "arm-1",
        },
    ]
    mock_path = tmp_path / "mock_scenes.json"
    scenes_output = tmp_path / "scenes.jsonl"
    pairs_output = tmp_path / "pairs.jsonl"
    mock_path.write_text(json.dumps(mock_rows), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_cdse_pairs.py",
            "--mock-scenes-json",
            str(mock_path),
            "--scenes-output",
            str(scenes_output),
            "--pairs-output",
            str(pairs_output),
            "--dry-run",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--pair-max-day-delta",
            "30",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )

    assert result.returncode == 0, result.stderr
    scene_rows = [json.loads(line) for line in scenes_output.read_text(encoding="utf-8").splitlines()]
    pair_rows = [json.loads(line) for line in pairs_output.read_text(encoding="utf-8").splitlines()]
    assert len(scene_rows) == 2
    assert pair_rows[0]["pair_id"] == "scene-a__scene-b"
