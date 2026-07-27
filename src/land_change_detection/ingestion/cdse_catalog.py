from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .schemas import SceneManifest, SceneSearchRequest


def _coerce_bbox(payload: Any) -> tuple[float, float, float, float]:
    if payload is None:
        raise ValueError("scene record is missing bbox")
    if isinstance(payload, dict):
        if "bbox" in payload:
            payload = payload["bbox"]
        elif {"minx", "miny", "maxx", "maxy"} <= payload.keys():
            payload = [payload["minx"], payload["miny"], payload["maxx"], payload["maxy"]]
    values = [float(value) for value in payload]
    if len(values) != 4:
        raise ValueError("bbox must contain exactly four values")
    return (values[0], values[1], values[2], values[3])


def normalize_scene_record(record: dict[str, Any], default_source: str = "cdse") -> SceneManifest:
    scene_id = record.get("scene_id") or record.get("id") or record.get("identifier")
    if scene_id is None:
        raise ValueError("scene record is missing scene_id")
    acquisition_date = (
        record.get("acquisition_date")
        or record.get("date")
        or record.get("datetime")
        or record.get("start_datetime")
    )
    if acquisition_date is None:
        raise ValueError("scene record is missing acquisition_date")
    source = str(record.get("source") or record.get("provider") or default_source)
    sensor = str(record.get("sensor") or record.get("collection") or "sentinel-2")
    cloud = record.get("cloud_percentage")
    if cloud is None:
        cloud = record.get("cloud_cover")
    if cloud is None:
        cloud = record.get("eo:cloud_cover")
    bands = tuple(record.get("bands") or record.get("band_names") or ())
    local_paths = dict(record.get("local_paths") or {})
    path = record.get("local_path") or record.get("cache_path")
    if path and "default" not in local_paths:
        local_paths["default"] = str(path)
    metadata = dict(record.get("metadata") or {})
    for key in ("platform", "tile_id", "orbit", "processing_level"):
        if key in record and key not in metadata:
            metadata[key] = record[key]
    return SceneManifest(
        scene_id=str(scene_id),
        acquisition_date=acquisition_date,
        source=source,
        sensor=sensor,
        bbox=_coerce_bbox(record.get("bbox") or record.get("geometry")),
        crs=str(record.get("crs") or "EPSG:4326"),
        cloud_percentage=None if cloud is None else float(cloud),
        bands=bands,
        region_id=record.get("region_id"),
        local_paths=local_paths,
        metadata=metadata,
    )


class SceneProvider(Protocol):
    def search(self, request: SceneSearchRequest) -> list[dict[str, Any]]:
        ...


@dataclass
class JsonSceneProvider:
    path: str | Path

    def search(self, request: SceneSearchRequest) -> list[dict[str, Any]]:
        payload = json.loads(Path(self.path).read_text())
        if not isinstance(payload, list):
            raise ValueError("mock scene payload must be a JSON list")
        return [dict(item) for item in payload]


@dataclass
class CDSECatalog:
    provider: SceneProvider

    def search(self, request: SceneSearchRequest) -> list[SceneManifest]:
        scenes = [normalize_scene_record(item, default_source=request.source) for item in self.provider.search(request)]
        filtered = [scene for scene in scenes if _matches_request(scene, request)]
        filtered.sort(key=lambda scene: (scene.acquisition_date, scene.scene_id))
        if request.limit is not None:
            return filtered[: request.limit]
        return filtered


def _matches_request(scene: SceneManifest, request: SceneSearchRequest) -> bool:
    date_from = request.normalized_date_from()
    date_to = request.normalized_date_to()
    if date_from is not None and scene.acquisition_date < date_from:
        return False
    if date_to is not None and scene.acquisition_date > date_to:
        return False
    if request.sensor is not None and scene.sensor != request.sensor:
        return False
    if request.region_id is not None and scene.region_id != request.region_id:
        return False
    if request.max_cloud_percentage is not None:
        if scene.cloud_percentage is None or scene.cloud_percentage > request.max_cloud_percentage:
            return False
    if request.bbox is not None and tuple(scene.bbox) != tuple(float(value) for value in request.bbox):
        return False
    return True
