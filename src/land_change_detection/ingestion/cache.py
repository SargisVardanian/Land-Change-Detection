from __future__ import annotations

from pathlib import Path

from .schemas import SceneManifest


def ensure_cache_dir(path: str | Path) -> Path:
    cache_dir = Path(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def build_cache_path(scene: SceneManifest, cache_dir: str | Path, suffix: str = ".tif") -> Path:
    root = ensure_cache_dir(cache_dir)
    sensor_name = scene.sensor.lower().replace(" ", "_")
    date_token = scene.acquisition_date.isoformat().replace("-", "")
    return root / f"{sensor_name}_{scene.scene_id}_{date_token}{suffix}"
