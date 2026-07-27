from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .cache import build_cache_path, ensure_cache_dir
from .schemas import SceneManifest


@dataclass(frozen=True)
class PatchExportRequest:
    cache_dir: str | Path
    suffix: str = ".tif"
    dry_run: bool = True


class SentinelHubPatchProvider(Protocol):
    def export(self, scene: SceneManifest, request: PatchExportRequest) -> dict[str, str]:
        ...


def simulate_patch_exports(
    scenes: list[SceneManifest],
    request: PatchExportRequest,
) -> list[SceneManifest]:
    ensure_cache_dir(request.cache_dir)
    exported: list[SceneManifest] = []
    for scene in scenes:
        simulated_path = build_cache_path(scene, request.cache_dir, suffix=request.suffix)
        local_paths = dict(scene.local_paths)
        local_paths.setdefault("default", str(simulated_path))
        metadata = dict(scene.metadata)
        metadata["patch_export_dry_run"] = request.dry_run
        exported.append(
            SceneManifest(
                scene_id=scene.scene_id,
                acquisition_date=scene.acquisition_date,
                source=scene.source,
                sensor=scene.sensor,
                bbox=scene.bbox,
                crs=scene.crs,
                cloud_percentage=scene.cloud_percentage,
                bands=scene.bands,
                region_id=scene.region_id,
                local_paths=local_paths,
                metadata=metadata,
            )
        )
    return exported
