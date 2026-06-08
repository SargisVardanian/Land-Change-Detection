from __future__ import annotations

from dataclasses import dataclass

from .schemas import PairManifest, SceneManifest


@dataclass(frozen=True)
class PairingConfig:
    max_day_delta: int | None = None
    require_same_region: bool = True
    require_same_bbox: bool = True
    require_same_sensor: bool = True
    max_cloud_percentage: float | None = None


def build_before_after_pairs(
    scenes: list[SceneManifest],
    config: PairingConfig | None = None,
) -> list[PairManifest]:
    pairing = config or PairingConfig()
    sorted_scenes = sorted(scenes, key=lambda scene: (scene.region_id or "", scene.acquisition_date, scene.scene_id))
    pairs: list[PairManifest] = []
    for index, before in enumerate(sorted_scenes):
        if not _cloud_ok(before, pairing):
            continue
        for after in sorted_scenes[index + 1 :]:
            if not _eligible(before, after, pairing):
                continue
            delta = (after.acquisition_date - before.acquisition_date).days
            if delta <= 0:
                continue
            if pairing.max_day_delta is not None and delta > pairing.max_day_delta:
                continue
            pairs.append(
                PairManifest(
                    pair_id=f"{before.scene_id}__{after.scene_id}",
                    before_scene_id=before.scene_id,
                    after_scene_id=after.scene_id,
                    before_date=before.acquisition_date,
                    after_date=after.acquisition_date,
                    date_delta_days=delta,
                    source=before.source,
                    sensor=before.sensor,
                    bbox=before.bbox,
                    crs=before.crs,
                    region_id=before.region_id,
                    before_cloud_percentage=before.cloud_percentage,
                    after_cloud_percentage=after.cloud_percentage,
                    bands=before.bands or after.bands,
                    before_local_paths=before.local_paths,
                    after_local_paths=after.local_paths,
                    metadata={
                        "before_month": before.month,
                        "after_month": after.month,
                        "before_season": before.season,
                        "after_season": after.season,
                    },
                )
            )
    return pairs


def _cloud_ok(scene: SceneManifest, config: PairingConfig) -> bool:
    if config.max_cloud_percentage is None:
        return True
    return scene.cloud_percentage is not None and scene.cloud_percentage <= config.max_cloud_percentage


def _eligible(before: SceneManifest, after: SceneManifest, config: PairingConfig) -> bool:
    if not _cloud_ok(after, config):
        return False
    if config.require_same_region and before.region_id != after.region_id:
        return False
    if config.require_same_bbox and before.bbox != after.bbox:
        return False
    if config.require_same_sensor and before.sensor != after.sensor:
        return False
    if before.crs != after.crs:
        return False
    return True
