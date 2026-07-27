from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


def _coerce_date(value: str | date | datetime) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value)[:10])


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


@dataclass(frozen=True)
class SceneSearchRequest:
    bbox: tuple[float, float, float, float] | None = None
    date_from: date | str | None = None
    date_to: date | str | None = None
    max_cloud_percentage: float | None = None
    sensor: str | None = None
    source: str = "cdse"
    region_id: str | None = None
    dry_run: bool = True
    limit: int | None = None

    def normalized_date_from(self) -> date | None:
        return None if self.date_from is None else _coerce_date(self.date_from)

    def normalized_date_to(self) -> date | None:
        return None if self.date_to is None else _coerce_date(self.date_to)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "date_from": self.normalized_date_from().isoformat() if self.date_from is not None else None,
            "date_to": self.normalized_date_to().isoformat() if self.date_to is not None else None,
            "max_cloud_percentage": self.max_cloud_percentage,
            "sensor": self.sensor,
            "source": self.source,
            "region_id": self.region_id,
            "dry_run": self.dry_run,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class SceneManifest:
    scene_id: str
    acquisition_date: date
    source: str
    sensor: str
    bbox: tuple[float, float, float, float]
    crs: str
    cloud_percentage: float | None = None
    bands: tuple[str, ...] = ()
    region_id: str | None = None
    local_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    month: int = field(init=False)
    season: str = field(init=False)

    def __post_init__(self) -> None:
        normalized = _coerce_date(self.acquisition_date)
        object.__setattr__(self, "acquisition_date", normalized)
        object.__setattr__(self, "bbox", tuple(float(value) for value in self.bbox))
        object.__setattr__(self, "bands", tuple(str(value) for value in self.bands))
        object.__setattr__(self, "month", normalized.month)
        object.__setattr__(self, "season", season_from_month(normalized.month))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "acquisition_date": self.acquisition_date.isoformat(),
            "source": self.source,
            "sensor": self.sensor,
            "bbox": list(self.bbox),
            "crs": self.crs,
            "cloud_percentage": self.cloud_percentage,
            "bands": list(self.bands),
            "region_id": self.region_id,
            "local_paths": dict(self.local_paths),
            "metadata": dict(self.metadata),
            "month": self.month,
            "season": self.season,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SceneManifest":
        return cls(
            scene_id=str(payload["scene_id"]),
            acquisition_date=_coerce_date(payload["acquisition_date"]),
            source=str(payload["source"]),
            sensor=str(payload["sensor"]),
            bbox=tuple(float(value) for value in payload["bbox"]),
            crs=str(payload.get("crs", "EPSG:4326")),
            cloud_percentage=(
                None if payload.get("cloud_percentage") is None else float(payload["cloud_percentage"])
            ),
            bands=tuple(str(value) for value in payload.get("bands", ())),
            region_id=payload.get("region_id"),
            local_paths=dict(payload.get("local_paths", {})),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class PairManifest:
    pair_id: str
    before_scene_id: str
    after_scene_id: str
    before_date: date
    after_date: date
    date_delta_days: int
    source: str
    sensor: str
    bbox: tuple[float, float, float, float]
    crs: str
    region_id: str | None = None
    before_cloud_percentage: float | None = None
    after_cloud_percentage: float | None = None
    bands: tuple[str, ...] = ()
    before_local_paths: dict[str, str] = field(default_factory=dict)
    after_local_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    month: int = field(init=False)
    season: str = field(init=False)

    def __post_init__(self) -> None:
        before = _coerce_date(self.before_date)
        after = _coerce_date(self.after_date)
        object.__setattr__(self, "before_date", before)
        object.__setattr__(self, "after_date", after)
        object.__setattr__(self, "bbox", tuple(float(value) for value in self.bbox))
        object.__setattr__(self, "bands", tuple(str(value) for value in self.bands))
        object.__setattr__(self, "month", after.month)
        object.__setattr__(self, "season", season_from_month(after.month))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "before_scene_id": self.before_scene_id,
            "after_scene_id": self.after_scene_id,
            "before_date": self.before_date.isoformat(),
            "after_date": self.after_date.isoformat(),
            "date_delta_days": self.date_delta_days,
            "source": self.source,
            "sensor": self.sensor,
            "bbox": list(self.bbox),
            "crs": self.crs,
            "region_id": self.region_id,
            "before_cloud_percentage": self.before_cloud_percentage,
            "after_cloud_percentage": self.after_cloud_percentage,
            "bands": list(self.bands),
            "before_local_paths": dict(self.before_local_paths),
            "after_local_paths": dict(self.after_local_paths),
            "metadata": dict(self.metadata),
            "month": self.month,
            "season": self.season,
        }

