from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..contracts import SegmentationArtifact
from ..semantic_surface import PRITHVI_BAND_NAMES, select_prithvi_bands
from ..training.prithvi_experimental import build_prithvi_runtime_bundle
from .base import BaseSegmentationBackend


@dataclass(frozen=True)
class PrithviBackendRuntime:
    model_dir: str
    device: str
    checkpoint_path: str | None
    terratorch_config_path: str | None
    terratorch_available: bool
    config_valid: bool
    runtime_ready: bool
    config_summary: dict | None
    module_summary: dict | None
    factory_summary: dict | None
    notes: list[str]
    loader_stage: str = "unresolved"
    runtime_object_loaded: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class PrithviTerratorchBackend(BaseSegmentationBackend):
    """Planned multispectral Prithvi backend.

    Prithvi-EO-2.0 is not a drop-in RGB replacement for the current
    Mask2Former path. It expects EO bands and a TerraTorch task head, so this
    backend is intentionally explicit until training/inference config is added.
    """

    def __init__(self, model_dir: str | Path, device: str):
        self.model_dir = Path(model_dir)
        self.device = device
        self._runtime = self._resolve_runtime()

    def inspect_runtime(self) -> PrithviBackendRuntime:
        return self._runtime

    def runtime_bundle(self) -> dict:
        return self._runtime.to_dict()

    def load_runtime_object(self) -> PrithviBackendRuntime:
        self._runtime = self._load_runtime_object()
        return self._runtime

    def predict(self, image: np.ndarray) -> SegmentationArtifact:
        if image.ndim != 3 or image.shape[0] < 13 or image.shape[-1] in {3, 4}:
            raise ValueError(
                "expected Prithvi input as channel-first EO stack with at least 13 bands; "
                f"got shape={image.shape}"
            )
        bands = select_prithvi_bands(image)
        if self._runtime.runtime_ready:
            raise NotImplementedError(
                "Prithvi runtime bundle is resolved, but a real TerraTorch semantic head is still not wired in this repository."
            )
        raise NotImplementedError(
            "prithvi_terratorch is a planned multispectral backend. "
            f"Received compatible bands {PRITHVI_BAND_NAMES} with shape {bands.shape}, "
            "but TerraTorch model/head wiring is not implemented yet."
        )

    def export_features(self, image: np.ndarray) -> SegmentationArtifact:
        if image.ndim != 3 or image.shape[0] < 13 or image.shape[-1] in {3, 4}:
            raise ValueError(
                "expected Prithvi input as channel-first EO stack with at least 13 bands; "
                f"got shape={image.shape}"
            )
        bands = select_prithvi_bands(image)
        return SegmentationArtifact(
            semantic_map=np.zeros(image.shape[1:], dtype=np.int32),
            confidence_map=None,
            label_summary=[],
            overlay_rgb=np.zeros((*image.shape[1:], 3), dtype=np.uint8),
            legend={0: "prithvi_feature_export_placeholder"},
            features=bands.astype(np.float32, copy=False),
            metadata={
                "backend": "prithvi_terratorch",
                "feature_export_only": True,
                "runtime_bundle": self.runtime_bundle(),
            },
        )

    def _resolve_runtime(self) -> PrithviBackendRuntime:
        bundle = build_prithvi_runtime_bundle(model_dir=self.model_dir)
        loader_stage = "bundle_ready" if bundle.config_valid or bundle.checkpoint_path or bundle.terratorch_config_path else "unresolved"
        return PrithviBackendRuntime(
            model_dir=str(self.model_dir),
            device=self.device,
            checkpoint_path=bundle.checkpoint_path,
            terratorch_config_path=bundle.terratorch_config_path,
            terratorch_available=bundle.terratorch_available,
            config_valid=bundle.config_valid,
            runtime_ready=bundle.runtime_ready,
            config_summary=bundle.config_summary,
            module_summary=bundle.module_summary,
            factory_summary=bundle.factory_summary,
            notes=bundle.notes,
            loader_stage=loader_stage,
            runtime_object_loaded=False,
        )

    def _load_runtime_object(self) -> PrithviBackendRuntime:
        runtime = self._runtime
        notes = list(runtime.notes)
        if not runtime.checkpoint_path or not runtime.terratorch_config_path:
            notes.append("Runtime object cannot be loaded because checkpoint or config is missing.")
            return PrithviBackendRuntime(
                **{**runtime.to_dict(), "notes": notes, "loader_stage": "unresolved", "runtime_object_loaded": False}
            )
        if not runtime.config_valid:
            notes.append("Runtime object cannot be loaded because TerraTorch config is invalid.")
            return PrithviBackendRuntime(
                **{**runtime.to_dict(), "notes": notes, "loader_stage": "bundle_ready", "runtime_object_loaded": False}
            )
        if not runtime.terratorch_available:
            notes.append("Runtime object cannot be loaded because TerraTorch is unavailable.")
            return PrithviBackendRuntime(
                **{**runtime.to_dict(), "notes": notes, "loader_stage": "loadable", "runtime_object_loaded": False}
            )
        factory_candidates = (runtime.factory_summary or {}).get("factory_candidates", [])
        if not factory_candidates:
            notes.append("Runtime object cannot be fully wired because no TerraTorch factory candidates were discovered.")
            return PrithviBackendRuntime(
                **{**runtime.to_dict(), "notes": notes, "loader_stage": "loadable", "runtime_object_loaded": False}
            )

        notes.append(
            "Checkpoint, config, and TerraTorch are all available. A real task/head loader is still not implemented, so a placeholder loaded state is returned."
        )
        return PrithviBackendRuntime(
            **{**runtime.to_dict(), "notes": notes, "loader_stage": "loaded_placeholder", "runtime_object_loaded": True}
        )
