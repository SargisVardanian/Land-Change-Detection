from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import SegmentationArtifact
from ..semantic_surface import PRITHVI_BAND_NAMES
from .base import BaseSegmentationBackend


class ExperimentalPrithviBackend(BaseSegmentationBackend):
    """Import-safe experimental Prithvi backend.

    This backend is intentionally not registered in the default runtime path.
    It exists so the research subsystem can validate a 6-band EO contract and
    later plug in TerraTorch-backed inference or feature extraction.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str,
        *,
        enable_experimental_prithvi: bool = False,
        feature_export_only: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.enable_experimental_prithvi = enable_experimental_prithvi
        self.feature_export_only = feature_export_only

    def predict(self, image: np.ndarray) -> SegmentationArtifact:
        self._validate_input(image)
        if not self.enable_experimental_prithvi:
            raise RuntimeError(
                "ExperimentalPrithviBackend is disabled. Set enable_experimental_prithvi=True "
                "to use this research-only backend."
            )

        terratorch_module = self._load_terratorch()
        if terratorch_module is None:
            raise RuntimeError(
                "Experimental Prithvi backend requires optional TerraTorch dependencies and model wiring. "
                "Install TerraTorch and provide compatible weights before enabling real inference."
            )

        if self.feature_export_only:
            features = image.astype(np.float32, copy=False)
            return SegmentationArtifact(
                semantic_map=np.zeros(image.shape[1:], dtype=np.int32),
                confidence_map=None,
                label_summary=[],
                overlay_rgb=np.zeros((*image.shape[1:], 3), dtype=np.uint8),
                legend={0: "experimental_prithvi_placeholder"},
                features=features,
                metadata={
                    "backend": "prithvi_terratorch_experimental",
                    "feature_export_only": True,
                    "terratorch_module": terratorch_module,
                },
            )

        raise NotImplementedError(
            "Experimental Prithvi inference head is not implemented yet. The platform is ready for "
            "real TerraTorch model/head integration, but no production decoder is wired here."
        )

    def _validate_input(self, image: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[0] != 6:
            raise ValueError(
                "Experimental Prithvi backend expects a channel-first 6-band EO tensor ordered as "
                f"{PRITHVI_BAND_NAMES}; got shape={image.shape}"
            )

    def _load_terratorch(self) -> str | None:
        try:
            import terratorch  # type: ignore  # noqa: F401
        except ModuleNotFoundError:
            return None
        return "terratorch"


def experimental_prithvi_capabilities() -> dict[str, Any]:
    return {
        "bands": list(PRITHVI_BAND_NAMES),
        "feature_export_only_supported": True,
        "requires_optional_dependency": "terratorch",
        "default_registered": False,
    }
