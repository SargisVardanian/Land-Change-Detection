from __future__ import annotations

import numpy as np
import pytest

from land_change_detection.segmentation_backends.prithvi_terratorch_experimental import (
    ExperimentalPrithviBackend,
    experimental_prithvi_capabilities,
)


def test_experimental_prithvi_requires_six_band_input():
    backend = ExperimentalPrithviBackend(model_dir=".", device="cpu")
    with pytest.raises(ValueError):
        backend.predict(np.zeros((13, 4, 4), dtype=np.float32))


def test_experimental_prithvi_disabled_by_default():
    backend = ExperimentalPrithviBackend(model_dir=".", device="cpu")
    with pytest.raises(RuntimeError, match="disabled"):
        backend.predict(np.zeros((6, 4, 4), dtype=np.float32))


def test_experimental_prithvi_capabilities_contract():
    capabilities = experimental_prithvi_capabilities()
    assert capabilities["bands"] == ["blue", "green", "red", "nir", "swir1", "swir2"]
    assert capabilities["default_registered"] is False
