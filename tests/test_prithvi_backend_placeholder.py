from __future__ import annotations

import numpy as np
import pytest

from land_change_detection.segmentation_backends.registry import get_segmentation_backend


def test_prithvi_backend_requires_multispectral_input():
    backend = get_segmentation_backend("prithvi_terratorch", model_dir="unused", device="cpu")
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel-first EO stack"):
        backend.predict(rgb)


def test_prithvi_backend_is_explicitly_not_implemented_yet():
    backend = get_segmentation_backend("prithvi_terratorch", model_dir="unused", device="cpu")
    multispectral = np.zeros((13, 16, 16), dtype=np.float32)

    with pytest.raises(NotImplementedError, match="planned multispectral backend"):
        backend.predict(multispectral)
