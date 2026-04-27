from __future__ import annotations

import numpy as np

from land_change_detection.contracts import SegmentationArtifact
from land_change_detection.grid_utils import build_cell_packs


def test_build_cell_packs_shapes():
    before = np.zeros((8, 8, 3), dtype=np.uint8)
    after = np.zeros((8, 8, 3), dtype=np.uint8)
    artifact = SegmentationArtifact(
        semantic_map=np.zeros((8, 8), dtype=np.int32),
        confidence_map=None,
        label_summary=[{"label": "background", "percent": 100.0}],
        overlay_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        legend={0: "background"},
    )
    packs = build_cell_packs(before, after, artifact, artifact, legend=artifact.legend, rows=4, cols=4)
    assert len(packs) == 16
    assert packs[0].before_crop.shape == (2, 2, 3)
    assert packs[-1].cell.cell_id == "D4"
