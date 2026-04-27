from __future__ import annotations

import numpy as np

from land_change_detection.contracts import CellPack, CellRegion, SegmentationArtifact


def test_segmentation_artifact_aliases():
    semantic = np.zeros((4, 4), dtype=np.int32)
    overlay = np.zeros((4, 4, 3), dtype=np.uint8)
    artifact = SegmentationArtifact(
        semantic_map=semantic,
        confidence_map=None,
        label_summary=[{"label": "background", "percent": 100.0}],
        overlay_rgb=overlay,
        legend={0: "background"},
    )
    assert artifact.class_map.shape == (4, 4)
    assert artifact.color_map.shape == (4, 4, 3)
    assert artifact.id2label[0] == "background"


def test_cell_pack_bbox():
    pack = CellPack(
        cell=CellRegion("A1", 0, 0, 8, 8),
        before_crop=np.zeros((8, 8, 3), dtype=np.uint8),
        after_crop=np.zeros((8, 8, 3), dtype=np.uint8),
        before_semantic_map=np.zeros((8, 8), dtype=np.int32),
        after_semantic_map=np.zeros((8, 8), dtype=np.int32),
        before_overlay=np.zeros((8, 8, 3), dtype=np.uint8),
        after_overlay=np.zeros((8, 8, 3), dtype=np.uint8),
        before_confidence=None,
        after_confidence=None,
        legend={0: "background"},
    )
    assert pack.cell.bbox == (0, 0, 8, 8)
