from __future__ import annotations

import numpy as np

from land_change_detection.change_interpretation.deterministic_reporting import (
    build_cell_report_rows_from_segmentation,
    build_scene_overview_from_rows,
)
from land_change_detection.contracts import CellPack, CellRegion


def test_segmentation_reporting_builds_full_row():
    pack = CellPack(
        cell=CellRegion("A1", 0, 0, 8, 8),
        before_crop=np.zeros((8, 8, 3), dtype=np.uint8),
        after_crop=np.zeros((8, 8, 3), dtype=np.uint8),
        before_semantic_map=np.zeros((8, 8), dtype=np.int32),
        after_semantic_map=np.ones((8, 8), dtype=np.int32),
        before_overlay=np.zeros((8, 8, 3), dtype=np.uint8),
        after_overlay=np.zeros((8, 8, 3), dtype=np.uint8),
        before_confidence=None,
        after_confidence=None,
        legend={0: "bareland", 1: "building"},
    )
    rows = build_cell_report_rows_from_segmentation([pack])
    assert len(rows) == 1
    assert rows[0]["cell"] == "A1"
    assert isinstance(rows[0]["likely_change"], str)
    assert isinstance(build_scene_overview_from_rows(rows), str)
