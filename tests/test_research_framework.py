from __future__ import annotations

import numpy as np

from land_change_detection.contracts import PipelineV2Result, SegmentationArtifact
from land_change_detection.grid_utils import build_cell_packs
from land_change_detection.research_framework import build_research_evidence_report


def _artifact(class_map: np.ndarray, legend: dict[int, str]) -> SegmentationArtifact:
    return SegmentationArtifact(
        semantic_map=class_map,
        confidence_map=None,
        label_summary=[
            {"label": "cropland", "percent": float((class_map == 1).mean() * 100.0)},
            {"label": "built_up", "percent": float((class_map == 2).mean() * 100.0)},
        ],
        overlay_rgb=np.zeros((*class_map.shape, 3), dtype=np.uint8),
        legend=legend,
    )


def test_research_report_contains_protocol_transitions_and_exports():
    before_map = np.ones((8, 8), dtype=np.int32)
    after_map = np.ones((8, 8), dtype=np.int32)
    after_map[:4, :4] = 2
    legend = {1: "cropland", 2: "built_up"}
    before_img = np.zeros((8, 8, 3), dtype=np.uint8)
    after_img = np.zeros((8, 8, 3), dtype=np.uint8)
    before_seg = _artifact(before_map, legend)
    after_seg = _artifact(after_map, legend)
    packs = build_cell_packs(
        before_img=before_img,
        after_img=after_img,
        before_seg=before_seg,
        after_seg=after_seg,
        legend=legend,
        rows=2,
        cols=2,
    )
    result = PipelineV2Result(
        before_segmentation=before_seg,
        after_segmentation=after_seg,
        cell_packs=packs,
        metadata={"rows": 2, "cols": 2, "segmentation_backend": "fake_semantic"},
    )

    report = build_research_evidence_report(result, scene_id="unit-test", bbox={"left": 0, "top": 0, "right": 8, "bottom": 8})

    assert report.protocol.research_questions[0].id == "RQ1"
    assert report.grid == {"rows": 2, "cols": 2}
    assert report.segmentation_backend == "fake_semantic"
    assert report.transitions[0]["before"] == "cropland"
    assert report.transitions[0]["after"] == "built_up"
    assert len(report.cell_rows) == 4
    assert "Semantic Land-Cover Change Research Framework" in report.to_markdown()
    assert report.to_dict()["scene_id"] == "unit-test"
