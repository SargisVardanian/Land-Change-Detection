from __future__ import annotations

import numpy as np

from land_change_detection.contracts import PipelineV2Result, SegmentationArtifact
from land_change_detection.pipeline_v2 import LandChangePipelineV2


class FakeSegmentationRuntime:
    backend_name = "fake"

    def run(self, image):
        h, w = image.shape[:2]
        return SegmentationArtifact(
            semantic_map=np.zeros((h, w), dtype=np.int32),
            confidence_map=None,
            label_summary=[{"label": "background", "percent": 100.0}],
            overlay_rgb=np.zeros((h, w, 3), dtype=np.uint8),
            legend={0: "background"},
        )


def test_pipeline_v2_smoke():
    pipeline = LandChangePipelineV2(segmentation_runtime=FakeSegmentationRuntime())
    before = np.zeros((16, 16, 3), dtype=np.uint8)
    after = np.zeros((16, 16, 3), dtype=np.uint8)
    result = pipeline.run(before, after, rows=4, cols=4)
    assert isinstance(result, PipelineV2Result)
    assert len(result.cell_packs) == 16
    assert result.before_segmentation.legend[0] == "background"
