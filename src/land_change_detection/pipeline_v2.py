from __future__ import annotations

import numpy as np

from .contracts import PipelineV2Result
from .grid_utils import build_cell_packs
from .segmentation_runtime import SegmentationRuntime


class LandChangePipelineV2:
    def __init__(self, segmentation_runtime: SegmentationRuntime, change_runtime=None):
        self.segmentation_runtime = segmentation_runtime
        self.change_runtime = change_runtime

    def run(self, before_img: np.ndarray, after_img: np.ndarray, rows: int = 4, cols: int = 4) -> PipelineV2Result:
        before_seg = self.segmentation_runtime.run(before_img)
        after_seg = self.segmentation_runtime.run(after_img)
        packs = build_cell_packs(
            before_img=before_img,
            after_img=after_img,
            before_seg=before_seg,
            after_seg=after_seg,
            legend=before_seg.legend,
            rows=rows,
            cols=cols,
            scene_thumbnail=before_img,
        )
        scene_report = self.change_runtime.interpret_scene(packs) if self.change_runtime is not None else None
        return PipelineV2Result(
            before_segmentation=before_seg,
            after_segmentation=after_seg,
            cell_packs=packs,
            scene_report=scene_report,
            metadata={"rows": rows, "cols": cols, "segmentation_backend": self.segmentation_runtime.backend_name},
        )
