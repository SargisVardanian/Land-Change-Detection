from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from land_change_detection.contracts import SegmentationArtifact
from land_change_detection.pipelines.evidence_bundle import serialize_evidence_bundle_for_llm
from land_change_detection.pipelines.research_pipeline import (
    ResearchPipeline,
    ResearchPipelineConfig,
    ResearchPipelineResult,
)


class FakeSegmentationRuntime:
    backend_name = "fake_segmentation"

    def run(self, image):
        h, w = image.shape[:2]
        semantic_map = np.ones((h, w), dtype=np.int32)
        semantic_map[: h // 2, : w // 2] = 2
        return SegmentationArtifact(
            semantic_map=semantic_map,
            confidence_map=None,
            label_summary=[
                {"label": "cropland", "percent": 75.0},
                {"label": "built_up", "percent": 25.0},
            ],
            overlay_rgb=np.zeros((h, w, 3), dtype=np.uint8),
            legend={1: "cropland", 2: "built_up"},
        )


@dataclass(frozen=True)
class FakeRetrievalArtifact:
    items: tuple[dict, ...]

    def to_dict(self):
        return {"items": list(self.items)}


class FakeRetrievalRuntime:
    def run(self, query):
        return FakeRetrievalArtifact(
            items=(
                {
                    "item_id": "pair-02",
                    "score": 0.72,
                    "dates": ["2023-06-01", "2024-06-05"],
                    "sensor": "Sentinel-2",
                    "source": "unit-test",
                    "transition_hints": ["cropland->built_up"],
                    "thumbnail_paths": ["/tmp/pair-02.png"],
                },
                {
                    "item_id": "pair-01",
                    "score": 0.91,
                    "dates": ["2022-04-10", "2023-04-11"],
                    "sensor": "Sentinel-2",
                    "source": "unit-test",
                    "transition_hints": ["cropland->built_up", "water->wetland"],
                    "thumbnail_paths": ["/tmp/pair-01.png"],
                },
            )
        )


class FakeExplainerRuntime:
    def explain(self, prompt: str, evidence: dict):
        return {"prompt_length": len(prompt), "retrieved_items": len(evidence["retrieved_items"])}


def test_research_pipeline_builds_serializable_bundle_and_prompt():
    pipeline = ResearchPipeline(
        segmentation_runtime=FakeSegmentationRuntime(),
        retrieval_runtime=FakeRetrievalRuntime(),
        explainer_runtime=FakeExplainerRuntime(),
        config=ResearchPipelineConfig(retrieval_mode="pair_analog", top_k=2, rows=2, cols=2),
    )
    before = np.zeros((8, 8, 3), dtype=np.uint8)
    after = np.zeros((8, 8, 3), dtype=np.uint8)
    result = pipeline.run(query={"text": "cropland to built_up"}, before_img=before, after_img=after)

    assert isinstance(result, ResearchPipelineResult)
    assert result.pipeline_result is not None
    assert result.evidence_bundle.retrieval_mode == "pair_analog"
    assert [item.item_id for item in result.evidence_bundle.retrieved_items] == ["pair-01", "pair-02"]
    assert result.evidence_bundle.segmentation_summaries["before"][0]["label"] == "cropland"
    assert "cropland->built_up" in result.evidence_bundle.transition_hints
    assert "item_id=pair-01" in result.llm_prompt
    assert result.to_dict()["evidence_bundle"]["retrieved_items"][0]["item_id"] == "pair-01"
    assert result.explanation == {"prompt_length": len(result.llm_prompt), "retrieved_items": 2}


def test_evidence_bundle_serializer_is_deterministic():
    pipeline = ResearchPipeline(
        retrieval_runtime=FakeRetrievalRuntime(),
        config=ResearchPipelineConfig(retrieval_mode="text_bitemporal", top_k=2),
    )
    result = pipeline.run(query="wetting")

    prompt_a = serialize_evidence_bundle_for_llm(result.evidence_bundle)
    prompt_b = serialize_evidence_bundle_for_llm(result.evidence_bundle)

    assert prompt_a == prompt_b
    assert "retrieval_mode: text_bitemporal" in prompt_a
    assert "item_id=pair-01" in prompt_a
