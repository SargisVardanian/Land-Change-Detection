from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from land_change_detection.contracts import SegmentationArtifact
from land_change_detection.pipelines.research_pipeline import (
    ResearchPipeline,
    ResearchPipelineConfig,
    ResearchPipelineResult,
)
from land_change_detection.retrieval.contracts import RetrievalMode, RetrievalQuery
from land_change_detection.retrieval.registry import available_retrieval_backends
from land_change_detection.retrieval.runtime import RetrievalRuntime


MODE_LABELS: dict[RetrievalMode, str] = {
    RetrievalMode.STATIC_REGION: "Static Region",
    RetrievalMode.PAIR_ANALOG: "Pair Analog",
    RetrievalMode.TEXT_BITEMPORAL: "Text Bitemporal",
    RetrievalMode.NOVELTY_SINGLE_IMAGE: "Novelty Single Image",
    RetrievalMode.TRANSITION_CONDITIONED: "Transition Conditioned",
    RetrievalMode.TRAJECTORY: "Trajectory",
}


class FakePageExplainerRuntime:
    def explain(self, prompt: str, evidence: dict[str, Any]) -> dict[str, Any]:
        retrieved = evidence.get("retrieved_items", [])
        top_item = retrieved[0]["item_id"] if retrieved else None
        top_score = retrieved[0]["score"] if retrieved else None
        return {
            "summary": f"Top evidence item: {top_item}" if top_item else "No retrieved evidence.",
            "top_score": top_score,
            "retrieved_count": len(retrieved),
            "prompt_preview": "\n".join(prompt.splitlines()[:6]),
        }


class FakePageSegmentationRuntime:
    backend_name = "page_fake_segmentation"

    def run(self, image: np.ndarray) -> SegmentationArtifact:
        height, width = image.shape[:2]
        semantic_map = np.ones((height, width), dtype=np.int32)
        if image.ndim == 3 and image.shape[2] >= 3:
            semantic_map[image[..., 1] > image[..., 2]] = 2
            semantic_map[image[..., 0] > image[..., 1]] = 3
        total = float(height * width) if height and width else 1.0
        label_summary = [
            {"label": "cropland", "percent": round(float(np.mean(semantic_map == 1) * 100.0), 2)},
            {"label": "tree_cover", "percent": round(float(np.mean(semantic_map == 2) * 100.0), 2)},
            {"label": "built_up", "percent": round(float(np.mean(semantic_map == 3) * 100.0), 2)},
        ]
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[semantic_map == 1] = (60, 140, 70)
        overlay[semantic_map == 2] = (20, 90, 30)
        overlay[semantic_map == 3] = (180, 70, 70)
        return SegmentationArtifact(
            semantic_map=semantic_map,
            confidence_map=np.full((height, width), fill_value=0.5, dtype=np.float32),
            label_summary=[row for row in label_summary if row["percent"] > 0.0 or total == 0.0],
            overlay_rgb=overlay,
            legend={1: "cropland", 2: "tree_cover", 3: "built_up"},
            metadata={"backend": self.backend_name},
        )


class QueryInputs:
    def __init__(
        self,
        *,
        retrieval_mode: RetrievalMode,
        top_k: int,
        backend_name: str,
        text_query: str | None = None,
        image_path: str | None = None,
        before_image_path: str | None = None,
        after_image_path: str | None = None,
        item_id: str | None = None,
        transition_hint: str | None = None,
        generate_explanation: bool = False,
    ):
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.backend_name = backend_name
        self.text_query = text_query
        self.image_path = image_path
        self.before_image_path = before_image_path
        self.after_image_path = after_image_path
        self.item_id = item_id
        self.transition_hint = transition_hint
        self.generate_explanation = generate_explanation


def resolve_retrieval_mode(raw_value: str | RetrievalMode) -> RetrievalMode:
    if isinstance(raw_value, RetrievalMode):
        return raw_value
    return RetrievalMode(raw_value)


def build_query(inputs: QueryInputs) -> RetrievalQuery:
    filters: dict[str, Any] = {}
    if inputs.transition_hint:
        filters["transition_hint"] = inputs.transition_hint
    return RetrievalQuery(
        mode=inputs.retrieval_mode,
        top_k=inputs.top_k,
        text=(inputs.text_query or None),
        item_id=(inputs.item_id or None),
        image_path=(inputs.image_path or None),
        before_image_path=(inputs.before_image_path or None),
        after_image_path=(inputs.after_image_path or None),
        filters=filters,
        metadata={"page": "Research_Retrieval"},
    )


def safe_backend_name(raw_backend_name: str) -> str:
    backends = available_retrieval_backends()
    if raw_backend_name in backends:
        return raw_backend_name
    if "fake" in backends:
        return "fake"
    return backends[0]


def build_research_pipeline(
    *,
    backend_name: str,
    retrieval_mode: RetrievalMode,
    top_k: int,
    include_segmentation: bool,
    generate_explanation: bool,
) -> ResearchPipeline:
    retrieval_runtime = RetrievalRuntime(backend_name=safe_backend_name(backend_name), model_dir=".", device="cpu")
    segmentation_runtime = FakePageSegmentationRuntime() if include_segmentation else None
    explainer_runtime = FakePageExplainerRuntime() if generate_explanation else None
    return ResearchPipeline(
        segmentation_runtime=segmentation_runtime,
        retrieval_runtime=retrieval_runtime,
        explainer_runtime=explainer_runtime,
        config=ResearchPipelineConfig(
            retrieval_mode=retrieval_mode.value,
            top_k=top_k,
        ),
    )


def load_image_array(source: Any) -> np.ndarray | None:
    if source is None:
        return None
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            return None
        image = Image.open(path).convert("RGB")
        return np.array(image)
    if hasattr(source, "read"):
        data = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        image = Image.open(BytesIO(data)).convert("RGB")
        return np.array(image)
    return None


def run_research_query(
    *,
    inputs: QueryInputs,
    before_image: np.ndarray | None = None,
    after_image: np.ndarray | None = None,
) -> ResearchPipelineResult:
    query = build_query(inputs)
    include_segmentation = before_image is not None and after_image is not None
    pipeline = build_research_pipeline(
        backend_name=inputs.backend_name,
        retrieval_mode=inputs.retrieval_mode,
        top_k=inputs.top_k,
        include_segmentation=include_segmentation,
        generate_explanation=inputs.generate_explanation,
    )
    return pipeline.run(
        query=query,
        before_img=before_image,
        after_img=after_image,
    )


def render_results(result: ResearchPipelineResult) -> None:
    import streamlit as st

    st.subheader("Retrieved Evidence")
    if not result.evidence_bundle.retrieved_items:
        st.info("No retrieval results.")
        return
    for item in result.evidence_bundle.retrieved_items:
        metadata = {
            "score": round(float(item.score), 4),
            "rank": item.rank,
            "sensor": item.sensor,
            "source": item.source,
            "dates": list(item.acquisition_dates),
            "transitions": list(item.transition_hints),
            "thumbnails": list(item.thumbnail_paths),
            "metadata": item.metadata,
        }
        with st.expander(f"#{item.rank} {item.item_id}", expanded=item.rank == 1):
            st.json(metadata)

    st.subheader("Evidence Bundle")
    st.json(result.evidence_bundle.to_dict())
    st.subheader("LLM Prompt")
    st.code(result.llm_prompt, language="text")
    if result.explanation is not None:
        st.subheader("Explanation")
        st.json(result.explanation)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Research Retrieval", layout="wide")
    st.title("Research Retrieval")
    st.caption("Research-only retrieval page. Default backend is deterministic fake runtime.")

    backend_options = available_retrieval_backends()
    default_backend_index = backend_options.index("fake") if "fake" in backend_options else 0
    mode = resolve_retrieval_mode(
        st.selectbox(
            "Retrieval Mode",
            options=[mode.value for mode in RetrievalMode],
            format_func=lambda value: MODE_LABELS[resolve_retrieval_mode(value)],
        )
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        backend_name = st.selectbox("Backend", options=backend_options, index=default_backend_index)
    with col_b:
        top_k = int(st.slider("Top K", min_value=1, max_value=10, value=5))
    with col_c:
        generate_explanation = st.checkbox("Generate Explanation", value=False)

    text_query = st.text_input("Text Query", value="" if mode is not RetrievalMode.TEXT_BITEMPORAL else "wetting")
    transition_hint = st.text_input("Transition Hint", value="")
    item_id = st.text_input("Reference Item ID", value="")

    image_path = st.text_input("Single Image Path", value="")
    before_image_path = st.text_input("Before Image Path", value="")
    after_image_path = st.text_input("After Image Path", value="")

    upload_col_a, upload_col_b = st.columns(2)
    with upload_col_a:
        before_upload = st.file_uploader("Upload Before Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    with upload_col_b:
        after_upload = st.file_uploader("Upload After Image", type=["png", "jpg", "jpeg", "tif", "tiff"])

    before_image = load_image_array(before_upload)
    if before_image is None:
        before_image = load_image_array(before_image_path)
    after_image = load_image_array(after_upload)
    if after_image is None:
        after_image = load_image_array(after_image_path)

    if st.button("Run Retrieval", type="primary"):
        inputs = QueryInputs(
            retrieval_mode=mode,
            top_k=top_k,
            backend_name=backend_name,
            text_query=text_query,
            image_path=image_path,
            before_image_path=before_image_path or getattr(before_upload, "name", None),
            after_image_path=after_image_path or getattr(after_upload, "name", None),
            item_id=item_id,
            transition_hint=transition_hint,
            generate_explanation=generate_explanation,
        )
        result = run_research_query(
            inputs=inputs,
            before_image=before_image,
            after_image=after_image,
        )
        render_results(result)


if __name__ == "__main__":
    main()
