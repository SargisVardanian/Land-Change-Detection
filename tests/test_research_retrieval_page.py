from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from land_change_detection.retrieval.contracts import RetrievalMode


PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "Research_Retrieval.py"


def _load_page_module():
    spec = importlib.util.spec_from_file_location("research_retrieval_page", PAGE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_research_page_builds_fake_query_and_pipeline_result():
    page = _load_page_module()
    inputs = page.QueryInputs(
        retrieval_mode=RetrievalMode.TEXT_BITEMPORAL,
        top_k=3,
        backend_name="fake",
        text_query="wetting",
        transition_hint="dry_to_wet",
        generate_explanation=True,
    )

    before = np.zeros((8, 8, 3), dtype=np.uint8)
    after = np.zeros((8, 8, 3), dtype=np.uint8)
    after[..., 1] = 255

    result = page.run_research_query(inputs=inputs, before_image=before, after_image=after)

    assert result.evidence_bundle.retrieval_mode == "text_bitemporal"
    assert len(result.evidence_bundle.retrieved_items) == 3
    assert result.evidence_bundle.retrieved_items[0].item_id == "text_bitemporal:wetting:1"
    assert "before" in result.evidence_bundle.segmentation_summaries
    assert result.explanation["retrieved_count"] == 3


def test_research_page_falls_back_to_fake_backend():
    page = _load_page_module()

    assert page.safe_backend_name("missing-backend") == "fake"
