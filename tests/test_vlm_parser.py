from __future__ import annotations

import json

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torchvision")

from land_change_detection.change_interpretation.parser import parse_cell_response
from land_change_detection.remote_sensing_vlm import (
    EXPECTED_GRID_CELLS,
    OllamaSemanticChangeExplainer,
    REASONING_PROFILES,
    cell_observations_quality_issues,
    has_complete_final_response,
    normalize_cell_observations,
    parse_model_response,
)


def test_parse_cell_response():
    raw = """
    {
      "cell_id": "B3",
      "before_observations": ["bare ground", "linear edge"],
      "after_observations": ["bare ground", "new rectilinear feature"],
      "stable_elements": ["main road edge"],
      "changed_elements": ["new compact surface"],
      "primary_transition": "possible prepared-ground to compact-surface change",
      "alternative_hypotheses": ["grading"],
      "uncertainty": "medium",
      "final_summary": "B3 shows a cautious compact-surface addition hypothesis."
    }
    """
    report = parse_cell_response(raw)
    assert report.cell_id == "B3"
    assert report.uncertainty == "medium"
    assert "new compact surface" in report.changed_elements


def test_parse_visual_summary_response():
    cells = [
        {"cell": cell, "observation": f"{cell} shows a specific visible before/after surface comparison near roads or ground texture.", "confidence": "medium"}
        for cell in EXPECTED_GRID_CELLS
    ]
    parsed = parse_model_response(
        json.dumps(
            {
                "scene_overview": "The crop is mostly stable, with local surface rework near the lower road edge.",
                "before_summary": "Before, the area is mostly pale bare ground with roads and scattered built surfaces.",
                "after_summary": "After, the same area remains mostly bare, but several cells show clearer road or plot traces.",
                "main_changes": [
                    "Possible road-edge or access-line rework is visible.",
                    "Some bare ground appears newly graded or compacted.",
                ],
                "cell_observations": cells,
            }
        )
    )
    assert has_complete_final_response(parsed)
    assert parsed["before_summary"].startswith("Before")
    assert parsed["main_changes"][0].startswith("Possible")
    assert parsed["cell_observations"][0]["cell"] == "A1"


def test_rejects_thinking_process_as_user_summary():
    parsed = parse_model_response(
        """
        {
          "scene_overview": "Here's a thinking process to arrive at the desired output.",
          "before_summary": "Before summary.",
          "after_summary": "After summary.",
          "main_changes": ["A visible change."],
          "cell_observations": [{"cell": "A1", "observation": "A visible change.", "confidence": "medium"}]
        }
        """
    )
    assert parsed["scene_overview"] == ""
    assert not has_complete_final_response(parsed)


def test_rejects_repeated_generic_cell_observations():
    rows = normalize_cell_observations(
        [
            {"cell": cell, "observation": "Possible expansion of low vegetation or cultivated surface.", "confidence": "high"}
            for cell in EXPECTED_GRID_CELLS
        ]
    )
    assert rows == []


def test_cell_observation_quality_requires_all_grid_cells():
    rows = [
        {"cell": "A1", "observation": "A1 shows stable road and ground layout with no clear structural change.", "confidence": "medium"},
        {"cell": "B3", "observation": "B3 shows a possible new track or graded ground line.", "confidence": "medium"},
    ]
    assert cell_observations_quality_issues(rows)


def test_normalize_cell_observations_from_strings():
    rows = normalize_cell_observations(["B3: possible road or plot-line rework", "bad placeholder"])
    assert rows[0] == {
        "cell": "B3",
        "observation": "possible road or plot-line rework",
        "confidence": "uncertain",
    }


def test_full_local_ollama_profile_has_larger_budget():
    profile = REASONING_PROFILES["full_local"]
    assert profile.label == "Full local analysis"
    assert profile.ollama_num_ctx == 8192
    assert profile.ollama_num_predict == 2048


def test_ollama_prompt_excludes_semantic_and_dino_context():
    explainer = OllamaSemanticChangeExplainer(reasoning_profile="full_local")
    prompt = explainer._final_prompt(
        semantic_context="Mask2Former says grass, bareland, DINO region, semantic transition.",
        visual_context="RGB-only visual measurement hints.",
        response_language="English",
        reasoning_text="A1 and B2 need inspection.",
        has_change_guide=False,
    )

    assert "RGB-only visual measurement hints." in prompt
    assert "Mask2Former says" not in prompt
    assert "grass, bareland" not in prompt
    assert "DINO region" not in prompt
    assert "change-guide heatmap" not in prompt


def test_ollama_repair_prompt_forces_complete_json_without_semantic_fallback():
    explainer = OllamaSemanticChangeExplainer(reasoning_profile="full_local")
    prompt = explainer._repair_prompt("partial answer", response_language="English")

    assert "strict JSON only" in prompt
    assert "exactly 16 objects" in prompt
    assert "Do not mention semantic segmentation" in prompt
