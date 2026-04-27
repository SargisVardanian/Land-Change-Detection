from __future__ import annotations

from land_change_detection.change_interpretation.parser import parse_cell_response
from land_change_detection.remote_sensing_vlm import (
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
    parsed = parse_model_response(
        """
        {
          "scene_overview": "The crop is mostly stable, with local surface rework near the lower road edge.",
          "before_summary": "Before, the area is mostly pale bare ground with roads and scattered built surfaces.",
          "after_summary": "After, the same area remains mostly bare, but several cells show clearer road or plot traces.",
          "main_changes": [
            "Possible road-edge or access-line rework is visible.",
            "Some bare ground appears newly graded or compacted."
          ],
          "cell_observations": [
            {"cell": "B3", "observation": "Possible compacted ground or service-line change.", "confidence": "medium"},
            {"cell": "D4", "observation": "Mostly stable aside from weak tone differences.", "confidence": "low"}
          ]
        }
        """
    )
    assert has_complete_final_response(parsed)
    assert parsed["before_summary"].startswith("Before")
    assert parsed["main_changes"][0].startswith("Possible")
    assert parsed["cell_observations"][0]["cell"] == "B3"


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


def test_normalize_cell_observations_from_strings():
    rows = normalize_cell_observations(["B3: possible road or plot-line rework", "bad placeholder"])
    assert rows[0] == {
        "cell": "B3",
        "observation": "possible road or plot-line rework",
        "confidence": "uncertain",
    }
