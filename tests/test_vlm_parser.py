from __future__ import annotations

from land_change_detection.change_interpretation.parser import parse_cell_response


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
