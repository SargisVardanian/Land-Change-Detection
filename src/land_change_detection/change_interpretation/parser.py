from __future__ import annotations

import json

from ..schema_validation import validate_uncertainty
from .contracts import CellInterpretation, SceneInterpretation


def parse_cell_response(raw_text: str) -> CellInterpretation:
    data = json.loads(raw_text)
    return CellInterpretation(
        cell_id=str(data["cell_id"]),
        before_observations=list(data.get("before_observations", [])),
        after_observations=list(data.get("after_observations", [])),
        stable_elements=list(data.get("stable_elements", [])),
        changed_elements=list(data.get("changed_elements", [])),
        primary_transition=str(data.get("primary_transition", "")),
        alternative_hypotheses=list(data.get("alternative_hypotheses", [])),
        uncertainty=validate_uncertainty(str(data.get("uncertainty", "medium"))),
        final_summary=str(data.get("final_summary", "")),
        raw_response=raw_text,
    )


def parse_scene_response(raw_text: str) -> SceneInterpretation:
    data = json.loads(raw_text)
    reports = [parse_cell_response(json.dumps(item)) for item in data.get("cell_reports", [])]
    return SceneInterpretation(
        scene_overview=str(data.get("scene_overview", "")),
        cell_reports=reports,
        global_uncertainty=validate_uncertainty(str(data.get("global_uncertainty", "medium"))),
        metadata={"raw_response": raw_text},
    )
