from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CellInterpretation:
    cell_id: str
    before_observations: list[str]
    after_observations: list[str]
    stable_elements: list[str]
    changed_elements: list[str]
    primary_transition: str
    alternative_hypotheses: list[str]
    uncertainty: str
    final_summary: str
    raw_response: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class SceneInterpretation:
    scene_overview: str
    cell_reports: list[CellInterpretation]
    global_uncertainty: str
    metadata: dict[str, Any] | None = None
