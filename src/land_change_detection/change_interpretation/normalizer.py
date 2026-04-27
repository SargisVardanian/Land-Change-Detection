from __future__ import annotations

from .contracts import CellInterpretation, SceneInterpretation


def normalize_cell_interpretation(report: CellInterpretation) -> CellInterpretation:
    report.before_observations = [item.strip() for item in report.before_observations if str(item).strip()]
    report.after_observations = [item.strip() for item in report.after_observations if str(item).strip()]
    report.stable_elements = [item.strip() for item in report.stable_elements if str(item).strip()]
    report.changed_elements = [item.strip() for item in report.changed_elements if str(item).strip()]
    report.alternative_hypotheses = [item.strip() for item in report.alternative_hypotheses if str(item).strip()]
    report.primary_transition = report.primary_transition.strip()
    report.final_summary = report.final_summary.strip()
    return report


def normalize_scene_interpretation(scene: SceneInterpretation) -> SceneInterpretation:
    scene.scene_overview = scene.scene_overview.strip()
    scene.cell_reports = [normalize_cell_interpretation(item) for item in scene.cell_reports]
    return scene
