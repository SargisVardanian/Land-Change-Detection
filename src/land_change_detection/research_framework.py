from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .change_interpretation.deterministic_reporting import (
    build_cell_report_rows_from_segmentation,
    build_scene_overview_from_rows,
)
from .contracts import PipelineV2Result
from .semantic_surface import summarize_transitions


@dataclass(frozen=True)
class ResearchQuestion:
    id: str
    prompt: str
    evidence: str
    decision_value: str


@dataclass(frozen=True)
class ResearchProtocol:
    name: str
    objective: str
    unit_of_analysis: str
    primary_pipeline: str
    baselines: tuple[str, ...]
    quality_checks: tuple[str, ...]
    research_questions: tuple[ResearchQuestion, ...]


@dataclass(frozen=True)
class ResearchEvidenceReport:
    protocol: ResearchProtocol
    scene_id: str
    bbox: dict[str, int]
    generated_at: str
    image_shape: tuple[int, int, int]
    grid: dict[str, int]
    segmentation_backend: str
    class_summary_before: list[dict[str, Any]]
    class_summary_after: list[dict[str, Any]]
    transitions: list[dict[str, Any]]
    cell_rows: list[dict[str, Any]]
    scene_overview: str
    evidence_limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.protocol.name}",
            "",
            f"- Scene: `{self.scene_id}`",
            f"- Generated: `{self.generated_at}`",
            f"- Crop bbox: `{self.bbox}`",
            f"- Crop shape: `{self.image_shape}`",
            f"- Grid: `{self.grid['rows']}x{self.grid['cols']}`",
            f"- Segmentation backend: `{self.segmentation_backend}`",
            "",
            "## Objective",
            self.protocol.objective,
            "",
            "## Research Questions",
        ]
        for question in self.protocol.research_questions:
            lines.append(f"- **{question.id}**: {question.prompt} Evidence: {question.evidence}. Decision value: {question.decision_value}.")
        lines.extend(["", "## Scene Overview", self.scene_overview, "", "## Top Semantic Transitions"])
        if self.transitions:
            for item in self.transitions[:12]:
                lines.append(f"- {item['before']} -> {item['after']}: {item['percent']:.2f}% ({item['pixels']} px)")
        else:
            lines.append("- No semantic transitions above the reporting threshold.")
        lines.extend(["", "## Highest Priority Cells"])
        for row in sorted(self.cell_rows, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:8]:
            lines.append(
                "- "
                + f"{row['cell']}: {row['likely_change']} "
                + f"(score {float(row.get('score', 0.0)):.1f}, confidence {row.get('confidence', 'unknown')}). "
                + f"Support: {row.get('support', 'none')}."
            )
        lines.extend(["", "## Evidence Limitations"])
        lines.extend(f"- {item}" for item in self.evidence_limitations)
        return "\n".join(lines).strip() + "\n"


DEFAULT_RESEARCH_PROTOCOL = ResearchProtocol(
    name="Semantic Land-Cover Change Research Framework",
    objective=(
        "Convert paired T1/T2 imagery into an interpretable land-cover transition study: "
        "segment each timestamp, compare class transitions, rank spatial cells, and separate "
        "pixel evidence from narrative explanation."
    ),
    unit_of_analysis="aligned crop plus 4x4 research grid cells",
    primary_pipeline="T1 semantic segmentation -> T2 semantic segmentation -> transition matrix -> ranked cell evidence -> report",
    baselines=(
        "visual RGB before/after inspection",
        "cell-level pixel-difference prioritization",
        "future binary changed/unchanged baseline for localization only",
    ),
    quality_checks=(
        "T1 and T2 crop dimensions must match",
        "semantic maps must be aligned to the crop grid",
        "transition claims must cite class support and cell location",
        "VLM prose must remain secondary to segmentation-derived evidence",
    ),
    research_questions=(
        ResearchQuestion(
            id="RQ1",
            prompt="What land-cover classes changed between T1 and T2?",
            evidence="semantic transition matrix",
            decision_value="identifies what changed into what",
        ),
        ResearchQuestion(
            id="RQ2",
            prompt="Where are the strongest localized changes?",
            evidence="ranked 4x4 cell reports",
            decision_value="prioritizes review and field validation targets",
        ),
        ResearchQuestion(
            id="RQ3",
            prompt="Which changes are policy-relevant for Armenia-focused environmental and agricultural monitoring?",
            evidence="class transitions involving cropland, vegetation, built-up, bare ground, and water",
            decision_value="supports practical monitoring narratives",
        ),
    ),
)


def build_research_evidence_report(
    pipeline_result: PipelineV2Result,
    *,
    scene_id: str,
    bbox: dict[str, int],
    protocol: ResearchProtocol = DEFAULT_RESEARCH_PROTOCOL,
    top_k: int = 16,
) -> ResearchEvidenceReport:
    before_map = pipeline_result.before_segmentation.class_map
    after_map = pipeline_result.after_segmentation.class_map
    if before_map.shape != after_map.shape:
        raise ValueError(f"semantic maps must match, got {before_map.shape} vs {after_map.shape}")

    rows = build_cell_report_rows_from_segmentation(pipeline_result.cell_packs)
    transitions = [
        item
        for item in summarize_transitions(
            before_map,
            after_map,
            top_k=top_k,
            id2label=pipeline_result.before_segmentation.legend,
        )
        if item["before"] != item["after"]
    ]
    metadata = pipeline_result.metadata or {}
    image_shape = tuple(int(value) for value in pipeline_result.cell_packs[0].before_crop.shape) if pipeline_result.cell_packs else (0, 0, 0)
    if pipeline_result.cell_packs:
        full_h = int(before_map.shape[0])
        full_w = int(before_map.shape[1])
        image_shape = (full_h, full_w, int(np.asarray(pipeline_result.cell_packs[0].before_crop).shape[2]))

    return ResearchEvidenceReport(
        protocol=protocol,
        scene_id=scene_id,
        bbox=bbox,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        image_shape=image_shape,
        grid={"rows": int(metadata.get("rows", 4)), "cols": int(metadata.get("cols", 4))},
        segmentation_backend=str(metadata.get("segmentation_backend", "unknown")),
        class_summary_before=pipeline_result.before_segmentation.label_summary,
        class_summary_after=pipeline_result.after_segmentation.label_summary,
        transitions=transitions,
        cell_rows=rows,
        scene_overview=build_scene_overview_from_rows(rows),
        evidence_limitations=(
            "This report assumes T1/T2 imagery is co-registered at the crop scale.",
            "RGB-only Mask2Former output is a current prototype baseline, not a multispectral final model.",
            "Cell rankings are research triage signals and should be checked against source imagery or field evidence.",
        ),
    )
