from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..contracts import PipelineV2Result
from ..grid_utils import build_cell_packs
from ..semantic_surface import summarize_transitions
from .evidence_bundle import EvidenceBundle, EvidenceRetrievedItem, serialize_evidence_bundle_for_llm


class SegmentationRuntimeLike(Protocol):
    backend_name: str

    def run(self, image: np.ndarray) -> Any: ...


class RetrievalRuntimeLike(Protocol):
    def run(self, query: Any) -> Any: ...


class ExplainerRuntimeLike(Protocol):
    def explain(self, prompt: str, evidence: dict[str, Any]) -> Any: ...


@dataclass(frozen=True)
class ResearchPipelineConfig:
    retrieval_mode: str
    top_k: int = 5
    rows: int = 4
    cols: int = 4
    include_segmentation_summaries: bool = True
    include_transition_hints: bool = True


@dataclass(frozen=True)
class ResearchPipelineResult:
    pipeline_result: PipelineV2Result | None
    retrieval_artifact: Any
    evidence_bundle: EvidenceBundle
    llm_prompt: str
    explanation: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_result_metadata": self.pipeline_result.metadata if self.pipeline_result is not None else None,
            "retrieval_artifact": _safe_serialize(self.retrieval_artifact),
            "evidence_bundle": self.evidence_bundle.to_dict(),
            "llm_prompt": self.llm_prompt,
            "explanation": _safe_serialize(self.explanation),
            "metadata": self.metadata,
        }


class ResearchPipeline:
    def __init__(
        self,
        *,
        segmentation_runtime: SegmentationRuntimeLike | None = None,
        retrieval_runtime: RetrievalRuntimeLike,
        explainer_runtime: ExplainerRuntimeLike | None = None,
        config: ResearchPipelineConfig | None = None,
    ):
        self.segmentation_runtime = segmentation_runtime
        self.retrieval_runtime = retrieval_runtime
        self.explainer_runtime = explainer_runtime
        self.config = config or ResearchPipelineConfig(retrieval_mode="static_region")

    def run(
        self,
        *,
        query: Any,
        before_img: np.ndarray | None = None,
        after_img: np.ndarray | None = None,
    ) -> ResearchPipelineResult:
        pipeline_result = self._build_pipeline_result(before_img=before_img, after_img=after_img)
        retrieval_artifact = self.retrieval_runtime.run(query)
        evidence_bundle = self._build_evidence_bundle(query=query, retrieval_artifact=retrieval_artifact, pipeline_result=pipeline_result)
        llm_prompt = serialize_evidence_bundle_for_llm(evidence_bundle)
        explanation = None
        if self.explainer_runtime is not None:
            explanation = self.explainer_runtime.explain(llm_prompt, evidence_bundle.to_dict())
        return ResearchPipelineResult(
            pipeline_result=pipeline_result,
            retrieval_artifact=retrieval_artifact,
            evidence_bundle=evidence_bundle,
            llm_prompt=llm_prompt,
            explanation=explanation,
            metadata={"retrieval_mode": self.config.retrieval_mode, "top_k": self.config.top_k},
        )

    def _build_pipeline_result(
        self,
        *,
        before_img: np.ndarray | None,
        after_img: np.ndarray | None,
    ) -> PipelineV2Result | None:
        if self.segmentation_runtime is None:
            return None
        if before_img is None or after_img is None:
            return None
        before_seg = self.segmentation_runtime.run(before_img)
        after_seg = self.segmentation_runtime.run(after_img)
        packs = build_cell_packs(
            before_img=before_img,
            after_img=after_img,
            before_seg=before_seg,
            after_seg=after_seg,
            legend=before_seg.legend,
            rows=self.config.rows,
            cols=self.config.cols,
            scene_thumbnail=before_img,
        )
        return PipelineV2Result(
            before_segmentation=before_seg,
            after_segmentation=after_seg,
            cell_packs=packs,
            scene_report=None,
            metadata={
                "rows": self.config.rows,
                "cols": self.config.cols,
                "segmentation_backend": getattr(self.segmentation_runtime, "backend_name", "unknown"),
            },
        )

    def _build_evidence_bundle(
        self,
        *,
        query: Any,
        retrieval_artifact: Any,
        pipeline_result: PipelineV2Result | None,
    ) -> EvidenceBundle:
        items = _coerce_retrieved_items(retrieval_artifact, self.config.retrieval_mode)
        top_items = tuple(items[: self.config.top_k])
        segmentation_summaries = _build_segmentation_summaries(pipeline_result, include=self.config.include_segmentation_summaries)
        transition_hints = _merge_transition_hints(
            _build_transition_hints(pipeline_result, include=self.config.include_transition_hints),
            tuple(hint for item in top_items for hint in item.transition_hints),
        )
        thumbnail_paths = tuple(path for item in top_items for path in item.thumbnail_paths)
        return EvidenceBundle(
            query=_stringify_query(query),
            retrieval_mode=self.config.retrieval_mode,
            top_k=self.config.top_k,
            retrieved_items=top_items,
            scores=tuple(float(item.score) for item in top_items),
            segmentation_summaries=segmentation_summaries,
            transition_hints=transition_hints,
            thumbnail_paths=thumbnail_paths,
            metadata={
                "segmentation_backend": (
                    str((pipeline_result.metadata or {}).get("segmentation_backend", "unknown")) if pipeline_result is not None else "none"
                ),
                "retrieval_item_count": len(top_items),
            },
        )


def _coerce_retrieved_items(retrieval_artifact: Any, retrieval_mode: str) -> list[EvidenceRetrievedItem]:
    raw_items = []
    if retrieval_artifact is None:
        return []
    if hasattr(retrieval_artifact, "items"):
        raw_items = list(getattr(retrieval_artifact, "items"))
    elif isinstance(retrieval_artifact, dict):
        raw_items = list(retrieval_artifact.get("items", []))
    elif isinstance(retrieval_artifact, (list, tuple)):
        raw_items = list(retrieval_artifact)
    items: list[EvidenceRetrievedItem] = []
    for index, raw in enumerate(raw_items, start=1):
        payload = _safe_serialize(raw)
        if not isinstance(payload, dict):
            payload = {"value": payload}
        item_id = str(payload.get("item_id") or payload.get("id") or f"item-{index}")
        score = float(payload.get("score", 0.0))
        dates = tuple(str(value) for value in payload.get("dates", payload.get("acquisition_dates", [])) if value is not None)
        transition_hints = tuple(str(value) for value in payload.get("transition_hints", []) if value is not None)
        thumbnail_paths = tuple(str(value) for value in payload.get("thumbnail_paths", []) if value is not None)
        metadata = {
            str(key): value
            for key, value in payload.items()
            if key
            not in {"item_id", "id", "score", "dates", "acquisition_dates", "sensor", "source", "transition_hints", "thumbnail_paths"}
        }
        items.append(
            EvidenceRetrievedItem(
                item_id=item_id,
                score=score,
                rank=int(payload.get("rank", index)),
                retrieval_mode=str(payload.get("retrieval_mode", retrieval_mode)),
                acquisition_dates=dates,
                sensor=str(payload["sensor"]) if payload.get("sensor") is not None else None,
                source=str(payload["source"]) if payload.get("source") is not None else None,
                transition_hints=transition_hints,
                thumbnail_paths=thumbnail_paths,
                metadata=metadata,
            )
        )
    return sorted(items, key=lambda item: (-float(item.score), item.item_id))


def _build_segmentation_summaries(
    pipeline_result: PipelineV2Result | None,
    *,
    include: bool,
) -> dict[str, list[dict[str, Any]]]:
    if not include or pipeline_result is None:
        return {}
    return {
        "before": list(pipeline_result.before_segmentation.label_summary),
        "after": list(pipeline_result.after_segmentation.label_summary),
    }


def _build_transition_hints(
    pipeline_result: PipelineV2Result | None,
    *,
    include: bool,
) -> tuple[str, ...]:
    if not include or pipeline_result is None:
        return ()
    transitions = summarize_transitions(
        pipeline_result.before_segmentation.class_map,
        pipeline_result.after_segmentation.class_map,
        top_k=5,
        id2label=pipeline_result.before_segmentation.legend,
    )
    hints = []
    for item in transitions:
        before_label = str(item.get("before", "unknown"))
        after_label = str(item.get("after", "unknown"))
        if before_label == after_label:
            continue
        hints.append(f"{before_label}->{after_label}")
    return tuple(hints)


def _merge_transition_hints(*hint_groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in hint_groups:
        for hint in group:
            if hint in seen:
                continue
            seen.add(hint)
            merged.append(hint)
    return tuple(merged)


def _stringify_query(query: Any) -> str:
    if isinstance(query, str):
        return query
    if isinstance(query, dict):
        if "text" in query:
            return str(query["text"])
        parts = [f"{key}={query[key]}" for key in sorted(query)]
        return ", ".join(parts)
    return str(query)


def _safe_serialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(item) for item in value]
    if hasattr(value, "to_dict"):
        return _safe_serialize(value.to_dict())
    if hasattr(value, "__dict__"):
        return {str(key): _safe_serialize(item) for key, item in vars(value).items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)
