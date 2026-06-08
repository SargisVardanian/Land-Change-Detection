from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _normalize_sequence(values: list[Any] | tuple[Any, ...] | None) -> list[Any]:
    if not values:
        return []
    return [_normalize_scalar(value) for value in values]


@dataclass(frozen=True)
class EvidenceRetrievedItem:
    item_id: str
    score: float
    rank: int
    retrieval_mode: str
    acquisition_dates: tuple[str, ...] = ()
    sensor: str | None = None
    source: str | None = None
    transition_hints: tuple[str, ...] = ()
    thumbnail_paths: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = float(self.score)
        payload["acquisition_dates"] = _normalize_sequence(list(self.acquisition_dates))
        payload["transition_hints"] = _normalize_sequence(list(self.transition_hints))
        payload["thumbnail_paths"] = _normalize_sequence(list(self.thumbnail_paths))
        payload["metadata"] = {str(key): _normalize_scalar(value) for key, value in self.metadata.items()}
        return payload


@dataclass(frozen=True)
class EvidenceBundle:
    query: str
    retrieval_mode: str
    top_k: int
    retrieved_items: tuple[EvidenceRetrievedItem, ...]
    scores: tuple[float, ...]
    segmentation_summaries: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    transition_hints: tuple[str, ...] = ()
    thumbnail_paths: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "retrieval_mode": self.retrieval_mode,
            "top_k": int(self.top_k),
            "retrieved_items": [item.to_dict() for item in self.retrieved_items],
            "scores": [float(score) for score in self.scores],
            "segmentation_summaries": self.segmentation_summaries,
            "transition_hints": _normalize_sequence(list(self.transition_hints)),
            "thumbnail_paths": _normalize_sequence(list(self.thumbnail_paths)),
            "metadata": {str(key): _normalize_scalar(value) for key, value in self.metadata.items()},
        }


def serialize_evidence_bundle_for_llm(bundle: EvidenceBundle) -> str:
    lines = [
        "Research Evidence Bundle",
        f"query: {bundle.query}",
        f"retrieval_mode: {bundle.retrieval_mode}",
        f"top_k: {bundle.top_k}",
    ]
    if bundle.transition_hints:
        lines.append("transition_hints: " + ", ".join(sorted(str(item) for item in bundle.transition_hints)))
    for key in sorted(bundle.segmentation_summaries):
        lines.append(f"{key}_segmentation_summary:")
        for row in bundle.segmentation_summaries[key]:
            label = str(row.get("label", "unknown"))
            percent = float(row.get("percent", 0.0))
            lines.append(f"  - {label}: {percent:.2f}%")
    lines.append("retrieved_items:")
    ordered_items = sorted(bundle.retrieved_items, key=lambda item: (-float(item.score), item.item_id))
    for item in ordered_items:
        dates = ", ".join(item.acquisition_dates) if item.acquisition_dates else "unknown"
        transitions = ", ".join(sorted(item.transition_hints)) if item.transition_hints else "none"
        thumbs = ", ".join(item.thumbnail_paths) if item.thumbnail_paths else "none"
        sensor = item.sensor or "unknown"
        source = item.source or "unknown"
        lines.append(
            f"  - rank={item.rank}; item_id={item.item_id}; score={float(item.score):.4f}; "
            f"dates={dates}; sensor={sensor}; source={source}; transitions={transitions}; thumbnails={thumbs}"
        )
    return "\n".join(lines)
