from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Protocol

from ..contracts import RetrievalArtifact, RetrievalItem, RetrievalMode, RetrievalQuery, RetrievalResult
from .base import RetrievalBackend


class TextEncoder(Protocol):
    def encode_text(self, text: str) -> list[float]:
        ...


class PairEncoder(Protocol):
    def encode_pair(
        self,
        *,
        item_id: str,
        before_image_path: str | None,
        after_image_path: str | None,
        metadata: dict[str, Any],
        region_hint: dict[str, Any] | None = None,
    ) -> list[float]:
        ...


class ProjectionHead(Protocol):
    def project(self, vector: list[float]) -> list[float]:
        ...


@dataclass(frozen=True)
class TextBitemporalArchiveItem:
    item_id: str
    before_image_path: str | None = None
    after_image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    region_hint: dict[str, Any] | None = None
    thumbnail_path: str | None = None
    transition_hint: str | None = None


class IdentityProjectionHead:
    def project(self, vector: list[float]) -> list[float]:
        return list(vector)


class DeterministicTextEncoder:
    def __init__(self, dims: int = 8):
        self.dims = dims

    def encode_text(self, text: str) -> list[float]:
        return _hashed_vector(text, dims=self.dims)


class DeterministicPairEncoder:
    def __init__(self, dims: int = 8):
        self.dims = dims

    def encode_pair(
        self,
        *,
        item_id: str,
        before_image_path: str | None,
        after_image_path: str | None,
        metadata: dict[str, Any],
        region_hint: dict[str, Any] | None = None,
    ) -> list[float]:
        parts = [
            item_id,
            before_image_path or "",
            after_image_path or "",
            metadata.get("transition_hint", ""),
            metadata.get("sensor", ""),
            metadata.get("acquisition_date", ""),
            _region_key(region_hint),
        ]
        return _hashed_vector("|".join(parts), dims=self.dims)


class TextBitemporalRemoteCLIPBackend(RetrievalBackend):
    def __init__(
        self,
        model_dir: str,
        device: str,
        *,
        archive_items: list[TextBitemporalArchiveItem] | None = None,
        text_encoder: TextEncoder | None = None,
        pair_encoder: PairEncoder | None = None,
        text_projection: ProjectionHead | None = None,
        pair_projection: ProjectionHead | None = None,
    ):
        super().__init__(model_dir=model_dir, device=device)
        self.archive_items = archive_items or []
        self.text_encoder = text_encoder or DeterministicTextEncoder()
        self.pair_encoder = pair_encoder or DeterministicPairEncoder()
        self.text_projection = text_projection or IdentityProjectionHead()
        self.pair_projection = pair_projection or IdentityProjectionHead()

    def retrieve(self, query: RetrievalQuery) -> RetrievalArtifact:
        if query.mode not in (RetrievalMode.TEXT_BITEMPORAL, RetrievalMode.TRANSITION_CONDITIONED):
            raise ValueError(
                "TextBitemporalRemoteCLIPBackend supports only text_bitemporal or transition_conditioned queries."
            )

        query_text = self._build_query_text(query)
        query_region_hint = query.region_hint or query.filters.get("region_hint")
        query_vector = self.text_projection.project(self.text_encoder.encode_text(query_text))
        pooled_query_vector = changed_region_pool(query_vector, query_region_hint)

        ranked: list[RetrievalItem] = []
        scored_items: list[tuple[TextBitemporalArchiveItem, float]] = []
        for archive_item in self.archive_items:
            pair_vector = self.pair_encoder.encode_pair(
                item_id=archive_item.item_id,
                before_image_path=archive_item.before_image_path,
                after_image_path=archive_item.after_image_path,
                metadata=archive_item.metadata,
                region_hint=archive_item.region_hint,
            )
            pair_vector = self.pair_projection.project(pair_vector)
            pair_vector = changed_region_pool(pair_vector, archive_item.region_hint)
            score = cosine_similarity(pooled_query_vector, pair_vector)
            if query_region_hint and archive_item.region_hint:
                score += 0.05 * region_overlap_score(query_region_hint, archive_item.region_hint)
            scored_items.append((archive_item, min(score, 1.0)))

        scored_items.sort(key=lambda item: (-item[1], item[0].item_id))
        for rank, (archive_item, score) in enumerate(scored_items[: query.top_k], start=1):
            ranked.append(
                RetrievalItem(
                    item_id=archive_item.item_id,
                    score=round(score, 6),
                    mode=query.mode,
                    rank=rank,
                    metadata={
                        **archive_item.metadata,
                        "query_text": query_text,
                        "localized_query": bool(query_region_hint),
                        "backend": "text_bitemporal_remoteclip",
                    },
                    thumbnail_path=archive_item.thumbnail_path,
                    before_image_path=archive_item.before_image_path,
                    after_image_path=archive_item.after_image_path,
                    region=archive_item.region_hint,
                    transition_hint=archive_item.transition_hint or archive_item.metadata.get("transition_hint"),
                )
            )

        result = RetrievalResult(
            query=query,
            items=ranked,
            backend_name="text_bitemporal_remoteclip",
            metadata={
                "deterministic": isinstance(self.text_encoder, DeterministicTextEncoder)
                and isinstance(self.pair_encoder, DeterministicPairEncoder),
                "archive_size": len(self.archive_items),
                "query_text": query_text,
            },
        )
        return RetrievalArtifact(
            mode=query.mode,
            result=result,
            metadata={
                "localized_query": bool(query_region_hint),
                "retrieved_count": len(ranked),
            },
        )

    def _build_query_text(self, query: RetrievalQuery) -> str:
        if query.text:
            return query.text.strip()

        from_class = query.filters.get("from_class")
        to_class = query.filters.get("to_class")
        transition = query.filters.get("transition_hint") or query.filters.get("transition")
        if from_class and to_class:
            return f"{from_class} to {to_class}"
        if transition:
            return str(transition)
        raise ValueError("Text retrieval query must provide free text or structured transition filters.")


def changed_region_pool(vector: list[float], region_hint: dict[str, Any] | None) -> list[float]:
    if not region_hint:
        return normalize_vector(vector)

    width = max(0.0, float(region_hint.get("x1", 1.0)) - float(region_hint.get("x0", 0.0)))
    height = max(0.0, float(region_hint.get("y1", 1.0)) - float(region_hint.get("y0", 0.0)))
    area_weight = max(0.1, min(1.0, width * height))
    pooled = [value * (1.0 + area_weight) for value in vector]
    return normalize_vector(pooled)


def region_overlap_score(query_region: dict[str, Any], item_region: dict[str, Any]) -> float:
    qx0 = float(query_region.get("x0", 0.0))
    qy0 = float(query_region.get("y0", 0.0))
    qx1 = float(query_region.get("x1", 0.0))
    qy1 = float(query_region.get("y1", 0.0))
    ix0 = float(item_region.get("x0", 0.0))
    iy0 = float(item_region.get("y0", 0.0))
    ix1 = float(item_region.get("x1", 0.0))
    iy1 = float(item_region.get("y1", 0.0))

    inter_w = max(0.0, min(qx1, ix1) - max(qx0, ix0))
    inter_h = max(0.0, min(qy1, iy1) - max(qy0, iy0))
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    query_area = max(0.0, (qx1 - qx0) * (qy1 - qy0))
    item_area = max(0.0, (ix1 - ix0) * (iy1 - iy0))
    union_area = query_area + item_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = normalize_vector(left)
    right_norm = normalize_vector(right)
    return sum(a * b for a, b in zip(left_norm, right_norm, strict=False))


def normalize_vector(vector: list[float]) -> list[float]:
    magnitude = sqrt(sum(value * value for value in vector))
    if magnitude == 0.0:
        return [0.0 for _ in vector]
    return [value / magnitude for value in vector]


def _hashed_vector(text: str, *, dims: int) -> list[float]:
    values = [0.0] * dims
    for index, char in enumerate(text.lower()):
        slot = index % dims
        values[slot] += ((ord(char) % 31) + 1) / 31.0
    return normalize_vector(values)


def _region_key(region_hint: dict[str, Any] | None) -> str:
    if not region_hint:
        return ""
    ordered = sorted((str(key), str(value)) for key, value in region_hint.items())
    return ";".join(f"{key}={value}" for key, value in ordered)
