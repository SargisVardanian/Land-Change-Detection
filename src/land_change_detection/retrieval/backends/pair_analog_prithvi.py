from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import RetrievalArtifact, RetrievalItem, RetrievalMode, RetrievalQuery, RetrievalResult
from ..indexing.manifests import RetrievalManifestItem
from .base import RetrievalBackend
from .static_region_prithvi import (
    EmbeddingBackend,
    FaissVectorIndex,
    NumpyVectorIndex,
    SearchIndexItem,
    _coerce_vector,
    _load_manifest_items,
    _normalize_metadata,
    _rerank_score,
    _text_signature,
    create_embedding_backend,
    resolve_index_path,
    vector_store_name,
)


def build_pair_manifest_item(
    payload: dict[str, Any],
    embedding_backend: EmbeddingBackend,
) -> RetrievalManifestItem:
    metadata = _normalize_metadata(payload.get("metadata"))
    metadata.setdefault("sensor", payload.get("sensor"))
    metadata.setdefault("source", payload.get("source"))
    metadata.setdefault("region_id", payload.get("region_id"))
    metadata.setdefault("geography", payload.get("geography"))
    metadata.setdefault("transition_label", payload.get("transition_label"))
    signature = _text_signature(
        payload.get("item_id"),
        payload.get("before_image_path"),
        payload.get("after_image_path"),
        payload.get("transition_label"),
        metadata,
    )
    vector = payload.get("vector")
    if vector is None:
        vector = embedding_backend.embed_signature(signature).tolist()
    return RetrievalManifestItem(
        item_id=str(payload["item_id"]),
        mode=RetrievalMode(payload.get("mode", RetrievalMode.PAIR_ANALOG.value)),
        vector=[float(value) for value in vector],
        metadata={
            **metadata,
            "before_image_path": payload.get("before_image_path"),
            "after_image_path": payload.get("after_image_path"),
            "thumbnail_path": payload.get("thumbnail_path"),
        },
    )


def _build_pair_query_signature(query: RetrievalQuery) -> str:
    return _text_signature(
        query.item_id,
        query.before_image_path,
        query.after_image_path,
        query.text,
        query.filters,
        query.metadata,
    )


class PairAnalogPrithviBackend(RetrievalBackend):
    index_filename = "pair_analog_index.json"

    def __init__(
        self,
        model_dir: str | Path,
        device: str,
        *,
        embedding_backend_name: str = "numpy",
        prefer_faiss: bool = False,
    ):
        super().__init__(model_dir=model_dir, device=device)
        self.embedding_backend = create_embedding_backend(embedding_backend_name)
        self.index_path = resolve_index_path(self.model_dir, self.index_filename)
        self._index_items = self._load_index_items()
        self.vector_store_type = vector_store_name(prefer_faiss)
        self.vector_index = self._build_vector_index(prefer_faiss=prefer_faiss)

    def _load_index_items(self) -> list[SearchIndexItem]:
        if self.index_path is None:
            return []
        items = _load_manifest_items(self.index_path)
        return [
            SearchIndexItem(
                manifest=item,
                vector=_coerce_vector(item.vector),
                metadata=_normalize_metadata(item.metadata),
            )
            for item in items
            if item.mode in {RetrievalMode.PAIR_ANALOG, RetrievalMode.TRANSITION_CONDITIONED}
        ]

    def _build_vector_index(self, *, prefer_faiss: bool) -> NumpyVectorIndex:
        if prefer_faiss:
            try:
                return FaissVectorIndex(self._index_items)
            except RuntimeError:
                pass
        return NumpyVectorIndex(self._index_items)

    def retrieve(self, query: RetrievalQuery) -> RetrievalArtifact:
        query_metadata = _normalize_metadata({**query.filters, **query.metadata})
        query_vector = self.embedding_backend.embed_signature(_build_pair_query_signature(query))
        raw_ranked = self.vector_index.search(query_vector, max(query.top_k * 3, query.top_k))
        rescored = []
        for item, score in raw_ranked:
            rescored.append((item, score + _rerank_score(query_metadata, item.metadata), score))
        rescored.sort(key=lambda row: (-row[1], -row[2], row[0].manifest.item_id))
        items: list[RetrievalItem] = []
        for rank, (index_item, score, raw_score) in enumerate(rescored[: query.top_k], start=1):
            metadata = dict(index_item.metadata)
            metadata.update(
                {
                    "raw_score": round(raw_score, 6),
                    "vector_store": self.vector_store_type,
                    "embedding_backend": self.embedding_backend.name,
                }
            )
            items.append(
                RetrievalItem(
                    item_id=index_item.manifest.item_id,
                    score=float(round(score, 6)),
                    mode=query.mode,
                    rank=rank,
                    metadata=metadata,
                    thumbnail_path=metadata.get("thumbnail_path"),
                    before_image_path=metadata.get("before_image_path"),
                    after_image_path=metadata.get("after_image_path"),
                    region=metadata.get("region"),
                    transition_hint=metadata.get("transition_label"),
                )
            )
        result = RetrievalResult(
            query=query,
            items=items,
            backend_name="pair_analog_prithvi",
            metadata={
                "index_path": str(self.index_path) if self.index_path is not None else None,
                "index_size": len(self._index_items),
                "vector_store": self.vector_store_type,
                "embedding_backend": self.embedding_backend.name,
            },
        )
        return RetrievalArtifact(
            mode=query.mode,
            result=result,
            metadata={"item_count": len(items)},
        )
