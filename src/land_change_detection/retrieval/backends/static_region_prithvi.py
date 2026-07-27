from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..contracts import RetrievalArtifact, RetrievalItem, RetrievalMode, RetrievalQuery, RetrievalResult
from ..indexing.manifests import RetrievalManifestItem
from .base import RetrievalBackend


def _stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _text_signature(*parts: Any) -> str:
    normalized: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, dict):
            for key in sorted(part):
                normalized.append(f"{key}={part[key]}")
            continue
        if isinstance(part, (list, tuple)):
            normalized.append(",".join(str(item) for item in part))
            continue
        normalized.append(str(part))
    return "|".join(normalized)


def _infer_season(month: Any) -> str | None:
    try:
        month_int = int(month)
    except (TypeError, ValueError):
        return None
    if month_int in (12, 1, 2):
        return "winter"
    if month_int in (3, 4, 5):
        return "spring"
    if month_int in (6, 7, 8):
        return "summer"
    if month_int in (9, 10, 11):
        return "autumn"
    return None


def _normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(metadata or {})
    month = payload.get("month")
    if month is not None:
        try:
            payload["month"] = int(month)
        except (TypeError, ValueError):
            pass
    if not payload.get("season"):
        inferred = _infer_season(payload.get("month"))
        if inferred:
            payload["season"] = inferred
    return payload


def _coerce_vector(values: list[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("Embedding vectors must be one-dimensional.")
    return vector


def _cosine_similarity(query_vector: np.ndarray, item_vector: np.ndarray) -> float:
    query_norm = float(np.linalg.norm(query_vector))
    item_norm = float(np.linalg.norm(item_vector))
    if query_norm == 0.0 or item_norm == 0.0:
        return 0.0
    return float(np.dot(query_vector, item_vector) / (query_norm * item_norm))


class EmbeddingBackend(Protocol):
    name: str
    dimension: int

    def embed_signature(self, signature: str) -> np.ndarray: ...


class FakeDeterministicEmbeddingBackend:
    name = "fake"

    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def embed_signature(self, signature: str) -> np.ndarray:
        seed = _stable_hash(signature)
        vector = np.zeros(self.dimension, dtype=float)
        for index in range(self.dimension):
            shift = (index * 7) % 53
            value = ((seed >> shift) & 0xFF) / 255.0
            vector[index] = value + ((index % 5) * 0.01)
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0.0 else vector / norm


class NumpyBaselineEmbeddingBackend:
    name = "numpy"

    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def embed_signature(self, signature: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=float)
        for token in signature.lower().split("|"):
            token = token.strip()
            if not token:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(self.dimension):
                vector[index] += digest[index % len(digest)] / 255.0
        if not np.any(vector):
            vector[0] = 1.0
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0.0 else vector / norm


@dataclass(frozen=True)
class SearchIndexItem:
    manifest: RetrievalManifestItem
    vector: np.ndarray
    metadata: dict[str, Any]


class NumpyVectorIndex:
    name = "numpy"

    def __init__(self, items: list[SearchIndexItem]):
        self.items = list(items)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[SearchIndexItem, float]]:
        ranked = [
            (item, _cosine_similarity(query_vector, item.vector))
            for item in self.items
        ]
        ranked.sort(key=lambda pair: (-pair[1], pair[0].manifest.item_id))
        return ranked[:top_k]


class FaissVectorIndex(NumpyVectorIndex):
    name = "faiss"

    def __init__(self, items: list[SearchIndexItem]):
        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("FAISS backend requested but faiss is not installed.") from exc
        super().__init__(items)
        if not items:
            self._faiss_index = None
            self._ordered_items = []
            return
        matrix = np.stack([item.vector.astype("float32") for item in items])
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self._faiss_index = index
        self._ordered_items = list(items)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[SearchIndexItem, float]]:
        if not getattr(self, "_ordered_items", None):
            return []
        vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        try:
            import faiss  # type: ignore
        except ImportError:  # pragma: no cover - guarded by __init__
            return super().search(query_vector, top_k)
        faiss.normalize_L2(vector)
        scores, indices = self._faiss_index.search(vector, min(top_k, len(self._ordered_items)))
        ranked: list[tuple[SearchIndexItem, float]] = []
        for score, item_index in zip(scores[0], indices[0], strict=True):
            if item_index < 0:
                continue
            ranked.append((self._ordered_items[int(item_index)], float(score)))
        ranked.sort(key=lambda pair: (-pair[1], pair[0].manifest.item_id))
        return ranked


def _load_manifest_items(path: Path) -> list[RetrievalManifestItem]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        raw_items = payload.get("items", payload)
        return [RetrievalManifestItem.from_dict(item) for item in raw_items]
    items: list[RetrievalManifestItem] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(RetrievalManifestItem.from_dict(json.loads(line)))
    return items


def _index_path_candidates(model_dir: Path, index_filename: str) -> list[Path]:
    if model_dir.is_file():
        return [model_dir]
    return [
        model_dir / index_filename,
        model_dir / "retrieval_index.json",
        model_dir / "retrieval_index.jsonl",
    ]


def resolve_index_path(model_dir: str | Path, index_filename: str) -> Path | None:
    root = Path(model_dir)
    for candidate in _index_path_candidates(root, index_filename):
        if candidate.exists():
            return candidate
    return None


def create_embedding_backend(name: str, dimension: int = 16) -> EmbeddingBackend:
    if name == "fake":
        return FakeDeterministicEmbeddingBackend(dimension=dimension)
    if name == "numpy":
        return NumpyBaselineEmbeddingBackend(dimension=dimension)
    raise ValueError(f"Unknown embedding backend: {name}")


def vector_store_name(prefer_faiss: bool) -> str:
    if not prefer_faiss:
        return "numpy"
    try:
        import faiss  # noqa: F401  # pragma: no cover - optional dependency
    except ImportError:
        return "numpy"
    return "faiss"


def build_static_manifest_item(
    payload: dict[str, Any],
    embedding_backend: EmbeddingBackend,
) -> RetrievalManifestItem:
    metadata = _normalize_metadata(payload.get("metadata"))
    metadata.setdefault("sensor", payload.get("sensor"))
    metadata.setdefault("source", payload.get("source"))
    metadata.setdefault("region_id", payload.get("region_id"))
    metadata.setdefault("transition_label", payload.get("transition_label"))
    signature = _text_signature(
        payload.get("item_id"),
        payload.get("image_path"),
        payload.get("text"),
        metadata,
    )
    vector = payload.get("vector")
    if vector is None:
        vector = embedding_backend.embed_signature(signature).tolist()
    mode = RetrievalMode(payload.get("mode", RetrievalMode.STATIC_REGION.value))
    return RetrievalManifestItem(
        item_id=str(payload["item_id"]),
        mode=mode,
        vector=[float(value) for value in vector],
        metadata={
            **metadata,
            "image_path": payload.get("image_path"),
            "thumbnail_path": payload.get("thumbnail_path", payload.get("image_path")),
        },
    )


def _build_static_query_signature(query: RetrievalQuery) -> str:
    return _text_signature(
        query.item_id,
        query.image_path,
        query.text,
        query.filters,
        query.metadata,
    )


def _rerank_score(query_metadata: dict[str, Any], item_metadata: dict[str, Any]) -> float:
    bonus = 0.0
    if query_metadata.get("month") is not None and query_metadata.get("month") == item_metadata.get("month"):
        bonus += 0.06
    elif query_metadata.get("season") and query_metadata.get("season") == item_metadata.get("season"):
        bonus += 0.03
    if query_metadata.get("sensor") and query_metadata.get("sensor") == item_metadata.get("sensor"):
        bonus += 0.05
    if query_metadata.get("region_id") and query_metadata.get("region_id") == item_metadata.get("region_id"):
        bonus += 0.04
    if query_metadata.get("geography") and query_metadata.get("geography") == item_metadata.get("geography"):
        bonus += 0.04
    if query_metadata.get("transition_label") and query_metadata.get("transition_label") == item_metadata.get("transition_label"):
        bonus += 0.05
    return bonus


class StaticRegionPrithviBackend(RetrievalBackend):
    index_filename = "static_region_index.json"

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
            if item.mode in {RetrievalMode.STATIC_REGION, RetrievalMode.TRANSITION_CONDITIONED}
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
        query_vector = self.embedding_backend.embed_signature(_build_static_query_signature(query))
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
                    before_image_path=metadata.get("image_path"),
                    after_image_path=None,
                    region=metadata.get("region"),
                    transition_hint=metadata.get("transition_label"),
                )
            )
        result = RetrievalResult(
            query=query,
            items=items,
            backend_name="static_region_prithvi",
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
