from __future__ import annotations

import math

from .manifests import RetrievalManifestItem


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same length.")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class InMemoryVectorStore:
    def __init__(self, items: list[RetrievalManifestItem] | None = None):
        self.items = items or []

    def add(self, item: RetrievalManifestItem) -> None:
        self.items.append(item)

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[RetrievalManifestItem, float]]:
        ranked = [
            (item, cosine_similarity(query_vector, item.vector))
            for item in self.items
        ]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]
