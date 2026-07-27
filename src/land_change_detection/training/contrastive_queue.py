from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class QueueBatch:
    embeddings: Tensor
    labels: list[str]


class ContrastiveQueue:
    def __init__(self, capacity: int = 8192, dim: int = 512, device: str | torch.device = "cpu"):
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.device = torch.device(device)
        self._embeddings = torch.empty((0, dim), device=self.device)
        self._labels: list[str] = []

    def __len__(self) -> int:
        return len(self._labels)

    def enqueue(self, embeddings: Tensor, labels: list[str]) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected embeddings [N,{self.dim}], got {tuple(embeddings.shape)}")
        if embeddings.shape[0] != len(labels):
            raise ValueError("labels length must match embeddings")
        normalized = F.normalize(embeddings.detach().to(self.device), dim=-1)
        self._embeddings = torch.cat([self._embeddings, normalized], dim=0)[-self.capacity :]
        self._labels = (self._labels + [str(label) for label in labels])[-self.capacity :]

    def snapshot(self) -> QueueBatch:
        return QueueBatch(embeddings=self._embeddings.clone(), labels=list(self._labels))

    def clear(self) -> None:
        self._embeddings = torch.empty((0, self.dim), device=self.device)
        self._labels = []
