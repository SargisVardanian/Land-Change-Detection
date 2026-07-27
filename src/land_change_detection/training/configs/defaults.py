from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 1e-3
    margin: float = 0.2
    temperature: float = 0.1
