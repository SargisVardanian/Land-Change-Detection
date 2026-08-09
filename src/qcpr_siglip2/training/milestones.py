"""Required checkpoint and evaluation milestones for SigLIP-2 phases."""

from __future__ import annotations

import math
from pathlib import Path


def steps_for_exposure(
    presentations_per_pair: int, *, unique_pairs: int, logical_batch_size: int
) -> int:
    """Convert a physical-pair exposure budget into optimizer steps."""

    if min(presentations_per_pair, unique_pairs, logical_batch_size) <= 0:
        raise ValueError("exposure, unique pair count and logical batch must be positive")
    return math.ceil(presentations_per_pair * unique_pairs / logical_batch_size)


def required_milestones(
    phase: str, *, unique_pairs: int, logical_batch_size: int
) -> tuple[int, ...]:
    """Return dynamic full-ranking milestones measured by pair exposure."""

    normalized = phase.upper()
    if normalized == "A":
        exposures = (8,)
    elif normalized == "B":
        exposures = (12, 16, 20, 24)
    else:
        raise ValueError(f"unsupported SigLIP-2 phase: {phase}")
    return tuple(
        steps_for_exposure(
            exposure,
            unique_pairs=unique_pairs,
            logical_batch_size=logical_batch_size,
        )
        for exposure in exposures
    )


def milestone_checkpoint_path(run_root: Path, step: int) -> Path:
    """Return the canonical path for a milestone checkpoint."""

    if step < 0:
        raise ValueError("milestone step must be non-negative")
    return run_root / "milestones" / f"checkpoint_step_{step}.pt"


def milestone_evaluation_path(run_root: Path, step: int) -> Path:
    """Return the canonical output directory for a milestone evaluation."""

    if step < 0:
        raise ValueError("milestone step must be non-negative")
    return run_root / "milestone_evaluations" / f"step_{step}"
