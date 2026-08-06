"""Required checkpoint and evaluation milestones for SigLIP-2 phases."""

from __future__ import annotations

from pathlib import Path


def required_milestones(phase: str) -> tuple[int, ...]:
    """Return the immutable full-ranking milestones for a training phase."""

    normalized = phase.upper()
    if normalized == "A":
        return (0, 128, 256)
    if normalized == "B":
        return (256, 512, 1024, 1792)
    raise ValueError(f"unsupported SigLIP-2 phase: {phase}")


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
