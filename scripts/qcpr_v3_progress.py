from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping


def write_progress(
    path: str | Path,
    *,
    stage: str,
    completed: int,
    total: int,
    started: float,
    complete: bool = False,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if completed < 0 or total < 0 or completed > total:
        raise ValueError("progress counters must satisfy 0 <= completed <= total")
    elapsed = max(time.perf_counter() - started, 0.0)
    fraction = completed / total if total else 0.0
    eta = elapsed * (total - completed) / completed if completed else None
    payload: dict[str, Any] = {
        "stage": stage,
        "completed": completed,
        "total": total,
        "completed_fraction": fraction,
        "elapsed_seconds": elapsed,
        "eta_seconds": 0.0 if complete else eta,
        "complete": complete,
        "updated_unix_seconds": time.time(),
    }
    if metrics:
        payload["metrics"] = dict(metrics)
    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite progress value for {key}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return payload
