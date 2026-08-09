from __future__ import annotations

from scripts.build_qcpr_coordination_followup import stable_sample


def test_stable_sample_is_deterministic() -> None:
    rows = [{"id": value} for value in ("c", "a", "b")]
    assert stable_sample(rows, 2, "id") == stable_sample(reversed(rows), 2, "id")
