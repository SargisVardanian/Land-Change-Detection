from __future__ import annotations

import json

import pytest

import ucv2_progress


def test_progress_payload_has_eta_and_metrics(monkeypatch) -> None:
    monkeypatch.setattr(ucv2_progress.time, "perf_counter", lambda: 15.0)
    payload = ucv2_progress.progress_payload(
        stage="training",
        completed=2,
        total=5,
        started=5.0,
        metrics={"loss": 1.25},
    )
    assert payload["completed_fraction"] == pytest.approx(0.4)
    assert payload["eta_seconds"] == pytest.approx(15.0)
    assert payload["metrics"] == {"loss": 1.25}


def test_write_progress_is_atomic_and_complete_has_zero_eta(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ucv2_progress.time, "perf_counter", lambda: 8.0)
    path = tmp_path / "progress.json"
    ucv2_progress.write_progress(
        path,
        stage="complete",
        completed=3,
        total=3,
        started=2.0,
        complete=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert payload["eta_seconds"] == 0.0
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_progress_counters_are_rejected() -> None:
    with pytest.raises(ValueError, match="0 <= completed <= total"):
        ucv2_progress.progress_payload(stage="bad", completed=2, total=1, started=0.0)
