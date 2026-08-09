from __future__ import annotations

import json
from pathlib import Path

from qcpr_data.reports.decision_package import summarize, write_markdown


def test_decision_package_is_deterministic_and_writable(tmp_path: Path) -> None:
    value = summarize(
        code_sha="abc",
        branch="codex/test",
        release_path=tmp_path / "release",
        source_registry=[{"source": "x"}],
        items=[{"source": "x", "frames": [{}, {}]}],
        queries=[{"query_scope": "exact", "training_enabled": True}],
        integrity={"passed": True},
        statuses={"exact": "READY"},
    )
    assert value["physical_item_count"] == 1
    assert value["frame_count"] == 2
    assert value["training_query_count"] == 1
    target = tmp_path / "decision.md"
    write_markdown(target, value)
    assert "Integrity passed: `True`" in target.read_text(encoding="utf-8")
    json.dumps(value, sort_keys=True)
