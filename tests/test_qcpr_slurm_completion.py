from __future__ import annotations

from pathlib import Path

from scripts.collect_qcpr_slurm_completion import _artifact_inventory


def test_completion_inventory_is_deterministic_and_excludes_self(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "completion.json").write_text("old\n", encoding="utf-8")
    (tmp_path / "completion.json.tmp").write_text("partial\n", encoding="utf-8")
    records = _artifact_inventory(tmp_path)
    assert [record["path"] for record in records] == ["a.txt", "b.txt"]
    assert all(len(str(record["sha256"])) == 64 for record in records)
