from __future__ import annotations

from pathlib import Path

from scripts.collect_qcpr_slurm_completion import _artifact_inventory, _job_record


def test_completion_inventory_is_deterministic_and_excludes_self(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "completion.json").write_text("old\n", encoding="utf-8")
    (tmp_path / "completion.json.tmp").write_text("partial\n", encoding="utf-8")
    records = _artifact_inventory(tmp_path)
    assert [record["path"] for record in records] == ["a.txt", "b.txt"]
    assert all(len(str(record["sha256"])) == 64 for record in records)


def test_job_record_uses_explicit_sacct_binary(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_check_output(command, text):
        observed["command"] = command
        observed["text"] = text
        return "210234|COMPLETED|0:0|01:51:00|gpu03|\n"

    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    record = _job_record("210234", "/opt/slurm/bin/sacct")
    assert observed["command"][0] == "/opt/slurm/bin/sacct"
    assert record["state"] == "COMPLETED"
