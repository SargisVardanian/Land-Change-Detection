from __future__ import annotations

import json
from pathlib import Path

from scripts.collect_qcpr_slurm_completion import _artifact_inventory, main
from scripts.enrich_qcpr_slurm_completion import main as enrich_main


def test_completion_inventory_is_deterministic_and_excludes_self(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "completion.json").write_text("old\n", encoding="utf-8")
    (tmp_path / "completion.json.tmp").write_text("partial\n", encoding="utf-8")
    records = _artifact_inventory(tmp_path)
    assert [record["path"] for record in records] == ["a.txt", "b.txt"]
    assert all(len(str(record["sha256"])) == 64 for record in records)


def test_compute_collector_is_idempotent_and_never_requires_accounting(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "metrics.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect",
            "--job-id",
            "210239",
            "--collector-job-id",
            "210240",
            "--run-root",
            str(tmp_path),
            "--expected-code-sha",
            "a" * 40,
            "--required-artifact",
            "metrics.json",
        ],
    )
    assert main() == 0
    first = (tmp_path / "completion.json").read_bytes()
    assert main() == 0
    assert (tmp_path / "completion.json").read_bytes() == first
    payload = json.loads(first)
    assert payload["accounting_status"] == "NOT_AVAILABLE_ON_COMPUTE"
    assert payload["required_artifacts_complete"] is True


def test_login_enrichment_calls_sacct_once(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "completion.json").write_text(
        json.dumps(
            {
                "schema_version": "qcpr-slurm-completion-v2",
                "upstream_job_id": "210239",
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_check_output(command, text):
        calls.append((command, text))
        return "210239|COMPLETED|0:0|00:00:38|gpu03|\n"

    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    monkeypatch.setattr(
        "sys.argv",
        ["enrich", "--run-root", str(tmp_path), "--sacct-bin", "sacct"],
    )
    assert enrich_main() == 0
    assert len(calls) == 1
    payload = json.loads(
        (tmp_path / "completion_accounting.json").read_text(encoding="utf-8")
    )
    assert payload["accounting_status"] == "RESOLVED_ON_LOGIN"
    assert payload["exit_code"] == "0:0"
