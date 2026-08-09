"""Safety tests for the immutable-release validator CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_validator_rejects_output_inside_release(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    release = tmp_path / "immutable_release"
    release.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/validate_qcpr_release_contract.py"),
            "--release",
            str(release),
            "--output",
            str(release / "audits" / "result.json"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "immutable releases are read-only" in result.stderr
    assert not (release / "audits" / "result.json").exists()
