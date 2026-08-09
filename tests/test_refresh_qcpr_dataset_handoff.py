"""Tests for the mutable shared-handoff refresh tool."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_refresh_uses_release_and_current_model_contract(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    release = tmp_path / "release.json"
    shared = tmp_path / "shared"
    requirements = tmp_path / "requirements.json"
    release.write_text(
        json.dumps({"release_name": "qcpr_bitemporal_v2_train_20260808_final_r19g"}),
        encoding="utf-8",
    )
    requirements.write_text('{"schema_version":"model-v3"}\n', encoding="utf-8")
    shared.mkdir()
    for name in ("dataset_status.json", "dataset_capabilities.json"):
        (shared / name).write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/refresh_qcpr_dataset_handoff.py"),
            "--release-handoff",
            str(release),
            "--shared-root",
            str(shared),
            "--model-requirements",
            str(requirements),
            "--model-branch",
            "codex/qcpr-model-v3-temporal-retrieval",
            "--model-sha",
            "d59cb90",
            "--dataset-code-sha",
            "6bce613",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    assert result.returncode == 0
    root = json.loads((shared / "dataset_final_to_model.json").read_text())
    nested = json.loads((shared / "handoff/dataset_final_to_model.json").read_text())
    assert root == nested
    assert root["dataset_code_sha"] == "6bce613"
    assert root["dataset_cluster_branch_sha"] == "6bce613"
    assert root["dataset_remote_sha"] is None
    assert root["dataset_remote_verification"] == "NOT_AVAILABLE_GITHUB_SSH_AUTH_REQUIRED"
    assert root["model_agent_sha"] == "d59cb90"
    assert root["model_requirements_sha256"] == hashlib.sha256(
        requirements.read_bytes()
    ).hexdigest()
    assert root["immutable_release_mutated"] is False
