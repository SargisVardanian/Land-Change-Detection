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


def test_refresh_merges_bounded_coordination_status(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    shared = tmp_path / "shared"
    (shared / "handoff").mkdir(parents=True)
    release = tmp_path / "release_handoff.json"
    release.write_text(
        json.dumps({"release_name": "qcpr_bitemporal_v2_train_20260808_final_r19g"})
    )
    requirements = tmp_path / "model_requirements.json"
    requirements.write_text("{}\n")
    for name in ("dataset_status.json", "dataset_capabilities.json"):
        (shared / name).write_text("{}\n")
    dubai = tmp_path / "dubai.json"
    dubai.write_text(json.dumps({"blocker": "license absent"}))
    artifact = {"path": "/artifact", "sha256": "a" * 64}
    coordination = tmp_path / "coordination.json"
    coordination.write_text(
        json.dumps(
            {
                "release": "qcpr_bitemporal_v2_train_20260808_final_r19g",
                "release_content_sha256": "c" * 64,
                "immutable_release_mutated": False,
                "MULTIPOSITIVE_SMOKE_READY": True,
                "multipositive_artifact": artifact,
                "HIGHRES_RUNTIME_STRESS_READY": True,
                "highres_physical_manifest": artifact,
                "VERIFIED_CROSS_SPLIT_LEAKAGE_COUNT": 0,
                "UNRESOLVED_HIGH_RISK_DUPLICATES": 0,
                "perceptual_confirmation_artifact": artifact,
                "FOREST_HUMAN_TRAIN_READY": True,
                "forest_human_candidate": artifact,
                "DUBAI_EXTERNAL_READY": False,
                "dubai_status": {"path": str(dubai), "sha256": "b" * 64},
                "AUTHORIZE_TEMPORALSIGLIP_CORE_TRAINING": True,
                "AUTHORIZE_TEMPORALSIGLIP_DOMAIN_EXPANDED_ABLATION": False,
                "r19g_remains_valid": True,
            }
        )
    )
    subprocess.run(
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
            "codex/model",
            "--model-sha",
            "1" * 40,
            "--dataset-code-sha",
            "2" * 40,
            "--coordination-status",
            str(coordination),
        ],
        check=True,
    )
    handoff = json.loads((shared / "dataset_final_to_model.json").read_text())
    assert handoff["MULTIPOSITIVE_SMOKE_READY"] is True
    assert handoff["release_content_sha256"] == "c" * 64
    assert handoff["HIGHRES_RUNTIME_STRESS_READY"] is True
    assert handoff["dubai_external_blocker"] == "license absent"
    assert handoff["compatibility_authority"] == "model_requirements_sha256"
    assert handoff["r19g_remains_valid"] is True


def test_refresh_merges_forest_extension_without_rewriting_model_head(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    shared = tmp_path / "shared"
    (shared / "handoff").mkdir(parents=True)
    release = tmp_path / "release_handoff.json"
    release.write_text(
        json.dumps(
            {
                "release_name": "qcpr_bitemporal_v2_train_20260808_final_r19g",
                "model_agent_branch": "frozen/model",
                "model_agent_sha": "f" * 40,
            }
        )
    )
    requirements = tmp_path / "model_requirements.json"
    requirements.write_text("{}\n")
    for name in ("dataset_status.json", "dataset_capabilities.json"):
        (shared / name).write_text("{}\n")
    artifact = {"path": "/artifact", "sha256": "a" * 64}
    extension = tmp_path / "extension.json"
    extension.write_text(
        json.dumps(
            {
                "core_release": "qcpr_bitemporal_v2_train_20260808_final_r19g",
                "core_unchanged": True,
                "FOREST_HUMAN_TRAIN_READY": True,
                "FOREST_LICENSE_STATUS": "PARTIAL",
                "FOREST_USE_SCOPE": "RESEARCH_ONLY_UNTIL_CLARIFIED",
                "CORE_PLUS_FOREST_HUMAN_TRAIN": artifact,
                "FOREST_HUMAN_TRAIN": artifact,
                "FOREST_FROZEN_DOMAIN_SHIFT_EVAL": artifact,
                "FOREST_PROVENANCE_LICENSE_AUDIT": artifact,
                "FOREST_SAMPLER_PROPOSAL": artifact,
                "SOURCE_SHORTCUT_CONTROL_METADATA": artifact,
                "AUTHORIZE_TEMPORALSIGLIP_DOMAIN_EXPANDED_ABLATION": True,
                "authorization_scope": "later matched research-only data ablation",
                "HIGHRES_PHYSICAL_READY": True,
                "HIGHRES_RUNTIME_STRESS_READY": True,
                "HIGHRES_EVAL_READY": False,
                "HIGHRES_TRAIN_READY": False,
                "DUBAI_EXTERNAL_READY": False,
                "DUBAI_TRAIN_READY": False,
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/refresh_qcpr_dataset_handoff.py"),
            "--release-handoff",
            str(release),
            "--shared-root",
            str(shared),
            "--model-requirements",
            str(requirements),
            "--dataset-code-sha",
            "2" * 40,
            "--forest-extension-status",
            str(extension),
        ],
        check=True,
    )
    handoff = json.loads((shared / "dataset_final_to_model.json").read_text())
    assert handoff["model_agent_branch"] == "frozen/model"
    assert handoff["model_agent_sha"] == "f" * 40
    assert handoff["FOREST_LICENSE_STATUS"] == "PARTIAL"
    assert handoff["AUTHORIZE_TEMPORALSIGLIP_DOMAIN_EXPANDED_ABLATION"] is True
    assert handoff["HIGHRES_EVAL_READY"] is False
    assert handoff["HIGHRES_TRAIN_READY"] is False
