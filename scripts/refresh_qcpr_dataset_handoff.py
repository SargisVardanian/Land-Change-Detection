#!/usr/bin/env python3
"""Refresh mutable shared handoff contracts from an immutable release handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any], *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-handoff", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--model-requirements", type=Path, required=True)
    parser.add_argument("--model-branch")
    parser.add_argument("--model-sha")
    parser.add_argument("--dataset-code-sha", required=True)
    parser.add_argument("--dataset-remote-sha")
    parser.add_argument("--coordination-status", type=Path)
    parser.add_argument("--forest-extension-status", type=Path)
    args = parser.parse_args()
    if bool(args.model_branch) != bool(args.model_sha):
        parser.error("--model-branch and --model-sha must be provided together")

    handoff = _read(args.release_handoff)
    release_name = str(handoff.get("release_name") or "")
    if not release_name.endswith("r19g"):
        parser.error("release handoff is not authoritative r19g")
    requirements_sha = _sha256(args.model_requirements)
    coordination: dict[str, Any] = {}
    if args.coordination_status:
        coordination = _read(args.coordination_status)
        if coordination.get("release") != release_name:
            parser.error("coordination status does not target authoritative r19g")
        if coordination.get("immutable_release_mutated") is not False:
            parser.error("coordination status does not preserve immutable r19g")

    coordination_fields: dict[str, Any] = {}
    if coordination:
        coordination_fields = {
            "release_content_sha256": coordination["release_content_sha256"],
            "MULTIPOSITIVE_SMOKE_READY": coordination["MULTIPOSITIVE_SMOKE_READY"],
            "multipositive_compatibility_artifact": coordination["multipositive_artifact"],
            "HIGHRES_RUNTIME_STRESS_READY": coordination["HIGHRES_RUNTIME_STRESS_READY"],
            "highres_physical_stress_manifest": coordination["highres_physical_manifest"],
            "VERIFIED_CROSS_SPLIT_LEAKAGE_COUNT": coordination[
                "VERIFIED_CROSS_SPLIT_LEAKAGE_COUNT"
            ],
            "UNRESOLVED_HIGH_RISK_DUPLICATES": coordination[
                "UNRESOLVED_HIGH_RISK_DUPLICATES"
            ],
            "perceptual_cross_split_confirmation": coordination[
                "perceptual_confirmation_artifact"
            ],
            "FOREST_HUMAN_CAPTION_IDENTIFIED": coordination["FOREST_HUMAN_TRAIN_READY"],
            "FOREST_HUMAN_TRAIN_READY": coordination["FOREST_HUMAN_TRAIN_READY"],
            "forest_human_only_candidate": coordination["forest_human_candidate"],
            "DUBAI_EXTERNAL_READY": coordination["DUBAI_EXTERNAL_READY"],
            "dubai_external_status": coordination["dubai_status"],
            "AUTHORIZE_TEMPORALSIGLIP_CORE_TRAINING": coordination[
                "AUTHORIZE_TEMPORALSIGLIP_CORE_TRAINING"
            ],
            "AUTHORIZE_TEMPORALSIGLIP_DOMAIN_EXPANDED_ABLATION": coordination[
                "AUTHORIZE_TEMPORALSIGLIP_DOMAIN_EXPANDED_ABLATION"
            ],
            "r19g_remains_valid": coordination["r19g_remains_valid"],
            "coordination_followup_status": {
                "path": str(args.coordination_status),
                "sha256": _sha256(args.coordination_status),
            },
        }
        dubai_status_path = Path(coordination["dubai_status"]["path"])
        if dubai_status_path.is_file():
            coordination_fields["dubai_external_blocker"] = _read(dubai_status_path).get(
                "blocker"
            )

    extension_fields: dict[str, Any] = {}
    if args.forest_extension_status:
        extension = _read(args.forest_extension_status)
        if extension.get("core_release") != release_name:
            parser.error("Forest extension does not target authoritative r19g")
        if extension.get("core_unchanged") is not True:
            parser.error("Forest extension does not preserve immutable r19g")
        extension_fields = {
            "FOREST_HUMAN_TRAIN_READY": extension["FOREST_HUMAN_TRAIN_READY"],
            "FOREST_LICENSE_STATUS": extension["FOREST_LICENSE_STATUS"],
            "FOREST_USE_SCOPE": extension["FOREST_USE_SCOPE"],
            "CORE_PLUS_FOREST_HUMAN_TRAIN": extension["CORE_PLUS_FOREST_HUMAN_TRAIN"],
            "FOREST_HUMAN_TRAIN": extension["FOREST_HUMAN_TRAIN"],
            "FOREST_FROZEN_DOMAIN_SHIFT_EVAL": extension[
                "FOREST_FROZEN_DOMAIN_SHIFT_EVAL"
            ],
            "FOREST_PROVENANCE_LICENSE_AUDIT": extension[
                "FOREST_PROVENANCE_LICENSE_AUDIT"
            ],
            "FOREST_SAMPLER_PROPOSAL": extension["FOREST_SAMPLER_PROPOSAL"],
            "SOURCE_SHORTCUT_CONTROL_METADATA": extension[
                "SOURCE_SHORTCUT_CONTROL_METADATA"
            ],
            "AUTHORIZE_TEMPORALSIGLIP_DOMAIN_EXPANDED_ABLATION": extension[
                "AUTHORIZE_TEMPORALSIGLIP_DOMAIN_EXPANDED_ABLATION"
            ],
            "domain_expanded_ablation_authorization_scope": extension[
                "authorization_scope"
            ],
            "HIGHRES_PHYSICAL_READY": extension["HIGHRES_PHYSICAL_READY"],
            "HIGHRES_RUNTIME_STRESS_READY": extension["HIGHRES_RUNTIME_STRESS_READY"],
            "HIGHRES_EVAL_READY": extension["HIGHRES_EVAL_READY"],
            "HIGHRES_TRAIN_READY": extension["HIGHRES_TRAIN_READY"],
            "DUBAI_EXTERNAL_READY": extension["DUBAI_EXTERNAL_READY"],
            "DUBAI_TRAIN_READY": extension["DUBAI_TRAIN_READY"],
            "forest_extension_status": {
                "path": str(args.forest_extension_status),
                "sha256": _sha256(args.forest_extension_status),
            },
        }
    model_fields = {}
    if args.model_branch and args.model_sha:
        model_fields = {
            "model_agent_branch": args.model_branch,
            "model_agent_sha": args.model_sha,
        }
    handoff.update(
        {
            "dataset_code_sha": args.dataset_code_sha,
            "dataset_source_head_sha": args.dataset_code_sha,
            "dataset_cluster_branch_sha": args.dataset_code_sha,
            "dataset_remote_sha": args.dataset_remote_sha,
            "dataset_remote_verification": (
                "VERIFIED" if args.dataset_remote_sha else "NOT_AVAILABLE_GITHUB_SSH_AUTH_REQUIRED"
            ),
            "model_requirements_sha256": requirements_sha,
            "compatibility_authority": "model_requirements_sha256",
            "shared_handoff_refresh_only": True,
            "immutable_release_mutated": False,
            **coordination_fields,
            **extension_fields,
            **model_fields,
        }
    )

    root_handoff = args.shared_root / "dataset_final_to_model.json"
    nested_handoff = args.shared_root / "handoff" / "dataset_final_to_model.json"
    jsonl_handoff = args.shared_root / "handoff" / "dataset_to_model.jsonl"
    _write(root_handoff, handoff)
    _write(nested_handoff, handoff)
    _write(jsonl_handoff, handoff, compact=True)
    handoff_sha = _sha256(root_handoff)

    for name in ("dataset_status.json", "dataset_capabilities.json"):
        path = args.shared_root / name
        value = _read(path)
        value["dataset_code_sha"] = args.dataset_code_sha
        value["dataset_cluster_branch_sha"] = args.dataset_code_sha
        value["dataset_remote_sha"] = args.dataset_remote_sha
        value["dataset_remote_verification"] = (
            "VERIFIED" if args.dataset_remote_sha else "NOT_AVAILABLE_GITHUB_SSH_AUTH_REQUIRED"
        )
        if name == "dataset_status.json":
            value["dataset_head_sha"] = args.dataset_code_sha
        value["model_requirements_sha256"] = requirements_sha
        value["compatibility_authority"] = "model_requirements_sha256"
        value.update(coordination_fields)
        value.update(extension_fields)
        value["handoff_path"] = "contracts/qcpr_shared/handoff/dataset_final_to_model.json"
        value["handoff_sha256"] = handoff_sha
        value["handoff_schema_complete"] = True
        value["immutable_release_mutated"] = False
        _write(path, value)

    print(
        json.dumps(
            {
                "dataset_code_sha": args.dataset_code_sha,
                "handoff_sha256": handoff_sha,
                "model_requirements_sha256": requirements_sha,
                "release_name": release_name,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
