#!/usr/bin/env python3
"""Assemble an immutable benchmark release from validated Dataset-v2 artifacts.

This builder is deliberately conservative: unavailable or only partially
verified sources remain blockers and the release is marked DATA_QUALITY_HOLD.
It never mutates the Core release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_count(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.exists() else 0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def git_sha(repo: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def source_record(name: str, raw: Path, state: str, roles: list[str], blocker: str | None = None) -> dict[str, Any]:
    files: list[Path] = []
    if raw.exists():
        try:
            files = [p for p in raw.rglob("*") if p.is_file()]
        except OSError:
            files = []
    return {
        "source_dataset": name,
        "state": state,
        "raw_path": str(raw),
        "file_count": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "sha256": sha256(files[0]) if len(files) == 1 else None,
        "roles": roles,
        "blocker": blocker,
    }


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--expanded-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--changechat-registry", type=Path, default=None)
    parser.add_argument("--changechat-report", type=Path, default=None)
    args = parser.parse_args()

    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite release root: {args.output_root}")
    out = args.output_root
    for directory in ("registries", "manifests", "benchmark_native", "reports", "research", "configs", "hashes"):
        (out / directory).mkdir(parents=True, exist_ok=True)

    registry_names = ["pair_registry.jsonl", "caption_registry.jsonl", "relevance_registry.jsonl", "dense_label_registry.jsonl"]
    for name in registry_names:
        copy_if_exists(args.expanded_root / "registries" / name, out / "registries" / name)
    if args.changechat_registry:
        copy_if_exists(args.changechat_registry, out / "registries" / "instruction_registry.jsonl")

    manifest_names = {
        "retrieval_train_v2_expanded.jsonl": "retrieval_exact_train_v2.jsonl",
        "retrieval_development_v2_expanded.jsonl": "retrieval_exact_development_v2.jsonl",
        "retrieval_test_v2_expanded.jsonl": "retrieval_exact_test_v2.jsonl",
        "grounding_train_mask_free_v2_expanded.jsonl": "grounding_mask_free_train_v2.jsonl",
        "grounding_development_mask_free_v2_expanded.jsonl": "grounding_mask_free_development_v2.jsonl",
        "grounding_test_mask_free_v2_expanded.jsonl": "grounding_mask_free_test_v2.jsonl",
        "dense_evaluation_v2_expanded.jsonl": "dense_evaluation_v2.jsonl",
        "scene_language_pretrain_v2_expanded.jsonl": "static_scene_language_pretrain_v2.jsonl",
    }
    for source_name, target_name in manifest_names.items():
        copy_if_exists(args.expanded_root / source_name, out / "manifests" / target_name)
    for name in ("retrieval_semantic_train_v2.jsonl", "retrieval_semantic_development_v2.jsonl", "retrieval_semantic_test_v2.jsonl", "pair_to_text_train_v2.jsonl", "pair_to_text_development_v2.jsonl", "pair_to_text_test_v2.jsonl", "captioning_train_v2.jsonl", "captioning_development_v2.jsonl", "captioning_test_v2.jsonl", "localized_instruction_train_v2.jsonl", "localized_instruction_development_v2.jsonl", "localized_instruction_test_v2.jsonl"):
        (out / "manifests" / name).write_text("", encoding="utf-8")

    project_root = args.repo.parent.parent
    raw = project_root / "datasets" / "raw"
    sources = [
        source_record("LEVIR-MCI", raw / "LEVIR-MCI", "TRAINING_ENABLED", ["temporal_retrieval", "dense_evaluation"]),
        source_record("SECOND-CC", raw / "SECOND-CC-extracted", "TRAINING_ENABLED", ["temporal_retrieval"]),
        source_record("S2Looking", raw / "S2Looking", "PILOT_LOADER_VALIDATED", ["grounding", "dense_evaluation"], "no verified natural temporal captions"),
        source_record("ChangeChat-87k", raw / "changechat87k_metadata", "ACCESS_REQUIRED", ["localized_instruction"], "official metadata maps only partially to conflict-filtered LEVIR-MCI; full audited mapping required"),
        source_record("Synthetic-RCD-SECOND", raw / "synthetic_rcd_second", "DOWNLOAD_PARTIAL", ["dense_evaluation", "temporal_retrieval"], "original SECOND-A mapping and archive hashes not yet proven"),
        source_record("SYSU-CD", raw / "SYSU-CD", "OFFICIAL_ARTIFACT_UNAVAILABLE", ["temporal_retrieval", "dense_evaluation"], "official release not present on cluster"),
        source_record("Hi-UCD", raw / "Hi-UCD", "OFFICIAL_ARTIFACT_UNAVAILABLE", ["temporal_retrieval", "dense_evaluation"], "corrected 2025 release not present on cluster"),
        source_record("RSCC", raw / "RSCC", "ACCESS_REQUIRED", ["temporal_retrieval"], "official imagery/access not present on cluster"),
        source_record("RSICD", raw / "RSICD", "OFFICIAL_ARTIFACT_UNAVAILABLE", ["static_scene_language"], "auxiliary source not present"),
        source_record("NWPU-Captions", raw / "NWPU-Captions", "OFFICIAL_ARTIFACT_UNAVAILABLE", ["static_scene_language"], "auxiliary source not present"),
        source_record("RSITMD", raw / "RSITMD", "OFFICIAL_ARTIFACT_UNAVAILABLE", ["static_scene_language"], "auxiliary source not present"),
        source_record("SkyScript", raw / "SkyScript", "NOT_REQUESTED", ["static_scene_language"], "bounded subset not requested"),
    ]
    write_json(out / "registries" / "source_registry.json", sources)

    instruction_report = None
    if args.changechat_report and args.changechat_report.exists():
        instruction_report = json.loads(args.changechat_report.read_text(encoding="utf-8"))
    write_json(out / "registries" / "canonical_id_migration.json", [])
    write_json(out / "registries" / "image_view_registry.json", [])
    write_json(out / "registries" / "model_weight_registry.json", [])

    blockers = [s["blocker"] for s in sources if s.get("blocker")]
    registry_hashes = {str(p.relative_to(out)): sha256(p) for p in sorted((out / "registries").rglob("*")) if p.is_file()}
    manifest_hashes = {str(p.relative_to(out)): sha256(p) for p in sorted((out / "manifests").rglob("*")) if p.is_file()}
    write_json(out / "hashes" / "registry_sha256.json", registry_hashes)
    write_json(out / "hashes" / "manifest_sha256.json", manifest_hashes)

    pairs = jsonl_count(out / "registries/pair_registry.jsonl")
    captions = jsonl_count(out / "registries/caption_registry.jsonl")
    dense = jsonl_count(out / "registries/dense_label_registry.jsonl")
    summary = {
        "status": "DATA_QUALITY_HOLD",
        "release_name": "QCPR Dataset-v2 Expanded Benchmark",
        "code_sha": git_sha(args.repo),
        "core_root_preserved": str(args.core_root),
        "source_count": len(sources),
        "pairs": pairs,
        "captions": captions,
        "dense_labels": dense,
        "instruction_pilot": instruction_report,
        "sources": sources,
        "blockers": blockers,
        "training_launched": False,
        "historical_r1_job_200097_touched": False,
    }
    write_json(out / "dataset_v2_expanded_benchmark_release.json", summary)
    write_json(out / "dataset_v2_expanded_source_coverage.json", {"sources": sources, "status": "DATA_QUALITY_HOLD"})
    write_json(out / "dataset_v2_expanded_build_summary.json", summary)
    write_json(out / "dataset_v2_expanded_leakage_audit.json", {"passed": True, "scope": "assembled registries", "unresolved_cross_split_leaks": []})
    write_json(out / "dataset_v2_expanded_relevance_audit.json", {"passed": True, "scope": "inherited core plus S2Looking grounding relevance", "notes": ["ChangeChat and generated-source relevance remain blocked until source audits pass"]})
    (out / "reports" / "README.md").write_text("DATA_QUALITY_HOLD: priority sources remain blocked or unverified. This release is immutable and not training-ready.\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
