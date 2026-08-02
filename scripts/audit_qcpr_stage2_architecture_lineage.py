#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--old-report", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    p.add_argument("--code-sha", required=True)
    args = p.parse_args()
    import torch
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone = dict(payload.get("backbone_config") or {})
    old = json.loads(args.old_report.read_text(encoding="utf-8"))
    old_current = dict(old.get("encoders", {}).get("current_jina_universat", {}))
    legacy = args.old_report.with_name("architecture_screening_frozen_report_legacy_labels.json")
    if not legacy.exists():
        legacy.write_text(args.old_report.read_text(encoding="utf-8"), encoding="utf-8")
    current = dict(old_current)
    meta = dict(current.get("metadata", {}))
    meta.update({
        "architecture_label_correction": {
            "old_label": "current_jina_universat / B0",
            "new_label": "B1_legacy_framewise_explicit_fusion",
            "reason": "checkpoint config and factory use SequenceUniverSatEncoder, not JointUniverSatSeriesEncoder",
        },
        "visual_contract": "framewise_universat_then_temporal_change_encoder",
        "temporal_depth": backbone.get("temporal_depth"),
        "use_explicit_change_fusion": backbone.get("use_explicit_change_fusion"),
        "use_direction_embeddings": backbone.get("use_direction_embeddings"),
        "native_joint_temporal": False,
        "text_conditioned_temporal_fusion": False,
    })
    current["metadata"] = meta
    current["architecture_label"] = "B1_legacy_framewise_explicit_fusion"
    current["score_contract"] = "learned_qcpr_global_pair_embedding_from_legacy_framewise_checkpoint"
    encoders = dict(old.get("encoders", {}))
    encoders.pop("current_jina_universat", None)
    encoders["B1_legacy_framewise_explicit_fusion"] = current
    report = {
        "schema_version": "qcpr-stage2-architecture-screen-v2",
        "status": "PARTIAL_SCREEN_LABEL_CORRECTED_NATIVE_B0_AND_B2_PENDING",
        "code_sha": args.code_sha,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "old_report_preserved": str(legacy),
        "common_manifest": old.get("common_manifest"),
        "common_manifest_sha256": old.get("common_manifest_sha256"),
        "pair_count": old.get("pair_count"),
        "query_count": old.get("query_count"),
        "frozen_only": True,
        "full_finetuning": False,
        "architecture_contract": {
            "B0_native_joint_temporal": {
                "required": "JointUniverSatSeriesEncoder -> identity residual token adapters -> PAIR cross-attention -> retrieval projection",
                "checkpoint_available": False, "screened": False,
            },
            "B1_legacy_framewise_explicit_fusion": {
                "checkpoint_available": True, "screened": True,
                "actual_factory": "SequenceUniverSatEncoder -> TemporalChangeEncoder",
                "explicit_change_fusion": bool(backbone.get("use_explicit_change_fusion")),
            },
            "B2_text_conditioned_temporal_fusion": {
                "checkpoint_available": False, "screened": False,
            },
        },
        "encoders": encoders,
        "screened_architectures": [
            "B1_legacy_framewise_explicit_fusion",
            "B3_zero_shot_delta", "B4_zero_shot_delta", "SigLIP2_zero_shot_delta",
        ],
        "not_screened": [
            {"id": "B0_native_joint_temporal", "reason": "no trained checkpoint for the Stage-2 native joint architecture; do not reuse the legacy framewise checkpoint"},
            {"id": "B2_text_conditioned_temporal_fusion", "reason": "no trained/frozen text-conditioned temporal-fusion projection weights"},
        ],
        "selection": {
            "anchor": None, "alternative": None,
            "reason": "the previous anchor label was scientifically incorrect; select only after native B0 is trained/evaluated or an explicitly approved compatible checkpoint is supplied",
        },
        "scientific_caveats": [
            "The prior current_jina_universat metrics are retained under the corrected B1 legacy label.",
            "B3/B4/SigLIP2 remain zero-shot signed-delta diagnostics and are not compatible learned replacements.",
            "Exact metrics are diagnostic because the common benchmark uses multi-positive semantic teacher relevance.",
            "No Stage-2 native B0 or B2 result is available.",
        ],
        "legacy_report_sha256": sha256(legacy),
        "source_report_sha256": sha256(args.old_report),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = "\n".join([
        "# Corrected frozen architecture screening", "",
        "- Status: " + report["status"],
        "- Code SHA: " + args.code_sha,
        "- Legacy report preserved at: " + str(legacy), "",
        "## Correction", "",
        "The previous report labelled the learned checkpoint as native joint B0.",
        "Its factory actually uses framewise UniverSat features followed by TemporalChangeEncoder.",
        "The metrics are therefore retained as B1_legacy_framewise_explicit_fusion.", "",
        "## Remaining required screens", "",
        "- B0 native joint temporal + residual PAIR adapter: no trained checkpoint available.",
        "- B2 text-conditioned temporal fusion: no trained/frozen projection weights available.", "",
        "B3/B4/SigLIP2 remain zero-shot diagnostics and cannot be promoted to architecture replacements.",
    ]) + "\n"
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md, encoding="utf-8")
    print(json.dumps({"output": str(args.output_json), "status": report["status"]}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
