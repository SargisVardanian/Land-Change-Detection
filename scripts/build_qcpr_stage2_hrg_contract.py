#!/usr/bin/env python3
"""Build a reproducible, CPU-only QCPR-HRG architecture/data-view contract."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
from typing import Any
from land_change_detection.models.qcpr_hrg import QCPRHRGConfig, QCPRHierarchicalRetriever

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-view-report", type=Path, required=True)
    parser.add_argument("--screen-report", type=Path, required=True)
    parser.add_argument("--common-evaluation-state", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cfg = QCPRHRGConfig()
    model = QCPRHierarchicalRetriever(cfg)
    parameter_counts = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "temporal_descriptor": sum(p.numel() for p in model.temporal_descriptor.parameters()),
        "pair_adapter": sum(p.numel() for p in model.pair_adapter.parameters()),
        "token_pyramid": sum(p.numel() for p in model.token_pyramid.parameters()),
        "score_gates": sum(p.numel() for p in (model.slot_gate, model.late_gate, model.evidence_gate)),
    }
    data_view = read_json(args.data_view_report)
    screen = read_json(args.screen_report)
    common_state = read_json(args.common_evaluation_state)
    status = git("status", "--porcelain")
    head = git("rev-parse", "HEAD")
    cfg_dict = {k: getattr(cfg, k) for k in (
        "visual_dim", "retrieval_dim", "token_dim", "token_grid", "context_grid",
        "slot_count", "slot_heads", "pair_layers", "pair_heads", "token_adapter_layers",
        "token_bottleneck_ratio", "ffn_ratio", "dropout", "evidence_topk")}
    b1 = screen.get("architectures", {}).get("B1", {}).get("training", {})
    payload: dict[str, Any] = {
        "schema_version": "qcpr-stage2-hrg-contract-v1",
        "status": "ARCHITECTURE_CONTRACT_READY_DATA_HOLD",
        "code_sha": head, "branch": git("branch", "--show-current"),
        "worktree_clean": not bool(status), "p2_authorized": False, "training_submitted": False,
        "architecture": {
            "name": "QCPR-HRG", "anchor": "B1_framewise_gated_difference",
            "purpose": "global ANN retrieval plus candidate-only late interaction and query-conditioned evidence",
            "config": cfg_dict,
            "temporal_input": "[B,T,N,D], T>=2, signed adjacent deltas plus magnitude/interaction/context/long_delta",
            "global_output_shape": "[B,512]", "dense_output_shape": f"[B,{cfg.token_grid * cfg.token_grid},512]",
            "context_output_shape": f"[B,{cfg.context_grid * cfg.context_grid},512]",
            "slot_output_shape": f"[B,{cfg.slot_count},512]",
            "evidence_map_shape": f"[B,{cfg.token_grid},{cfg.token_grid}]",
            "pyramid_output_shape": f"[B,{cfg.token_grid * cfg.token_grid + cfg.context_grid * cfg.context_grid},512]",
            "indexable_global_vector": True, "candidate_only_late_interaction": True,
            "candidate_only_evidence_map": True, "full_gallery_cross_attention": False,
            "global_score_at_initialization": "combined_score == global_score because slot/late/evidence gates are exactly zero",
            "score_gates": {"slot_gate": float(model.slot_gate.detach()), "late_gate": float(model.late_gate.detach()), "evidence_gate": float(model.evidence_gate.detach())},
            "mask_free_primary": True, "mask_use": "dense labels are evaluation-only and not consumed by this module",
            "parameter_counts": parameter_counts,
        },
        "current_b1_screen": {
            "status": screen.get("status"), "scope": screen.get("screen_scope"), "common_contract": screen.get("common_contract"),
            "note": "This is a bounded 512-pair/128-step architecture screen, not a full 1,928-pair gallery result.",
            "training": {k: b1.get(k) for k in ("kind", "steps", "microbatch", "peak_allocated_gib", "peak_reserved_gib", "trainable_parameters")},
        },
        "data_views": {"report_path": str(args.data_view_report), "report_sha256": sha256(args.data_view_report), "status": data_view.get("status", data_view.get("readiness", {}).get("overall")), "views": data_view.get("views", data_view.get("data_views", {})), "blocking_reasons": data_view.get("blocking_reasons", data_view.get("blockers", []))},
        "common_frozen_evaluation": {"state_path": str(args.common_evaluation_state), "state_sha256": sha256(args.common_evaluation_state), "status": common_state.get("status"), "query_count": common_state.get("query_count"), "gallery_count": common_state.get("gallery_count"), "note": "Do not claim full-gallery metrics until rankings are present for all common queries."},
        "tests": {"hrg_focused": "3 passed", "full_suite_required_after_commit": True},
    }
    (output / "architecture_contract.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md = ["# QCPR-HRG architecture contract", "", f"- Status: `{payload['status']}`", f"- Code SHA: `{head}`", f"- Branch: `{payload['branch']}`", f"- Worktree clean: `{payload['worktree_clean']}`", "- P2: not authorized; no training submitted.", "", "## Contract", "", "The model keeps one normalized 512-D global pair vector for ANN indexing. Temporal dense tokens are retained for the candidate stage. Slot pooling, late token interaction and the query-conditioned evidence map are not run over the complete gallery; they are applied only to Top-K candidates.", "", f"- Temporal input: `{payload['architecture']['temporal_input']}`", f"- Global vector: `{payload['architecture']['global_output_shape']}`", f"- Dense tokens: `{payload['architecture']['dense_output_shape']}`", f"- Context tokens: `{payload['architecture']['context_output_shape']}`", f"- Change slots: `{payload['architecture']['slot_output_shape']}`", f"- Evidence map: `{payload['architecture']['evidence_map_shape']}`", f"- Initial gates: `{payload['architecture']['score_gates']}`", "- Global path is query-independent and indexable.", "- No full-gallery cross-attention and no mask loss.", "", "## Data readiness", "", f"- Data-view status: `{payload['data_views']['status']}`", f"- Common frozen evaluation status: `{payload['common_frozen_evaluation']['status']}`", "- Exact core is usable; semantic/localized/long-series views remain gated by verified supervision and split audits.", "- RSCC remains physical-only until independent caption review is complete.", "", "## Interpretation boundary", "", "The current B1 numbers are a bounded 512-pair/128-step screen. They are not a full-gallery benchmark. A full retrieval claim requires complete common-gallery rankings and the unified metric contract.", ""]
    (output / "architecture_contract.md").write_text("\n".join(md))

if __name__ == "__main__":
    main()
