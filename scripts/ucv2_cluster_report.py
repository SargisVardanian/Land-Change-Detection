from __future__ import annotations

import json
from pathlib import Path

import torch

from ucv2_cluster_common import run_metadata


def finalize(output_dir: Path, device_name: str) -> dict:
    report_path = output_dir / "smoke_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    cuda = torch.cuda.is_available() and device_name.startswith("cuda")
    gpu = torch.cuda.get_device_name(0) if cuda else "cpu"
    bf16 = bool(cuda and config.get("use_bf16") and torch.cuda.is_bf16_supported())
    report.update(run_metadata())
    report.update({
        "device_type": "cuda" if cuda else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu,
        "bf16_requested": bool(config.get("use_bf16")),
        "bf16_active": bf16,
        "image_size": config.get("image_size"),
        "output_grid": config.get("output_grid"),
        "batch_size": config.get("batch_size"),
        "temporal_depth": config.get("temporal_depth"),
        "use_direction_embeddings": config.get("use_direction_embeddings"),
        "use_explicit_change_fusion": config.get("use_explicit_change_fusion"),
        "trainable_temperature": config.get("trainable_temperature"),
        "text_adapter_enabled": config.get("use_text_adapter"),
    })
    report["exact_steps_passed"] = report.get("steps_completed") == config.get("max_steps")
    report["real_cluster_smoke_passed"] = bool(
        report.get("status") == "PASS"
        and not report.get("fake_backbones")
        and cuda and bf16 and "H100" in gpu
        and report.get("steps_completed") == 10
        and report.get("finite_loss")
        and report.get("train_val_disjoint")
        and report.get("frozen_grad_violations") == []
        and report.get("missing_gradients") == []
        and report.get("checkpoint_roundtrip_passed")
        and report.get("git_commit") and report.get("slurm_job_id")
    )
    report["cluster_ready"] = report["real_cluster_smoke_passed"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
