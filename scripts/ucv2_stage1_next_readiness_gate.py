from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> int:
    smoke_path = Path(sys.argv[1])
    memory_path = Path(sys.argv[2])
    current_commit = sys.argv[3]
    world_size = int(sys.argv[4])
    minimum_global_batch = int(sys.argv[5])
    if not smoke_path.exists() or not memory_path.exists():
        raise SystemExit("Missing Stage-1-next smoke or memory report")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    smoke_checks = {
        "real_cluster_smoke_passed": True,
        "git_commit": current_commit,
        "steps_completed": 10,
        "finite_loss": True,
        "device_type": "cuda",
        "bf16_active": True,
        "fake_backbones": False,
        "train_val_disjoint": True,
        "frozen_grad_violations": [],
        "missing_gradients": [],
        "checkpoint_roundtrip_passed": True,
        "image_size": 256,
        "output_grid": 32,
        "stage1_next": True,
        "loss": "multi_positive_set_info_nce",
        "stable_caption_groups": True,
        "use_direction_embeddings": True,
        "use_explicit_change_fusion": True,
        "trainable_temperature": True,
    }
    errors = [
        f"smoke {key}={smoke.get(key)!r}"
        for key, expected in smoke_checks.items()
        if smoke.get(key) != expected
    ]
    if "H100" not in smoke.get("gpu_name", ""):
        errors.append("Stage-1-next smoke did not run on H100")
    memory_checks = {
        "status": "PASS",
        "git_commit": current_commit,
        "stage1_next": True,
        "loss": "multi_positive_set_info_nce",
        "stable_caption_groups": True,
        "use_direction_embeddings": True,
        "use_explicit_change_fusion": True,
        "trainable_temperature": True,
    }
    errors.extend(
        f"memory {key}={memory.get(key)!r}"
        for key, expected in memory_checks.items()
        if memory.get(key) != expected
    )
    recommended = int(memory.get("recommended_batch_size") or 0)
    required_local = math.ceil(minimum_global_batch / world_size)
    if recommended < required_local:
        errors.append(f"required local batch {required_local} did not fit; probe recommended {recommended}")
    if errors:
        raise SystemExit("UniChange v2 Stage-1-next readiness gate failed: " + "; ".join(errors))
    print(required_local)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
