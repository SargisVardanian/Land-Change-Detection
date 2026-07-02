from __future__ import annotations

import json
import math
import sys
import argparse
import hashlib
from pathlib import Path


def _file_fingerprint(path: str | Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_row_count(path: str | Path) -> int:
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def _manifest_dataset_names(path: str | Path) -> set[str]:
    names: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            names.add(str(json.loads(line).get("dataset_name", "")))
    return names


def _parse_weight(value: str) -> tuple[str, float]:
    name, raw = value.split("=", 1)
    return name, float(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoke_report", type=Path)
    parser.add_argument("memory_report", type=Path)
    parser.add_argument("current_commit")
    parser.add_argument("world_size", type=int)
    parser.add_argument("minimum_global_batch", type=int)
    parser.add_argument("expected_temporal_depth", type=int, nargs="?", default=6)
    parser.add_argument("--expected-data-mode", choices=("levir_only", "mixed"), default="levir_only")
    parser.add_argument("--expected-train-manifest", action="append", default=[])
    parser.add_argument("--expected-val-manifest", action="append", default=[])
    parser.add_argument("--expected-dataset-weight", action="append", default=[])
    parser.add_argument("--expected-dataset-config", type=Path, default=None)
    return parser.parse_args()


def _expected_from_args(args: argparse.Namespace) -> dict[str, object]:
    train_manifests = [str(path) for path in args.expected_train_manifest]
    val_manifests = [str(path) for path in args.expected_val_manifest]
    weights = dict(_parse_weight(value) for value in args.expected_dataset_weight)
    if args.expected_dataset_config:
        payload = json.loads(args.expected_dataset_config.read_text(encoding="utf-8"))
        train_manifests = [str(path) for path in payload.get("train_manifests", train_manifests)]
        val_manifests = [str(path) for path in payload.get("val_manifests", val_manifests)]
        weights = {str(key): float(value) for key, value in payload.get("dataset_sampling_weights", weights).items()}
    dataset_names: set[str] = set()
    for path in [*train_manifests, *val_manifests]:
        dataset_names.update(_manifest_dataset_names(path))
    return {
        "train_manifests": train_manifests,
        "val_manifests": val_manifests,
        "manifest_fingerprints": {
            "train": {path: _file_fingerprint(path) for path in train_manifests},
            "validation": {path: _file_fingerprint(path) for path in val_manifests},
        },
        "dataset_names": sorted(dataset_names),
        "dataset_weights": weights if args.expected_data_mode == "mixed" else {},
        "train_row_count": sum(_manifest_row_count(path) for path in train_manifests),
        "validation_row_count": sum(_manifest_row_count(path) for path in val_manifests),
    }


def main() -> int:
    args = parse_args()
    smoke_path = args.smoke_report
    memory_path = args.memory_report
    current_commit = args.current_commit
    world_size = args.world_size
    minimum_global_batch = args.minimum_global_batch
    expected_temporal_depth = args.expected_temporal_depth
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
        "temporal_depth": expected_temporal_depth,
        "text_adapter_enabled": True,
    }
    errors = [
        f"smoke {key}={smoke.get(key)!r}"
        for key, expected in smoke_checks.items()
        if smoke.get(key) != expected
    ]
    if "H100" not in smoke.get("gpu_name", ""):
        errors.append("Stage-1-next smoke did not run on H100")
    if smoke.get("data_mode", "levir_only") != args.expected_data_mode:
        errors.append(f"smoke data_mode={smoke.get('data_mode')!r}; expected {args.expected_data_mode!r}")
    if args.expected_data_mode == "mixed":
        expected = _expected_from_args(args)
        for key in ("manifest_fingerprints", "dataset_names", "dataset_weights", "train_row_count", "validation_row_count"):
            if smoke.get(key) != expected[key]:
                errors.append(f"smoke {key}={smoke.get(key)!r}; expected {expected[key]!r}")
        if smoke.get("mixed_smoke") is not True:
            errors.append("mixed training requires a smoke report with mixed_smoke=true")
        if len(smoke.get("dataset_names", [])) < 2:
            errors.append("mixed smoke did not include at least two datasets")
    memory_checks = {
        "status": "PASS",
        "git_commit": current_commit,
        "stage1_next": True,
        "loss": "multi_positive_set_info_nce",
        "stable_caption_groups": True,
        "use_direction_embeddings": True,
        "use_explicit_change_fusion": True,
        "trainable_temperature": True,
        "temporal_depth": expected_temporal_depth,
        "text_adapter_enabled": True,
    }
    errors.extend(
        f"memory {key}={memory.get(key)!r}"
        for key, expected in memory_checks.items()
        if memory.get(key) != expected
    )
    if memory.get("memory_data_mode") != "shape_probe":
        errors.append(f"memory memory_data_mode={memory.get('memory_data_mode')!r}; expected 'shape_probe'")
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
