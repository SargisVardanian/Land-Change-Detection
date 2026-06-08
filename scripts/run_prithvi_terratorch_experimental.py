from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Prithvi/TerraTorch experimental training path, with explicit fallback to the 6-band baseline."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--allow-fallback-baseline", action="store_true")
    parser.add_argument("--require-terratorch", action="store_true")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--terratorch-config-path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from land_change_detection.training.prithvi_experimental import (
        build_prithvi_runtime_bundle,
        PrithviExperimentalStatus,
        count_manifest_rows,
        terratorch_available,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "terratorch_env_diagnostics.json"
    available = terratorch_available()
    manifest_rows = count_manifest_rows(args.manifest)
    runtime = build_prithvi_runtime_bundle(
        checkpoint_path=args.checkpoint_path,
        terratorch_config_path=args.terratorch_config_path,
    )
    notes: list[str] = []
    notes.extend(runtime.notes)

    if runtime.runtime_ready:
        notes.append(
            "TerraTorch, config, and checkpoint were all discovered. A real TerraTorch head is still not wired in this repository, "
            "so the runner preserves the resolved runtime contract but executes the current 6-band baseline."
        )
    elif available:
        notes.append(
            "TerraTorch is installed, but the full Prithvi runtime contract is incomplete or the real training head is not wired. "
            "Falling back to the current 6-band semantic baseline while preserving diagnostics."
        )
    else:
        notes.append(
            "TerraTorch is not installed. The experimental runner can only use the current 6-band semantic baseline fallback."
        )

    if args.require_terratorch and not available:
        status = PrithviExperimentalStatus(
            mode="blocked_missing_terratorch",
        terratorch_available=False,
        manifest_rows=manifest_rows,
            output_dir=str(output_dir),
            run_tag="prithvi_terratorch_experimental",
            notes=notes,
            checkpoint_path=runtime.checkpoint_path,
            terratorch_config_path=runtime.terratorch_config_path,
            runtime_ready=runtime.runtime_ready,
        )
        _write_diagnostics(diagnostics_path)
        (output_dir / "prithvi_experimental_status.json").write_text(
            json.dumps(status.to_dict(), indent=2),
            encoding="utf-8",
        )
        raise RuntimeError("TerraTorch is required for this run, but it is not installed.")

    if not args.allow_fallback_baseline and not available:
        status = PrithviExperimentalStatus(
            mode="blocked_no_fallback",
            terratorch_available=False,
            manifest_rows=manifest_rows,
            output_dir=str(output_dir),
            run_tag="prithvi_terratorch_experimental",
            notes=notes,
            checkpoint_path=runtime.checkpoint_path,
            terratorch_config_path=runtime.terratorch_config_path,
            runtime_ready=runtime.runtime_ready,
        )
        _write_diagnostics(diagnostics_path)
        (output_dir / "prithvi_experimental_status.json").write_text(
            json.dumps(status.to_dict(), indent=2),
            encoding="utf-8",
        )
        raise RuntimeError("Fallback baseline is disabled and TerraTorch is unavailable.")

    subprocess.run(
        [
            sys.executable,
            "scripts/run_prithvi_semantic_train_eval.py",
            "--manifest",
            args.manifest,
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--num-classes",
            str(args.num_classes),
        ],
        check=True,
    )

    status = PrithviExperimentalStatus(
        mode="fallback_baseline" if not runtime.runtime_ready else "runtime_resolved_fallback",
        terratorch_available=available,
        manifest_rows=manifest_rows,
        output_dir=str(output_dir),
        run_tag="prithvi_terratorch_experimental",
        notes=notes,
        checkpoint_path=runtime.checkpoint_path,
        terratorch_config_path=runtime.terratorch_config_path,
        runtime_ready=runtime.runtime_ready,
    )
    _write_diagnostics(diagnostics_path)
    (output_dir / "prithvi_experimental_status.json").write_text(
        json.dumps(status.to_dict(), indent=2),
        encoding="utf-8",
    )
    return 0


def _write_diagnostics(path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_terratorch_env.py",
            "--output",
            str(path),
        ],
        check=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
